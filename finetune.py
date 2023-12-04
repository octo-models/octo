import copy
import datetime
from functools import partial
import json
import os

from absl import app, flags, logging
import flax
from flax.training import orbax_utils
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_flags, ConfigDict
import numpy as np
import optax
import orbax.checkpoint
import tensorflow as tf
import tqdm
import wandb

from orca.data.dataset import make_single_dataset
from orca.data.utils.text_processing import text_processors
from orca.utils.jax_utils import initialize_compilation_cache
from orca.utils.pretrained_utils import _verify_shapes, PretrainedModel
from orca.utils.train_utils import (
    batched_apply,
    check_config_diff,
    create_optimizer,
    format_name_with_config,
    Timer,
    TrainState,
)
from orca.utils.visualization_lib import Visualizer

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")

default_config_file = os.path.join(
    os.path.dirname(__file__), "experiments/dibya/finetune_config.py:image_conditioned"
)
config_flags.DEFINE_config_file(
    "config",
    default_config_file,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def main(_):
    initialize_compilation_cache()
    devices = jax.devices()
    logging.info(
        f"""
        ORCA Finetuning Script
        ======================
        Pretrained model: {FLAGS.config.pretrained_path}
        Finetuning Dataset: {FLAGS.config.dataset_kwargs.name}
        Data dir: {FLAGS.config.dataset_kwargs.data_dir}
        Task Modality: {FLAGS.config.modality}
        Finetuning Mode: {FLAGS.config.finetuning_mode}

        # Devices: {jax.device_count()}
        Batch size: {FLAGS.config.batch_size} ({FLAGS.config.batch_size // len(devices) } per device)
        # Steps: {FLAGS.config.num_steps}
    """
    )

    #########
    #
    # Setup Jax Data Parallelism
    #
    #########

    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(jax.devices(), axis_names="batch")
    # Our batches will be data-parallel sharded -- each device will get a slice of the batch
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))
    # Our model will be replicated across devices (we are only doing data parallelism, not model parallelism)
    replicated_sharding = NamedSharding(mesh, PartitionSpec())

    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    #########
    #
    # Setup WandB
    #
    #########

    name = format_name_with_config(
        FLAGS.name,
        FLAGS.config.to_dict(),
    )
    wandb_id = "{name}_{time}".format(
        name=name,
        time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    wandb.init(
        config=FLAGS.config.to_dict(),
        id=wandb_id,
        name=name,
        mode="disabled" if FLAGS.debug else None,
        **FLAGS.config.wandb,
    )

    #########
    #
    # Load Pretraining Config + optionally modify
    #
    #########

    orig_config = PretrainedModel.load_config(FLAGS.config.pretrained_path)
    flat_config = flax.traverse_util.flatten_dict(
        orig_config.to_dict(), keep_empty_nodes=True
    )
    for d_key in flax.traverse_util.flatten_dict(
        FLAGS.config.get("config_delete_keys", ConfigDict()).to_dict()
    ):
        for c_key in list(flat_config.keys()):
            if ".".join(c_key).startswith(".".join(d_key)):
                del flat_config[c_key]

    config = ConfigDict(flax.traverse_util.unflatten_dict(flat_config))
    config.update(FLAGS.config.get("update_config", ConfigDict()))
    check_config_diff(config, orig_config)

    #########
    #
    # Setup Data Loader
    #
    #########

    # create text processor
    if config["text_processor"] is None:
        text_processor = None
    else:
        text_processor = text_processors[config["text_processor"]](
            **config["text_processor_kwargs"]
        )

    def process_text(batch):
        if text_processor is None:
            batch["tasks"].pop("language_instruction")
        elif FLAGS.config.modality == "image_conditioned":
            batch["tasks"]["language_instruction"] = text_processor.encode(
                ["" for s in batch["tasks"]["language_instruction"]]
            )  # Multimodal model expects empty string for image_conditioned
        else:
            batch["tasks"]["language_instruction"] = text_processor.encode(
                [s.decode("utf-8") for s in batch["tasks"]["language_instruction"]]
            )
        del batch["dataset_name"]
        return batch

    dataset = make_single_dataset(
        FLAGS.config.dataset_kwargs,
        FLAGS.config.traj_transform_kwargs,
        FLAGS.config.frame_transform_kwargs,
        train=True,
        frame_transform_threads=FLAGS.config.frame_transform_threads,
    )
    val_dataset = make_single_dataset(
        FLAGS.config.dataset_kwargs,
        FLAGS.config.traj_transform_kwargs,
        FLAGS.config.frame_transform_kwargs,
        train=False,
        frame_transform_threads=FLAGS.config.frame_transform_threads,
    )
    visualizer = Visualizer(
        val_dataset, text_processor=text_processor, freeze_trajs=False
    )

    def create_iterator(dataset):
        dataset = (
            dataset.repeat()
            .unbatch()
            .shuffle(FLAGS.config.shuffle_buffer_size)
            .batch(FLAGS.config.batch_size)
        )  # Trajs -> Transitions -> Shuffle -> Batches
        iterator = map(process_text, dataset.iterator())  # Process text
        return iterator

    train_data_iter = create_iterator(dataset)
    val_data_iter = create_iterator(val_dataset)

    example_batch = next(train_data_iter)

    #########
    #
    # Load Pretrained Model
    #
    #########

    model = PretrainedModel.load_pretrained(
        FLAGS.config.pretrained_path,
        config=config,
        example_batch=example_batch,
        text_processor=text_processor,
        step=FLAGS.config.pretrained_step,
    )

    #########
    #
    # Setup Optimizer and Train State
    #
    #########

    rng = jax.random.PRNGKey(FLAGS.config.seed)

    params = model.params
    if FLAGS.config.optimizer.frozen_keys is None:
        FLAGS.config.optimizer.frozen_keys = model.config["optimizer"]["frozen_keys"]

    tx, lr_callable, param_norm_callable = create_optimizer(
        params,
        FLAGS.config.optimizer.to_dict(),
    )
    train_state = TrainState.create(
        apply_fn=model.model_def.apply,
        params=params,
        tx=tx,
        rng=rng,
    )

    #########
    #
    # Save all metadata
    #
    #########

    if FLAGS.config.save_dir is not None:
        save_dir = tf.io.gfile.join(
            FLAGS.config.save_dir,
            FLAGS.config.wandb.project,
            FLAGS.config.wandb.group or "",
            wandb_id,
        )
        wandb.config.update(dict(save_dir=save_dir), allow_val_change=True)
        logging.info("Saving to %s", save_dir)
        tf.io.gfile.makedirs(save_dir)

        # Save model config
        new_config = ConfigDict(model.config)
        new_config.window_size = example_batch["observation"]["pad_mask"].shape[1]

        fname = tf.io.gfile.join(save_dir, "config.json")
        with tf.io.gfile.GFile(fname, "w") as config_file:
            config_file.write(new_config.to_json_best_effort())

        # Save finetuning config
        fname = tf.io.gfile.join(save_dir, "finetune_config.json")
        with tf.io.gfile.GFile(fname, "w") as config_file:
            config_file.write(FLAGS.config.to_json_best_effort())

        tf.io.gfile.copy(
            os.path.join(FLAGS.config.pretrained_path, "example_batch.msgpack"),
            os.path.join(save_dir, "example_batch.msgpack"),
        )

        # Save dataset statistics
        fname = os.path.join(save_dir, "dataset_statistics.json")
        with tf.io.gfile.GFile(fname, "w") as f:
            stats = jax.tree_map(lambda x: x.tolist(), dataset.dataset_statistics)
            json.dump(stats, f)

        # Save example batch to verify shapes later
        with tf.io.gfile.GFile(
            os.path.join(save_dir, "example_batch.msgpack"), "wb"
        ) as f:
            f.write(flax.serialization.msgpack_serialize(example_batch))

        example_batch_spec = jax.tree_map(
            lambda arr: (arr.shape, str(arr.dtype)), example_batch
        )
        wandb.config.update(
            dict(example_batch_spec=example_batch_spec), allow_val_change=True
        )
        # Setup Orbax checkpointers
        state_checkpointer = orbax.checkpoint.CheckpointManager(
            tf.io.gfile.join(save_dir, "state"),
            orbax.checkpoint.PyTreeCheckpointer(),
            options=orbax.checkpoint.CheckpointManagerOptions(
                max_to_keep=1,
            ),
        )  # only keep latest full TrainState

        params_checkpointer = orbax.checkpoint.CheckpointManager(
            save_dir,
            orbax.checkpoint.PyTreeCheckpointer(),
        )  # keep every params checkpoint
    else:
        save_dir = None
        logging.warning("save_dir not passed in, not saving checkpoints")

    #########
    #
    # Define loss, train_step, and eval_step
    #
    #########

    def loss_fn(params, state, batch, rng, train=True):
        def get_loss(model):
            transformer_embeddings = model.orca_transformer(
                batch["observation"],
                batch["tasks"],
                batch["observation"]["pad_mask"],
                train=train,
            )
            action_loss, action_metrics = model.heads["action"].loss(
                transformer_embeddings,  # Action head knows to pull out the action readout_key
                batch["action"],
                pad_mask=batch["observation"]["pad_mask"],
                train=train,
            )
            return action_loss, action_metrics

        return state.apply_fn(
            {"params": params},
            rngs={"dropout": rng},
            method=get_loss,
        )

    # Data parallelism
    # Model is replicated across devices, data is split across devices
    @partial(
        jax.jit,
        in_shardings=[replicated_sharding, dp_sharding],
    )
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, state, batch, dropout_rng, train=True
        )
        # Gradient Metrics (TODO: Does the finetuner need these?) ###
        grad_norm = optax.global_norm(grads)
        updates, _ = state.tx.update(grads, state.opt_state, state.params)
        update_norm = optax.global_norm(updates)
        info.update(
            {
                "grad_norm": grad_norm,
                "update_norm": update_norm,
                "param_norm": param_norm_callable(state.params),
                "learning_rate": lr_callable(state.step),
            }
        )
        # End Debug Metrics #

        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    @partial(
        jax.jit,
        in_shardings=[replicated_sharding, dp_sharding],
    )
    def eval_step(state, batch):
        return loss_fn(state.params, state, batch, rng=state.rng, train=False)[1]

    SAMPLES_FOR_VIZ = 8

    @jax.jit
    def sample_actions(state, observations, tasks):
        new_model = model.replace(params=state.params)  # Put new params in model
        actions = new_model.sample_actions(
            observations,
            tasks,
            sample_shape=(SAMPLES_FOR_VIZ,),
            rng=state.rng,
        )
        actions = actions[..., 0, :]  # get prediction for current action
        actions = jnp.moveaxis(actions, 0, 1)  # (batch_size, n_samples, action_dim)
        return actions

    #########
    #
    # Train loop
    #
    #########

    def wandb_log(info, step):
        wandb.log(flatten_dict(info, sep="/"), step=step)

    timer = Timer()
    for i in tqdm.tqdm(
        range(0, int(FLAGS.config.num_steps)),
        total=int(FLAGS.config.num_steps),
        dynamic_ncols=True,
    ):
        timer.tick("total")

        with timer("dataset"):
            batch = next(train_data_iter)

        with timer("train"):
            train_state, update_info = train_step(train_state, batch)

        timer.tock("total")

        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_log(
                {"training": update_info, "timer": timer.get_average_times()}, step=i
            )

        if (i + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Evaluating...")
            with timer("val"):
                metrics = []
                for _, batch in zip(range(FLAGS.config.num_val_batches), val_data_iter):
                    metrics.append(eval_step(train_state, batch))
                metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
                wandb_log({"validation": metrics}, step=i)

            with timer("visualize"):
                policy_fn = batched_apply(
                    partial(
                        sample_actions,
                        train_state,
                    ),
                    FLAGS.config.batch_size,
                )
                raw_infos = visualizer.raw_evaluations(policy_fn, max_trajs=100)
                metrics = visualizer.metrics_for_wandb(raw_infos)
                images = visualizer.visualize_for_wandb(policy_fn, max_trajs=8)
                wandb_log(
                    {
                        "offline_metrics": metrics,
                        "visualizations": images,
                    },
                    step=i,
                )

        if (i + 1) % FLAGS.config.save_interval == 0 and save_dir is not None:
            logging.info("Saving checkpoint...")

            params_checkpointer.save(
                i + 1,
                train_state.params,
                {"save_args": orbax_utils.save_args_from_target(train_state.params)},
            )
            state_checkpointer.save(
                i + 1,
                train_state,
                {"save_args": orbax_utils.save_args_from_target(train_state)},
            )


if __name__ == "__main__":
    app.run(main)
