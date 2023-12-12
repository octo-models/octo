import datetime
from functools import partial
import imp
import json
import os

from absl import app, flags, logging
import flax
from flax.traverse_util import flatten_dict
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_flags, ConfigDict
import optax
import tensorflow as tf
import tqdm
import wandb

from orca.data.dataset import make_single_dataset
from orca.utils.jax_utils import initialize_compilation_cache
from orca.utils.pretrained_utils import ORCAModel
from orca.utils.spec import ModuleSpec
from orca.utils.train_callbacks import (
    RolloutVisualizationCallback,
    SaveCallback,
    ValidationCallback,
    VisualizationCallback,
)
from orca.utils.train_utils import (
    check_config_diff,
    create_optimizer,
    format_name_with_config,
    merge_params,
    process_text,
    Timer,
    TrainState,
)

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

    orig_config = ORCAModel.load_config(FLAGS.config.pretrained_path)
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
        text_processor = ModuleSpec.instantiate(config["text_processor"])()

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    # load standardize_fn from `path/to/file.py:fn_name` format
    if (
        standardize_fn := FLAGS.config["dataset_kwargs"].get("standardize_fn", None)
    ) is not None:
        path, name = standardize_fn.split(":")
        # imp is deprecated, but it's also what ml_collections uses
        standardize_fn = getattr(imp.load_source("standardize_fn", path), name)
        del FLAGS.config["dataset_kwargs"]["standardize_fn"]
        FLAGS.config["dataset_kwargs"]["standardize_fn"] = standardize_fn

    dataset = make_single_dataset(
        FLAGS.config.dataset_kwargs,
        FLAGS.config.traj_transform_kwargs,
        FLAGS.config.frame_transform_kwargs,
        train=True,
    )
    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(FLAGS.config.shuffle_buffer_size)
        .batch(FLAGS.config.batch_size)
        .iterator()
    )
    train_data_iter = map(process_batch, train_data_iter)
    example_batch = next(train_data_iter)

    #########
    #
    # Load Pretrained Model
    #
    #########

    pretrained_model = ORCAModel.load_pretrained(
        FLAGS.config.pretrained_path,
        step=FLAGS.config.pretrained_step,
    )
    model = ORCAModel.from_config(config, example_batch, text_processor)
    merged_params = merge_params(model.params, pretrained_model.params)
    model = model.replace(params=merged_params)
    del pretrained_model
    model_def = model.model_def

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
        **FLAGS.config.optimizer.to_dict(),
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
        save_callback = SaveCallback(save_dir)

        # Save model config
        new_config = ConfigDict(model.config)
        new_config.window_size = example_batch["observation"]["pad_mask"].shape[1]

        with save_callback.open("config.json", "w") as config_file:
            config_file.write(new_config.to_json_best_effort())

        # Save finetuning config
        with save_callback.open("finetune_config.json", "w") as config_file:
            config_file.write(FLAGS.config.to_json_best_effort())

        with save_callback.open("dataset_statistics.json", "w") as f:
            stats = jax.tree_map(lambda x: x.tolist(), dataset.dataset_statistics)
            json.dump(stats, f)

        # Save example batch to verify shapes later
        with save_callback.open("example_batch.msgpack", "wb") as f:
            f.write(flax.serialization.msgpack_serialize(example_batch))
    else:
        save_dir = None
        save_callback = SaveCallback(None)
        logging.warning("save_dir not passed in, not saving checkpoints")

    example_batch_spec = jax.tree_map(
        lambda arr: (arr.shape, str(arr.dtype)), example_batch
    )
    wandb.config.update(
        dict(example_batch_spec=example_batch_spec), allow_val_change=True
    )

    #########
    #
    # Define loss, train_step, and eval_step
    #
    #########

    def loss_fn(params, state, batch, rng, train=True):
        model = model_def.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = model.orca_transformer(
            batch["observation"],
            batch["task"],
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

    #########
    #
    # Build validation & visualization callbacks
    #
    #########

    if FLAGS.config.modality == "image_conditioned":
        modes_to_evaluate = ["image_conditioned"]
    elif FLAGS.config.modality == "text_conditioned":
        modes_to_evaluate = ["text_conditioned"]
    elif FLAGS.config.modality == "multimodal":
        modes_to_evaluate = ["image_conditioned", "text_conditioned"]
    else:
        modes_to_evaluate = ["base"]

    dataset_kwargs_list = [FLAGS.config.dataset_kwargs]

    val_callback = ValidationCallback(
        loss_fn=loss_fn,
        process_batch_fn=process_batch,
        text_processor=text_processor,
        val_dataset_kwargs_list=dataset_kwargs_list,
        dataset_kwargs=FLAGS.config,
        modes_to_evaluate=modes_to_evaluate,
        **FLAGS.config.val_kwargs,
    )

    viz_callback = VisualizationCallback(
        text_processor=text_processor,
        val_dataset_kwargs_list=dataset_kwargs_list,
        dataset_kwargs=FLAGS.config,
        modes_to_evaluate=modes_to_evaluate,
        **FLAGS.config.viz_kwargs,
    )

    #########
    #
    # Optionally build visualizers for sim env evals
    #
    #########

    if "rollout_kwargs" in FLAGS.config:
        rollout_callback = RolloutVisualizationCallback(
            text_processor=text_processor,
            history_length=FLAGS.config["window_size"],
            model_pred_horizon=config["model"]["heads"]["action"]["kwargs"].get(
                "pred_horizon", 1
            ),
            **FLAGS.config.rollout_kwargs.to_dict(),
        )
    else:
        rollout_callback = None

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
                val_metrics = val_callback(train_state, i + 1)
                wandb_log(val_metrics, step=i)

            with timer("visualize"):
                viz_metrics = viz_callback(train_state, i + 1)
                wandb_log(viz_metrics, step=i)

            if rollout_callback is not None:
                with timer("rollout"):
                    rollout_metrics = rollout_callback(train_state, i + 1)
                    wandb_log(rollout_metrics, step=i)

        if (i + 1) % FLAGS.config.save_interval == 0 and save_dir is not None:
            logging.info("Saving checkpoint...")
            save_callback(train_state, i + 1)


if __name__ == "__main__":
    app.run(main)
