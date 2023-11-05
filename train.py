import copy
import datetime
from functools import partial
import json
import os
import os.path as osp

from absl import app, flags, logging
import flax
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_flags
import numpy as np
import optax
import orbax.checkpoint
import tensorflow as tf
import tqdm
import wandb

import orca
from orca.data.dataset import make_interleaved_dataset, make_single_dataset
from orca.data.utils.text_processing import text_processors
from orca.model import create_model_def, OrcaModel
from orca.model.components.hf_weight_loaders import weights_loaders
from orca.utils.jax_utils import initialize_compilation_cache
from orca.utils.train_utils import (
    batched_apply,
    create_train_state,
    filter_eval_datasets,
    format_name_with_config,
    Timer,
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

config_dir = os.path.join(os.path.dirname(__file__), "configs")
config_flags.DEFINE_config_file(
    "config",
    os.path.join(config_dir, "config.py:transformer_bc"),
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

NUM_ACTIONS_FOR_VIS = 8


def main(_):
    initialize_compilation_cache()

    assert FLAGS.config.batch_size % jax.device_count() == 0

    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(jax.devices(), axis_names="batch")
    # replicated sharding -- does not shard arrays
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    # data-parallel sharding -- shards arrays along the first axis
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # set up wandb and logging
    if FLAGS.config.wandb_resume_id is None:
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
            with tf.io.gfile.GFile(
                os.path.join(save_dir, "config.json"), "w"
            ) as config_file:
                config_file.write(FLAGS.config.to_json_best_effort())
        else:
            save_dir = None
            logging.info("save_dir not passed in, not saving checkpoints")
    else:
        # resume previous run
        wandb_run = wandb.Api().run(FLAGS.config.wandb_resume_id)
        wandb.init(
            project=wandb_run.project,
            id=wandb_run.id,
            resume="must",
        )
        save_dir = wandb_run.config["save_dir"]

    if save_dir is not None:
        # make checkpointers
        options = orbax.checkpoint.CheckpointManagerOptions(
            max_to_keep=1,
        )
        state_checkpointer = orbax.checkpoint.CheckpointManager(
            save_dir, orbax.checkpoint.PyTreeCheckpointer(), options=options
        )
        params_checkpointer = orbax.checkpoint.CheckpointManager(
            tf.io.gfile.join(save_dir, "params"),
            orbax.checkpoint.PyTreeCheckpointer(),
        )

    codebase_directory = osp.abspath(osp.join(osp.dirname(orca.__file__), ".."))
    wandb.run.log_code(codebase_directory)  # TODO: replace w/ codesave_library?

    # set up text tokenization (this needs to happen after batching but before sharding)
    if FLAGS.config.text_processor is None:
        text_processor = None
    else:
        text_processor = text_processors[FLAGS.config.text_processor](
            **FLAGS.config.text_processor_kwargs
        )

    def process_text(batch):
        if text_processor is None:
            batch["tasks"].pop("language_instruction")
        else:
            batch["tasks"]["language_instruction"] = text_processor.encode(
                [s.decode("utf-8") for s in batch["tasks"]["language_instruction"]]
            )
        return batch

    # load datasets
    sample_weights = (
        FLAGS.config.dataset_kwargs["sample_weights"]
        if "sample_weights" in FLAGS.config.dataset_kwargs
        else None
    )
    train_data = make_interleaved_dataset(
        FLAGS.config.dataset_kwargs["common_kwargs"],
        FLAGS.config.dataset_kwargs["data_kwargs_list"],
        FLAGS.config.dataset_kwargs["transform_kwargs"],
        train=True,
        sample_weights=sample_weights,
        shuffle_buffer_size=FLAGS.config.shuffle_buffer_size,
    ).batch(FLAGS.config.batch_size)
    val_datas = []
    visualizers = []
    val_datasets_kwargs, val_datasets_sample_weights = filter_eval_datasets(
        FLAGS.config.dataset_kwargs["data_kwargs_list"],
        sample_weights,
        FLAGS.config.eval_datasets,
    )
    for dataset_kwargs in val_datasets_kwargs:
        val_data_kwargs = copy.deepcopy(dataset_kwargs)
        val_data_kwargs.update(**FLAGS.config.dataset_kwargs["common_kwargs"])
        val_data_kwargs["shuffle"] = False
        val_dataset = make_single_dataset(
            val_data_kwargs,
            FLAGS.config.dataset_kwargs["transform_kwargs"],
            train=False,
        )
        action_proprio_metadata = val_dataset.action_proprio_metadata
        val_datas.append(
            val_dataset.unbatch()
            # TODO: doesn't this mean every single tiny dataset has a huge shuffle buffer?
            .shuffle(FLAGS.config.shuffle_buffer_size)
            .repeat()
            .batch(FLAGS.config.batch_size)
        )
        visualizers.append(
            Visualizer(val_dataset, text_processor=text_processor, cache_trajs=False)
        )

        # save normalization constants for evaluation
        if save_dir is not None:
            with tf.io.gfile.GFile(
                os.path.join(
                    save_dir, f"action_proprio_metadata_{val_data_kwargs['name']}.json"
                ),
                "w",
            ) as f:
                json.dump(
                    jax.tree_map(
                        lambda x: [float(e) for e in x.numpy()], action_proprio_metadata
                    ),
                    f,
                )

    train_data_iter = map(process_text, train_data.iterator())
    val_data_iters = [map(process_text, val_data.iterator()) for val_data in val_datas]

    example_batch = next(train_data_iter)
    logging.info(f"Batch size: {example_batch['action'].shape[0]}")
    logging.info(f"Number of devices: {jax.device_count()}")
    logging.info(
        f"Batch size per device: {example_batch['action'].shape[0] // jax.device_count()}"
    )

    # truncate batch size for faster init
    example_batch = jax.tree_map(lambda x: x[:1], example_batch)

    # set up model, optimizer, loss
    model_def = create_model_def(
        **FLAGS.config.model.to_dict(),
    )

    # pretrained weights to load
    pretrained_loaders = [weights_loaders[w] for w in FLAGS.config.pretrained_weights]

    optimizer_kwargs = FLAGS.config.optimizer.to_dict()
    if isinstance(optimizer_kwargs["learning_rate"], dict):
        optimizer_kwargs["learning_rate"] = optax.warmup_cosine_decay_schedule(
            **optimizer_kwargs["learning_rate"]
        )
    tx = optax.chain(
        optax.clip_by_global_norm(optimizer_kwargs.pop("clip_gradient")),
        optax.adamw(mu_dtype=jnp.bfloat16, **optimizer_kwargs),
    )

    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, construct_rng = jax.random.split(rng)

    train_state = create_train_state(
        construct_rng,
        model_def,
        tx,
        init_args=(
            example_batch["observation"],
            example_batch["tasks"],
            example_batch["observation"]["pad_mask"],
        ),
        init_kwargs=dict(train=False, verbose=True),
        pretrained_loaders=pretrained_loaders,
    )

    if save_dir is not None:
        # Saving example batch for future checkpoint loading
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

    if FLAGS.config.wandb_resume_id is not None:
        train_state = state_checkpointer.restore(
            state_checkpointer.latest_step(), items=train_state
        )
        checkpoint_step = int(train_state.step)
        logging.info("Restored checkpoint from %s", save_dir)
        if FLAGS.config.start_step is not None:
            start_step = FLAGS.config.start_step  # start_step overrides checkpoint
        else:
            start_step = checkpoint_step
        logging.info("Starting training from step %d", start_step)
    else:
        start_step = FLAGS.config.start_step or 0

    horizon = (
        FLAGS.config.dataset_kwargs.transform_kwargs.window_size
        - FLAGS.config.model.heads["action"]["kwargs"]["pred_horizon"]
        + 1
    )  # Ensures that there is a full horizon of actions to predict for each timestep

    def loss_fn(params, state, batch, rng, train=True):
        def get_loss(model: OrcaModel, observations, tasks, actions, train):
            # only use first horizon timesteps as input to transformer
            # to ensure that there is a full horizon of actions to predict for each timestep
            observations = jax.tree_map(lambda x: x[:, :horizon], observations)

            transformer_embeddings = model.orca_transformer(
                observations, tasks, observations["pad_mask"], train=train
            )
            action_loss, action_metrics = model.heads["action"].loss(
                transformer_embeddings,  # Action head knows to pull out the action readout_key
                actions,
                pad_mask=observations["pad_mask"],
                train=train,
            )

            return action_loss, action_metrics

        return state.apply_fn(
            {"params": params},
            batch["observation"],
            batch["tasks"],
            batch["action"],
            train=train,
            rngs={"dropout": rng},
            method=get_loss,
        )

    @partial(
        jax.jit,
        # state is replicated, batch is data-parallel
        in_shardings=(replicated_sharding, dp_sharding),
        out_shardings=(replicated_sharding, replicated_sharding),
        # allows jax to modify `state` in-place, saving a lot of memory
        donate_argnums=0,
    )
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, state, batch, dropout_rng, train=True
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    @partial(
        jax.jit,
        # state is replicated, batch is data-parallel
        in_shardings=(replicated_sharding, dp_sharding),
        out_shardings=replicated_sharding,
    )
    def eval_step(state, batch):
        loss, info = loss_fn(state.params, state, batch, state.rng, train=False)
        return info

    @partial(
        jax.jit,
        in_shardings=(replicated_sharding, dp_sharding, dp_sharding),
        out_shardings=dp_sharding,
    )
    def get_policy_sampled_actions(state, observations, tasks):
        # only use first horizon timesteps as input to predict_action
        observations = jax.tree_map(lambda x: x[:, -horizon:], observations)

        def get_actions(model, observations, tasks, train):
            transformer_embeddings = model.orca_transformer(
                observations,
                tasks,
                observations["pad_mask"],
                train=train,
            )

            actions = model.heads["action"].predict_action(
                transformer_embeddings,  # Action head knows to pull out the action readout_key
                train=train,
                argmax=False,
                sample_shape=(NUM_ACTIONS_FOR_VIS,),
                rng=state.rng,
            )
            return actions

        actions = state.apply_fn(
            {"params": state.params},
            observations,
            tasks,
            train=False,
            method=get_actions,
            rngs={"dropout": state.rng},
        )  # We could also have used run_head here, but this is easier to read

        # actions is (NUM_ACTIONS_FOR_VIS, batch_size, pred_horizon, action_dim)
        # where actions[:, :, i] predicts the action at timestep "window_size + i"
        actions = actions[..., 0, :]

        # viz expects (batch_size, n_samples, action_dim)
        actions = jnp.moveaxis(actions, 0, 1)
        return actions

    def wandb_log(info, step):
        wandb.log(flatten_dict(info, sep="/"), step=step)

    timer = Timer()
    for i in tqdm.tqdm(
        range(start_step, int(FLAGS.config.num_steps)),
        total=int(FLAGS.config.num_steps),
        initial=start_step,
        dynamic_ncols=True,
    ):
        timer.tick("total")

        with timer("dataset"):
            batch = next(train_data_iter)

        with timer("train"):
            train_state, update_info = train_step(train_state, batch)

        if (i + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Evaluating...")
            timer.tick("val")
            per_dataset_metrics = []
            for data_kwargs, val_data_iter in zip(val_datasets_kwargs, val_data_iters):
                metrics = []
                for _, batch in zip(range(FLAGS.config.num_val_batches), val_data_iter):
                    metrics.append(eval_step(train_state, batch))
                metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
                wandb_log({f"validation_{data_kwargs['name']}": metrics}, step=i)
                per_dataset_metrics.append(metrics)

            # log weighted aggregate metrics
            val_datasets_sample_weights = (
                val_datasets_sample_weights
                if val_datasets_sample_weights is not None
                else [1.0] * len(per_dataset_metrics)
            )
            val_datasets_sample_weights = val_datasets_sample_weights / np.sum(
                val_datasets_sample_weights
            )  # normalize to sum to 1
            agg_metrics = jax.tree_map(
                lambda *xs: np.sum(xs),
                *[
                    jax.tree_map(lambda x: x * weight, metric)
                    for metric, weight in zip(
                        per_dataset_metrics, val_datasets_sample_weights
                    )
                ],
            )
            wandb_log({"validation_aggregate": agg_metrics}, step=i)
            timer.tock("val")

            logging.info("Visualizing...")
            timer.tick("visualize")
            policy_fn = batched_apply(
                partial(get_policy_sampled_actions, train_state),
                FLAGS.config.batch_size,
            )
            for data_kwargs, visualizer in zip(val_datasets_kwargs, visualizers):
                raw_infos = visualizer.raw_evaluations(policy_fn, max_trajs=100)
                metrics = visualizer.metrics_for_wandb(raw_infos)
                images = visualizer.visualize_for_wandb(policy_fn, max_trajs=8)
                wandb_log(
                    {
                        f"offline_metrics_{data_kwargs['name']}": metrics,
                        f"visualizations_{data_kwargs['name']}": images,
                    },
                    step=i,
                )
            timer.tock("visualize")

        if (i + 1) % FLAGS.config.save_interval == 0 and save_dir is not None:
            logging.info("Saving checkpoint...")
            params_checkpointer.save(i + 1, train_state.params)
            checkpoint_path = state_checkpointer.save(i + 1, train_state)
            logging.info("Saved checkpoint to %s", checkpoint_path)

        timer.tock("total")

        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_log(
                {"training": update_info, "timer": timer.get_average_times()}, step=i
            )


if __name__ == "__main__":
    app.run(main)
