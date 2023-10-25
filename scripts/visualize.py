#############################################
#
#
#   Code to do post-hoc analysis on a directory of checkpoints
#
#
#############################################

import copy
import datetime
from functools import partial
import json
import os

from absl import app, flags, logging
from flax.training import checkpoints
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_flags, ConfigDict
from ml_collections.config_dict import placeholder
import numpy as np
import optax
import tensorflow as tf
import wandb

from orca.data.dataset import make_dataset
from orca.data.utils.text_processing import text_processors
from orca.model import create_model_def
from orca.model.components.hf_weight_loaders import weights_loaders
from orca.utils.jax_utils import initialize_compilation_cache
from orca.utils.train_utils import (
    batched_apply,
    create_train_state,
    filter_eval_datasets,
)
from orca.utils.visualization_lib import Visualizer

FLAGS = flags.FLAGS
flags.DEFINE_bool("dummy", False, "Dummy visualization run.")
flags.DEFINE_string("checkpoint", None, "Checkpoint to visualize.")
flags.DEFINE_string(
    "checkpoints",
    None,
    "Path to directory of checkpoints to visualize. ",
)
flags.DEFINE_integer(
    "eval_every", None, "If not None, skip any steps not divisible by eval_every."
)
flags.DEFINE_multi_string(
    "modes",
    ["image"],
    "List of modes to visualize and evaluate.",
)

config_dir = os.path.join(os.path.dirname(__file__), "configs")
config_flags.DEFINE_config_file(
    "config",
    os.path.join(config_dir, "config.py:transformer_bc_bridge"),
    "File path used to get the dataset kwargs.",
    lock_config=False,
)

flags.DEFINE_bool("run_eval", False, "If True, runs eval on all provided datasets.")

wandb_config = ConfigDict(
    dict(
        project="orca_evaluation",
        group=placeholder(str),
        entity=placeholder(str),
        name="evaluation",
        mode="disabled",
    )
)
config_flags.DEFINE_config_dict("wandb", wandb_config, "wandb config")


def dummy_main(_):
    visualizer = Visualizer(FLAGS.config.dataset_kwargs)

    def policy_fn(observations, tasks):
        batch_size = next(iter(jax.tree_leaves(observations))).shape[0]
        return np.random.rand(batch_size, 10, 7) * 2 - 1

    images = visualizer.visualize_for_wandb(policy_fn, n_trajs=1)
    info = visualizer.raw_evaluations(policy_fn, max_trajs=100)
    bridge_metrics = visualizer.metrics_for_wandb(info)
    wandb.init(name="dummy", group="orca", project="test")
    wandb.log(images)
    wandb.log(bridge_metrics)

    print(bridge_metrics)


def main(_):
    if FLAGS.dummy:
        return dummy_main(_)

    initialize_compilation_cache()

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")
    if FLAGS.config.text_processor is None:
        text_processor = None
    else:
        text_processor = text_processors[FLAGS.config.text_processor](
            **FLAGS.config.text_processor_kwargs
        )

    val_datas = []
    visualizers = []
    val_datasets_kwargs, _ = filter_eval_datasets(
        FLAGS.config.dataset_kwargs["data_kwargs_list"],
        [1.0] * len(FLAGS.config.dataset_kwargs["data_kwargs_list"]),
        FLAGS.config.eval_datasets,
    )
    for dataset_kwargs in val_datasets_kwargs:
        val_data_kwargs = copy.deepcopy(dataset_kwargs)
        val_data_kwargs.update(**FLAGS.config.dataset_kwargs["common_kwargs"])
        val_dataset = make_dataset(**val_data_kwargs, train=False)
        val_datas.append(
            val_dataset.unbatch()
            .shuffle(FLAGS.config.shuffle_buffer_size)
            .repeat()
            .batch(FLAGS.config.batch_size)
        )
        visualizers.append(Visualizer(val_data_kwargs, text_processor=text_processor))

    assert (FLAGS.checkpoint is None) ^ (
        FLAGS.checkpoints is None
    ), "Must pass in exactly one of checkpoint or checkpoints"

    if FLAGS.checkpoints:
        list_of_checkpoints = checkpoints._all_checkpoints(FLAGS.checkpoints)
        config_path = tf.io.gfile.join(FLAGS.checkpoints, "config.json")
    else:
        list_of_checkpoints = [FLAGS.checkpoint_path]
        config_path = tf.io.gfile.join(FLAGS.checkpoint_path, "..", "config.json")

    logging.info(list_of_checkpoints)
    logging.info(f"Loading config from {config_path}")
    with tf.io.gfile.GFile(config_path, "r") as config_file:
        old_config = ConfigDict(json.load(config_file))
    logging.info(old_config)

    def process_text(batch):
        if text_processor is None:
            batch["tasks"].pop("language_instruction")
        else:
            batch["tasks"]["language_instruction"] = text_processor.encode(
                [s.decode("utf-8") for s in batch["tasks"]["language_instruction"]]
            )
        return batch

    val_data_iters = [map(process_text, val_data.iterator()) for val_data in val_datas]
    example_batch = next(val_data_iters[0])

    # truncate batch size for faster init
    example_batch = jax.tree_map(lambda x: x[:1], example_batch)

    model_def = create_model_def(
        action_dim=example_batch["action"].shape[-1],
        window_size=example_batch["observation"]["image_0"].shape[1],
        **old_config.model.to_dict(),
    )

    # pretrained weights to load
    pretrained_loaders = [weights_loaders[w] for w in old_config.pretrained_weights]

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=old_config.optimizer.learning_rate,
        warmup_steps=old_config.optimizer.warmup_steps,
        decay_steps=old_config.optimizer.decay_steps,
        end_value=0.0,
    )
    tx = optax.adam(lr_schedule)
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, construct_rng = jax.random.split(rng)

    train_state = create_train_state(
        construct_rng,
        model_def,
        tx,
        init_args=(
            example_batch["observation"],
            example_batch["tasks"],
            example_batch["action"],
        ),
        pretrained_loaders=pretrained_loaders,
    )

    @partial(jax.jit, static_argnames=("argmax", "n"))
    def get_policy_sampled_actions(state, observations, tasks, *, argmax=False, n=1):
        actions = state.apply_fn(
            {"params": state.params},
            observations,
            tasks,
            train=False,
            argmax=argmax,
            sample_shape=(n,),
            rng=state.rng,
            method="predict_action",
            rngs={"dropout": state.rng},
        )
        actions = actions[..., 0, :]

        # viz expects (batch_size, n_samples, action_dim)
        if argmax:
            actions = actions[:, None]
        else:
            actions = jnp.moveaxis(actions, 0, 1)
        return actions

    if FLAGS.run_eval:
        # create a 1D mesh with a single axis named "batch"
        mesh = Mesh(jax.devices(), axis_names="batch")
        # replicated sharding -- does not shard arrays
        replicated_sharding = NamedSharding(mesh, PartitionSpec())
        # data-parallel sharding -- shards arrays along the first axis
        dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))

        def loss_fn(params, state, batch, rng, train=True):
            info = state.apply_fn(
                {"params": params},
                batch["observation"],
                batch["tasks"],
                batch["action"],
                train=train,
                rngs={"dropout": rng},
            )
            return info["loss"], info

        @partial(
            jax.jit,
            # state is replicated, batch is data-parallel
            in_shardings=(replicated_sharding, dp_sharding),
            out_shardings=replicated_sharding,
        )
        def eval_step(state, batch):
            loss, info = loss_fn(state.params, state, batch, state.rng, train=False)
            return info

    wandb_id = "{name}_{time}".format(
        name=FLAGS.wandb.name,
        time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    wandb.init(
        id=wandb_id,
        **FLAGS.wandb,
    )

    def wandb_log(info, step):
        wandb.log(flatten_dict(info, sep="/"), step=step)

    for checkpoint in list_of_checkpoints:
        step = int(checkpoints._checkpoint_path_step(checkpoint))
        if FLAGS.eval_every is not None and step % FLAGS.eval_every != 0:
            continue
        print(f"Loading checkpoint {step}: ", checkpoint)
        train_state = checkpoints.restore_checkpoint(checkpoint, target=train_state)
        policy_fn = batched_apply(
            partial(get_policy_sampled_actions, train_state, argmax=False, n=8),
            FLAGS.config.batch_size,
            devices=jax.devices(),
        )
        for mode in FLAGS.modes:
            for data_kwargs, visualizer in zip(
                FLAGS.config.dataset_kwargs["data_kwargs_list"], visualizers
            ):
                images = visualizer.visualize_for_wandb(policy_fn, max_trajs=3)
                wandb_log({f"{mode}_{data_kwargs['name']}": images}, step=step)

                info = visualizer.raw_evaluations(policy_fn, max_trajs=100)
                metrics = visualizer.metrics_for_wandb(info)
                wandb_log({f"{mode}_{data_kwargs['name']}": metrics}, step=step)
        if FLAGS.run_eval:
            for data_kwargs, val_data_iter in zip(
                FLAGS.config.dataset_kwargs["data_kwargs_list"], val_data_iters
            ):
                metrics = []
                for _, batch in zip(range(FLAGS.config.num_val_batches), val_data_iter):
                    metrics.append(eval_step(train_state, batch))
                metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
                wandb_log({f"validation_{data_kwargs['name']}": metrics}, step=step)


if __name__ == "__main__":
    app.run(main)
