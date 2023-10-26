#############################################
#
#
#   Code to do post-hoc analysis on a directory of checkpoints
#
#
#############################################

import datetime
from functools import partial
import os

from absl import app, flags, logging
from flax.training import checkpoints
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
from ml_collections import config_flags, ConfigDict
from ml_collections.config_dict import placeholder
import numpy as np
import orbax.checkpoint as ocp
import tensorflow as tf
import wandb

from orca.utils.jax_utils import initialize_compilation_cache
from orca.utils.pretrained_utils import PretrainedModel
from orca.utils.train_utils import batched_apply
from orca.utils.visualization_lib import Visualizer

FLAGS = flags.FLAGS
flags.DEFINE_bool("dummy", False, "Dummy visualization run.")
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
    os.path.join(config_dir, "config.py:gc_bridge"),
    "File path used to get the dataset kwargs.",
    lock_config=False,
)

wandb_config = ConfigDict(
    dict(
        project="orca_evaluation",
        group=placeholder(str),
        entity=placeholder(str),
        name="evaluation",
        mode="online",
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
    model = PretrainedModel.load_pretrained(FLAGS.checkpoints)
    text_processor = model.text_processor

    visualizers = []
    for dataset_kwargs in FLAGS.config.dataset_kwargs["data_kwargs_list"]:
        val_data_kwargs = {
            **dataset_kwargs,
            **FLAGS.config.dataset_kwargs["common_kwargs"],
        }
        visualizers.append(
            Visualizer(
                val_data_kwargs, text_processor=text_processor, cache_trajs=False
            )
        )

    list_of_checkpoints = checkpoints._all_checkpoints(FLAGS.checkpoints)

    @partial(jax.jit, static_argnames=("argmax", "n"))
    def get_policy_sampled_actions(
        model: PretrainedModel, observations, tasks, *, argmax=False, n=1
    ):
        horizon = (
            model.config["dataset_kwargs"]["common_kwargs"]["window_size"]
            - model.config["model"]["heads"]["action"]["kwargs"]["pred_horizon"]
            + 1
        )  # This is the horizon the model was trained for
        observations = jax.tree_map(lambda x: x[:, -1 * horizon :], observations)

        actions = model.sample_actions(
            observations,
            tasks,
            argmax=argmax,
            sample_shape=(n,),
            rng=jax.random.PRNGKey(0),
        )
        actions = actions[..., 0, :]  # get first prediction

        # viz expects (batch_size, n_samples, action_dim)
        if argmax:
            actions = actions[:, None]
        else:
            actions = jnp.moveaxis(actions, 0, 1)
        return actions

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

    checkpointer = ocp.PyTreeCheckpointer()
    from flax.training import orbax_utils

    custom_restore_args = orbax_utils.restore_args_from_target(model)

    for checkpoint in list_of_checkpoints:
        step = int(checkpoints._checkpoint_path_step(checkpoint))
        if FLAGS.eval_every is not None and step % FLAGS.eval_every != 0:
            continue
        print(f"Loading checkpoint {step}: ", checkpoint)
        model = checkpointer.restore(
            checkpoint,
            item=model,
            transforms={},
            restore_args=custom_restore_args,
        )

        policy_fn = batched_apply(
            partial(get_policy_sampled_actions, model, argmax=False, n=8),
            FLAGS.config.batch_size,
            devices=jax.devices(),
        )

        for data_kwargs, visualizer in zip(
            FLAGS.config.dataset_kwargs["data_kwargs_list"], visualizers
        ):
            raw_infos = visualizer.raw_evaluations(policy_fn, max_trajs=100)
            metrics = visualizer.metrics_for_wandb(raw_infos)
            images = visualizer.visualize_for_wandb(policy_fn, max_trajs=8)
            wandb_log(
                {
                    f"offline_metrics_{data_kwargs['name']}": metrics,
                    f"visualizations_{data_kwargs['name']}": images,
                },
                step=step,
            )


if __name__ == "__main__":
    app.run(main)
