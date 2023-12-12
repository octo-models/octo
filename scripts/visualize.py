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
import flax
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
from ml_collections import config_flags, ConfigDict
from ml_collections.config_dict import placeholder
import numpy as np
import orbax.checkpoint as ocp
import tensorflow as tf
import wandb

from orca.data.dataset import make_single_dataset
from orca.model.orca_model import ORCAModel
from orca.utils.jax_utils import initialize_compilation_cache
from orca.utils.train_utils import batched_apply, filter_eval_datasets
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
flags.DEFINE_integer(
    "samples_per_timestep", 8, "Number of action samples to use at each timestep"
)
flags.DEFINE_integer(
    "horizon",
    None,
    "What horizon policies should be evaluated at. Defaults to the horizon it was trained at",
)
flags.DEFINE_float("temperature", 1.0, "Temperature to sample actions at")
flags.DEFINE_list(
    "policy_modes",
    None,
    "Which policy modes to evaluate. Defaults to the modes that the policy was trained for",
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
    model = ORCAModel.load_pretrained(FLAGS.checkpoints)
    text_processor = model.text_processor
    if text_processor is not None:
        zero_text = text_processor.encode([""])[0]

    visualizers = []
    val_datasets_kwargs, _ = filter_eval_datasets(
        FLAGS.config.dataset_kwargs["data_kwargs_list"],
        [1.0] * len(FLAGS.config.dataset_kwargs["data_kwargs_list"]),
        FLAGS.config.eval_datasets,
    )
    for dataset_kwargs in val_datasets_kwargs:
        val_data_kwargs = {
            **dataset_kwargs,
            **FLAGS.config.dataset_kwargs["common_kwargs"],
            "shuffle": False,
        }
        val_dataset = make_single_dataset(
            val_data_kwargs,
            FLAGS.config.dataset_kwargs["transform_kwargs"],
            train=False,
        )
        visualizers.append(
            Visualizer(val_dataset, text_processor=text_processor, freeze_trajs=False)
        )

    list_of_checkpoints = ocp.utils.checkpoint_steps_paths(FLAGS.checkpoints)
    list_of_checkpoints = sorted(
        list_of_checkpoints,
        key=lambda path: ocp.utils.step_from_checkpoint_name(path.name),
    )
    logging.info(list_of_checkpoints)

    horizon = FLAGS.horizon or (
        model.config["dataset_kwargs"]["transform_kwargs"]["window_size"]
        - model.config["model"]["heads"]["action"]["kwargs"]["pred_horizon"]
        + 1
    )  # This is the horizon the model was trained for

    def remove_text(tasks):
        if text_processor is not None:
            new_language = jax.tree_map(
                lambda x, example: jnp.broadcast_to(example[None], x.shape),
                tasks["language_instruction"],
                zero_text,
            )
            tasks = flax.core.copy(tasks, {"language_instruction": new_language})
        return tasks

    def remove_images(tasks):
        new_images = {k: jnp.zeros_like(v) for k, v in tasks.items() if "image" in k}
        return flax.core.copy(tasks, new_images)

    @partial(jax.jit, static_argnames=("argmax", "n", "policy_mode"))
    def get_policy_sampled_actions(
        model: ORCAModel,
        observations,
        tasks,
        *,
        argmax=False,
        n=1,
        policy_mode=None,
    ):
        observations = jax.tree_map(lambda x: x[:, -1 * horizon :], observations)
        if policy_mode is None:
            pass
        elif policy_mode == "text_conditioned":
            tasks = remove_images(tasks)
        elif policy_mode == "image_conditioned":
            tasks = remove_text(tasks)
        elif policy_mode == "unconditioned":
            tasks = remove_text(remove_images(tasks))
        else:
            raise NotImplementedError()

        actions = model.sample_actions(
            observations,
            tasks,
            argmax=argmax,
            sample_shape=(n,),
            rng=jax.random.PRNGKey(0),
            temperature=FLAGS.temperature,
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

    for path in list_of_checkpoints:
        step = ocp.utils.step_from_checkpoint_name(path.name)
        if FLAGS.eval_every is not None and step % FLAGS.eval_every != 0:
            continue
        print(f"Loading checkpoint {step}: ", path)
        params = checkpointer.restore(tf.io.gfile.join(path, "default"), model.params)
        model = model.replace(params=params)
        if FLAGS.policy_modes is not None:
            modes_to_evaluate = FLAGS.policy_modes
        elif text_processor is not None:
            modes_to_evaluate = [
                "text_conditioned",
                "image_conditioned",
                "unconditioned",
            ]
        else:
            modes_to_evaluate = ["image_conditioned"]

        modal_policy_fns = {
            k: batched_apply(
                partial(
                    get_policy_sampled_actions,
                    model,
                    argmax=False,
                    n=FLAGS.samples_per_timestep,
                    policy_mode=k,
                ),
                min(128, FLAGS.config.batch_size),
            )
            for k in modes_to_evaluate
        }

        for data_kwargs, visualizer in zip(val_datasets_kwargs, visualizers):
            for mode, policy_fn in modal_policy_fns.items():
                raw_infos = visualizer.raw_evaluations(policy_fn, max_trajs=100)
                metrics = visualizer.metrics_for_wandb(raw_infos)
                images = visualizer.visualize_for_wandb(policy_fn, max_trajs=8)
                wandb_log(
                    {
                        f"offline_metrics_{data_kwargs['name']}/{mode}": metrics,
                        f"visualizations_{data_kwargs['name']}/{mode}": images,
                    },
                    step=step,
                )


if __name__ == "__main__":
    app.run(main)
