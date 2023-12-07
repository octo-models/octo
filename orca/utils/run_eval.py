#!/usr/bin/env python3

from datetime import datetime
from functools import partial
import json
import os
import time
from typing import Callable

from absl import flags, logging
import click
import cv2
import flax
import gym
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from orca.utils.eval_utils import (
    download_checkpoint_from_gcs,
    load_jaxrlm_checkpoint,
    supply_rng,
)
from orca.utils.gym_wrappers import HistoryWrapper, RHCWrapper
from orca.utils.pretrained_utils import PretrainedModel

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    "checkpoint_weights_path", None, "Path to checkpoint", required=True
)
flags.DEFINE_multi_integer("checkpoint_step", None, "Checkpoint step", required=True)
flags.DEFINE_bool("add_jaxrlm_baseline", False, "Also compare to jaxrl_m baseline")
flags.DEFINE_string(
    "checkpoint_cache_dir", "/tmp/", "Where to cache checkpoints downloaded from GCS"
)
flags.DEFINE_string(
    "modality",
    "",
    "Either 'g', 'goal', 'l', 'language' (leave empty to prompt when running)",
)

flags.DEFINE_integer("im_size", None, "Image size", required=True)
flags.DEFINE_string("video_save_path", None, "Path to save video")
flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_integer("horizon", 1, "Observation history length")
flags.DEFINE_integer("pred_horizon", 1, "Length of action sequence from model")
flags.DEFINE_integer("exec_horizon", 1, "Length of action sequence to execute")
flags.DEFINE_float("temperature", 1.0, "Temperature for sampling actions")
flags.DEFINE_bool("deterministic", False, "Whether to sample action deterministically")

# show image flag
flags.DEFINE_bool("show_image", False, "Show image")


##############################################################################


# TODO: use class UnnormalizeActionProprio instead
@partial(jax.jit, static_argnames="argmax")
def sample_unormalized_actions(
    pretrained_model: PretrainedModel,
    observations,
    tasks,
    mean,
    std,
    rng,
    argmax=False,
    temperature=1.0,
):
    # add batch dim to observations
    observations = jax.tree_map(lambda x: x[None], observations)
    # tasks = jax.tree_map(lambda x: x[None], tasks)
    logging.warning(
        "observations: %s", flax.core.pretty_repr(jax.tree_map(jnp.shape, observations))
    )
    logging.warning("tasks: %s", flax.core.pretty_repr(jax.tree_map(jnp.shape, tasks)))
    actions = pretrained_model.sample_actions(
        observations,
        tasks,
        rng=rng,
        argmax=argmax,
        temperature=temperature,
    )
    # remove batch dim
    return actions[0] * std + mean


def load_checkpoint(weights_path: str, step: int, dataset_name: str = "bridge_dataset"):
    model = PretrainedModel.load_pretrained(weights_path, step=int(step))
    metadata_path = os.path.join(
        weights_path, f"dataset_statistics_{dataset_name}.json"
    )
    with open(metadata_path, "r") as f:
        action_proprio_metadata = json.load(f)
    action_mean = jnp.array(action_proprio_metadata["action"]["mean"])
    action_std = jnp.array(action_proprio_metadata["action"]["std"])

    policy_fn = supply_rng(
        partial(
            sample_unormalized_actions,
            model,
            argmax=FLAGS.deterministic,
            mean=action_mean,
            std=action_std,
            temperature=FLAGS.temperature,
        ),
    )
    return (policy_fn, model)


def run_eval_loop(env: gym.Env, get_goal_condition: Callable, step_duration: float):
    """
    Main evaluation loop.
    : param env: gym.Env
    : param get_goal_condition:     Callable to get the image goal during the
                                    init of the eval run
    : param step_duration:          Duration of each step
    """
    assert len(FLAGS.checkpoint_weights_path) == len(FLAGS.checkpoint_step)
    FLAGS.modality = FLAGS.modality[:1]
    if FLAGS.modality not in ["g", "l", ""]:
        FLAGS.modality = ""

    # policies is a dict from run_name to policy function
    policies = {}
    for (checkpoint_weights_path, checkpoint_step,) in zip(
        FLAGS.checkpoint_weights_path,
        FLAGS.checkpoint_step,
    ):
        checkpoint_weights_path, checkpoint_step = download_checkpoint_from_gcs(
            checkpoint_weights_path,
            checkpoint_step,
            FLAGS.checkpoint_cache_dir,
        )
        assert tf.io.gfile.exists(checkpoint_weights_path), checkpoint_weights_path
        run_name = checkpoint_weights_path.rpartition("/")[2]
        policies[f"{run_name}-{checkpoint_step}"] = load_checkpoint(
            checkpoint_weights_path,
            checkpoint_step,
        )

    # TODO: prevent having custom impl
    if FLAGS.add_jaxrlm_baseline:
        base_dir = "/mount/harddrive/homer/bridgev2_packaged/bridgev2policies/"
        policies["jaxrl_gcbc"] = load_jaxrlm_checkpoint(
            weights_path=f"{base_dir}gcbc_256/checkpoint_300000/",
            config_path=f"{base_dir}gcbc_256/gcbc_256_config.json",
            code_path=f"{base_dir}bridge_data_v2.zip",
        )

    env = HistoryWrapper(env, FLAGS.horizon)
    # env = TemporalEnsembleWrapper(env, FLAGS.pred_horizon)
    env = RHCWrapper(env, FLAGS.pred_horizon, FLAGS.exec_horizon)

    goal_image = jnp.zeros((FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8)
    goal_instruction = ""

    # goal sampling loop
    while True:
        # ask for which policy to use
        if len(policies) == 1:
            policy_idx = 0
            print("Using default policy 0: ", list(policies.keys())[policy_idx])
        else:
            print("policies:")
            for i, name in enumerate(policies.keys()):
                print(f"{i}) {name}")
            policy_idx = click.prompt("Select policy", type=int)

        policy_name = list(policies.keys())[policy_idx]
        policy_fn, model = policies[policy_name]
        model: PretrainedModel  # type hinting

        modality = FLAGS.modality
        if not modality:
            modality = click.prompt(
                "Language or goal image?", type=click.Choice(["l", "g"])
            )

        if modality == "g":
            if click.confirm("Take a new goal?", default=True):
                obs = get_goal_condition()
                obs.pop("proprio")  # TODO: remove this fix
                goal = jax.tree_map(lambda x: x[None], obs)

            task = model.create_tasks(goals=goal)
            goal_image = goal["image_0"][0]
            goal_instruction = ""
        elif modality == "l":
            print("Current instruction: ", goal_instruction)
            if click.confirm("Take a new instruction?", default=True):
                text = input("Instruction?")

            task = model.create_tasks(texts=[text])
            goal_instruction = text
            goal_image = jnp.zeros_like(goal_image)
        else:
            raise NotImplementedError()

        # reset env
        obs, _ = env.reset()
        obs.pop("proprio")  # TODO: remove this fix
        time.sleep(2.0)

        input("Press [Enter] to start.")

        # do rollout
        last_tstep = time.time()
        images = []
        goals = []
        t = 0
        while t < FLAGS.num_timesteps:
            if time.time() > last_tstep + step_duration:
                last_tstep = time.time()

                # save images
                images.append(obs["image_0"][-1])
                goals.append(goal_image)

                if FLAGS.show_image:
                    bgr_img = cv2.cvtColor(obs["image_0"][-1], cv2.COLOR_RGB2BGR)
                    cv2.imshow("img_view", bgr_img)
                    cv2.waitKey(20)

                # get action
                forward_pass_time = time.time()
                action = np.array(policy_fn(obs, task), dtype=np.float64)
                print("forward pass time: ", time.time() - forward_pass_time)

                # perform environment step
                start_time = time.time()
                obs, _, _, truncated, _ = env.step(action)
                print("step time: ", time.time() - start_time)
                obs.pop("proprio")  # TODO: remove this fix

                t += 1

                if truncated:
                    break

        # save video
        if FLAGS.video_save_path is not None:
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                FLAGS.video_save_path,
                f"{curr_time}_{policy_name}.mp4",
            )
            video = np.concatenate([np.stack(goals), np.stack(images)], axis=1)
            imageio.mimsave(save_path, video, fps=1.0 / step_duration * 3)
