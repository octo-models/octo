#!/usr/bin/env python3

import json
import os
import time
from datetime import datetime
from functools import partial

import cv2
import flax
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from absl import app, flags, logging

# bridge_data_robot imports
from widowx_envs.widowx_env_service import WidowXClient, WidowXStatus, WidowXConfigs

from orca.utils.pretrained_utils import PretrainedModel
from widowx_wrapper import WidowXGym, convert_obs, wait_for_obs, state_to_eep
from orca.utils.gym_wrappers import (
    HistoryWrapper,
    RHCWrapper,
    TemporalEnsembleWrapper,
)

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    "checkpoint_weights_path", None, "Path to checkpoint", required=True
)
flags.DEFINE_multi_string(
    "checkpoint_config_path", None, "Path to checkpoint config JSON", required=True
)
flags.DEFINE_multi_string(
    "checkpoint_metadata_path", None, "Path to checkpoint metadata JSON", required=True
)
flags.DEFINE_multi_string(
    "checkpoint_example_batch_path",
    None,
    "Path to checkpoint metadata JSON",
    required=True,
)
flags.DEFINE_integer("im_size", None, "Image size", required=True)
flags.DEFINE_string("video_save_path", None, "Path to save video")
flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_bool("blocking", False, "Use the blocking controller")
flags.DEFINE_spaceseplist("goal_eep", [0.3, 0.0, 0.15], "Goal position")
flags.DEFINE_spaceseplist("initial_eep", [0.3, 0.0, 0.15], "Initial position")
flags.DEFINE_integer("horizon", 1, "Observation history length")
flags.DEFINE_integer("pred_horizon", 1, "Observation history length")
flags.DEFINE_integer("exec_horizon", 1, "Action sequence length")
flags.DEFINE_bool("deterministic", False, "Whether to sample action deterministically")
flags.DEFINE_float("temperature", 1.0, "Temperature for sampling actions")
flags.DEFINE_string("ip", "localhost", "IP address of the robot")
flags.DEFINE_integer("port", 5556, "Port of the robot")

# show image flag
flags.DEFINE_bool("show_image", False, "Show image")

##############################################################################

STEP_DURATION = 0.4
STICKY_GRIPPER_NUM_STEPS = 1
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
CAMERA_TOPICS = [{"name": "/blue/image_raw"}]
ENV_PARAMS = {
    "camera_topics": CAMERA_TOPICS,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "move_duration": STEP_DURATION,
}

##############################################################################


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped


@partial(jax.jit, static_argnames="argmax")
def sample_actions(
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


def load_checkpoint(weights_path, config_path, metadata_path, example_batch_path):
    model = PretrainedModel.load_pretrained(
        weights_path, config_path, example_batch_path
    )

    with open(metadata_path, "r") as f:
        action_proprio_metadata = json.load(f)
    action_mean = jnp.array(action_proprio_metadata["action"]["mean"])
    action_std = jnp.array(action_proprio_metadata["action"]["std"])

    policy_fn = supply_rng(
        partial(
            sample_actions,
            model,
            argmax=FLAGS.deterministic,
            mean=action_mean,
            std=action_std,
            temperature=FLAGS.temperature,
        ),
    )
    return (policy_fn, model)


def main(_):
    assert (
        len(FLAGS.checkpoint_weights_path)
        == len(FLAGS.checkpoint_config_path)
        == len(FLAGS.checkpoint_metadata_path)
        == len(FLAGS.checkpoint_example_batch_path)
    )

    # policies is a dict from run_name to policy function
    policies = {}
    for (
        checkpoint_weights_path,
        checkpoint_config_path,
        checkpoint_metadata_path,
        checkpoint_example_batch_path,
    ) in zip(
        FLAGS.checkpoint_weights_path,
        FLAGS.checkpoint_config_path,
        FLAGS.checkpoint_metadata_path,
        FLAGS.checkpoint_example_batch_path,
    ):
        assert tf.io.gfile.exists(checkpoint_weights_path), checkpoint_weights_path
        checkpoint_num = int(checkpoint_weights_path.split("_")[-1])
        run_name = checkpoint_config_path.split("/")[-2]
        policies[f"{run_name}-{checkpoint_num}"] = load_checkpoint(
            checkpoint_weights_path,
            checkpoint_config_path,
            checkpoint_metadata_path,
            checkpoint_example_batch_path,
        )

    if FLAGS.initial_eep is not None:
        assert isinstance(FLAGS.initial_eep, list)
        initial_eep = [float(e) for e in FLAGS.initial_eep]
        start_state = np.concatenate([initial_eep, [0, 0, 0, 1]])
    else:
        start_state = None

    # set up environment
    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(ENV_PARAMS)
    env_params["state_state"] = list(start_state)
    widowx_client = WidowXClient(host=FLAGS.ip, port=FLAGS.port)
    widowx_client.init(env_params, image_size=FLAGS.im_size)
    env = WidowXGym(
        widowx_client, FLAGS.im_size, FLAGS.blocking, STICKY_GRIPPER_NUM_STEPS
    )
    env = HistoryWrapper(env, FLAGS.horizon)
    # env = TemporalEnsembleWrapper(env, FLAGS.pred_horizon)
    env = RHCWrapper(env, FLAGS.pred_horizon, FLAGS.exec_horizon)

    task = {
        "image_0": jnp.zeros((FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8),
    }

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
            policy_idx = int(input("select policy: "))

        policy_name = list(policies.keys())[policy_idx]
        policy_fn, model = policies[policy_name]
        model: PretrainedModel  # type hinting

        modality = input("Language or goal image? [l/g]")
        if modality == "g":
            # ask for new goal
            if task["image_0"] is None:
                print("Taking a new goal...")
                ch = "y"
            else:
                ch = input("Take a new goal? [y/n]")
            if ch == "y":
                assert isinstance(FLAGS.goal_eep, list)
                _eep = [float(e) for e in FLAGS.goal_eep]
                goal_eep = state_to_eep(_eep, 0)
                widowx_client.move_gripper(1.0)  # open gripper

                move_status = None
                while move_status != WidowXStatus.SUCCESS:
                    move_status = widowx_client.move(goal_eep, duration=1.5)

                input("Press [Enter] when ready for taking the goal image. ")
                obs = wait_for_obs(widowx_client)
                goals = jax.tree_map(lambda x: x[None], convert_obs(obs, FLAGS.im_size))
                task = model.create_tasks(goals=goals)
        else:
            # ask for new instruction
            if "language_instruction" not in task or ["language_instruction"] is None:
                ch = "y"
            else:
                ch = input("New instruction? [y/n]")
            if ch == "y":
                text = input("Instruction?")
                task = model.create_tasks(text=[text])

        # reset env
        widowx_client.reset()
        time.sleep(2.5)
        obs, _ = env.reset()

        # move to initial position
        if FLAGS.initial_eep is not None:
            assert isinstance(FLAGS.initial_eep, list)
            initial_eep = [float(e) for e in FLAGS.initial_eep]
            eep = state_to_eep(initial_eep, 0)
            widowx_client.move_gripper(1.0)  # open gripper

            # retry move action until success
            move_status = None
            while move_status != WidowXStatus.SUCCESS:
                move_status = widowx_client.move(eep, duration=1.5)

        input("Press [Enter] to start.")

        # do rollout
        last_tstep = time.time()
        images = []
        goals = []
        t = 0
        while t < FLAGS.num_timesteps:
            if time.time() > last_tstep + STEP_DURATION or FLAGS.blocking:
                last_tstep = time.time()

                # save images
                images.append(obs["image_0"][-1])
                goals.append(task["image_0"][0])

                if FLAGS.show_image:
                    bgr_img = cv2.cvtColor(obs["full_image"][-1], cv2.COLOR_RGB2BGR)
                    cv2.imshow("img_view", bgr_img)
                    cv2.waitKey(10)

                # get action
                action = np.array(policy_fn(obs, task))

                # perform environment step
                obs, _, _, truncated, _ = env.step(action)

                t += 1

                if truncated:
                    break

        # save video
        if FLAGS.video_save_path is not None:
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                FLAGS.video_save_path,
                f"{curr_time}_{policy_name}_sticky_{STICKY_GRIPPER_NUM_STEPS}.mp4",
            )
            video = np.concatenate([np.stack(goals), np.stack(images)], axis=1)
            imageio.mimsave(save_path, video, fps=1.0 / STEP_DURATION * 3)


if __name__ == "__main__":
    app.run(main)
