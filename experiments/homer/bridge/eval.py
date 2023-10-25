#!/usr/bin/env python3

import json
import os
import sys
import time
import traceback
from collections import deque
from datetime import datetime
from functools import partial

import cv2
import flax
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from absl import app, flags, logging
from experiments.homer.bridge.pretrained_utils import PretrainedModelWrapper
from PIL import Image
from pyquaternion import Quaternion

# bridge_data_robot imports
from widowx_envs.widowx_env_service import WidowXClient, WidowXStatus

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
flags.DEFINE_integer("act_exec_horizon", 1, "Action sequence length")
flags.DEFINE_bool("deterministic", False, "Whether to sample action deterministically")
flags.DEFINE_string("ip", "localhost", "IP address of the robot")
flags.DEFINE_integer("port", 5556, "Port of the robot")

# show image flag
flags.DEFINE_bool("show_image", False, "Show image")

##############################################################################

STEP_DURATION = 0.2
NO_PITCH_ROLL = False
NO_YAW = False
STICKY_GRIPPER_NUM_STEPS = 1
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.1, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
CAMERA_TOPICS = [{"name": "/blue/image_raw"}]
FIXED_STD = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

##############################################################################


def stack_and_pad_obs(fn, horizon):
    """
    TODO: Replace with env wrapper
    This turns a function that takes a fixed length observation history into a function that
    takes just the current observation (or sequence of observations since the last policy call).
    The full observation history is saved inside this function. This function handles stacking
    the list of observation dictionaries to form a dictionary of arrays. This function also pads
    the observation history to the full horizon length. A `pad_mask` key is added to the final
    observation dictionary that denotes which timesteps are padding.
    """

    full_history = []

    def stack_obs(obs):
        dict_list = {k: [dic[k] for dic in obs] for k in obs[0]}
        return jax.tree_map(
            lambda x: np.stack(x), dict_list, is_leaf=lambda x: type(x) == list
        )

    def wrapped_fn(obs, *args, **kwargs):
        nonlocal full_history
        if isinstance(obs, list):
            full_history.extend(obs)
        else:
            full_history.append(obs)
        history = full_history[-horizon:]
        pad_length = horizon - len(history)
        pad_mask = np.ones(horizon)
        pad_mask[:pad_length] = 0
        history = [history[0]] * pad_length + history
        full_obs = stack_obs(history)
        full_obs["pad_mask"] = pad_mask
        return fn(full_obs, *args, **kwargs)

    return wrapped_fn


def state_to_eep(xyz_coor, zangle: float):
    """
    Implement the state to eep function.
    Refered to `bridge_data_robot`'s `widowx_controller/widowx_controller.py`
    return a 4x4 matrix
    """
    assert len(xyz_coor) == 3
    DEFAULT_ROTATION = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
    new_pose = np.eye(4)
    new_pose[:3, -1] = xyz_coor
    new_quat = Quaternion(axis=np.array([0.0, 0.0, 1.0]), angle=zangle) * Quaternion(
        matrix=DEFAULT_ROTATION
    )
    new_pose[:3, :3] = new_quat.rotation_matrix
    # yaw, pitch, roll = quat.yaw_pitch_roll
    return new_pose


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped


def convert_obs(obs):
    image_obs = (
        obs["image"].reshape(3, FLAGS.im_size, FLAGS.im_size).transpose(1, 2, 0) * 255
    ).astype(np.uint8)
    return {"image_0": image_obs, "proprio": obs["state"]}


@partial(jax.jit, static_argnames="argmax")
def sample_actions(
    pretrained_model: PretrainedModelWrapper,
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
        seed=rng,
        argmax=argmax,
        temperature=temperature,
    )
    # remove batch dim
    return actions[0] * std + mean


def load_checkpoint(weights_path, config_path, metadata_path, example_batch_path):
    model = PretrainedModelWrapper.load_pretrained(
        weights_path, config_path, example_batch_path
    )

    with open(metadata_path, "r") as f:
        action_proprio_metadata = json.load(f)
    action_mean = jnp.array(action_proprio_metadata["action"]["mean"])
    action_std = jnp.array(action_proprio_metadata["action"]["std"])

    policy_fn = stack_and_pad_obs(
        supply_rng(
            partial(
                sample_actions,
                model,
                argmax=FLAGS.deterministic,
                mean=action_mean,
                std=action_std,
            ),
        ),
        horizon=model.config["dataset_kwargs"]["common_kwargs"]["window_size"],
    )
    return (policy_fn, model)


def wait_for_obs(widowx_client):
    obs = widowx_client.get_observation()
    while obs is None:
        print("Waiting for observations...")
        obs = widowx_client.get_observation()
        time.sleep(1)
    return obs


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
        run_name = checkpoint_config_path.split("/")[-1]
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
    env_params = {
        "fix_zangle": 0.1,
        "move_duration": 0.2,
        "adaptive_wait": True,
        "move_to_rand_start_freq": 1,
        "override_workspace_boundaries": WORKSPACE_BOUNDS,
        "action_clipping": "xyz",
        "catch_environment_except": False,
        "start_state": list(start_state),
        "return_full_image": False,
        "camera_topics": CAMERA_TOPICS,
    }

    widowx_client = WidowXClient(FLAGS.ip, FLAGS.port)
    widowx_client.init(env_params, image_size=FLAGS.im_size)

    # goal sampling loop
    while True:
        # ask for which policy to use
        if len(policies) == 1:
            policy_idx = 0
            print("Use default policy 1: ", list(policies.keys())[policy_idx])
        else:
            print("policies:")
            for i, name in enumerate(policies.keys()):
                print(f"{i}) {name}")
            policy_idx = int(input("select policy: "))

        policy_name = list(policies.keys())[policy_idx]
        policy_fn, model = policies[policy_name]
        model: PretrainedModelWrapper  # type hinting

        modality = input("Language or goal image? [l/g]")
        if modality == "g":
            # ask for new goal
            if task["image_0"] is None:
                print("Taking a new goal...")
                ch = "y"
            else:
                ch = input("Taking a new goal? [y/n]")
            if ch == "y":
                assert isinstance(FLAGS.goal_eep, list)
                _eep = [float(e) for e in FLAGS.goal_eep]
                goal_eep = state_to_eep(_eep, 0)

                move_status = None
                while move_status != WidowXStatus.SUCCESS:
                    move_status = widowx_client.move(goal_eep, duration=1.5)

                input("Press [Enter] when ready for taking the goal image. ")
                obs = wait_for_obs(widowx_client)
                goals = jax.tree_map(lambda x: x[None], convert_obs(obs))
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

        # move to initial position
        if FLAGS.initial_eep is not None:
            assert isinstance(FLAGS.initial_eep, list)
            initial_eep = [float(e) for e in FLAGS.initial_eep]
            eep = state_to_eep(initial_eep, 0)

            # retry move action until success
            move_status = None
            while move_status != WidowXStatus.SUCCESS:
                move_status = widowx_client.move(eep, duration=1.5)

        input("start?")

        # do rollout
        obs_hist = []
        last_tstep = time.time()
        images = []
        goals = []
        t = 0

        # keep track of our own gripper state to implement sticky gripper
        is_gripper_closed = False
        num_consecutive_gripper_change_actions = 0
        try:
            while t < FLAGS.num_timesteps:
                if time.time() > last_tstep + STEP_DURATION or FLAGS.blocking:
                    last_tstep = time.time()

                    raw_obs = wait_for_obs(widowx_client)
                    obs = convert_obs(raw_obs)
                    obs_hist.append(obs)

                    if FLAGS.show_image:
                        bgr_img = cv2.cvtColor(raw_obs["full_image"], cv2.COLOR_RGB2BGR)
                        cv2.imshow("img_view", bgr_img)
                        cv2.waitKey(10)

                    actions = np.array(policy_fn(obs_hist, task))
                    assert len(actions) >= FLAGS.act_exec_horizon

                    obs_hist = []
                    for i in range(FLAGS.act_exec_horizon):
                        action = actions[i]
                        action += np.random.normal(0, FIXED_STD)

                        # sticky gripper logic
                        if (action[-1] < 0.5) != is_gripper_closed:
                            num_consecutive_gripper_change_actions += 1
                        else:
                            num_consecutive_gripper_change_actions = 0

                        if (
                            num_consecutive_gripper_change_actions
                            >= STICKY_GRIPPER_NUM_STEPS
                        ):
                            is_gripper_closed = not is_gripper_closed
                            num_consecutive_gripper_change_actions = 0

                        action[-1] = 0.0 if is_gripper_closed else 1.0

                        # remove degrees of freedom
                        if NO_PITCH_ROLL:
                            action[3] = 0
                            action[4] = 0
                        if NO_YAW:
                            action[5] = 0

                        # perform environment step
                        widowx_client.step_action(action, blocking=FLAGS.blocking)

                        # save image
                        images.append(obs["image_0"])
                        goals.append(task["image_0"])

                        t += 1
        except Exception as e:
            print(traceback.format_exc(), file=sys.stderr)

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
