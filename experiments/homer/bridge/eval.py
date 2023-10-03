import sys
import os
import time
from datetime import datetime
import traceback
from collections import deque
import json

from absl import app, flags, logging

import numpy as np
import tensorflow as tf

import jax
import jax.numpy as jnp
from PIL import Image
import imageio
from functools import partial

from flax.training import checkpoints
from orca.model import create_model_def
from orca.utils.train_utils import create_train_state
from orca.data.utils.text_processing import text_processors
import optax

# bridge_data_robot imports
from widowx_envs.widowx_env import BridgeDataRailRLPrivateWidowX
from multicam_server.topic_utils import IMTopic

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
flags.DEFINE_integer("im_size", None, "Image size", required=True)
flags.DEFINE_string("video_save_path", None, "Path to save video")
flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_bool("blocking", False, "Use the blocking controller")
flags.DEFINE_spaceseplist("goal_eep", None, "Goal position")
flags.DEFINE_spaceseplist("initial_eep", None, "Initial position")
flags.DEFINE_integer("act_exec_horizon", 1, "Action sequence length")
flags.DEFINE_bool("deterministic", False, "Whether to sample action deterministically")

##############################################################################

STEP_DURATION = 0.2
NO_PITCH_ROLL = False
NO_YAW = False
STICKY_GRIPPER_NUM_STEPS = 1
WORKSPACE_BOUNDS = np.array([[0.1, -0.15, -0.1, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]])
CAMERA_TOPICS = [IMTopic("/blue/image_raw")]
FIXED_STD = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

##############################################################################


def stack_and_pad_obs(fn, horizon):
    """
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
    observations, tasks, state, mean, std, rng, argmax=False, temperature=1.0
):
    # add batch dim
    observations = jax.tree_map(lambda x: x[None], observations)
    tasks = jax.tree_map(lambda x: x[None], tasks)
    actions = state.apply_fn(
        {"params": state.params},
        observations,
        tasks,
        train=False,
        argmax=argmax,
        rng=rng,
        temperature=temperature,
        method="predict_action",
    )
    # remove batch dim
    return actions[0] * std + mean


def load_checkpoint(weights_path, config_path, metadata_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    window_size = config["dataset_kwargs"]["common_kwargs"]["window_size"]
    horizon = window_size - config["model"]["policy_kwargs"]["pred_horizon"] + 1

    example_actions = jnp.zeros((1, window_size, 7), dtype=np.float32)
    example_obs = {
        "image_0": jnp.zeros(
            (1, window_size, FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8
        ),
        "pad_mask": jnp.ones((1, window_size)),
    }

    if config["text_processor"] is None:
        text_processor = None
        language_embed = None
    else:
        text_processor = text_processors[config["text_processor"]](
            **config["text_processor_kwargs"]
        )
        language_embed = text_processor.encode([[""]])

    example_batch = {
        "observations": example_obs,
        "tasks": {
            "image_0": jnp.zeros((1, FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8),
            "language_instruction": language_embed,
        },
        "actions": example_actions,
    }

    # create train_state
    rng = jax.random.PRNGKey(0)
    rng, construct_rng = jax.random.split(rng)

    model_def = create_model_def(
        action_dim=example_batch["actions"].shape[-1],
        window_size=example_batch["observations"]["image_0"].shape[1],
        **config["model"],
    )

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config["optimizer"]["learning_rate"],
        warmup_steps=config["optimizer"]["warmup_steps"],
        decay_steps=config["optimizer"]["decay_steps"],
        end_value=0.0,
    )

    tx = optax.adam(lr_schedule)
    train_state = create_train_state(
        construct_rng,
        model_def,
        tx,
        init_args=(
            example_batch["observations"],
            example_batch["tasks"],
            example_batch["actions"],
        ),
    )

    # hydrate train_state with parameters from checkpoint
    train_state = checkpoints.restore_checkpoint(weights_path, train_state)

    with open(metadata_path, "r") as f:
        action_proprio_metadata = json.load(f)
    action_mean = jnp.array(action_proprio_metadata["action"]["mean"])
    action_std = jnp.array(action_proprio_metadata["action"]["std"])

    rng, policy_rng = jax.random.split(rng)

    policy_fn = stack_and_pad_obs(
        supply_rng(
            partial(
                sample_actions,
                state=train_state,
                argmax=FLAGS.deterministic,
                mean=action_mean,
                std=action_std,
            ),
            rng=policy_rng,
        ),
        horizon=horizon,
    )

    return policy_fn, text_processor


def main(_):
    assert (
        len(FLAGS.checkpoint_weights_path)
        == len(FLAGS.checkpoint_config_path)
        == len(FLAGS.checkpoint_metadata_path)
    )

    # policies is a dict from run_name to policy function
    policies = {}
    for (
        checkpoint_weights_path,
        checkpoint_config_path,
        checkpoint_metadata_path,
    ) in zip(
        FLAGS.checkpoint_weights_path,
        FLAGS.checkpoint_config_path,
        FLAGS.checkpoint_metadata_path,
    ):
        assert tf.io.gfile.exists(checkpoint_weights_path), checkpoint_weights_path
        checkpoint_num = int(checkpoint_weights_path.split("_")[-1])
        run_name = checkpoint_config_path.split("/")[-1]
        policies[f"{run_name}-{checkpoint_num}"] = load_checkpoint(
            checkpoint_weights_path, checkpoint_config_path, checkpoint_metadata_path
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
        "start_state": start_state,
        "return_full_image": False,
        "camera_topics": CAMERA_TOPICS,
    }
    env = BridgeDataRailRLPrivateWidowX(env_params, fixed_image_size=FLAGS.im_size)

    task = {"image_0": jnp.zeros((FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8)}

    # goal sampling loop
    while True:
        # ask for which policy to use
        if len(policies) == 1:
            policy_idx = 0
            input("Press [Enter] to start.")
        else:
            print("policies:")
            for i, name in enumerate(policies.keys()):
                print(f"{i}) {name}")
            policy_idx = int(input("select policy: "))

        policy_name = list(policies.keys())[policy_idx]
        policy_fn, text_processor = policies[policy_name]

        modality = input("Language or goal image? [l/g]")
        if modality == "g":
            # ask for new goal
            if task["image_0"] is None:
                print("Taking a new goal...")
                ch = "y"
            else:
                ch = input("Taking a new goal? [y/n]")
            if ch == "y":
                if FLAGS.goal_eep is not None:
                    assert isinstance(FLAGS.goal_eep, list)
                    goal_eep = [float(e) for e in FLAGS.goal_eep]
                else:
                    low_bound = WORKSPACE_BOUNDS[0][:3] + 0.03
                    high_bound = WORKSPACE_BOUNDS[1][:3] - 0.03
                    goal_eep = np.random.uniform(low_bound, high_bound)
                env.controller().open_gripper(True)
                try:
                    env.controller().move_to_state(goal_eep, 0, duration=1.5)
                    env._reset_previous_qpos()
                except Exception as e:
                    continue
                input("Press [Enter] when ready for taking the goal image. ")
                obs = env.current_obs()
                task = convert_obs(obs)

                # create a dummy language input if this model also expects language
                if text_processor is not None:
                    task["language_instruction"] = text_processor.encode("")

        else:
            # ask for new instruction
            if task["language_instruction"] is None:
                ch = "y"
            else:
                ch = input("New instruction? [y/n]")
            if ch == "y":
                task["language_instruction"] = text_processor.encode(
                    input("Instruction?")
                )

        try:
            env.reset()
            env.start()
        except Exception as e:
            continue

        # move to initial position
        try:
            if FLAGS.initial_eep is not None:
                assert isinstance(FLAGS.initial_eep, list)
                initial_eep = [float(e) for e in FLAGS.initial_eep]
                env.controller().move_to_state(initial_eep, 0, duration=1.5)
                env._reset_previous_qpos()
        except Exception as e:
            continue

        input("start?")

        # do rollout
        obs = env.current_obs()
        obs = convert_obs(obs)
        obs_hist = [obs]
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
                        obs, _, _, _ = env.step(
                            action, last_tstep + STEP_DURATION, blocking=FLAGS.blocking
                        )
                        obs = convert_obs(obs)
                        obs_hist.append(obs)

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
