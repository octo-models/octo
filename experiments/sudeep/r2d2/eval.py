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

import cv2
import jax
import jax.numpy as jnp
import imageio
from functools import partial

from flax.training import checkpoints
from orca.model import create_model_def
from orca.utils.train_utils import create_train_state
from orca.data.utils.text_processing import text_processors
import optax

# r2d2 robot imports
from r2d2.controllers.oculus_controller import VRPolicy
from r2d2.robot_env import RobotEnv
from r2d2.user_interface.data_collector import DataCollecter
from r2d2.user_interface.eval_gui import EvalGUI


np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_string("video_save_path", None, "Path to save video")
flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_bool("blocking", False, "Use the blocking controller")
flags.DEFINE_spaceseplist("goal_eep", None, "Goal position")
flags.DEFINE_spaceseplist("initial_eep", None, "Initial position")
flags.DEFINE_integer("act_exec_horizon", 1, "Action sequence length")
flags.DEFINE_bool("deterministic", False, "Whether to sample action deterministically")


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


def _resize_img(img):
    img = cv2.resize(img[:,:,:3], (320, 180), interpolation=cv2.INTER_AREA)
    return img


def _null_goal():
    return np.zeros((180, 320, 3), dtype=np.uint8)


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
            (1, window_size, 180, 320, 3), dtype=np.uint8
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
            "image_0": jnp.zeros((1, 180, 320, 3), dtype=np.uint8),
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

class OrcaPolicy:
    def __init__(self, policy_fn, img_key='16291792_left'):
        self.policy_fn = policy_fn
        self.set_goal(_null_goal())
        self.img_key = img_key
        self._last_time = None

    def set_goal(self, goal_img):
        self.task = dict(image_0=goal_img)

    def _convert_obs(self, observation):
        image = _resize_img(observation['image'][self.img_key])
        state = np.array(observation['robot_state']['joint_positions']).astype(np.float32)
        return {"image_0": image, "proprio": state}

    def forward(self, observation):
        obs_hist = [self._convert_obs(observation)]
        action   = np.array(self.policy_fn(obs_hist, self.task))[0]

        cur_time = time.time()
        if self._last_time is not None:
            print('Effective HZ:', 1.0 / (cur_time - self._last_time))
        self._last_time = cur_time

        return np.clip(action, -1, 1)

    def load_goal_img_dir(self, goal_img_dir):
        print(f"loaded goal imag dir: {goal_img_dir}")
        img_path = os.path.join(goal_img_dir, '0.png')
        goal_img = _resize_img(cv2.imread(img_path))
        self.set_goal(goal_img)

    def load_lang_conditioning(self, text):
        return


def main(_):
    checkpoint_weights_path  = '/home/sdasari/orca/checkpoints/orca/orca_r2d2_pen_mix_20231011_164455/checkpoint_40000/'
    checkpoint_config_path   = '/home/sdasari/orca/checkpoints/orca/orca_r2d2_pen_mix_20231011_164455/config.json'
    checkpoint_metadata_path = '/home/sdasari/orca/checkpoints/orca/orca_r2d2_pen_mix_20231011_164455/action_proprio_metadata_r2_d2_pen_ourlab.json'

    policy_fn, _ = load_checkpoint(
            checkpoint_weights_path, checkpoint_config_path, checkpoint_metadata_path
        )

    policy = OrcaPolicy(policy_fn)

    # compile model
    observation = dict(image={policy.img_key: np.ones((180, 320, 3), dtype=np.uint8) * 255},
                       robot_state=dict(joint_positions=[0 for _ in range(7)]))
    policy.forward(observation)

    # start up R2D2 eval gui
    EvalGUI(policy=policy)


if __name__ == "__main__":
    app.run(main)
