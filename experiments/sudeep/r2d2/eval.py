import os
import time
import flax
from datetime import datetime
import json

from absl import app, flags, logging

import numpy as np

import cv2
import jax
import jax.numpy as jnp
from functools import partial

from octo.model.octo_model import OctoModel
from octo.data.utils.data_utils import StateEncoding
import optax

# r2d2 robot imports
from r2d2.user_interface.eval_gui import EvalGUI


np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS
flags.DEFINE_bool("deterministic", False, "Whether to sample action deterministically")
flags.DEFINE_float("temperature", 1.0, "Temperature for sampling actions")
flags.DEFINE_integer("img_width", 128, "Width of input image")
flags.DEFINE_integer("img_height", 128, "Height of input image")


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


_CAMERA_MAPPINGS = {
    "16291792_left": "image_2",
    "22246076_left": "image_0",
    "26638268_left": "image_1",
}


def _resize_img(img):
    img = cv2.resize(
        img[:, :, :3], (FLAGS.img_width, FLAGS.img_height), interpolation=cv2.INTER_AREA
    )
    return img


def _null_goal():
    obs = {
        k: np.zeros((FLAGS.img_height, FLAGS.img_width, 3), dtype=np.uint8)
        for k in _CAMERA_MAPPINGS.keys()
    }
    return obs


def _null_obs():
    obs = dict()
    obs["image"] = {
        k: np.zeros((FLAGS.img_height, FLAGS.img_width, 3), dtype=np.uint8)
        for k in _CAMERA_MAPPINGS.keys()
    }
    obs["robot_state"] = dict(joint_positions=[0 for _ in range(7)])
    return obs


@partial(jax.jit, static_argnames="argmax")
def sample_actions(
    pretrained_model: OctoModel,
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
    model = OctoModel.load_pretrained(weights_path)

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
                temperature=FLAGS.temperature,
            ),
        ),
        horizon=model.config["dataset_kwargs"]["transform_kwargs"]["window_size"],
    )
    return (policy_fn, model)


class OctoPolicy:
    def __init__(self, policy_fn, img_mapping=_CAMERA_MAPPINGS):
        self.policy_fn = policy_fn
        self.img_mapping = img_mapping
        self.load_goal_imgs(_null_goal())
        self._last_time = None

    def _convert_obs(self, observation):
        obs = {
            v: _resize_img(observation["image"][k]) for k, v in self.img_mapping.items()
        }
        raw_proprio = [StateEncoding.NONE] + observation["robot_state"][
            "joint_positions"
        ]
        obs["proprio"] = np.array(raw_proprio).astype(np.float32)
        return obs

    def forward(self, observation):
        obs_hist = [self._convert_obs(observation)]
        action = np.array(self.policy_fn(obs_hist, self.goal))[0]

        cur_time = time.time()
        if self._last_time is not None:
            print("Effective HZ:", 1.0 / (cur_time - self._last_time))
        self._last_time = cur_time

        return np.clip(action, -1, 1)

    def load_goal_imgs(self, goal_dict):
        self.goal = {
            v: _resize_img(goal_dict[k])[None].copy()
            for k, v in self.img_mapping.items()
        }
        if "robot_state" in goal_dict:
            raw_proprio = [StateEncoding.NONE] + goal_dict["robot_state"][
                "joint_positions"
            ]
            self.goal["proprio"] = np.array(raw_proprio).astype(np.float32)

    def load_lang(self, text):
        pass


def main(_):
    checkpoint_weights_path = "/home/sdasari/octo/octo_gc_res128/400000/"
    checkpoint_config_path = "/home/sdasari/octo/octo_gc_res128/config.json"
    checkpoint_metadata_path = "/home/sdasari/octo/octo_gc_res128/action_proprio_metadata_r2_d2_play_cmu_rgb.json"
    checkpoint_example_batch = "/home/sdasari/octo/octo_gc_res128/example_batch.msgpack"

    policy_fn, _ = load_checkpoint(
        checkpoint_weights_path,
        checkpoint_config_path,
        checkpoint_metadata_path,
        checkpoint_example_batch,
    )

    policy = OctoPolicy(policy_fn)
    # compile the policy and run through with a null observation
    policy.forward(_null_obs())

    # start up R2D2 eval gui
    EvalGUI(policy=policy)


if __name__ == "__main__":
    app.run(main)
