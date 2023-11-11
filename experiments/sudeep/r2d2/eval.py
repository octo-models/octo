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

from orca.utils.pretrained_utils import PretrainedModel
from orca.data.utils.data_utils import StateEncoding
import optax

# r2d2 robot imports
from r2d2.user_interface.eval_gui import EvalGUI


np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS
flags.DEFINE_bool("deterministic", False, "Whether to sample action deterministically")
flags.DEFINE_float("temperature", 1.0, "Temperature for sampling actions")


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
        horizon=model.config["dataset_kwargs"]["common_kwargs"]["window_size"],
    )
    return (policy_fn, model)


class OrcaPolicy:
    def __init__(self, policy_fn, img_key='16291792_left'):
        self.policy_fn = policy_fn
        self.set_goal(_null_goal())
        self.img_key = img_key
        self._last_time = None

    def set_goal(self, goal_img):
        self.task = dict(image_0=goal_img[None].copy())

    def _convert_obs(self, observation):
        image = _resize_img(observation['image'][self.img_key])
        obs = [StateEncoding.NONE] + observation['robot_state']['joint_positions']
        state = np.array(obs).astype(np.float32)
        print(image.shape, image.dtype, state.shape, state.dtype)
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
    checkpoint_weights_path  = '/home/sdasari/orca/reverify/orca/test_20231102_024636/checkpoint_40000/'
    checkpoint_config_path   = '/home/sdasari/orca/reverify/orca/test_20231102_024636/config.json'
    checkpoint_metadata_path = '/home/sdasari/orca/reverify/orca/test_20231102_024636/action_proprio_metadata_r2_d2_pen_ourlab.json'
    checkpoint_example_batch = '/home/sdasari/orca/reverify/orca/test_20231102_024636/example_batch.msgpack'

    policy_fn, _ = load_checkpoint(
            checkpoint_weights_path, checkpoint_config_path, checkpoint_metadata_path, checkpoint_example_batch
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
