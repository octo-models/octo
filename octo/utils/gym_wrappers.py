from collections import deque
import logging
from typing import Dict, Optional, Sequence, Tuple

import gym
import gym.spaces
import jax
import numpy as np
import tensorflow as tf


def stack_and_pad(history: deque, num_obs: int):
    """
    Converts a list of observation dictionaries (`history`) into a single observation dictionary
    by stacking the values. Adds a padding mask to the observation that denotes which timesteps
    represent padding based on the number of observations seen so far (`num_obs`).
    """
    horizon = len(history)
    full_obs = {k: np.stack([dic[k] for dic in history]) for k in history[0]}
    pad_length = horizon - min(num_obs, horizon)
    timestep_pad_mask = np.ones(horizon)
    timestep_pad_mask[:pad_length] = 0
    full_obs["timestep_pad_mask"] = timestep_pad_mask
    return full_obs


def space_stack(space: gym.Space, repeat: int):
    """
    Creates new Gym space that represents the original observation/action space
    repeated `repeat` times.
    """

    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=space.dtype,
        )
    elif isinstance(space, gym.spaces.Discrete):
        return gym.spaces.MultiDiscrete([space.n] * repeat)
    elif isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict(
            {k: space_stack(v, repeat) for k, v in space.spaces.items()}
        )
    else:
        raise ValueError(f"Space {space} is not supported by Octo Gym wrappers.")


def listdict2dictlist(LD):
    return {k: [dic[k] for dic in LD] for k in LD[0]}


def add_octo_env_wrappers(
    env: gym.Env,
    action_proprio_metadata: dict,
    horizon: int,
    exec_horizon: int,
    resize_size: Optional[Dict[str, Tuple]] = None,
    use_temp_ensembling: bool = True,
):
    """Adds env wrappers for proprio normalization, action prediction,
    image resizing, and history stacking.

    Arguments:
        env: gym Env
        action_proprio_metadata: dict containing proprio stats for NormalizeProprio
        horizon: int for HistoryWrapper
        exec_horizon: int for RHCWrapper or TemporalEnsembleWrapper
        resize_size: None or tuple or list of tuples for ResizeImageWrapper
        use_temp_ensembling: whether to use TemporalEnsembleWrapper or RHCWrapper
    """
    env = NormalizeProprio(env, action_proprio_metadata)
    env = ResizeImageWrapper(env, resize_size)

    env = HistoryWrapper(env, horizon)

    if use_temp_ensembling:
        env = TemporalEnsembleWrapper(env, exec_horizon)
    else:
        env = RHCWrapper(env, exec_horizon)

    return env


class HistoryWrapper(gym.Wrapper):
    """
    Accumulates the observation history into `horizon` size chunks. If the length of the history
    is less than the length of the horizon, we pad the history to the full horizon length.
    A `timestep_pad_mask` key is added to the final observation dictionary that denotes which timesteps
    are padding.
    """

    def __init__(self, env: gym.Env, horizon: int):
        super().__init__(env)
        self.horizon = horizon

        self.history = deque(maxlen=self.horizon)
        self.num_obs = 0

        self.observation_space = space_stack(self.env.observation_space, self.horizon)

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        self.num_obs += 1
        self.history.append(obs)
        assert len(self.history) == self.horizon
        full_obs = stack_and_pad(self.history, self.num_obs)

        return full_obs, reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.num_obs = 1
        self.history.extend([obs] * self.horizon)
        full_obs = stack_and_pad(self.history, self.num_obs)

        return full_obs, info


class RHCWrapper(gym.Wrapper):
    """
    Performs receding horizon control. The policy returns `pred_horizon` actions and
    we execute `exec_horizon` of them.
    """

    def __init__(self, env: gym.Env, exec_horizon: int):
        super().__init__(env)
        self.exec_horizon = exec_horizon

    def step(self, actions):
        if self.exec_horizon == 1 and len(actions.shape) == 1:
            actions = actions[None]
        assert len(actions) >= self.exec_horizon
        rewards = []
        observations = []
        infos = []

        for i in range(self.exec_horizon):
            obs, reward, done, trunc, info = self.env.step(actions[i])
            observations.append(obs)
            rewards.append(reward)
            infos.append(info)

            if done or trunc:
                break

        infos = listdict2dictlist(infos)
        infos["rewards"] = rewards
        infos["observations"] = observations

        return obs, np.sum(rewards), done, trunc, infos


class TemporalEnsembleWrapper(gym.Wrapper):
    """
    Performs temporal ensembling from https://arxiv.org/abs/2304.13705
    At every timestep we execute an exponential weighted average of the last
    `pred_horizon` predictions for that timestep.
    """

    def __init__(self, env: gym.Env, pred_horizon: int, exp_weight: int = 0):
        super().__init__(env)
        self.pred_horizon = pred_horizon
        self.exp_weight = exp_weight

        self.act_history = deque(maxlen=self.pred_horizon)

        self.action_space = space_stack(self.env.action_space, self.pred_horizon)

    def step(self, actions):
        assert len(actions) >= self.pred_horizon

        self.act_history.append(actions[: self.pred_horizon])
        num_actions = len(self.act_history)

        # select the predicted action for the current step from the history of action chunk predictions
        curr_act_preds = np.stack(
            [
                pred_actions[i]
                for (i, pred_actions) in zip(
                    range(num_actions - 1, -1, -1), self.act_history
                )
            ]
        )

        # more recent predictions get exponentially *less* weight than older predictions
        weights = np.exp(-self.exp_weight * np.arange(num_actions))
        weights = weights / weights.sum()
        # compute the weighted average across all predictions for this timestep
        action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return self.env.step(action)

    def reset(self, **kwargs):
        self.act_history = deque(maxlen=self.pred_horizon)
        return self.env.reset(**kwargs)


class ResizeImageWrapper(gym.ObservationWrapper):
    """
    Resizes images from a robot environment to the size the model expects.

    We attempt to match the resizing operations done in the model's data pipeline.
    First, we resize the image using lanczos interpolation to match the resizing done
    when converting the raw data into RLDS. Then, we crop and resize the image with
    bilinear interpolation to match the average of the crop and resize image augmentation
    performed during training.
    """

    def __init__(
        self,
        env: gym.Env,
        resize_size: Optional[Dict[str, Tuple]] = None,
        augmented_keys: Sequence[str] = ("image_primary",),
        avg_scale: float = 0.9,
        avg_ratio: float = 1.0,
    ):
        super().__init__(env)
        assert isinstance(
            self.observation_space, gym.spaces.Dict
        ), "Only Dict observation spaces are supported."
        spaces = self.observation_space.spaces
        self.resize_size = resize_size
        self.augmented_keys = augmented_keys
        if len(self.augmented_keys) > 0:
            new_height = tf.clip_by_value(tf.sqrt(avg_scale / avg_ratio), 0, 1)
            new_width = tf.clip_by_value(tf.sqrt(avg_scale * avg_ratio), 0, 1)
            height_offset = (1 - new_height) / 2
            width_offset = (1 - new_width) / 2
            self.bounding_box = tf.stack(
                [
                    height_offset,
                    width_offset,
                    height_offset + new_height,
                    width_offset + new_width,
                ],
            )

        if resize_size is None:
            self.keys_to_resize = {}
        else:
            self.keys_to_resize = {
                f"image_{i}": resize_size[i] for i in resize_size.keys()
            }
        logging.info(f"Resizing images: {self.keys_to_resize}")
        for k, size in self.keys_to_resize.items():
            spaces[k] = gym.spaces.Box(
                low=0,
                high=255,
                shape=size + (3,),
                dtype=np.uint8,
            )
        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, observation):
        for k, size in self.keys_to_resize.items():
            image = tf.image.resize(
                observation[k], size=size, method="lanczos3", antialias=True
            )

            # if this image key was augmented with random resizes and crops,
            # we perform the average of the augmentation here
            if k in self.augmented_keys:
                image = tf.image.crop_and_resize(
                    image[None], self.bounding_box[None], [0], size
                )[0]

            image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()

            observation[k] = image
        return observation


class NormalizeProprio(gym.ObservationWrapper):
    """
    Un-normalizes the proprio.
    """

    def __init__(
        self,
        env: gym.Env,
        action_proprio_metadata: dict,
    ):
        self.action_proprio_metadata = jax.tree_map(
            lambda x: np.array(x),
            action_proprio_metadata,
            is_leaf=lambda x: isinstance(x, list),
        )
        super().__init__(env)

    def normalize(self, data, metadata):
        mask = metadata.get("mask", np.ones_like(metadata["mean"], dtype=bool))
        return np.where(
            mask,
            (data - metadata["mean"]) / (metadata["std"] + 1e-8),
            data,
        )

    def observation(self, obs):
        if "proprio" in self.action_proprio_metadata:
            obs["proprio"] = self.normalize(
                obs["proprio"], self.action_proprio_metadata["proprio"]
            )
        else:
            assert "proprio" not in obs, "Cannot normalize proprio without metadata."
        return obs
