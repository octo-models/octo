from collections import deque
import logging
from typing import Optional, Sequence, Tuple, Union

import gym
import gym.spaces
import jax
import numpy as np
import tensorflow as tf


def stack_and_pad(history: list, num_obs: int):
    """
    Converts a list of observation dictionaries (`history`) into a single observation dictionary
    by stacking the values. Adds a padding mask to the observation that denotes which timesteps
    represent padding based on the number of observations seen so far (`num_obs`).
    """
    horizon = len(history)
    full_obs = {k: np.stack([dic[k] for dic in history]) for k in history[0]}
    pad_length = horizon - max(num_obs, horizon)
    pad_mask = np.ones(horizon)
    pad_mask[:pad_length] = 0
    full_obs["pad_mask"] = pad_mask
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
    env: gym.Env, config: dict, dataset_statistics: dict, **kwargs
):
    """Adds env wrappers for action normalization, multi-action
    future prediction, image resizing, and history stacking.

    Uses defaults from model config, but all can be overridden through kwargs.

    Arguments:
        env: gym Env
        config: PretrainedModel.config
        dataset_statistics: from PretrainedModel.load_dataset_statistics
        # Additional (optional) kwargs
        normalization_type: str for UnnormalizeActionProprio
        exec_horizon: int for RHCWrapper
        resize_size: None or tuple or list of tuples for ResizeImageWrapper
        horizon: int for HistoryWrapper
    """
    normalization_type = kwargs.get(
        "normalization_type",
        config["dataset_kwargs"]["common_dataset_kwargs"][
            "action_proprio_normalization_type"
        ],
    )

    logging.info(
        "Unnormalizing proprio and actions w/ statistics: ", dataset_statistics
    )
    env = UnnormalizeActionProprio(env, dataset_statistics, normalization_type)
    exec_horizon = kwargs.get(
        "exec_horizon", config["model"]["heads"]["action"]["kwargs"]["pred_horizon"]
    )

    logging.info("Running receding horizon control with exec_horizon: ", exec_horizon)
    env = RHCWrapper(env, exec_horizon)
    resize_size = kwargs.get(
        "resize_size",
        config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"],
    )

    logging.info("Resizing images w/ parameters", resize_size)
    env = ResizeImageWrapper(env, resize_size)

    horizon = kwargs.get("horizon", config["window_size"])
    logging.info("Adding history of size: ", horizon)
    env = HistoryWrapper(env, horizon)

    logging.info("New observation space: ", env.observation_space)
    return env


class HistoryWrapper(gym.Wrapper):
    """
    Accumulates the observation history into `horizon` size chunks. If the length of the history
    is less than the length of the horizon, we pad the history to the full horizon length.
    A `pad_mask` key is added to the final observation dictionary that denotes which timesteps
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


class ResizeImageWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        resize_size: Optional[Union[Tuple, Sequence[Tuple]]],
    ):
        super().__init__(env)
        assert isinstance(
            self.observation_space, gym.spaces.Dict
        ), "Only Dict observation spaces are supported."
        spaces = self.observation_space.spaces
        self.resize_size = resize_size

        if resize_size is None:
            self.keys_to_resize = {}
        elif isinstance(resize_size, tuple):
            self.keys_to_resize = {k: resize_size for k in spaces if "image_" in k}
        else:
            self.keys_to_resize = {
                f"image_{i}": resize_size[i] for i in range(len(resize_size))
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
            image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8).numpy()
            observation[k] = image
        return observation


class UnnormalizeActionProprio(gym.ActionWrapper, gym.ObservationWrapper):
    """
    Un-normalizes the action and proprio.
    """

    def __init__(
        self,
        env: gym.Env,
        action_proprio_metadata: dict,
        normalization_type: str,
    ):
        self.action_proprio_metadata = jax.tree_map(
            lambda x: np.array(x),
            action_proprio_metadata,
            is_leaf=lambda x: isinstance(x, list),
        )
        self.normalization_type = normalization_type
        super().__init__(env)

    def unnormalize(self, data, metadata):
        mask = metadata.get("mask", np.ones_like(metadata["mean"], dtype=bool))
        if self.normalization_type == "normal":
            return np.where(
                mask,
                (data * metadata["std"]) + metadata["mean"],
                data,
            )
        elif self.normalization_type == "bounds":
            return np.where(
                mask,
                ((data + 1) / 2 * (metadata["max"] - metadata["min"] + 1e-8))
                + metadata["min"],
                data,
            )
        else:
            raise ValueError(
                f"Unknown action/proprio normalization type: {self.normalization_type}"
            )

    def normalize(self, data, metadata):
        mask = metadata.get("mask", np.ones_like(metadata["mean"], dtype=bool))
        if self.normalization_type == "normal":
            return np.where(
                mask,
                (data - metadata["mean"]) / (metadata["std"] + 1e-8),
                data,
            )
        elif self.normalization_type == "bounds":
            return np.where(
                mask,
                np.clip(
                    2
                    * (data - metadata["min"])
                    / (metadata["max"] - metadata["min"] + 1e-8)
                    - 1,
                    -1,
                    1,
                ),
                data,
            )
        else:
            raise ValueError(
                f"Unknown action/proprio normalization type: {self.normalization_type}"
            )

    def action(self, action):
        return self.unnormalize(action, self.action_proprio_metadata["action"])

    def observation(self, obs):
        obs["proprio"] = self.normalize(
            obs["proprio"], self.action_proprio_metadata["proprio"]
        )
        return obs
