# Taken from
# https://github.com/denisyarats/dmc2gym
# and modified to exclude duplicated code.

import copy
from typing import OrderedDict

import dm_env
import gym
import numpy as np
from gym import spaces


def filter_info(obs):
    new_obs = {}
    info = {}
    for k, v in obs.items():
        if "info" in k:
            info_key = k.split("/")[-1]
            info[info_key] = v
        else:
            new_obs[k] = v
    return new_obs, info


def dmc_spec2gym_space(spec):
    if isinstance(spec, OrderedDict) or isinstance(spec, dict):
        new_spec = OrderedDict()
        for k, v in spec.items():
            if "info" in k:
                continue
            new_spec[k] = dmc_spec2gym_space(v)
        return spaces.Dict(new_spec)
    elif isinstance(spec, dm_env.specs.BoundedArray):
        low = np.broadcast_to(spec.minimum, spec.shape)
        high = np.broadcast_to(spec.maximum, spec.shape)
        return spaces.Box(low=low, high=high, shape=spec.shape, dtype=spec.dtype)
    elif isinstance(spec, dm_env.specs.Array):
        if np.issubdtype(spec.dtype, np.integer):
            low = np.iinfo(spec.dtype).min
            high = np.iinfo(spec.dtype).max
        elif np.issubdtype(spec.dtype, np.inexact):
            low = float("-inf")
            high = float("inf")
        else:
            raise ValueError()

        return spaces.Box(low=low, high=high, shape=spec.shape, dtype=spec.dtype)
    else:
        raise NotImplementedError


def dmc_obs2gym_obs(obs):
    if isinstance(obs, OrderedDict) or isinstance(obs, dict):
        obs = copy.copy(obs)
        for k, v in obs.items():
            obs[k] = dmc_obs2gym_obs(v)
        return obs
    else:
        return np.asarray(obs)


class DMCGYM(gym.core.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env: dm_env.Environment):
        self._env = env

        self.action_space = dmc_spec2gym_space(self._env.action_spec())

        self.observation_space = dmc_spec2gym_space(self._env.observation_spec())

        self.viewer = None

    def _get_viewer(self):
        if self.viewer is None:
            from gym.envs.mujoco.mujoco_rendering import Viewer

            self.viewer = Viewer(
                self._env.physics.model.ptr, self._env.physics.data.ptr
            )
        return self.viewer

    def __getattr__(self, name):
        return getattr(self._env, name)

    def seed(self, seed: int):
        if hasattr(self._env, "random_state"):
            self._env.random_state.seed(seed)
        else:
            self._env.task.random.seed(seed)

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        time_step = self._env.step(action)
        reward = time_step.reward
        done = time_step.last()
        obs, info = filter_info(time_step.observation)

        trunc = done and time_step.discount == 1.0

        return dmc_obs2gym_obs(obs), reward, done, trunc, info

    def reset(self):
        time_step = self._env.reset()
        obs, info = filter_info(time_step.observation)
        return dmc_obs2gym_obs(obs), info

    def render(
        self, mode="rgb_array", height: int = 128, width: int = 128, camera_id: int = 0
    ):
        assert mode in ["human", "rgb_array"], (
            "only support rgb_array and human mode, given %s" % mode
        )
        if mode == "rgb_array":
            return self._env.physics.render(
                height=height, width=width, camera_id=camera_id
            )
        elif mode == "human":
            self._get_viewer().render()
