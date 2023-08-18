from collections import deque

import gym
import jax
import numpy as np


def stack_obs(obs):
    dict_list = {k: [dic[k] for dic in obs] for k in obs[0]}
    return jax.tree_map(
        lambda x: np.stack(x), dict_list, is_leaf=lambda x: type(x) == list
    )


class ActionChunkingWrapper(gym.Wrapper):
    """
    Enables receding horizon control.

    Executes act_exec_horizon actions in the environment.
    """

    def __init__(self, env: gym.Env, act_exec_horizon: int):
        super().__init__(env)
        self.env = env
        self.act_exec_horizon = act_exec_horizon

    def step(self, action, *args):
        assert len(action) >= self.act_exec_horizon

        for i in range(self.act_exec_horizon):
            obs, reward, done, trunc, info = self.env.step(action[i], *args)
        return obs, reward, done, trunc, info


class ObsHistoryWrapper(gym.Wrapper):
    """
    Enables observation histories.

    Accumulates observations into obs_horizon size chunks. Starts by repeating the first obs.

    """

    def __init__(self, env: gym.Env, obs_horizon: int):
        super().__init__(env)
        self.env = env
        self.obs_horizon = obs_horizon
        self.current_obs = deque(maxlen=self.obs_horizon)

    def step(self, *args, **kwargs):
        obs, reward, done, trunc, info = self.env.step(*args, **kwargs)
        self.current_obs.append(obs)
        return stack_obs(self.current_obs), reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        self.current_obs.extend([obs] * self.obs_horizon)
        return stack_obs(self.current_obs), info
