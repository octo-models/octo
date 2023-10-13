from collections import deque

import gym
import gym.spaces
import jax
import numpy as np


def stack_and_pad(history, num_obs, horizon):
    dict_list = {k: [dic[k] for dic in history] for k in history[0]}
    full_obs = jax.tree_map(
        lambda x: np.stack(x), dict_list, is_leaf=lambda x: type(x) is list
    )
    pad_length = horizon - max(num_obs, horizon)
    pad_mask = np.ones(horizon)
    pad_mask[:pad_length] = 0
    full_obs["pad_mask"] = pad_mask
    return full_obs


def space_stack(space: gym.Space, repeat: int):
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
        raise TypeError()


class HistoryWrapper(gym.Wrapper):
    """
    Accumulates the observation history into `horizon` size chunks. If the length of the history
    is less than the length of the horizon, we pad the history to the full horizon length.
    A `pad_mask` key is added to the final observation dictionary that denotes which timesteps
    are padding.
    """

    def __init__(self, env: gym.Env, horizon: int, pred_horizon: int):
        super().__init__(env)
        self.horizon = horizon
        self.pred_horizon = pred_horizon

        self.history = deque(maxlen=self.horizon)
        self.num_obs = 0

        self.observation_space = space_stack(self.env.observation_space, self.horizon)

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        self.num_obs += 1
        self.history.append(obs)
        full_obs = stack_and_pad(self.history, self.num_obs, self.horizon)

        return full_obs, reward, done, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.num_obs = 1
        self.history.extend([obs] * self.horizon)
        full_obs = stack_and_pad(self.history, self.num_obs, self.horizon)

        return full_obs, info


class RHCWrapper(gym.Wrapper):
    """
    Performs receding horizon control. The policy returns `pred_horizon` actions and
    we execute `exec_horizon` of them.
    """

    def __init__(
        self, env: gym.Env, horizon: int, pred_horizon: int, exec_horizon: int
    ):
        super().__init__(env)
        assert exec_horizon <= pred_horizon

        self.horizon = horizon
        self.pred_horizon = pred_horizon
        self.exec_horizon = exec_horizon

        self.action_space = space_stack(self.env.action_space, self.pred_horizon)

    def step(self, actions, *args):
        assert len(actions) == self.pred_horizon

        for i in range(self.exec_horizon):
            obs, reward, done, trunc, info = self.env.step(actions[i], *args)
            if done or trunc:
                break

        return obs, reward, done, trunc, info


class TemporalEnsembleWrapper(gym.Wrapper):
    """
    Performs temporal ensembling from https://arxiv.org/abs/2304.13705
    At every timestep we execute an exponential weighted average of the last
    `pred_horizon` predictions for that timestep.
    """

    def __init__(
        self, env: gym.Env, horizon: int, pred_horizon: int, exp_weight: int = 0
    ):
        super().__init__(env)
        self.horizon = horizon
        self.pred_horizon = pred_horizon
        self.weights = np.exp(-exp_weight * np.arange(pred_horizon)) / pred_horizon
        self.avg_actions = None

        self.action_space = space_stack(self.env.action_space, self.pred_horizon)

    def step(self, actions):
        assert len(actions) == self.pred_horizon

        if self.avg_actions is None:
            self.avg_actions = np.zeros_like(actions)

        # shift the averaged actions to the left by one
        shifted_actions = np.concatenate(
            [self.avg_actions[1:], np.zeros_like(self.avg_actions[-1:])]
        )
        # weight the current predicted actions and add them to the averaged actions
        self.avg_actions = shifted_actions + (self.weights[:, None] * actions)

        # execute the first action
        action = self.avg_actions[0]
        obs, reward, done, trunc, info = self.env.step(action)

        return obs, reward, done, trunc, info


class UnnormalizeActionProprio(gym.Wrapper):
    """
    Un-normalizes the action and proprio.
    """

    def __init__(
        self, env: gym.Env, action_proprio_metadata: dict, normalization_type: str
    ):
        self.action_proprio_metadata = action_proprio_metadata
        self.normalization_type = normalization_type
        super().__init__(env)

    def step(self, action, *args, **kwargs):
        if self.normalization_type == "normal":
            action = (
                action * self.action_proprio_metadata["action"]["std"]
            ) + self.action_proprio_metadata["action"]["mean"]
            obs, reward, done, trunc, info = self.env.step(action, *args, **kwargs)
            obs["proprio"] = (
                obs["proprio"] * self.action_proprio_metadata["proprio"]["std"]
            ) + self.action_proprio_metadata["proprio"]["mean"]
        elif self.normalization_type == "bounds":
            action = (
                action
                * (
                    self.action_proprio_metadata["action"]["max"]
                    - self.action_proprio_metadata["action"]["min"]
                )
            ) + self.action_proprio_metadata["action"]["min"]
            obs, reward, done, trunc, info = self.env.step(action, *args, **kwargs)
            obs["proprio"] = (
                obs["proprio"]
                * (
                    self.action_proprio_metadata["proprio"]["max"]
                    - self.action_proprio_metadata["proprio"]["min"]
                )
            ) + self.action_proprio_metadata["proprio"]["min"]
        else:
            raise ValueError

        return obs, reward, done, trunc, info
