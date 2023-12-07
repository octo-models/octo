from collections import deque

import gym
import gym.spaces
import numpy as np


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
        raise ValueError(f"Space {space} is not supported by ORCA Gym wrappers.")


def listdict2dictlist(LD):
    return {k: [dic[k] for dic in LD] for k in LD[0]}


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
        weights = np.exp(-self.exp_weight * np.arange(num_actions)) / num_actions

        # compute the weighted average across all predictions for this timestep
        action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return self.env.step(action)


class UnnormalizeActionProprio(gym.ActionWrapper, gym.ObservationWrapper):
    """
    Un-normalizes the action and proprio.
    """

    def __init__(
        self, env: gym.Env, action_proprio_metadata: dict, normalization_type: str
    ):
        self.action_proprio_metadata = action_proprio_metadata
        self.normalization_type = normalization_type
        super().__init__(env)

    def unnormalize(self, data, metadata):
        if self.normalization_type == "normal":
            return (data * metadata["std"]) + metadata["mean"]
        elif self.normalization_type == "bounds":
            return (data * (metadata["max"] - metadata["min"])) + metadata["min"]
        else:
            raise ValueError(
                f"Unknown action/proprio normalization type: {self.normalization_type}"
            )

    def normalize(self, data, metadata):
        if self.normalization_type == "normal":
            return (data / (metadata["std"] + 1e-8)) - metadata["mean"]
        elif self.normalization_type == "bounds":
            return (
                (data + 1) / (2 * (metadata["max"] - metadata["min"] + 1e-8))
            ) + metadata["min"]
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
