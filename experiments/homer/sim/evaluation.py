from collections import defaultdict, deque
from typing import Dict

import gym
import jax
import numpy as np


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


def flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def filter_info(info):
    filter_keys = [
        "object_names",
        "target_object",
        "initial_positions",
        "target_position",
        "goal",
    ]
    for k in filter_keys:
        if k in info:
            del info[k]
    return info


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate_gc(
    policy_fn,
    env: gym.Env,
    action_exec_horizon: int,
    num_episodes: int,
) -> Dict[str, float]:
    stats = defaultdict(list)

    for _ in range(num_episodes):
        obs, info = env.reset()
        goal = info["goal"]
        add_to(stats, flatten(filter_info(info)))
        done = False

        obs_history = [obs]
        while not done:
            # stack along time dimension
            action = policy_fn(obs_history, goal)
            assert len(action) >= action_exec_horizon

            for i in range(action_exec_horizon):
                next_obs, _, terminated, truncated, info = env.step(action[i])
                obs_history = []
                obs_history.append(next_obs)
                if terminated or truncated:
                    break

            goal = info["goal"]
            done = terminated or truncated
            add_to(stats, flatten(filter_info(info)))

        add_to(stats, flatten(filter_info(info), parent_key="final"))

    stats = {k: np.mean(v) for k, v in stats.items() if not isinstance(v[0], str)}

    return stats
