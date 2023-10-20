from collections import defaultdict, deque
from typing import Dict

import gym
import jax
import numpy as np


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
    num_episodes: int,
) -> Dict[str, float]:
    stats = defaultdict(list)

    for _ in range(num_episodes):
        obs, info = env.reset()
        goal = info["goal"]
        add_to(stats, flatten(filter_info(info)))
        done = False

        while not done:
            action = policy_fn(obs, goal)

            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

            goal = info["goal"]
            done = terminated or truncated
            add_to(stats, flatten(filter_info(info)))

        add_to(stats, flatten(filter_info(info), parent_key="final"))

    stats = {k: np.mean(v) for k, v in stats.items() if not isinstance(v[0], str)}

    return stats
