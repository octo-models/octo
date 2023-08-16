from collections import defaultdict
from typing import Dict

import gym
import jax
import numpy as np


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

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


def evaluate(policy_fn, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    stats = defaultdict(list)
    for _ in range(num_episodes):
        observation, info = env.reset()
        add_to(stats, flatten(info))
        done = False
        while not done:
            action = policy_fn(observation)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            add_to(stats, flatten(info))
        add_to(stats, flatten(info, parent_key="final"))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats


def evaluate_with_trajectories(
    policy_fn, env: gym.Env, num_episodes: int
) -> Dict[str, float]:
    trajectories = []
    stats = defaultdict(list)

    for _ in range(num_episodes):
        trajectory = defaultdict(list)
        observation, info = env.reset()
        add_to(stats, flatten(info))
        done = False
        while not done:
            action = policy_fn(observation)
            next_observation, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=r,
                done=done,
                info=info,
            )
            add_to(trajectory, transition)
            add_to(stats, flatten(info))
            observation = next_observation
        add_to(stats, flatten(info, parent_key="final"))
        trajectories.append(trajectory)

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, trajectories


def evaluate_gc(
    policy_fn,
    env: gym.Env,
    num_episodes: int,
    return_trajectories: bool = False,
) -> Dict[str, float]:
    stats = defaultdict(list)

    if return_trajectories:
        trajectories = []

    for _ in range(num_episodes):
        if return_trajectories:
            trajectory = defaultdict(list)

        observation, info = env.reset()
        goal = info["goal"]
        add_to(stats, flatten(filter_info(info)))
        done = False

        while not done:
            action = policy_fn(observation, goal)
            next_observation, r, terminated, truncated, info = env.step(action)
            goal = info["goal"]
            done = terminated or truncated
            transition = dict(
                observation=observation,
                next_observation=next_observation,
                goal=goal,
                action=action,
                reward=r,
                done=done,
                info=info,
            )

            add_to(stats, flatten(filter_info(info)))

            if return_trajectories:
                add_to(trajectory, transition)

            observation = next_observation

        add_to(stats, flatten(filter_info(info), parent_key="final"))
        if return_trajectories:
            trajectory["steps_remaining"] = list(
                np.arange(len(trajectory["action"]))[::-1]
            )
            trajectories.append(trajectory)

    stats = {k: np.mean(v) for k, v in stats.items() if not isinstance(v[0], str)}

    if return_trajectories:
        return stats, trajectories
    else:
        return stats
