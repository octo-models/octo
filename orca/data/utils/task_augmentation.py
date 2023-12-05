"""
Contains basic logic for randomly zero-ing out keys in the task specification.
"""

from fnmatch import fnmatch
from typing import Any, Dict, List, Tuple

import tensorflow as tf


def drop_keys_independent(
    traj: Dict[str, Any],
    drop_key_groups_probs: List[Tuple[List[str], float]],
    allow_drop_all: bool = False,
) -> Dict[str, Any]:
    """
    Independently drop keys in the tasks dictionary.

    :param traj: A dictionary containing trajectory data. should have a "tasks" key.
    :param drop_key_groups_probs: A list of tuples, where each tuple contains a list of keys and a dropout probability.
    :param allow_drop_all: If True, allow dropping all keys. Otherwise, if all keys are dropped, return the original
    :return: A dictionary with keys dropped out according to the specified probabilities.
    """

    # don't drop keys if there is no language instruction
    if tf.math.reduce_all(traj["tasks"]["language_instruction"] == ""):
        return traj

    tasks = traj["tasks"]
    new_tasks = tasks.copy()
    dropped_all = True
    image_keys = [key for key in tasks.keys() if "image" in key]

    for key_group, prob in drop_key_groups_probs:
        if not all(key in tasks for key in key_group):
            raise KeyError(
                f"keys {key_group} are not all present in tasks dictionary. tasks keys: {tasks.keys()}"
            )

        drop_group = tf.random.uniform([]) < prob
        dropped_all = dropped_all and drop_group

        # When no goal images are present, the goal timestep becomes the final timestep
        if all([image_key in key_group for image_key in image_keys]):
            new_tasks["goal_timestep"] = tf.where(
                drop_group,
                tasks["end_timestep"],
                tasks["goal_timestep"],
            )

        for key in key_group:
            new_tasks[key] = tf.where(
                drop_group,
                tf.zeros_like(tasks[key])
                if tf.debugging.is_numeric_tensor(tasks[key])
                else "",
                tasks[key],
            )

    if not allow_drop_all and dropped_all:
        return traj

    traj["tasks"] = new_tasks
    return traj


def delete_task_conditioning(
    traj: Dict[str, Any],
    delete_key_groups_probs: List[Tuple[List[str], float]],
):
    """
    Randomly chooses one group, and deletes all the keys in the tasks dictionary matching this pattern.

    :param traj: A dictionary containing trajectory data. should have a "tasks" key.
    :param switch_key_groups_probs: A list of tuples, where each tuple contains a list of patterns and their probability.
    :return: A dictionary with keys zeroed out according to the specified probabilities.
    """
    traj_len = tf.shape(traj["action"])[0]
    traj["tasks"]["pad_mask_dict"] = {
        k: tf.ones([traj_len], dtype=tf.bool) for k in traj["tasks"].keys()
    }

    if tf.math.reduce_all(traj["tasks"]["language_instruction"] == ""):
        return traj

    tasks = traj["tasks"]
    new_tasks = tasks.copy()

    delete_probs = [prob for _, prob in delete_key_groups_probs]
    delete_group_idx = tf.random.categorical(tf.math.log([delete_probs]), 1)[0, 0]

    image_keys = [key for key in tasks.keys() if "image" in key]

    for i, (delete_key_patterns, _) in enumerate(delete_key_groups_probs):
        matching_keys = [
            key
            for key in tasks.keys()
            if any(fnmatch(key, pattern) for pattern in delete_key_patterns)
        ]

        # When no goal images are present, the goal timestep becomes the final timestep
        if all([image_key in matching_keys for image_key in image_keys]):
            new_tasks["goal_timestep"] = tf.where(
                i == delete_group_idx,
                tasks["end_timestep"],
                tasks["goal_timestep"],
            )

        for key in matching_keys:
            new_tasks[key] = tf.where(
                i == delete_group_idx,
                tf.zeros_like(tasks[key])
                if tf.debugging.is_numeric_tensor(tasks[key])
                else "",
                tasks[key],
            )
            new_tasks["pad_mask_dict"][key] = tf.where(
                i == delete_group_idx,
                tf.zeros([traj_len], dtype=tf.bool),
                new_tasks["pad_mask_dict"][key],
            )

    traj["tasks"] = new_tasks
    return traj
