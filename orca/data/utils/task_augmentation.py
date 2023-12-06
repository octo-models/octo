"""
Contains basic logic for randomly zero-ing out keys in the task specification.
"""

from fnmatch import fnmatch
from typing import Any, Dict, List, Tuple

import tensorflow as tf


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
                tf.zeros_like(tasks["pad_mask_dict"][key]),
                new_tasks["pad_mask_dict"][key],
            )

    traj["tasks"] = new_tasks
    return traj
