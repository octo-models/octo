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
    Randomly chooses one group, and deletes all the keys in the task dictionary matching this pattern.

    Args:
        traj: A dictionary containing trajectory data. should have a "task" key.
        delete_key_groups_probs: A list of tuples, where each tuple contains a list of patterns and their probability.
    """
    if tf.math.reduce_all(traj["task"]["language_instruction"] == ""):
        return traj

    task = traj["task"]
    new_task = task.copy()

    delete_probs = [prob for _, prob in delete_key_groups_probs]
    delete_group_idx = tf.random.categorical(tf.math.log([delete_probs]), 1)[0, 0]

    image_keys = [key for key in task.keys() if "image" in key]

    for i, (delete_key_patterns, _) in enumerate(delete_key_groups_probs):
        matching_keys = [
            key
            for key in task.keys()
            if any(fnmatch(key, pattern) for pattern in delete_key_patterns)
        ]

        # When no goal images are present, the goal timestep becomes the final timestep
        if all([image_key in matching_keys for image_key in image_keys]):
            new_task["goal_timestep"] = tf.where(
                i == delete_group_idx,
                task["end_timestep"],
                task["goal_timestep"],
            )

        for key in matching_keys:
            new_task[key] = tf.where(
                i == delete_group_idx,
                tf.zeros_like(task[key])
                if tf.debugging.is_numeric_tensor(task[key])
                else "",
                task[key],
            )
            new_task["pad_mask_dict"][key] = tf.where(
                i == delete_group_idx,
                tf.zeros_like(task["pad_mask_dict"][key]),
                new_task["pad_mask_dict"][key],
            )

    traj["task"] = new_task
    return traj
