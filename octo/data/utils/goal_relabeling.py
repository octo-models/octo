"""
Contains simple goal relabeling logic for BC use-cases where rewards and next_observations are not required.
Each function should add entries to the "task" dict.
"""

from typing import Optional

import tensorflow as tf

from octo.data.utils.data_utils import tree_merge


def uniform(traj: dict, max_goal_distance: Optional[int] = None) -> dict:
    """
    Relabels with a true uniform distribution over future states.
    Optionally caps goal distance.
    """
    traj_len = tf.shape(tf.nest.flatten(traj["observation"])[0])[0]

    # select a random future index for each transition i in the range [i, traj_len)
    rand = tf.random.uniform([traj_len])
    low = tf.cast(tf.range(traj_len), tf.float32)
    if max_goal_distance is not None:
        high = tf.cast(
            tf.minimum(tf.range(traj_len) + max_goal_distance, traj_len), tf.float32
        )
    else:
        high = tf.cast(traj_len, tf.float32)
    goal_idxs = tf.cast(rand * (high - low) + low, tf.int32)

    # sometimes there are floating-point errors that cause an out-of-bounds
    goal_idxs = tf.minimum(goal_idxs, traj_len - 1)

    # adds keys to "task" mirroring "observation" keys (must do a tree merge to combine "pad_mask_dict" from
    # "observation" and "task" properly)
    goal = tf.nest.map_structure(lambda x: tf.gather(x, goal_idxs), traj["observation"])
    traj["task"] = tree_merge(traj["task"], goal)

    return traj
