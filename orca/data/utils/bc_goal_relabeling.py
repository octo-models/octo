"""
Contains simple goal relabeling logic for BC use-cases where rewards and next_observations are not required.
"""

import tensorflow as tf


def uniform(traj):
    """
    Relabels with a true uniform distribution over future states.
    """
    traj_len = tf.shape(tf.nest.flatten(traj["observation"])[0])[0]

    # select a random future index for each transition i in the range [i + 1, traj_len)
    rand = tf.random.uniform([traj_len])
    low = tf.cast(tf.range(traj_len) + 1, tf.float32)
    high = tf.cast(traj_len, tf.float32)
    goal_idxs = tf.cast(rand * (high - low) + low, tf.int32)

    # sometimes there are floating-point errors that cause an out-of-bounds
    goal_idxs = tf.minimum(goal_idxs, traj_len - 1)

    traj["tasks"] = tf.nest.map_structure(
        lambda x: tf.gather(x, goal_idxs),
        traj["observation"],
    )

    return traj
