"""
Contains simple goal relabeling logic for BC use-cases where rewards and next_observations are not required.
"""

import tensorflow as tf


def uniform(traj, dropout_keys_probs=None):
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

    if dropout_keys_probs is not None:
        traj["tasks"] = dropout_keys(traj["tasks"], dropout_keys_probs)

    return traj


def dropout_keys(
    tasks: Dict[str, Any], dropout_keys_probs: Dict[str, float]
) -> Dict[str, Any]:
    """
    Applies dropout to specified keys in the tasks dictionary.

    :param tasks: A dictionary containing task information.
    :param dropout_keys_probs: A dictionary specifying the dropout probability for each key in tasks.
    :return: A dictionary with keys dropped out according to the specified probabilities.
    """
    new_tasks = tasks.copy()
    for key in dropout_keys_probs:
        if key not in tasks:
            raise KeyError(f"{key} is not present in tasks dictionary.")

        new_tasks[key] = tf.where(
            tf.random.uniform([tf.shape(tasks[key])[0]]) < dropout_keys_probs[key],
            tf.zeros_like(tasks[key]),
            tasks[key],
        )
    return new_tasks
