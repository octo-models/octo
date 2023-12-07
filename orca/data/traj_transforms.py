"""
Contains trajectory transforms used in the orca data pipeline. Trajectory transforms operate on a dictionary
that represents a single trajectory, meaning each tensor has the same leading dimension (the trajectory
length).
"""
import tensorflow as tf


def chunk_act_obs(
    traj: dict,
    window_size: int,
    additional_action_window_size: int = 0,
) -> dict:
    """
    Chunks actions and observations into the given window_size.

    The "action" and "observation" keys are each given a new axis (at index 1) of size `window_size`.
    """
    traj_len = tf.shape(traj["action"])[0]
    chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1), [traj_len, window_size]
    ) + tf.broadcast_to(tf.range(traj_len)[:, None], [traj_len, window_size])

    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + additional_action_window_size),
        [traj_len, window_size + additional_action_window_size],
    ) + tf.broadcast_to(
        tf.range(traj_len)[:, None],
        [traj_len, window_size + additional_action_window_size],
    )

    floored_chunk_indices = tf.maximum(chunk_indices, 0)

    if "task" in traj:
        goal_timestep = traj["task"]["goal_timestep"]
    else:
        goal_timestep = tf.fill([traj_len], traj_len, dtype=tf.int32)

    floored_action_chunk_indices = tf.minimum(
        tf.maximum(action_chunk_indices, 0), goal_timestep[:, None] - 1
    )

    traj["observation"] = tf.nest.map_structure(
        lambda x: tf.gather(x, floored_chunk_indices), traj["observation"]
    )
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)

    # indicates whether an entire observation is padding
    traj["observation"]["pad_mask"] = chunk_indices >= 0

    # Actions past the goal timestep become no-ops
    action_past_goal = action_chunk_indices > goal_timestep[:, None] - 1
    # zero_actions = make_neutral_actions(traj["action"], action_encoding)
    # traj["action"] = tf.where(
    #     action_past_goal[:, :, None], zero_actions, traj["action"]
    # )
    return traj


def subsample(traj: dict, subsample_length: int) -> dict:
    """Subsamples trajectories to the given length."""
    traj_len = tf.shape(traj["action"])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)
    return traj


def add_pad_mask_dict(traj: dict) -> dict:
    """Adds a dictionary indicating which elements of the observation are padding.

    traj["observation"]["pad_mask_dict"] = {k: traj["observation"][k] is not padding}
    """
    traj_len = tf.shape(traj["action"])[0]
    pad_masks = {}
    for key in traj["observation"]:
        if traj["observation"][key].dtype == tf.string:
            pad_masks[key] = tf.strings.length(traj["observation"][key]) != 0
        else:
            pad_masks[key] = tf.ones([traj_len], dtype=tf.bool)
    traj["observation"]["pad_mask_dict"] = pad_masks
    return traj
