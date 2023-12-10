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
    """Chunks actions and observations into the given window_size.

    "observation" keys are given a new axis (at index 1) of size `window_size`. "action" is given a new axis
    (at index 1) of size `window_size + additional_action_window_size`.
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

    if "timestep" in traj["task"]:
        goal_timestep = traj["task"]["timestep"]
    else:
        goal_timestep = tf.fill([traj_len], traj_len - 1)

    floored_action_chunk_indices = tf.minimum(
        tf.maximum(action_chunk_indices, 0), goal_timestep[:, None]
    )

    traj["observation"] = tf.nest.map_structure(
        lambda x: tf.gather(x, floored_chunk_indices), traj["observation"]
    )
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)

    # indicates whether an entire observation is padding
    traj["observation"]["pad_mask"] = chunk_indices >= 0

    # Actions past the goal timestep become no-ops
    action_past_goal = action_chunk_indices > goal_timestep[:, None]
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
    """Adds a dictionary indicating which elements of the observation/task should be treated as padding.

    traj["observation"|"task"]["pad_mask_dict"] = {k: traj["observation"|"task"][k] is not padding}
    """
    traj_len = tf.shape(traj["action"])[0]
    for key in ["observation", "task"]:
        pad_mask_dict = {}
        for subkey in traj[key]:
            if traj[key][subkey].dtype == tf.string:
                # handles "language_instruction", "image_*", and "depth_*"
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0
            else:
                # all other keys should not be treated as padding
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)
        traj[key]["pad_mask_dict"] = pad_mask_dict
    return traj
