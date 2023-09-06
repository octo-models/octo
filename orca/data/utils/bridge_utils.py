from typing import Any, Dict

import tensorflow as tf


@tf.function(jit_compile=True)
def binarize_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """Converts gripper actions from continous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near
    1.0) or fully closed (near 0.0). As it transitions between the two, it
    sometimes passes through a few intermediate values. We relabel those
    intermediate values based on the state that is reached _after_ those
    intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we
    give up on binarizing and relabel that chunk of intermediate values as
    the last action in the trajectory.

    The scan implements the following code:

    new_actions = np.empty_like(actions)
    carry = actions[-1]
    for i in reversed(range(actions.shape[0])):
        if in_between_mask[i]:
            carry = carry
        else:
            carry = float(open_mask[i])
        new_actions[i] = carry
    """
    open_mask = actions > 0.95
    closed_mask = actions < 0.05
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))

    is_open_float = tf.cast(open_mask, tf.float32)

    def scan_fn(carry, i):
        return tf.cond(
            in_between_mask[i],
            lambda: tf.cast(carry, tf.float32),
            lambda: is_open_float[i],
        )

    new_actions = tf.scan(
        scan_fn, tf.range(tf.shape(actions)[0]), actions[-1], reverse=True
    )
    return new_actions


def relabel_actions(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Relabels the actions to use the reached proprio instead. Discards the last timestep of the
    trajectory (since we don't have a next state to compute the action.)
    """
    # relabel the first 6 action dims (xyz position, xyz rotation) using the reached proprio
    movement_actions = (
        traj["observation"]["state"][1:, :6] - traj["observation"]["state"][:-1, :6]
    )

    # discard the last timestep of the trajectory
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)

    # recombine to get full actions
    traj_truncated["action"] = tf.concat(
        [movement_actions, traj["action"][:-1, -1:]],
        axis=1,
    )

    return traj_truncated
