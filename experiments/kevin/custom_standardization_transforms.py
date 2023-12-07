"""Episode transforms for custom (non-OXE) RLDS datasets to canonical dataset definition."""
from typing import Any, Dict

import tensorflow as tf


def r2_d2_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["action"] = tf.concat(
        (
            trajectory["action_dict"]["cartesian_velocity"],
            trajectory["action_dict"]["gripper_velocity"],
        ),
        axis=-1,
    )
    return trajectory


def fmb_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["state"],
            trajectory["observation"]["gripper_state"][..., None],
        ),
        axis=-1,
    )
    return trajectory


def aloha_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory
