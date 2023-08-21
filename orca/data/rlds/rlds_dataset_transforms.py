"""Episode transforms for different RLDS datasets to canonical dataset definition."""
from typing import Any, Dict

import tensorflow as tf


def stanford_kuka_multimodal_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    # pad action to be 7-dimensional to fit Bridge data convention (add zero rotation action)
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros_like(trajectory["action"][:, :3]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def r2_d2_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["action"] = tf.concat(
        (
            trajectory["action_dict"]["cartesian_position"],
            trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


RLDS_TRAJECTORY_MAP_TRANSFORMS = {
    "stanford_kuka_multimodal_dataset": stanford_kuka_multimodal_dataset_transform,
    "r2_d2": r2_d2_dataset_transform,
    "r2_d2_pen": r2_d2_dataset_transform,
}
