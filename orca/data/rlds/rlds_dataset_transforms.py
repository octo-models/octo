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

    # map image observations
    trajectory["observation"]["image_0"] = trajectory["observation"]["image"]
    trajectory["observation"]["depth_0"] = trajectory["observation"]["depth_image"]
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
    keep_keys = [
        "observation",
        "action",
        "language_instruction",
        "is_terminal",
        "is_last",
        "_traj_index",
    ]
    trajectory = {k: v for k, v in trajectory.items() if k in keep_keys}

    # map image observations
    trajectory["observation"]["image_0"] = trajectory["observation"][
        "exterior_image_1_left"
    ]
    trajectory["observation"]["image_1"] = trajectory["observation"][
        "exterior_image_2_left"
    ]
    trajectory["observation"]["image_2"] = trajectory["observation"]["wrist_image_left"]
    return trajectory


def fmb_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observations"]["state"] = tf.concat(
        (
            trajectory["observations"]["state"],
            trajectory["observations"]["gripper_state"][..., None],
        ),
        axis=-1,
    )

    # map image observations
    trajectory["observation"]["image_0"] = trajectory["observation"]["image_side_1"]
    trajectory["observation"]["image_1"] = trajectory["observation"]["image_side_2"]
    trajectory["observation"]["image_2"] = trajectory["observation"]["image_wrist_1"]
    trajectory["observation"]["image_3"] = trajectory["observation"]["image_wrist_2"]
    trajectory["observation"]["depth_0"] = trajectory["observation"][
        "image_side_1_depth"
    ]
    trajectory["observation"]["depth_1"] = trajectory["observation"][
        "image_side_2_depth"
    ]
    trajectory["observation"]["depth_2"] = trajectory["observation"][
        "image_wrist_1_depth"
    ]
    trajectory["observation"]["depth_3"] = trajectory["observation"][
        "image_wrist_2_depth"
    ]
    return trajectory


def bridge_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    keep_keys = [
        "observation",
        "action",
        "language_instruction",
        "is_terminal",
        "is_last",
        "_traj_index",
    ]
    trajectory = {k: v for k, v in trajectory.items() if k in keep_keys}
    return trajectory


RLDS_TRAJECTORY_MAP_TRANSFORMS = {
    "stanford_kuka_multimodal_dataset": stanford_kuka_multimodal_dataset_transform,
    "r2_d2": r2_d2_dataset_transform,
    "r2_d2_pen": r2_d2_dataset_transform,
    "fmb_dataset": fmb_dataset_transform,
    "bridge_dataset": bridge_dataset_transform,
}
