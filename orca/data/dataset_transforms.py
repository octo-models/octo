"""Episode transforms for different RLDS datasets to canonical dataset definition."""
from typing import Any, Dict

import tensorflow as tf

import orca.data.bridge.bridge_utils as bridge


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
            trajectory["action_dict"]["cartesian_velocity"],
            trajectory["action_dict"]["gripper_velocity"],
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


def bridge_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            bridge.binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    # TODO (homer) commit to relabeling actions or just removing the last timestep
    trajectory = bridge.relabel_actions(trajectory)
    # trajectory = tf.nest.map_structure(lambda y: y[:-1], trajectory)
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
