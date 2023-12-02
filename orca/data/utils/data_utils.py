from enum import IntEnum
import hashlib
import inspect
import json
import logging
import os
from typing import Any, Callable, Dict, List

import dlimp as dl
import numpy as np
import tensorflow as tf
from tensorflow_datasets.core.dataset_builder import DatasetBuilder
import tqdm


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()
    }


class StateEncoding(IntEnum):
    """Defines supported proprio state encoding schemes for different datasets."""

    NONE = -1  # no state provided
    POS_EULER = 1  # EEF XYZ + roll-pitch-yaw + 1 x pad + gripper open/close
    POS_QUAT = 2  # EEF XYZ + quaternion + gripper open/close
    JOINT = 3  # 7 x joint angles (padding added if fewer) + gripper open/close
    JOINT_BIMANUAL = 4  # 2 x [6 x joint angles + gripper open/close]


class ActionEncoding(IntEnum):
    """Defines supported action encoding schemes for different datasets."""

    EEF_POS = 1  # EEF delta XYZ + roll-pitch-yaw + gripper open/close
    JOINT_POS = 2  # 7 x joint delta position + gripper open/close
    JOINT_POS_BIMANUAL = 3  # 2 x [6 x joint pos + gripper]


def state_encoding_length(state_encoding):
    if state_encoding == StateEncoding.NONE:
        return 0
    # TODO: remove hack that POS_EULER pads 0 to match length
    elif state_encoding in [
        StateEncoding.POS_EULER,
        StateEncoding.POS_QUAT,
        StateEncoding.JOINT,
    ]:
        return 8
    elif state_encoding in [StateEncoding.JOINT_BIMANUAL]:
        return 14
    else:
        raise ValueError(f"State encoding {state_encoding} not supported.")


def action_encoding_length(action_encoding):
    if action_encoding in [ActionEncoding.EEF_POS]:
        return 7
    elif action_encoding in [ActionEncoding.JOINT_POS]:
        return 8
    elif action_encoding in [ActionEncoding.JOINT_POS_BIMANUAL]:
        return 14
    else:
        raise ValueError(f"Action encoding {action_encoding} not supported.")


def make_zero_actions(action, action_encoding):
    """
    Returns neutral action for action encoding, matches shape of input action.
    Zero-action 0s out all relative actions and retains value of absolute actions like gripper open/close.
    """
    assert action.shape[-1] == action_encoding_length(action_encoding), (
        f"For action encoding {action_encoding} expected {action_encoding_length(action_encoding)}-dim action,"
        f" but got {action.shape[-1]}-dim action."
    )
    if action_encoding == ActionEncoding.EEF_POS:
        is_absolute_action = tf.range(action.shape[-1]) >= 6
    elif action_encoding == ActionEncoding.JOINT_POS:
        is_absolute_action = tf.range(action.shape[-1]) >= 7
    elif action_encoding == ActionEncoding.JOINT_POS_BIMANUAL:
        is_absolute_action = tf.math.logical_or(
            tf.range(action.shape[-1]) == 6,
            tf.range(action.shape[-1]) == 13,
        )
    else:
        raise ValueError(f"Action encoding {action_encoding} not supported.")

    return tf.where(
        is_absolute_action[None, None, :],
        action,
        tf.zeros_like(action),
    )


def pprint_data_mixture(
    dataset_kwargs_list: List[Dict[str, Any]], dataset_weights: List[int]
) -> None:
    print(
        "\n######################################################################################"
    )
    print(
        f"# Loading the following {len(dataset_kwargs_list)} datasets (incl. sampling weight):{'': >24} #"
    )
    for dataset_kwargs, weight in zip(dataset_kwargs_list, dataset_weights):
        pad = 80 - len(dataset_kwargs["name"])
        print(f"# {dataset_kwargs['name']}: {weight:=>{pad}f} #")
    print(
        "######################################################################################\n"
    )


def get_dataset_statistics(
    builder: DatasetBuilder,
    state_obs_keys: List[str],
    restructure_fn: Callable,
    transform_fn: Callable,
) -> dict:
    """Either computes the statistics of a dataset or loads them from a cache file if this function
    has been called before with the same arguments. Currently, the statistics include the
    min/max/mean/std of the actions and proprio as well as the number of transitions and
    trajectories in the dataset.
    """
    # compute a hash of the dataset info, state observation keys, and transform function
    # to determine the name of the cache file
    data_info_hash = hashlib.sha256(
        (
            str(builder.info)
            + str(state_obs_keys)
            + str(inspect.getsource(restructure_fn))
            + str(inspect.getsource(transform_fn))
        ).encode("utf-8")
    ).hexdigest()
    path = tf.io.gfile.join(
        builder.info.data_dir, f"dataset_statistics_{data_info_hash}.json"
    )
    # fallback local path for when data_dir is not writable
    local_path = os.path.expanduser(
        os.path.join(
            "~",
            ".cache",
            "orca",
            builder.name,
            f"dataset_statistics_{data_info_hash}.json",
        )
    )

    # check if cache file exists and load
    if tf.io.gfile.exists(path):
        logging.info(f"Loading existing dataset statistics from {path}.")
        with tf.io.gfile.GFile(path, "r") as f:
            metadata = json.load(f)
        return metadata

    if os.path.exists(local_path):
        logging.info(f"Loading existing dataset statistics from {local_path}.")
        with open(local_path, "r") as f:
            metadata = json.load(f)
        return metadata

    if "val" not in builder.info.splits:
        split = "train[:95%]"
        expected_trajs = int(builder.info.splits["train"].num_examples * 0.95)
    else:
        split = "train"
        expected_trajs = builder.info.splits["train"].num_examples
    dataset = (
        dl.DLataset.from_rlds(builder, split=split, shuffle=False)
        .map(restructure_fn)
        .map(
            lambda traj: {
                "action": traj["action"],
                "proprio": traj["observation"]["proprio"],
            }
        )
    )
    logging.info(
        f"Computing dataset statistics for {builder.name}. This may take awhile, "
        "but should only need to happen once."
    )
    actions = []
    proprios = []
    num_transitions = 0
    num_trajectories = 0
    for traj in tqdm.tqdm(
        dataset.iterator(),
        total=expected_trajs,
    ):
        actions.append(traj["action"])
        proprios.append(traj["proprio"])
        num_transitions += traj["action"].shape[0]
        num_trajectories += 1
    actions = np.concatenate(actions)
    proprios = np.concatenate(proprios)
    metadata = {
        "action": {
            "mean": actions.mean(0).tolist(),
            "std": actions.std(0).tolist(),
            "max": actions.max(0).tolist(),
            "min": actions.min(0).tolist(),
        },
        "proprio": {
            "mean": proprios.mean(0).tolist(),
            "std": proprios.std(0).tolist(),
            "max": proprios.max(0).tolist(),
            "min": proprios.min(0).tolist(),
        },
        "num_transitions": num_transitions,
        "num_trajectories": num_trajectories,
    }

    try:
        with tf.io.gfile.GFile(path, "w") as f:
            json.dump(metadata, f)
    except tf.errors.PermissionDeniedError:
        logging.warning(
            f"Could not write dataset statistics to {path}. "
            f"Writing to {local_path} instead."
        )
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w") as f:
            json.dump(metadata, f)

    return metadata


def normalize_action_and_proprio(traj, metadata, normalization_type):
    # maps keys of `metadata` to corresponding keys in `traj`
    keys_to_normalize = {
        "action": "action",
        "proprio": "observation/proprio",
    }
    if normalization_type == "normal":
        # normalize to mean 0, std 1
        for key, traj_key in keys_to_normalize.items():
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: (x - metadata[key]["mean"])
                / (metadata[key]["std"] + 1e-8),
            )
        return traj

    if normalization_type == "bounds":
        # normalize to [-1, 1]
        for key, traj_key in keys_to_normalize.items():
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: tf.clip_by_value(
                    2
                    * (x - metadata[key]["min"])
                    / (metadata[key]["max"] - metadata[key]["min"] + 1e-8)
                    - 1,
                    -1,
                    1,
                ),
            )
        return traj

    raise ValueError(f"Unknown normalization type {normalization_type}")


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


def rel2abs_gripper_actions(actions: tf.Tensor):
    """
    Converts relative actions (-1 for closing, +1 for opening) to absolute gripper actions in range [0...1].
    """
    abs_actions = tf.math.cumsum(actions, axis=0)
    abs_actions = tf.clip_by_value(abs_actions, 0, 1)
    return abs_actions


def invert_gripper_actions(actions: tf.Tensor):
    return 1 - actions


def allocate_threads(n: int, weights: np.ndarray):
    """
    Allocates an integer n across an array based on weights. The final array sums to n, but each
    element is no less than 1.
    """
    assert np.all(weights >= 0), "Weights must be non-negative"
    assert (
        len(weights) <= n
    ), "Number of threads must be at least as large as length of weights"
    weights = np.array(weights) / np.sum(weights)

    allocation = np.zeros_like(weights, dtype=int)
    while True:
        # give the remaining elements that would get less than 1 a 1
        mask = (weights * n < 1) & (weights > 0)
        if not mask.any():
            break
        n -= mask.sum()
        allocation += mask.astype(int)
        # recompute the distribution over the remaining elements
        weights[mask] = 0
        weights = weights / weights.sum()
    # allocate the remaining elements
    fractional, integral = np.modf(weights * n)
    allocation += integral.astype(int)
    n -= integral.sum()
    for i in np.argsort(fractional)[::-1][: int(n)]:
        allocation[i] += 1
    return allocation
