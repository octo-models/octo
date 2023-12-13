from enum import Enum
import hashlib
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import dlimp as dl
import numpy as np
import tensorflow as tf
import tqdm


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()
    }


def tree_merge(*trees: dict) -> dict:
    """Merges a list of nested dictionaries, with later dictionaries overriding earlier ones."""
    merged = {}
    for tree in trees:
        for k, v in tree.items():
            if isinstance(v, dict):
                merged[k] = tree_merge(merged.get(k, {}), v)
            else:
                merged[k] = v
    return merged


class NormalizationType(str, Enum):
    """Defines supported normalization schemes for action and proprio."""

    NORMAL = "normal"  # normalize to mean 0, std 1
    BOUNDS = "bounds"  # normalize to [-1, 1]


def to_padding(tensor: tf.Tensor) -> tf.Tensor:
    if tf.debugging.is_numeric_tensor(tensor):
        return tf.zeros_like(tensor)
    elif tensor.dtype == tf.string:
        return tf.fill(tf.shape(tensor), "")
    else:
        raise ValueError(f"Cannot generate padding for tensor of type {tensor.dtype}.")


def make_neutral_actions(
    action: tf.Tensor, absolute_action_mask: tf.Tensor
) -> tf.Tensor:
    """Returns "neutral" actions, meaning relative actions are zeroed and absolute actions are retained.
    `absolute_action_mask` should be a 1D boolean mask that indicates which action dimensions are absolute.
    """
    return tf.where(
        absolute_action_mask[(None,) * (action.ndim - 1)],
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
    dataset: dl.DLataset,
    hash_dependencies: Tuple[str, ...],
    save_dir: Optional[str] = None,
) -> dict:
    """Either computes the statistics of a dataset or loads them from a cache file if this function has been
    called before with the same `hash_dependencies`. Currently, the statistics include the min/max/mean/std of
    the actions and proprio as well as the number of transitions and trajectories in the dataset.
    """
    unique_hash = hashlib.sha256(
        "".join(hash_dependencies).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()

    # fallback local path for when data_dir is not writable or not provided
    local_path = os.path.expanduser(
        os.path.join(
            "~",
            ".cache",
            "octo",
            f"dataset_statistics_{unique_hash}.json",
        )
    )

    if save_dir is not None:
        path = tf.io.gfile.join(save_dir, f"dataset_statistics_{unique_hash}.json")
    else:
        path = local_path

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

    dataset = dataset.traj_map(
        lambda traj: {
            "action": traj["action"],
            "proprio": traj["observation"]["proprio"]
            if "proprio" in traj["observation"]
            else tf.zeros_like(traj["action"]),
        }
    )

    cardinality = dataset.cardinality().numpy()
    if cardinality == tf.data.INFINITE_CARDINALITY:
        raise ValueError("Cannot compute dataset statistics for infinite datasets.")

    logging.info(
        "Computing dataset statistics. This may take awhile, but should only need to happen "
        "once for each dataset."
    )
    actions = []
    proprios = []
    num_transitions = 0
    num_trajectories = 0
    for traj in tqdm.tqdm(
        dataset.iterator(),
        total=cardinality if cardinality != tf.data.UNKNOWN_CARDINALITY else None,
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


def normalize_action_and_proprio(
    traj: dict, metadata: dict, normalization_type: NormalizationType
):
    """Normalizes the action and proprio fields of a trajectory using the given metadata."""
    # maps keys of `metadata` to corresponding keys in `traj`
    keys_to_normalize = {
        "action": "action",
        "proprio": "observation/proprio",
    }
    if normalization_type == NormalizationType.NORMAL:
        # normalize to mean 0, std 1
        for key, traj_key in keys_to_normalize.items():
            mask = metadata[key].get(
                "mask", tf.ones_like(metadata[key]["mean"], dtype=tf.bool)
            )
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: tf.where(
                    mask, (x - metadata[key]["mean"]) / (metadata[key]["std"] + 1e-8), x
                ),
            )
        return traj

    if normalization_type == NormalizationType.BOUNDS:
        # normalize to [-1, 1]
        for key, traj_key in keys_to_normalize.items():
            mask = metadata[key].get(
                "mask", tf.ones_like(metadata[key]["min"], dtype=tf.bool)
            )
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: tf.where(
                    mask,
                    tf.clip_by_value(
                        2
                        * (x - metadata[key]["min"])
                        / (metadata[key]["max"] - metadata[key]["min"] + 1e-8)
                        - 1,
                        -1,
                        1,
                    ),
                    x,
                ),
            )
        return traj

    raise ValueError(f"Unknown normalization type {normalization_type}")


def binarize_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """Converts gripper actions from continous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near 1.0) or fully closed (near
    0.0). As it transitions between the two, it sometimes passes through a few intermediate values. We relabel
    those intermediate values based on the state that is reached _after_ those intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we give up on binarizing and relabel
    that chunk of intermediate values as the last action in the trajectory.

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


def rel_open_or_closed(actions: tf.Tensor):
    """
    Returns the initial absolute gripper state, given relative actions (-1 for closing, +1 for opening)
    Returns 1 if the gripper is initially open, 0 if it is initially closed.
    If nothing taken, assumes gripper is initially open.

    """
    opening_mask = actions > 1e-3
    closing_mask = actions < -1e-3
    old_state_mask = tf.where(opening_mask, -1, tf.where(closing_mask, -1, 0))
    # old_state_mask is 1 if closing, -1 if opening, 0 if no change

    def scan_fn(carry, i):
        return tf.cond(
            old_state_mask[i] == 0,
            lambda: tf.cast(carry, tf.float32),
            lambda: (tf.cast(old_state_mask[i], tf.float32) + 1) / 2,
        )

    return tf.scan(
        scan_fn,
        tf.range(tf.shape(actions)[0]),
        tf.zeros_like(actions[-1]),
        reverse=True,
    )[0]


def rel2abs_gripper_actions(actions: tf.Tensor):
    """
    Converts relative gripper actions (+1 for closing, -1 for opening) to absolute gripper actions
    (0 for closed, 1 for open). Assumes that the first relative gripper is not redundant
    (i.e. close when already closed).
    """
    opening_mask = actions < -0.1
    closing_mask = actions > 0.1

    # -1 for closing, 1 for opening, 0 for no change
    thresholded_actions = tf.where(opening_mask, 1, tf.where(closing_mask, -1, 0))

    def scan_fn(carry, i):
        return tf.cond(
            thresholded_actions[i] == 0,
            lambda: carry,
            lambda: thresholded_actions[i],
        )

    # if no relative grasp, assumes open for whole trajectory
    start = -1 * thresholded_actions[tf.argmax(thresholded_actions != 0, axis=0)]
    start = tf.cond(start == 0, lambda: 1, lambda: start)
    # -1 for closed, 1 for open
    new_actions = tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), start)

    new_actions = tf.cast(new_actions, tf.float32) / 2 + 0.5
    return new_actions


def invert_gripper_actions(actions: tf.Tensor):
    return 1 - actions


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


def allocate_threads(n: Optional[int], weights: np.ndarray):
    """Allocates an integer number of threads across datasets based on weights. The final array sums to `n`,
    but each element is no less than 1. If `n` is None, then every dataset is assigned a value of AUTOTUNE.
    """
    if n is None:
        return np.array([tf.data.AUTOTUNE] * len(weights))

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
