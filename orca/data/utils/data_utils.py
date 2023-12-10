from enum import Enum
import hashlib
import inspect
import json
import logging
import os
from typing import Any, Callable, Dict, List, Tuple

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
    builder: DatasetBuilder,
    restructure_fn: Callable[[dict], dict],
    hash_dependencies: Tuple[str, ...],
) -> dict:
    """Either computes the statistics of a dataset or loads them from a cache file if this function has been
    called before with the same arguments. Currently, the statistics include the min/max/mean/std of the
    actions and proprio as well as the number of transitions and trajectories in the dataset.
    """
    # compute a hash of the dataset info, restructure function source code, and any additional dependencies
    unique_hash = hashlib.sha256(
        "".join(
            (str(builder.info), inspect.getsource(restructure_fn), *hash_dependencies)
        ).encode(),
        usedforsecurity=False,
    ).hexdigest()
    path = tf.io.gfile.join(
        builder.info.data_dir, f"dataset_statistics_{unique_hash}.json"
    )
    # fallback local path for when data_dir is not writable
    local_path = os.path.expanduser(
        os.path.join(
            "~",
            ".cache",
            "orca",
            builder.name,
            f"dataset_statistics_{unique_hash}.json",
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

    dataset = (
        dl.DLataset.from_rlds(builder, split="train", shuffle=False)
        .traj_map(restructure_fn)
        .traj_map(
            lambda traj: {
                "action": traj["action"],
                "proprio": traj["observation"]["proprio"]
                if "proprio" in traj["observation"]
                else tf.zeros_like(traj["action"]),
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
        total=builder.info.splits["train"].num_examples,
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
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=lambda x: (x - metadata[key]["mean"])
                / (metadata[key]["std"] + 1e-8),
            )
        return traj

    if normalization_type == NormalizationType.BOUNDS:
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


def rel2abs_gripper_actions(actions: tf.Tensor):
    """Converts relative actions (-1 for closing, +1 for opening) to absolute gripper actions in range
    [0...1].
    """
    abs_actions = tf.math.cumsum(actions, axis=0)
    abs_actions = tf.clip_by_value(abs_actions, 0, 1)
    return abs_actions


def invert_gripper_actions(actions: tf.Tensor):
    return 1 - actions


def allocate_threads(n: int, weights: np.ndarray):
    """Allocates an integer n across an array based on weights. The final array sums to n, but each element is
    no less than 1.
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
