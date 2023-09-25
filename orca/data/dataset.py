import hashlib
import inspect
import json
import logging
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
from tensorflow_datasets.core.dataset_builder import DatasetBuilder

from orca.data.dataset_transforms import RLDS_TRAJECTORY_MAP_TRANSFORMS
from orca.data.utils import bc_goal_relabeling, task_augmentation


def get_action_proprio_stats(
    builder: DatasetBuilder,
    dataset: tf.data.Dataset,
    proprio_keys: List[str],
    transform_fcn: Any,
) -> Dict[str, Dict[str, List[float]]]:
    # get statistics file path --> embed unique hash that catches if dataset info / keys / transform changed
    transform_str = inspect.getsource(transform_fcn) if transform_fcn else ""
    data_info_hash = hashlib.sha256(
        (str(builder.info) + str(proprio_keys) + str(transform_str)).encode("utf-8")
    ).hexdigest()
    path = tf.io.gfile.join(
        builder.info.data_dir, f"action_proprio_stats_{data_info_hash}.json"
    )

    # check if stats already exist and load, otherwise compute
    if tf.io.gfile.exists(path):
        logging.info(f"Loading existing statistics for normalization from {path}.")
        with tf.io.gfile.GFile(path, "r") as f:
            metadata = json.load(f)
    else:
        logging.info("Computing action/proprio statistics for normalization...")
        actions = []
        proprios = []
        for episode in tqdm.tqdm(dataset.take(1000)):
            actions.append(episode["action"].numpy())
            proprios.append(episode["observation"]["proprio"].numpy())
        actions = np.concatenate(actions)
        proprios = np.concatenate(proprios)
        metadata = {
            "action": {
                "mean": [float(e) for e in actions.mean(0)],
                "std": [float(e) for e in actions.std(0)],
                "max": [float(e) for e in actions.max(0)],
                "min": [float(e) for e in actions.min(0)],
            },
            "proprio": {
                "mean": [float(e) for e in proprios.mean(0)],
                "std": [float(e) for e in proprios.std(0)],
                "max": [float(e) for e in proprios.max(0)],
                "min": [float(e) for e in proprios.min(0)],
            },
        }
        del actions
        del proprios
        with tf.io.gfile.GFile(path, "w") as f:
            json.dump(metadata, f)
        logging.info("Done!")

    return {
        k: {k2: tf.convert_to_tensor(v2, dtype=tf.float32) for k2, v2 in v.items()}
        for k, v in metadata.items()
    }


def _normalize_action_and_proprio(traj, metadata, normalization_type):
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
                match=traj_key,
                map_fn=lambda x: (x - metadata[key]["mean"]) / metadata[key]["std"],
            )
        return traj

    if normalization_type == "bounds":
        # normalize to [-1, 1]
        for key, traj_key in keys_to_normalize.items():
            traj = dl.transforms.selective_tree_map(
                traj,
                match=traj_key,
                map_fn=lambda x: tf.clip_by_value(
                    2
                    * (x - metadata[key]["min"])
                    / (metadata[key]["max"] - metadata[key]["min"])
                    - 1,
                    -1,
                    1,
                ),
            )
        return traj

    raise ValueError(f"Unknown normalization type {normalization_type}")


def _chunk_act_obs(traj, horizon):
    """
    Chunks actions and observations into the given horizon.

    The "action" and "observation" keys are each given a new axis (at index 1) of size `horizon`.
    """
    traj_len = tf.shape(traj["action"])[0]
    chunk_indices = tf.broadcast_to(
        tf.range(-horizon + 1, 1), [traj_len, horizon]
    ) + tf.broadcast_to(tf.range(traj_len)[:, None], [traj_len, horizon])
    # pads by repeating the first timestep
    chunk_indices = tf.maximum(chunk_indices, 0)
    traj["observation"] = tf.nest.map_structure(
        lambda x: tf.gather(x, chunk_indices), traj["observation"]
    )
    traj["action"] = tf.nest.map_structure(
        lambda x: tf.gather(x, chunk_indices), traj["action"]
    )
    return traj


def apply_common_transforms(
    dataset: tf.data.Dataset,
    *,
    train: bool,
    goal_relabeling_strategy: Optional[str] = None,
    goal_relabeling_kwargs: dict = {},
    image_augment_kwargs: dict = {},
    task_augmentation_strategy: Optional[str] = None,
    task_augmentation_kwargs: dict = {},
    resize_size: Optional[Tuple[int, int]] = None,
    horizon: int = 2,
    skip_unlabeled: bool = False,
    action_proprio_metadata: Optional[dict] = None,
    action_proprio_normalization_type: Optional[str] = None,
):
    """Common transforms shared between all datasets.

    Args:
        dataset (tf.data.Dataset): The dataset to transform.
        train (bool): Whether the dataset is for training (affects augmentation).
        goal_relabeling_strategy (Optional[str], optional): The goal relabeling strategy to use, or None for no goal
            relabeling. See `bc_goal_relabeling.py`.
        goal_relabeling_kwargs (dict, optional): Additional keyword arguments to pass to the goal relabeling function.
        image_augment_kwargs (dict, optional): Keyword arguments to pass to the augmentation function. See
            `dlimp.augmentations.augment_image` for documentation.
        resize_size (tuple, optional): target (height, width) for all RGB and depth images, default to no resize.
        horizon (int, optional): The length of the snippets that trajectories are chunked into.
        skip_unlabeled (bool, optional): Whether to skip trajectories with no language labels.
        action_proprio_metadata (Optional[dict], optional): A dictionary containing metadata about the action and
            proprio statistics. If None, no normalization is performed.
        action_proprio_normalization_type (Optional[str], optional): The type of normalization to perform on the action,
            proprio, or both. Can be "normal" (mean 0, std 1) or "bounds" (normalized to [-1, 1]).
    """
    if skip_unlabeled:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_any(x["language_instruction"] != "")
        )

    if action_proprio_metadata is not None:
        dataset = dataset.map(
            partial(
                _normalize_action_and_proprio,
                metadata=action_proprio_metadata,
                normalization_type=action_proprio_normalization_type,
            )
        )

    # decodes string keys with name "image", resizes "image" and "depth"
    dataset = dataset.frame_map(dl.transforms.decode_images)
    if resize_size:
        dataset = dataset.frame_map(
            partial(dl.transforms.resize_images, size=resize_size)
        )
        dataset = dataset.frame_map(
            partial(dl.transforms.resize_depth_images, size=resize_size)
        )

    if train:
        # augments the entire trajectory with the same seed
        dataset = dataset.frame_map(
            partial(
                dl.transforms.augment,
                augment_kwargs=image_augment_kwargs,
            )
        )

    # adds the "tasks" key
    if goal_relabeling_strategy is not None:
        dataset = dataset.map(
            partial(
                getattr(bc_goal_relabeling, goal_relabeling_strategy),
                **goal_relabeling_kwargs,
            )
        )

        def move_language_instruction_to_tasks(traj):
            traj["tasks"]["language_instruction"] = traj.pop("language_instruction")
            return traj
        
        dataset = dataset.map(move_language_instruction_to_tasks)

    if task_augmentation_strategy is not None:
        dataset = dataset.map(
            partial(
                getattr(task_augmentation, task_augmentation_strategy),
                **task_augmentation_kwargs,
            )
        )

    # chunks actions and observations
    assert (
        horizon >= 2
    ), "Horizon must be at least 2 to provide a timestep for conditioning and a timestep for prediction."
    dataset = dataset.map(partial(_chunk_act_obs, horizon=horizon))

    return dataset


def make_dataset(
    name: str,
    data_dir: str,
    train: bool,
    shuffle: bool = True,
    image_obs_keys: Union[str, List[str]] = [],
    depth_obs_keys: Union[str, List[str]] = [],
    state_obs_keys: Union[str, List[str]] = [],
    action_proprio_metadata: Optional[dict] = None,
    resize_size: Optional[Tuple[int, int]] = None,
    **kwargs,
) -> tf.data.Dataset:
    """Creates a dataset from the RLDS format.

    Args:
        name (str): The name of the RLDS dataset (usually "name" or "name:version").
        data_dir (str): The path to the data directory.
        train (bool): Whether to use the training or validation set.
        shuffle (bool, optional): Whether to shuffle the order of tfrecords.
        image_obs_keys (str, List[str], optional): List of image observation keys to be decoded. Mapped to "image_XXX".
            Inserts padding image for each None key.
        depth_obs_keys (str, List[str], optional): List of depth observation keys to be decoded. Mapped to "depth_XXX".
            Inserts padding image for each None key.
        state_obs_keys (str, List[str], optional): List of low-dim observation keys to be decoded.
            Get concatenated and mapped to "proprio". Inserts 1d padding for each None key.
        action_proprio_metadata (dict, optional): dict with min/max/mean/std for action and proprio normalization.
            If not provided, will get computed on the fly.
        resize_size (tuple, optional): target (height, width) for all RGB and depth images, default to no resize.
        **kwargs: Additional keyword arguments to pass to `apply_common_transforms`.
    Returns:
        Dataset of trajectories where each step has the following fields:
        - observation:
            - image_[0...N]        # RGB image observations
            - depth_[0...N]        # depth image observations
            - proprio              # concatenated low-dim observations
        - action                   # action vector
        - language_instruction     # language instruction string
        - is_last                  # boolean indicator, 1 on last step
        - is_terminal              # boolean indicator, 1 on last step *if not timeout*
    """
    builder = tfds.builder(name, data_dir=data_dir)
    if "val" not in builder.info.splits:
        split = "train[:95%]" if train else "train[95%:]"
    else:
        split = "train" if train else "val"

    dataset = dl.DLataset.from_rlds(builder, split=split, shuffle=shuffle)

    image_obs_keys = (
        [image_obs_keys] if not isinstance(image_obs_keys, Sequence) else image_obs_keys
    )
    depth_obs_keys = (
        [depth_obs_keys] if not isinstance(depth_obs_keys, Sequence) else depth_obs_keys
    )
    state_obs_keys = (
        [state_obs_keys] if not isinstance(state_obs_keys, Sequence) else state_obs_keys
    )

    def restructure(traj):
        # apply any dataset-specific transforms
        if name in RLDS_TRAJECTORY_MAP_TRANSFORMS:
            traj = RLDS_TRAJECTORY_MAP_TRANSFORMS[name](traj)

        # extracts RGB images, depth images and proprio based on provided keys, pad for all None keys
        orig_obs = traj.pop("observation")
        traj_len = tf.shape(traj["action"])[0]
        traj["observation"] = {}
        for i, key in enumerate(image_obs_keys):
            if key is None:
                pad_shape = (
                    (traj_len, resize_size[0], resize_size[1], 3)
                    if resize_size
                    else traj["observation"]["image_0"].shape
                )
                traj["observation"][f"image_{i}"] = tf.zeros(pad_shape, dtype=tf.uint8)
            else:
                traj["observation"][f"image_{i}"] = orig_obs[key]
        for i, key in enumerate(depth_obs_keys):
            if key is None:
                pad_shape = (
                    (traj_len, resize_size[0], resize_size[1])
                    if resize_size
                    else traj["observation"]["depth_0"].shape
                )
                traj["observation"][f"depth_{i}"] = tf.zeros(
                    pad_shape, dtype=tf.float32
                )
            else:
                traj["observation"][f"depth_{i}"] = orig_obs[key]
        if state_obs_keys:
            proprio = []
            for key in state_obs_keys:
                if key is None:
                    proprio.append(tf.zeros((traj_len, 1), dtype=tf.float32))
                else:
                    proprio.append(tf.cast(orig_obs[key], tf.float32))
            traj["observation"]["proprio"] = tf.concat(proprio, axis=-1)

        traj["action"] = tf.cast(traj["action"], tf.float32)

        # check that all other keys are present
        for key in ["action", "language_instruction", "is_last", "is_terminal"]:
            if key not in traj:
                raise ValueError(f"Key {key} is missing from trajectory: {traj}")

        return traj

    dataset = dataset.map(restructure)
    if action_proprio_metadata is None:
        action_proprio_metadata = get_action_proprio_stats(
            builder,
            dataset,
            state_obs_keys,
            RLDS_TRAJECTORY_MAP_TRANSFORMS.get(name, None),
        )

    dataset = apply_common_transforms(
        dataset,
        train=train,
        action_proprio_metadata=action_proprio_metadata,
        resize_size=resize_size,
        **kwargs,
    )
    dataset.action_proprio_metadata = action_proprio_metadata

    return dataset


def make_interleaved_dataset(
    common_dataset_args: dict,
    dataset_kwargs_list: List[dict],
    train: bool,
    sample_weights: Optional[List[float]] = None,
    shuffle_buffer_size: int = 100,
):
    """Creates an interleaved dataset from list of dataset kwargs.

    Args:
        common_dataset_args: shared arguments that get copied into every dataset (image size, shuffling etc)
        dataset_kwargs_list: list of kwargs, each element is passed to 'make_dataset' for individual datasets.
        train: whether this is a training or validation dataset.
        sample_weights: sampling weights for each dataset in list, values need to be >= 1.
        shuffle_buffer_size: base size of the dataset shuffle buffer for each dataset,
            gets multiplied by sampling weight.
    """
    # update dataset kwargs & create datasets
    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)
    assert len(sample_weights) == len(dataset_kwargs_list)
    assert (
        min(sample_weights) >= 1.0
    )  # convention to ensure sufficient shuffle buffer size

    datasets = []
    for i, data_kwargs in enumerate(dataset_kwargs_list):
        data_kwargs.update(**common_dataset_args)
        datasets.append(
            make_dataset(**data_kwargs, train=train)
            .unbatch()
            .shuffle(int(shuffle_buffer_size * sample_weights[i]))
            .repeat()
        )

    # interleave datasets with sampling weights
    dataset = tf.data.Dataset.sample_from_datasets(datasets, sample_weights)
    return dataset
