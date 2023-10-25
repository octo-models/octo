from collections import defaultdict
from functools import partial
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm

from orca.data.dataset_transforms import RLDS_TRAJECTORY_MAP_TRANSFORMS
from orca.data.utils import bc_goal_relabeling, task_augmentation
from orca.data.utils.data_utils import (
    ActionEncoding,
    get_action_proprio_stats,
    maybe_decode_depth_images,
    normalize_action_and_proprio,
    pprint_data_mixture,
    StateEncoding,
)


def _chunk_act_obs(traj, window_size):
    """
    Chunks actions and observations into the given window_size.

    The "action" and "observation" keys are each given a new axis (at index 1) of size `window_size`.
    """
    traj_len = tf.shape(traj["action"])[0]
    chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1), [traj_len, window_size]
    ) + tf.broadcast_to(tf.range(traj_len)[:, None], [traj_len, window_size])
    floored_chunk_indices = tf.maximum(chunk_indices, 0)
    for key in ["observation", "action"]:
        traj[key] = tf.nest.map_structure(
            lambda x: tf.gather(x, floored_chunk_indices), traj[key]
        )
    # out of bounds indices will be masked in transformer
    traj["observation"]["pad_mask"] = chunk_indices >= 0

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
    window_size: int = 1,
    resize_size: Optional[Tuple[int, int]] = None,
    skip_unlabeled: bool = False,
    **unused_kwargs,
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
        task_augmentation_strategy (Optional[str], optional): The task augmentation strategy to use, or None for no task
            augmentation. See `task_augmentation.py`.
        task_augmentation_kwargs (dict, optional): Additional keyword arguments to pass to the task augmentation
        resize_size (tuple, optional): target (height, width) for all RGB and depth images, default to no resize.
        window_size (int, optional): The length of the snippets that trajectories are chunked into.
        skip_unlabeled (bool, optional): Whether to skip trajectories with no language labels.
    """
    if skip_unlabeled:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_any(x["language_instruction"] != "")
        )

    # decodes string keys with names "image" & "depth", resizes "image" and "depth"
    dataset = dataset.frame_map(dl.transforms.decode_images)
    dataset = dataset.frame_map(maybe_decode_depth_images)
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
    dataset = dataset.map(partial(_chunk_act_obs, window_size=window_size))

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
    state_encoding: Optional[StateEncoding] = None,
    action_encoding: Optional[ActionEncoding] = None,
    ram_budget: Optional[int] = None,
    action_proprio_normalization_type: Optional[str] = None,
    apply_transforms: bool = True,
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
        state_encoding (StateEncoding, optional): type of state encoding used, e.g. joint angles vs EEF pose.
        action_encoding (ActionEncoding, optional): type of action encoding used, e.g. joint delta vs EEF delta.
        ram_budget (int, optional): limits the RAM used by tf.data.AUTOTUNE, unit: GB, forwarded to AutotuneOptions.
        action_proprio_normalization_type (Optional[str], optional): The type of normalization to perform on the action,
            proprio, or both. Can be "normal" (mean 0, std 1) or "bounds" (normalized to [-1, 1]).
        apply_transforms (bool): If True, applies common transforms like augmentations and chunking to episode dataset.
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

    dataset = dl.DLataset.from_rlds(
        builder, split=split, shuffle=shuffle, num_parallel_reads=8
    )
    if ram_budget:
        dataset = dataset.with_ram_budget(ram_budget)

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

        # remove unused keys
        keep_keys = [
            "observation",
            "action",
            "language_instruction",
            "is_terminal",
            "is_last",
            "_traj_index",
        ]
        traj = {k: v for k, v in traj.items() if k in keep_keys}

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
                traj["observation"][f"image_{i}"] = tf.io.encode_png(
                    tf.zeros(pad_shape, dtype=tf.uint8)
                )
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
            # pre-pend info which state encoding is used
            proprio = [
                tf.ones_like(proprio[0][:, :1]) * float(state_encoding)
            ] + proprio
            traj["observation"]["proprio"] = tf.concat(proprio, axis=-1)

        # TODO: support other action encodings as well
        assert (
            action_encoding == ActionEncoding.EEF_POS
        ), "Only support EEF pose delta actions for now."
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
        dataset = dataset.map(
            partial(
                normalize_action_and_proprio,
                metadata=action_proprio_metadata,
                normalization_type=action_proprio_normalization_type,
            )
        )

    if apply_transforms:
        dataset = apply_common_transforms(
            dataset,
            train=train,
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
    shuffle_buffer_size: int = 10000,
):
    """Creates an interleaved dataset from list of dataset kwargs.

    Args:
        common_dataset_args: shared arguments that get copied into every dataset (image size, shuffling etc)
        dataset_kwargs_list: list of kwargs, each element is passed to 'make_dataset' for individual datasets.
            Will get merged with and overwritten by common_dataset_args.
        train: whether this is a training or validation dataset.
        sample_weights: sampling weights for each dataset in list, values need to be >= 1.
        shuffle_buffer_size: size of the dataset shuffle buffer for interleaved dataset.
    """
    # update dataset kwargs & create datasets
    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)
    assert len(sample_weights) == len(dataset_kwargs_list)
    pprint_data_mixture(dataset_kwargs_list, sample_weights)

    datasets = []
    for i, data_kwargs in enumerate(
        tqdm.tqdm(dataset_kwargs_list, desc="Generating individual datasets...")
    ):
        data_kwargs.update(**common_dataset_args)
        datasets.append(
            make_dataset(**data_kwargs, train=train, apply_transforms=False).repeat()
        )

    # interleave datasets with sampling weights
    dataset = dl.DLataset.sample_from_datasets(datasets, sample_weights)

    # apply common transforms like augmentation, chunking etc on interleaved episode dataset
    # first interleaving episodes and then applying transforms is more memory efficient
    dataset = apply_common_transforms(
        dataset,
        train=train,
        **common_dataset_args,
    )

    dataset = dataset.flatten(num_parallel_calls=8).shuffle(shuffle_buffer_size)
    return dataset
