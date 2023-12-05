import copy
from functools import partial
import json
from typing import Callable, List, Optional, Sequence, Tuple, Union

from absl import logging
import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from orca.data.dataset_transforms import RLDS_TRAJECTORY_MAP_TRANSFORMS
from orca.data.utils import bc_goal_relabeling, task_augmentation
from orca.data.utils.data_utils import (
    action_encoding_length,
    ActionEncoding,
    allocate_threads,
    get_dataset_statistics,
    make_zero_actions,
    normalize_action_and_proprio,
    pprint_data_mixture,
    state_encoding_length,
    StateEncoding,
    tree_map,
)


def _chunk_act_obs(
    traj,
    window_size,
    additional_action_window_size=0,
    action_encoding: ActionEncoding = ActionEncoding.EEF_POS,
):
    """
    Chunks actions and observations into the given window_size.

    The "action" and "observation" keys are each given a new axis (at index 1) of size `window_size`.
    """
    traj_len = tf.shape(traj["action"])[0]
    chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1), [traj_len, window_size]
    ) + tf.broadcast_to(tf.range(traj_len)[:, None], [traj_len, window_size])

    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + additional_action_window_size),
        [traj_len, window_size + additional_action_window_size],
    ) + tf.broadcast_to(
        tf.range(traj_len)[:, None],
        [traj_len, window_size + additional_action_window_size],
    )

    floored_chunk_indices = tf.maximum(chunk_indices, 0)

    if "tasks" in traj:
        goal_timestep = traj["tasks"]["goal_timestep"]
    else:
        goal_timestep = tf.fill([traj_len], traj_len, dtype=tf.int32)

    floored_action_chunk_indices = tf.minimum(
        tf.maximum(action_chunk_indices, 0), goal_timestep[:, None] - 1
    )

    traj["observation"] = tf.nest.map_structure(
        lambda x: tf.gather(x, floored_chunk_indices), traj["observation"]
    )
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)

    # indicates whether an entire observation is padding
    traj["observation"]["pad_mask"] = chunk_indices >= 0

    # Actions past the goal timestep become no-ops
    action_past_goal = action_chunk_indices > goal_timestep[:, None] - 1
    zero_actions = make_zero_actions(traj["action"], action_encoding)
    traj["action"] = tf.where(
        action_past_goal[:, :, None], zero_actions, traj["action"]
    )
    return traj


def _subsample(traj, subsample_length):
    """Subsamples trajectories to the given length."""
    traj_len = tf.shape(traj["action"])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)
    return traj


def _add_pad_masks(traj):
    traj_len = tf.shape(traj["action"])[0]
    pad_masks = {}
    for key in traj["observation"]:
        if traj["observation"][key].dtype == tf.string:
            pad_masks[key] = tf.strings.length(traj["observation"][key]) != 0
        else:
            pad_masks[key] = tf.ones([traj_len], dtype=tf.bool)
    traj["observation"]["pad_mask_dict"] = pad_masks
    return traj


def _decode_images(obs: dict) -> dict:
    """Decodes images and depth images, marking empty strings as padding."""
    for key in obs:
        if "image" in key:
            if obs[key].dtype == tf.string:
                if tf.strings.length(obs[key]) == 0:
                    # this is a padding image
                    obs[key] = tf.zeros((1, 1, 3), dtype=tf.uint8)
                else:
                    obs[key] = tf.io.decode_image(
                        obs[key], expand_animations=False, dtype=tf.uint8
                    )
            elif obs[key].dtype == tf.uint8:
                pass
            else:
                raise ValueError(
                    f"Unsupported image dtype: found {key} with dtype {obs[key].dtype}"
                )
        elif "depth" in key:
            if obs[key].dtype == tf.string:
                if tf.strings.length(obs[key]) == 0:
                    # this is a padding image
                    obs[key] = tf.zeros((1, 1), dtype=tf.float32)
                else:
                    obs[key] = tf.io.decode_image(
                        obs[key], expand_animations=False, dtype=tf.float32
                    )[..., 0]
            elif obs[key].dtype == tf.float32:
                pass
            else:
                raise ValueError(
                    f"Unsupported depth dtype: found {key} with dtype {obs[key].dtype}"
                )

    # obs["pad_mask_dict"] = pad_mask_dict
    return obs


def _augment(obs: dict, seed, augment_kwargs) -> dict:
    """Augments images, skipping padding images."""
    for key in obs:
        if "image" in key:
            if obs["pad_mask_dict"][key]:
                obs[key] = dl.transforms.augment_image(
                    obs[key], **augment_kwargs, seed=seed
                )
    return obs


def apply_trajectory_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    goal_relabeling_strategy: Optional[str] = None,
    goal_relabeling_kwargs: dict = {},
    task_augmentation_strategy: Optional[str] = None,
    task_augmentation_kwargs: dict = {},
    window_size: int = 1,
    additional_action_window_size: int = 0,
    action_encoding: ActionEncoding = ActionEncoding.EEF_POS,
    subsample_length: Optional[int] = None,
    skip_unlabeled: bool = False,
    max_action: Optional[float] = None,
    max_proprio: Optional[float] = None,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """
    Applies common transforms that happen at a trajectory level. Such transforms are usually some
    sort of "relabeling" (e.g. filtering, chunking, adding goals, dropping keys). Transforms that
    happen in this function should have the following properties:

    - They require access to an entire trajectory (i.e. they cannot be applied in a frame-wise manner).
    - They are generally not CPU-intensive, mostly involving moving and copying data.
    - They do not require decoded images.

    Args:
        dataset (dl.DLataset): The dataset to transform.
        train (bool): Whether the dataset is for training (affects task augmentation).
        goal_relabeling_strategy (str, optional): The goal relabeling strategy to use, or None for
            no goal relabeling. See `bc_goal_relabeling.py`.
        goal_relabeling_kwargs (dict, optional): Additional keyword arguments to pass to the goal relabeling function.
        task_augmentation_strategy (Optional[str], optional): The task augmentation strategy to use, or None for no task
            augmentation. See `task_augmentation.py`.
        task_augmentation_kwargs (dict, optional): Additional keyword arguments to pass to the task augmentation function.
        window_size (int, optional): The length of the snippets that trajectories are chunked into.
        additional_action_window_size (int, optional): The number of additional actions beyond window_size to include
            in the chunked actions.
        action_encoding (ActionEncoding): type of action encoding used, e.g. joint delta vs EEF delta.
        subsample_length (int, optional): If provided, trajectories longer than this will be
            subsampled to this length (after goal relabeling).
        skip_unlabeled (bool, optional): Whether to skip trajectories with no language labels.
        max_action: (float, optional): If provided, trajectories in which *any* action dimension
            of *any* transition has an absolute value larger than this will be skipped.
        max_proprio: (float, optional): If provided, trajectories in which *any* proprio dimension
            of *any* transition has an absolute value larger than this will be skipped.
        num_parallel_calls (int, optional): number of parallel calls for map operations. Default to AUTOTUNE.
    """
    if skip_unlabeled:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_any(x["language_instruction"] != "")
        )

    if max_action is not None:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_all(tf.math.abs(x["action"]) <= max_action)
        )

    if max_proprio is not None:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_all(
                tf.math.abs(x["observation"]["proprio"]) <= max_proprio
            )
        )
    dataset = dataset.map(_add_pad_masks, num_parallel_calls)

    # adds the "tasks" key
    if goal_relabeling_strategy is not None:
        dataset = dataset.map(
            partial(
                getattr(bc_goal_relabeling, goal_relabeling_strategy),
                **goal_relabeling_kwargs,
            ),
            num_parallel_calls,
        )

        def move_language_instruction_to_tasks(traj):
            traj["tasks"]["language_instruction"] = traj.pop("language_instruction")
            return traj

        dataset = dataset.map(move_language_instruction_to_tasks, num_parallel_calls)

    if train and subsample_length is not None:
        dataset = dataset.map(
            partial(_subsample, subsample_length=subsample_length), num_parallel_calls
        )

    def add_pad_mask_dict(traj):
        traj["tasks"]["pad_mask_dict"]["language_instruction"] = (
            tf.strings.length(traj["tasks"]["language_instruction"]) != 0
        )
        return traj

    dataset = dataset.map(add_pad_mask_dict, num_parallel_calls)

    if train and task_augmentation_strategy is not None:
        dataset = dataset.map(
            partial(
                getattr(task_augmentation, task_augmentation_strategy),
                **task_augmentation_kwargs,
            ),
            num_parallel_calls,
        )

    dataset = dataset.map(
        partial(
            _chunk_act_obs,
            window_size=window_size,
            additional_action_window_size=additional_action_window_size,
            action_encoding=action_encoding,
        ),
        num_parallel_calls,
    )

    return dataset


def get_frame_transforms(
    train: bool,
    image_augment_kwargs: Optional[dict] = None,
    resize_size: Optional[Tuple[int, int]] = None,
) -> List[Callable[[dict], dict]]:
    """
    Returns a list of functions to be applied to each frame. These transforms are usually
    more CPU-intensive, (e.g. decoding or resizing images).

    Args:
        train (bool): Whether the dataset is for training (affects image augmentation).
        image_augment_kwargs (dict): Keyword arguments to pass to the image augmentation function. See
            `dlimp.transforms.augment_image` for documentation.
        resize_size (Tuple[int, int], optional): If provided, images will be resized to this size.
    """

    # convenience wrapper that takes a function that operates on a non-chunked "observation" dict
    # and applies it to the chunked "observation" dict as well as the "tasks" dict
    def apply_obs_transform(fn: Callable[[dict], dict], frame):
        # tasks is not chunked -- apply fn directly
        frame["tasks"] = fn(frame["tasks"])
        # observation is chunked -- apply fn along first axis
        frame["observation"] = dl.vmap(fn)(frame["observation"])
        return frame

    transforms = []

    # decode images (and depth images), marking empty strings as padding
    transforms.append(partial(apply_obs_transform, _decode_images))

    # resize images, if requested
    if resize_size is not None:
        transforms.append(
            partial(
                apply_obs_transform,
                partial(dl.transforms.resize_images, size=resize_size),
            )
        )
        transforms.append(
            partial(
                apply_obs_transform,
                partial(dl.transforms.resize_depth_images, size=resize_size),
            )
        )

    if train and image_augment_kwargs is not None:
        # augment all images with the same seed, skipping padding images
        def aug(frame):
            seed = tf.random.uniform([2], maxval=tf.dtypes.int32.max, dtype=tf.int32)
            aug_fn = partial(_augment, seed=seed, augment_kwargs=image_augment_kwargs)
            return apply_obs_transform(aug_fn, frame)

        transforms.append(aug)

    return transforms


def make_dataset_from_rlds(
    name: str,
    data_dir: str,
    train: bool,
    shuffle: bool = True,
    image_obs_keys: Union[str, List[str]] = [],
    depth_obs_keys: Union[str, List[str]] = [],
    state_obs_keys: Union[str, List[str]] = [],
    state_encoding: StateEncoding = StateEncoding.NONE,
    action_encoding: ActionEncoding = ActionEncoding.EEF_POS,
    action_proprio_normalization_type: Optional[str] = None,
    dataset_statistics: Optional[Union[dict, str]] = None,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> Tuple[dl.DLataset, Optional[dict]]:
    """This function is responsible for loading a specific RLDS dataset from storage and getting it into a
    standardized format (see below). Yields a dataset of trajectories. Does not include CPU-intensive operations.

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
        state_encoding (StateEncoding): type of state encoding used, e.g. joint angles vs EEF pose.
        action_encoding (ActionEncoding): type of action encoding used, e.g. joint delta vs EEF delta.
        action_proprio_normalization_type (Optional[str], optional): The type of normalization to perform on the action,
            proprio, or both. Can be "normal" (mean 0, std 1) or "bounds" (normalized to [-1, 1]).
        dataset_statistics: (dict|str, optional): dict (or path to JSON file) that contains dataset
            statistics for normalization. If `action_proprio_normalization_type` is "normal", this
            should contain "mean" and "std" keys. If `action_proprio_normalization_type` is "bounds",
            this should contain "min" and "max" keys. May also provide "num_transitions" and
            "num_trajectories" keys for downstream usage (e.g., for `make_interleaved_dataset`). If
            not provided, the statistics will be computed on the fly based on the train split of the
            dataset.
        num_parallel_reads: number of parallel read workers. Default to AUTOTUNE.
        num_parallel_calls: number of parallel calls for map operations. Default to AUTOTUNE.
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
        builder, split=split, shuffle=shuffle, num_parallel_reads=num_parallel_reads
    )

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
        rlds_transform = RLDS_TRAJECTORY_MAP_TRANSFORMS[name]

        # skip None transforms
        if rlds_transform is not None:
            traj = rlds_transform(traj)

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
                # pad with empty string
                traj["observation"][f"image_{i}"] = tf.repeat("", traj_len)
            else:
                traj["observation"][f"image_{i}"] = orig_obs[key]
        for i, key in enumerate(depth_obs_keys):
            if key is None:
                # pad with empty string
                traj["observation"][f"depth_{i}"] = tf.repeat("", traj_len)
            else:
                traj["observation"][f"depth_{i}"] = orig_obs[key]
        if state_obs_keys:
            proprio = []
            for key in state_obs_keys:
                if key is None:
                    # pad with zero
                    proprio.append(tf.zeros((traj_len, 1), dtype=tf.float32))
                else:
                    proprio.append(tf.cast(orig_obs[key], tf.float32))
            traj["observation"]["proprio"] = tf.concat(proprio, axis=-1)
            # make sure state encoding has correct length
            if state_encoding != StateEncoding.NONE:
                assert traj["observation"]["proprio"].shape[
                    -1
                ] == state_encoding_length(state_encoding), (
                    f"State encoding {state_encoding} for dataset {name} expects {state_encoding_length(state_encoding)}-dim proprio"
                    f" but got {traj['observation']['proprio'].shape[-1]}."
                )

        # make sure action encoding has correct length
        assert traj["action"].shape[-1] == action_encoding_length(action_encoding), (
            f"Action encoding {action_encoding} for dataset {name} expects {action_encoding_length(action_encoding)}-dim actions"
            f" but got {traj['action'].shape[-1]}."
        )
        traj["action"] = tf.cast(traj["action"], tf.float32)

        # check that all other keys are present
        for key in ["action", "language_instruction", "is_last", "is_terminal"]:
            if key not in traj:
                raise ValueError(f"Key {key} is missing from trajectory: {traj}")

        # add state and action encoding info
        traj["observation"]["state_encoding"] = tf.repeat(state_encoding, traj_len)[
            ..., None
        ]
        traj["observation"]["action_encoding"] = tf.repeat(action_encoding, traj_len)[
            ..., None
        ]

        # add timestep info
        traj["observation"]["timestep"] = tf.range(traj_len) + 1

        # add name of dataset
        traj["dataset_name"] = tf.repeat(name, traj_len)

        return traj

    dataset = dataset.map(restructure, num_parallel_calls)

    if isinstance(dataset_statistics, str):
        with tf.io.gfile.GFile(dataset_statistics, "r") as f:
            dataset_statistics = json.load(f)
    elif dataset_statistics is None:
        # tries to load from cache, otherwise computes on the fly
        dataset_statistics = get_dataset_statistics(
            builder, state_obs_keys, restructure, RLDS_TRAJECTORY_MAP_TRANSFORMS[name]
        )
    dataset_statistics = tree_map(np.array, dataset_statistics)

    dataset = dataset.map(
        partial(
            normalize_action_and_proprio,
            metadata=dataset_statistics,
            normalization_type=action_proprio_normalization_type,
        ),
        num_parallel_calls,
    )

    return dataset, dataset_statistics


def make_single_dataset(
    dataset_kwargs: dict,
    traj_transform_kwargs: dict,
    frame_transform_kwargs: dict,
    train: bool,
    frame_transform_threads: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """Creates a single dataset from kwargs. Returns a dataset of trajectories.

    Args:
        dataset_kwargs: kwargs passed to `make_dataset_from_rlds` that are dataset-specific.
        traj_transform_kwargs: kwargs passed to 'apply_trajectory_transforms'.
        frame_transform_kwargs: kwargs passed to 'get_frame_transforms'.
        train: whether this is a training or validation dataset.
        frame_transform_threads: number of parallel calls for frame transforms. Default to AUTOTUNE.
    """
    dataset, dataset_statistics = make_dataset_from_rlds(**dataset_kwargs, train=train)
    dataset = apply_trajectory_transforms(dataset, **traj_transform_kwargs, train=train)

    for fn in get_frame_transforms(**frame_transform_kwargs, train=train):
        dataset = dataset.frame_map(fn, frame_transform_threads)

    # this seems to reduce memory usage without affecting speed
    dataset = dataset.with_ram_budget(1)

    # save for later
    dataset.dataset_statistics = dataset_statistics
    return dataset


def make_interleaved_dataset(
    *,
    dataset_kwargs_list: List[dict],
    traj_transform_kwargs: dict,
    frame_transform_kwargs: dict,
    train: bool,
    sample_weights: Optional[List[float]],
    balance_weights: bool,
    shuffle_buffer_size: int,
    batch_size: int,
    traj_transform_threads: Optional[int],
    traj_read_threads: Optional[int],
    frame_transform_threads: int,
) -> dl.DLataset:
    """Creates an interleaved dataset from list of dataset kwargs. Returns a dataset of batched frames.

    Args:
        dataset_kwargs_list: list of kwargs, each element of which is passed to `make_dataset_from_rlds`.
            "num_parallel_calls" and "num_parallel_reads" are ignored.
        traj_transform_kwargs: kwargs passed to 'apply_trajectory_transforms'. "num_parallel_calls" is ignored.
        frame_transform_kwargs: kwargs passed to 'get_frame_transforms'.
        train: whether this is a training or validation dataset.
        sample_weights: sampling weights for each dataset in list. If None, defaults to uniform.
        balance_weights: if True, the sample weights are multiplied by the number of frames in
            each dataset. This makes it so that, if all the sample weights are equal, one full iteration
            through the interleaved dataset will correspond to one full iteration through each
            individual dataset (only in expectation, since in practice the sampling is random).
        shuffle_buffer_size: size of the dataset shuffle buffer (in number of frames).
        batch_size: batch size.
        traj_transform_threads: total number of parallel calls for trajectory transforms, distributed across
            datasets according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
        traj_read_threads: total number of parallel read workers for trajectory transforms, distributed across
            datasets according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
        frame_transform_threads: number of parallel calls for frame transforms, which happen after datasets
            are interleaved. Default to AUTOTUNE.
    """
    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)
    assert len(sample_weights) == len(dataset_kwargs_list)

    # go through datasets once to get sizes
    dataset_sizes = []
    all_dataset_statistics = []
    for dataset_kwargs in dataset_kwargs_list:
        _, dataset_statistics = make_dataset_from_rlds(**dataset_kwargs, train=train)
        dataset_sizes.append(dataset_statistics["num_transitions"])
        all_dataset_statistics.append(dataset_statistics)

    # balance and normalize weights
    if balance_weights:
        sample_weights = np.array(sample_weights) * np.array(dataset_sizes)
    sample_weights = np.array(sample_weights) / np.sum(sample_weights)
    pprint_data_mixture(dataset_kwargs_list, sample_weights)

    # allocate threads based on weights
    if traj_transform_threads is None:
        threads_per_dataset = [tf.data.AUTOTUNE] * len(dataset_kwargs_list)
    else:
        threads_per_dataset = allocate_threads(traj_transform_threads, sample_weights)
    if traj_read_threads is None:
        reads_per_dataset = [tf.data.AUTOTUNE] * len(dataset_kwargs_list)
    else:
        reads_per_dataset = allocate_threads(traj_read_threads, sample_weights)

    logging.info("Threads per dataset: %s", threads_per_dataset)
    logging.info("Reads per dataset: %s", reads_per_dataset)

    # construct datasets
    datasets = []
    action_encodings = []
    for dataset_kwargs, dataset_statistics, threads, reads in zip(
        dataset_kwargs_list,
        all_dataset_statistics,
        threads_per_dataset,
        reads_per_dataset,
    ):
        dataset, _ = make_dataset_from_rlds(
            **dataset_kwargs,
            train=train,
            num_parallel_calls=threads,
            num_parallel_reads=reads,
            dataset_statistics=dataset_statistics,
        )
        dataset = apply_trajectory_transforms(
            dataset.repeat(),
            **traj_transform_kwargs,
            num_parallel_calls=threads,
            train=train,
        ).flatten(num_parallel_calls=threads)
        action_encodings.append(
            dataset_kwargs.get("action_encoding", ActionEncoding.EEF_POS)
        )
        datasets.append(dataset)

    # TODO: support interleaving datasets with different action encodings
    assert (
        len(set(action_encodings)) == 1
    ), f"Need action encodings of all datasets to be identical, currently used encodings: {action_encodings}."

    # interleave at the transition level and then shuffle
    dataset: dl.DLataset = dl.DLataset.sample_from_datasets(
        datasets, sample_weights
    ).shuffle(shuffle_buffer_size)

    # apply frame transforms
    for fn in get_frame_transforms(**frame_transform_kwargs, train=train):
        dataset = dataset.map(fn, frame_transform_threads)

    dataset = dataset.batch(batch_size)

    # this seems to reduce memory usage without affecting speed
    dataset = dataset.with_ram_budget(1)

    # save for later
    dataset.dataset_statistics = all_dataset_statistics
    return dataset
