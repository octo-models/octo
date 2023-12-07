import copy
from functools import partial
import inspect
import json
from typing import Callable, List, Optional, Sequence, Tuple, Union

from absl import logging
import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from orca.data.standardization_transforms import RLDS_STANDARDIZATION_TRANSFORMS
from orca.data.utils import bc_goal_relabeling, task_augmentation
from orca.data.utils.data_utils import (
    allocate_threads,
    get_dataset_statistics,
    make_neutral_actions,
    NormalizationType,
    normalize_action_and_proprio,
    pprint_data_mixture,
    tree_map,
)


def _chunk_act_obs(
    traj,
    window_size,
    additional_action_window_size=0,
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

    if "task" in traj:
        goal_timestep = traj["task"]["goal_timestep"]
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
    # zero_actions = make_neutral_actions(traj["action"], action_encoding)
    # traj["action"] = tf.where(
    #     action_past_goal[:, :, None], zero_actions, traj["action"]
    # )
    return traj


def _subsample(traj, subsample_length):
    """Subsamples trajectories to the given length."""
    traj_len = tf.shape(traj["action"])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)
    return traj


def _add_pad_mask_dict(traj):
    """Adds a dictionary indicating which elements of the observation are padding.

    traj["observation"]["pad_mask_dict"] = {k: traj["observation"][k] is not padding}
    """
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
    """Decodes images and depth images."""
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
    return obs


def _augment(obs: dict, seed, augment_kwargs) -> dict:
    """Augments images, skipping padding images."""
    num_image_keys = sum(["image" in key for key in obs])

    if not isinstance(augment_kwargs, Sequence):
        augment_kwargs = [copy.deepcopy(augment_kwargs)] * num_image_keys

    for i in range(num_image_keys):
        if augment_kwargs[i] is not None:
            key = f"image_{i}"
            if obs["pad_mask_dict"][key]:
                obs[key] = dl.transforms.augment_image(
                    obs[key], **augment_kwargs[i], seed=seed + i
                )
    return obs


def _resize(obs: dict, resize_size, depth_resize_size) -> dict:
    """Resizes images and depth images."""
    num_image_keys = sum(["image" in key for key in obs])
    num_depth_keys = sum(["depth" in key for key in obs])

    if resize_size is None or isinstance(resize_size[0], int):
        resize_size = [resize_size] * num_image_keys
    if depth_resize_size is None or isinstance(depth_resize_size[0], int):
        depth_resize_size = [depth_resize_size] * num_depth_keys

    for i in range(num_image_keys):
        if resize_size[i] is not None:
            key = f"image_{i}"
            obs[key] = dl.transforms.resize_image(obs[key], size=resize_size[i])

    for i in range(num_depth_keys):
        if depth_resize_size[i] is not None:
            key = f"depth_{i}"
            obs[key] = dl.transforms.resize_depth_image(
                obs[key], size=depth_resize_size[i]
            )
    return obs


def apply_trajectory_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    goal_relabeling_strategy: Optional[str] = None,
    goal_relabeling_kwargs: dict = {},
    window_size: int = 1,
    additional_action_window_size: int = 0,
    subsample_length: Optional[int] = None,
    skip_unlabeled: bool = False,
    max_action: Optional[float] = None,
    max_proprio: Optional[float] = None,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """Applies common transforms that happen at a trajectory level. Such transforms are usually some sort of
    "relabeling" (e.g. filtering, chunking, adding goals, dropping keys). Transforms that happen in this
    function should have the following properties:

    - They require access to an entire trajectory (i.e. they cannot be applied in a frame-wise manner).
    - They are generally not CPU-intensive, mostly involving moving and copying data.
    - They do not require decoded images.

    Args:
        dataset (dl.DLataset): The dataset to transform.
        train (bool): Whether the dataset is for training (affects subsampling).
        goal_relabeling_strategy (str, optional): The goal relabeling strategy to use, or None for
            no goal relabeling. See `bc_goal_relabeling.py`.
        goal_relabeling_kwargs (dict, optional): Additional keyword arguments to pass to the goal relabeling function.
        window_size (int, optional): The length of the snippets that trajectories are chunked into.
        additional_action_window_size (int, optional): The number of additional actions beyond window_size to include
            in the chunked actions.
        subsample_length (int, optional): If provided, trajectories longer than this will be
            subsampled to this length (after goal relabeling).
        skip_unlabeled (bool, optional): Whether to skip trajectories with no language labels.
        max_action: (float, optional): If provided, trajectories in which *any* action dimension
            of *any* transition has an absolute value larger than this will be skipped.
        max_proprio: (float, optional): If provided, trajectories in which *any* proprio dimension
            of *any* transition has an absolute value larger than this will be skipped.
        num_parallel_calls (int, optional): number of parallel calls for map operations. Default to AUTOTUNE.
    """
    if skip_unlabeled and "language_instruction" in dataset.element_spec:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_any(x["language_instruction"] != "")
        )

    if max_action is not None:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_all(tf.math.abs(x["action"]) <= max_action)
        )

    if max_proprio is not None and "proprio" in dataset.element_spec["observation"]:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_all(
                tf.math.abs(x["observation"]["proprio"]) <= max_proprio
            )
        )

    # marks which observations are padding
    dataset = dataset.traj_map(_add_pad_mask_dict, num_parallel_calls)

    # adds the "task" key
    if goal_relabeling_strategy is not None:
        dataset = dataset.traj_map(
            partial(
                getattr(bc_goal_relabeling, goal_relabeling_strategy),
                **goal_relabeling_kwargs,
            ),
            num_parallel_calls,
        )

        if "language_instruction" in dataset.element_spec:

            def move_language_instruction_to_task(traj):
                traj["task"]["language_instruction"] = traj.pop("language_instruction")
                traj["task"]["pad_mask_dict"]["language_instruction"] = (
                    tf.strings.length(traj["task"]["language_instruction"]) != 0
                )
                return traj

            dataset = dataset.traj_map(
                move_language_instruction_to_task, num_parallel_calls
            )

    if train and subsample_length is not None:
        dataset = dataset.traj_map(
            partial(_subsample, subsample_length=subsample_length), num_parallel_calls
        )

    dataset = dataset.traj_map(
        partial(
            _chunk_act_obs,
            window_size=window_size,
            additional_action_window_size=additional_action_window_size,
        ),
        num_parallel_calls,
    )

    return dataset


def apply_frame_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    image_augment_kwargs: Union[Optional[dict], Sequence[Optional[dict]]] = None,
    resize_size: Union[
        Optional[Tuple[int, int]], Sequence[Optional[Tuple[int, int]]]
    ] = None,
    depth_resize_size: Union[
        Optional[Tuple[int, int]], Sequence[Optional[Tuple[int, int]]]
    ] = None,
    task_augment_strategy: Optional[str] = None,
    task_augment_kwargs: dict = {},
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """Applies common transforms that happen at a frame level. These transforms are usually more
    CPU-intensive, (e.g. decoding or resizing images).

    Args:
        train (bool): Whether the dataset is for training (affects image augmentation).
        dataset (dl.DLataset): The dataset to transform.
        image_augment_kwargs (dict|Sequence[dict]): Keyword arguments to pass to the image augmentation
            function. See `dlimp.transforms.augment_image` for documentation of these kwargs. If a list of dicts
            is provided, then the ith entry will be used for "image_i" (order determined by `image_obs_keys` in
            `make_dataset_from_rlds`). A None list entry will skip image augmentation for the corresponding
            image(s).
        resize_size (Tuple[int, int]|Sequence[Tuple[int, int]]): If provided, images will be
            resized to this size. If a list of tuples is provided, then the ith entry will be used for
            "image_i" and "depth_i" (order determined by `image_obs_keys` and `depth_obs_keys`, respectively,
            in `make_dataset_from_rlds`). A value of None or a None list entry will skip resizing for the
            corresponding image(s).
        depth_resize_size (Tuple[int, int]|Sequence[Tuple[int, int]]): Same as resize_size, but for depth
            images.
        task_augmentation_strategy (str, optional): The task augmentation strategy to use, or None for no task
            augmentation. See `task_augmentation.py`.
        task_augmentation_kwargs (dict, optional): Additional keyword arguments to pass to the task
            augmentation function.
        num_parallel_calls (int): number of parallel calls for frame_map operations. Default to AUTOTUNE.
    """

    # convenience wrapper that takes a function that operates on a non-chunked "observation" dict and applies
    # it to the chunked "observation" dict as well as the non-chunked "task" dict
    def apply_obs_transform(fn: Callable[[dict], dict], frame):
        # task is not chunked -- apply fn directly
        frame["task"] = fn(frame["task"])
        # observation is chunked -- apply fn along first axis
        frame["observation"] = dl.vmap(fn)(frame["observation"])
        return frame

    if train and task_augment_strategy is not None:
        # perform task augmentation (e.g., dropping keys)
        dataset = dataset.frame_map(
            partial(
                getattr(task_augmentation, task_augment_strategy),
                **task_augment_kwargs,
            ),
            num_parallel_calls,
        )

    # decode images (and depth images)
    dataset = dataset.frame_map(
        partial(apply_obs_transform, _decode_images), num_parallel_calls
    )

    # resize images (and depth images)
    dataset = dataset.frame_map(
        partial(
            apply_obs_transform,
            partial(
                _resize, resize_size=resize_size, depth_resize_size=depth_resize_size
            ),
        ),
        num_parallel_calls,
    )

    if train:
        # augment all images with the same seed, skipping padding images
        def aug(frame):
            seed = tf.random.uniform([2], maxval=tf.dtypes.int32.max, dtype=tf.int32)
            aug_fn = partial(_augment, seed=seed, augment_kwargs=image_augment_kwargs)
            return apply_obs_transform(aug_fn, frame)

        dataset = dataset.frame_map(aug, num_parallel_calls)

    return dataset


def make_dataset_from_rlds(
    name: str,
    data_dir: str,
    train: bool,
    standardize_fn: Optional[Callable[[dict], dict]] = None,
    shuffle: bool = True,
    image_obs_keys: Union[str, Sequence[str]] = (),
    depth_obs_keys: Union[str, Sequence[str]] = (),
    state_obs_keys: Union[str, Sequence[str]] = (),
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
    dataset_statistics: Optional[Union[dict, str]] = None,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> Tuple[dl.DLataset, dict]:
    """This function is responsible for loading a specific RLDS dataset from storage and getting it into a
    standardized format. Yields a dataset of trajectories. Does not include CPU-intensive operations.

    If `standardize_fn` is provided, it will be applied to each trajectory. This function should get the
    trajectory into a standard format, which includes the keys "observation", "action", "is_terminal", and
    "is_last". "observation" should be a dictionary containing some number of additional keys, which will be
    extracted into an even more standardized numbered format according to the "*_obs_keys" arguments.

    For example, if the "observation" dict has the keys "image_workspace" and "image_wrist" after
    `standardize_fn`, and `image_obs_keys=("image_workspace", None, "image_wrist")`, then the resulting
    dataset will have an "observation" dict containing the keys "image_0", "image_1", and "image_2", where
    "image_0" corresponds to "image_workspace", "image_1" is a padding image, and "image_2" corresponds to
    "image_wrist".

    Args:
        name (str): The name of the RLDS dataset (usually "name" or "name:version").
        data_dir (str): The path to the data directory.
        train (bool): Whether to use the training or validation set.
        shuffle (bool, optional): Whether to shuffle the file read order (does NOT fully shuffle directories,
            since one file usually contains many trajectories!).
        image_obs_keys (str|Sequence[str], optional): List of keys to be extracted from the "observation"
            dict and mapped to "image_{i}". Inserts padding (an empty string) for each None entry.
        depth_obs_keys (str|Sequence[str], optional): List of keys to be extracted from the "observation"
            dict and mapped to "depth_{i}". Inserts padding (an empty string) for each None entry.
        state_obs_keys (str|Sequence[str], optional): List of 1-dimensional proprioception keys to be
            extracted from the "observation" dict, concatenated, and mapped to "proprio". Inserts 1 element of
            padding (zero) for each None entry.
        action_proprio_normalization_type (str, optional): The type of normalization to perform on the action,
            proprio, or both. Can be "normal" (mean 0, std 1) or "bounds" (normalized to [-1, 1]).
        dataset_statistics: (dict|str, optional): dict (or path to JSON file) that contains dataset statistics
            for normalization. If `action_proprio_normalization_type` is "normal", this should contain "mean" and
            "std" keys. If `action_proprio_normalization_type` is "bounds", this should contain "min" and "max"
            keys. May also provide "num_transitions" and "num_trajectories" keys for downstream usage (e.g., for
            `make_interleaved_dataset`). If not provided, the statistics will be computed on the fly.
        num_parallel_reads (int): number of parallel read workers. Default to AUTOTUNE.
        num_parallel_calls (int): number of parallel calls for traj_map operations. Default to AUTOTUNE.
    Returns:
        Dataset of trajectories where each step has the following fields:
        - observation:
            - image_{0, 1, ..., N} # RGB image observations
            - depth_{0, 1, ..., N} # depth image observations
            - proprio              # 1-dimensional array of proprioceptive observations
        - action                   # action vector
        - is_last                  # boolean indicator, 1 on last step
        - is_terminal              # boolean indicator, 1 on last step *if not timeout*
        - language_instruction     # string language instruction (optional)
    """
    builder = tfds.builder(name, data_dir=data_dir)
    if "val" not in builder.info.splits:
        split = "train[:95%]" if train else "train[95%:]"
    else:
        split = "train" if train else "val"

    dataset = dl.DLataset.from_rlds(
        builder, split=split, shuffle=shuffle, num_parallel_reads=num_parallel_reads
    )

    if not isinstance(image_obs_keys, Sequence):
        image_obs_keys = [image_obs_keys]
    if not isinstance(depth_obs_keys, Sequence):
        depth_obs_keys = [depth_obs_keys]
    if not isinstance(state_obs_keys, Sequence):
        state_obs_keys = [state_obs_keys]

    def restructure(traj):
        standard_keys = {
            "observation",
            "action",
            "is_terminal",
            "is_last",
        }

        # apply a standardization function, if provided
        if standardize_fn is not None:
            traj = standardize_fn(traj)

        if not all(k in traj for k in standard_keys):
            raise ValueError(
                f"Trajectory is missing keys: {standard_keys - set(traj.keys())}. "
                "Did you write a `standardize_fn`?"
            )

        # filter out keys that are not needed
        allowed_keys = standard_keys | {"language_instruction"}
        traj = {k: v for k, v in traj.items() if k in allowed_keys}

        # extracts images, depth images and proprio from the "observation" dict
        traj_len = tf.shape(traj["action"])[0]
        old_obs = traj["observation"]
        new_obs = {}
        for i, key in enumerate(image_obs_keys):
            if key is None:
                new_obs[f"image_{i}"] = tf.repeat("", traj_len)  # padding
            else:
                new_obs[f"image_{i}"] = old_obs[key]

        for i, key in enumerate(depth_obs_keys):
            if key is None:
                new_obs[f"depth_{i}"] = tf.repeat("", traj_len)  # padding
            else:
                new_obs[f"depth_{i}"] = old_obs[key]

        if state_obs_keys:
            new_obs["proprio"] = tf.concat(
                [
                    tf.zeros((traj_len, 1), dtype=tf.float32)  # padding
                    if key is None
                    else tf.cast(old_obs[key], tf.float32)
                    for key in state_obs_keys
                ],
                axis=1,
            )

        # add timestep info
        new_obs["timestep"] = tf.range(traj_len) + 1

        traj["action"] = tf.cast(traj["action"], tf.float32)
        traj["observation"] = new_obs

        # add name of dataset
        traj["dataset_name"] = tf.repeat(name, traj_len)

        return traj

    # load or compute dataset statistics
    if isinstance(dataset_statistics, str):
        with tf.io.gfile.GFile(dataset_statistics, "r") as f:
            dataset_statistics = json.load(f)
    elif dataset_statistics is None:
        # tries to load from cache, otherwise computes on the fly
        dataset_statistics = get_dataset_statistics(
            builder,
            restructure,
            hash_dependencies=(
                str(state_obs_keys),
                inspect.getsource(standardize_fn) if standardize_fn is not None else "",
            ),
        )
    dataset_statistics = tree_map(np.array, dataset_statistics)

    dataset = dataset.traj_map(restructure, num_parallel_calls)
    dataset = dataset.traj_map(
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
) -> dl.DLataset:
    """Creates a single dataset from kwargs. Returns a dataset of trajectories.

    Args:
        dataset_kwargs: kwargs passed to `make_dataset_from_rlds` that are dataset-specific.
        traj_transform_kwargs: kwargs passed to 'apply_trajectory_transforms'.
        frame_transform_kwargs: kwargs passed to 'get_frame_transforms'.
        train: whether this is a training or validation dataset.
    """
    dataset, dataset_statistics = make_dataset_from_rlds(
        **dataset_kwargs,
        standardize_fn=RLDS_STANDARDIZATION_TRANSFORMS.get(
            dataset_kwargs["name"], None
        ),
        train=train,
    )
    dataset = apply_trajectory_transforms(dataset, **traj_transform_kwargs, train=train)
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)

    # this seems to reduce memory usage without affecting speed
    dataset = dataset.with_ram_budget(1)

    # save for later
    dataset.dataset_statistics = dataset_statistics
    return dataset


def make_interleaved_dataset(
    *,
    dataset_kwargs_list: Sequence[dict],
    traj_transform_kwargs: dict,
    frame_transform_kwargs: dict,
    train: bool,
    sample_weights: Optional[List[float]],
    balance_weights: bool,
    shuffle_buffer_size: int,
    batch_size: int,
    traj_transform_threads: Optional[int],
    traj_read_threads: Optional[int],
) -> dl.DLataset:
    """Creates an interleaved dataset from list of dataset kwargs. Returns a dataset of batched frames.

    Args:
        dataset_kwargs_list: list of kwargs, each element of which is passed to `make_dataset_from_rlds`.
            "num_parallel_calls" and "num_parallel_reads" are ignored.
        traj_transform_kwargs: kwargs passed to 'apply_trajectory_transforms'. "num_parallel_calls" is ignored.
        frame_transform_kwargs: kwargs passed to 'get_frame_transforms'.
        train: whether this is a training or validation dataset.
        sample_weights: sampling weights for each dataset in list. If None, defaults to uniform.
        balance_weights: if True, the sample weights are multiplied by the number of frames in each dataset.
            This makes it so that, if all the sample weights are equal, one full iteration through the interleaved
            dataset will correspond to one full iteration through each individual dataset (only in expectation,
            since in practice the sampling is random).
        shuffle_buffer_size: size of the dataset shuffle buffer (in number of frames).
        batch_size: batch size.
        traj_transform_threads: total number of parallel calls for trajectory transforms, distributed across
            datasets according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
        traj_read_threads: total number of parallel read workers for trajectory transforms, distributed across
            datasets according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
    """
    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)
    assert len(sample_weights) == len(dataset_kwargs_list)

    # go through datasets once to get sizes
    dataset_sizes = []
    all_dataset_statistics = []
    for dataset_kwargs in dataset_kwargs_list:
        _, dataset_statistics = make_dataset_from_rlds(
            **dataset_kwargs,
            standardize_fn=RLDS_STANDARDIZATION_TRANSFORMS.get(
                dataset_kwargs["name"], None
            ),
            train=train,
        )
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
    for dataset_kwargs, dataset_statistics, threads, reads in zip(
        dataset_kwargs_list,
        all_dataset_statistics,
        threads_per_dataset,
        reads_per_dataset,
    ):
        dataset, _ = make_dataset_from_rlds(
            **dataset_kwargs,
            standardize_fn=RLDS_STANDARDIZATION_TRANSFORMS.get(
                dataset_kwargs["name"], None
            ),
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
        datasets.append(dataset)

    # interleave at the transition level and then shuffle
    dataset: dl.DLataset = dl.DLataset.sample_from_datasets(
        datasets, sample_weights
    ).shuffle(shuffle_buffer_size)

    # apply frame transforms
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)

    # sequential batch (parallel batch seems to use much more memory)
    dataset = dataset.batch(batch_size)

    # this seems to reduce memory usage without affecting speed
    dataset = dataset.with_ram_budget(1)

    # save for later
    dataset.dataset_statistics = all_dataset_statistics
    return dataset
