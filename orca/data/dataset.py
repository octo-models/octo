from functools import partial
import inspect
import json
from typing import Callable, List, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from orca.data import obs_transforms, traj_transforms
from orca.data.utils import goal_relabeling, task_augmentation
from orca.data.utils.data_utils import (
    allocate_threads,
    get_dataset_statistics,
    NormalizationType,
    normalize_action_and_proprio,
    pprint_data_mixture,
    tree_map,
)


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
            no goal relabeling. See `goal_relabeling.py`.
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
    dataset = dataset.traj_map(traj_transforms.add_pad_mask_dict, num_parallel_calls)

    # adds the "task" key
    if goal_relabeling_strategy is not None:
        dataset = dataset.traj_map(
            partial(
                getattr(goal_relabeling, goal_relabeling_strategy),
                **goal_relabeling_kwargs,
            ),
            num_parallel_calls,
        )

    if "language_instruction" in dataset.element_spec:

        def process_language_instruction(traj):
            # move the "language_instruction" key into the "task" dict
            if "task" not in traj:
                traj["task"] = {}
            traj["task"]["language_instruction"] = traj.pop("language_instruction")
            # mark whether the language instruction is padding
            traj["task"]["pad_mask_dict"]["language_instruction"] = (
                tf.strings.length(traj["task"]["language_instruction"]) != 0
            )
            return traj

        dataset = dataset.traj_map(process_language_instruction, num_parallel_calls)

    dataset = dataset.traj_map(
        partial(
            traj_transforms.chunk_act_obs,
            window_size=window_size,
            additional_action_window_size=additional_action_window_size,
        ),
        num_parallel_calls,
    )

    if train and subsample_length is not None:
        dataset = dataset.traj_map(
            partial(traj_transforms.subsample, subsample_length=subsample_length),
            num_parallel_calls,
        )

    return dataset


def apply_frame_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    image_augment_kwargs: Union[dict, Mapping[str, dict]] = {},
    resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]] = {},
    depth_resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]] = {},
    task_augment_strategy: Optional[str] = None,
    task_augment_kwargs: dict = {},
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """Applies common transforms that happen at a frame level. These transforms are usually more
    CPU-intensive, (e.g. decoding or resizing images).

    Args:
        train (bool): Whether the dataset is for training (affects image augmentation).
        dataset (dl.DLataset): The dataset to transform.
        image_augment_kwargs (dict|Mapping[str, dict]): Keyword arguments to pass to the image augmentation
            function. See `dlimp.transforms.augment_image` for documentation of these kwargs. If a dict of
            dicts is provided, then key "k" will be used for "image_{k}" (names determined by `image_obs_keys`
            in `make_dataset_from_rlds`). Augmentation will be skipped for missing keys (so pass an empty dict
            to skip augmentation for all images).
        resize_size (Tuple[int, int]|Mapping[str, Tuple[int, int]]): If provided, images will be resized to
            this size. If a dict of tuples is provided, then key "k" will be used for "image_{k}" (names
            determined by `image_obs_keys` in `make_dataset_from_rlds`). Resizing will be skipped for missing
            keys (so pass an empty dict to skip resizing for all images).
        depth_resize_size (Tuple[int, int]|Mapping[str, Tuple[int, int]]): Same as resize_size, but for depth
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
        if "task" in frame:
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
        partial(apply_obs_transform, obs_transforms.decode_images), num_parallel_calls
    )

    # resize images (and depth images)
    dataset = dataset.frame_map(
        partial(
            apply_obs_transform,
            partial(
                obs_transforms.resize,
                resize_size=resize_size,
                depth_resize_size=depth_resize_size,
            ),
        ),
        num_parallel_calls,
    )

    if train:
        # augment all images with the same seed, skipping padding images
        def aug(frame):
            seed = tf.random.uniform([2], maxval=tf.dtypes.int32.max, dtype=tf.int32)
            aug_fn = partial(
                obs_transforms.augment, seed=seed, augment_kwargs=image_augment_kwargs
            )
            return apply_obs_transform(aug_fn, frame)

        dataset = dataset.frame_map(aug, num_parallel_calls)

    return dataset


def make_dataset_from_rlds(
    name: str,
    data_dir: str,
    train: bool,
    standardize_fn: Optional[Callable[[dict], dict]] = None,
    shuffle: bool = True,
    image_obs_keys: Mapping[str, Optional[str]] = {},
    depth_obs_keys: Mapping[str, Optional[str]] = {},
    state_obs_keys: Sequence[Optional[str]] = (),
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
    dataset_statistics: Optional[Union[dict, str]] = None,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> Tuple[dl.DLataset, dict]:
    """This function is responsible for loading a specific RLDS dataset from storage and getting it into a
    standardized format. Yields a dataset of trajectories. Does not include CPU-intensive operations.

    If `standardize_fn` is provided, it will be applied to each trajectory. This function should get the
    trajectory into a standard format, which includes the keys "observation" and "action". "observation"
    should be a dictionary containing some number of additional keys, which will be extracted into an even
    more standardized format according to the "*_obs_keys" arguments.

    The `image_obs_keys` and `depth_obs_keys` arguments are mappings from new names to old names, or None in
    place of an old name to insert padding. For example, if after `standardize_fn`, your "observation" dict
    has RGB images called "workspace" and "wrist", and `image_obs_keys={"primary": "workspace", "secondary":
    None, "wrist": "wrist"}`, then the resulting dataset will have an "observation" dict containing the keys
    "image_primary", "image_secondary", and "image_wrist", where "image_primary" corresponds to "workspace",
    "image_secondary" is a padding image, and "image_wrist" corresponds to "wrist".

    `state_obs_keys` is a list of 1-dimensional proprioceptive keys to concatenate into a single array, which
    will be placed in the "proprio" key of the "observation" dict. A single padding element (zero) will be
    inserted for each None entry.

    Args:
        name (str): The name of the RLDS dataset (usually "name" or "name:version").
        data_dir (str): The path to the data directory.
        train (bool): Whether to use the training or validation set.
        shuffle (bool, optional): Whether to shuffle the file read order (does NOT fully shuffle directories,
            since one file usually contains many trajectories!).
        image_obs_keys (Mapping[str, str|None]): Mapping from {new: old} indicating which RGB images to
            extract from the "observation" dict. `new_obs = {f"image_{new}": old_obs[old] for new, old in
            image_obs_keys.items()}`. If a value of `old` is None, inserts a padding image instead (empty
            string).
        depth_obs_keys (Mapping[str, str|None]): Same as `image_obs_keys`, but for depth images. Keys will be
            prefixed with "depth_" instead of "image_".
        state_obs_keys (Sequence[str|None]): List of 1-dimensional proprioception keys to be extracted from
            the "observation" dict, concatenated, and mapped to "proprio". Inserts 1 element of padding (zero) for
            each None entry.
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
            - image_{name1, name2, ...} # RGB image observations
            - depth_{name1, name2, ...} # depth image observations
            - proprio                   # 1-dimensional array of proprioceptive observations
        - action                        # action vector
        - language_instruction          # string language instruction (optional)
    """
    builder = tfds.builder(name, data_dir=data_dir)
    if "val" not in builder.info.splits:
        split = "train[:95%]" if train else "train[95%:]"
    else:
        split = "train" if train else "val"

    dataset = dl.DLataset.from_rlds(
        builder, split=split, shuffle=shuffle, num_parallel_reads=num_parallel_reads
    )

    def restructure(traj):
        standard_keys = {
            "observation",
            "action",
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
        for new, old in image_obs_keys.items():
            if old is None:
                new_obs[f"image_{new}"] = tf.repeat("", traj_len)  # padding
            else:
                new_obs[f"image_{new}"] = old_obs[old]

        for new, old in depth_obs_keys.items():
            if old is None:
                new_obs[f"depth_{new}"] = tf.repeat("", traj_len)  # padding
            else:
                new_obs[f"depth_{new}"] = old_obs[old]

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
        new_obs["timestep"] = tf.range(traj_len)

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
