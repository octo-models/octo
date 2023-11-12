import copy
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union

import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm

from orca.data.dataset_transforms import RLDS_TRAJECTORY_MAP_TRANSFORMS
from orca.data.utils import bc_goal_relabeling, task_augmentation
from orca.data.utils.data_utils import (
    ActionEncoding,
    get_dataset_statistics,
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
    # indicates whether or not an entire observation is padding
    traj["observation"]["pad_mask"] = chunk_indices >= 0

    return traj


def _decode_images(frame):
    """Decodes images and depth images, marking empty strings as padding."""
    obs = frame["observation"]
    # indicates which keys in the observation dict are padding
    pad_mask_dict = {}
    for key in obs:
        if "image" in key:
            if obs[key].dtype == tf.string:
                if tf.strings.length(obs[key]) == 0:
                    # this is a padding image
                    obs[key] = tf.zeros((1, 1, 3), dtype=tf.uint8)
                    pad_mask_dict[key] = False
                else:
                    obs[key] = tf.io.decode_image(
                        obs[key], expand_animations=False, dtype=tf.uint8
                    )
                    pad_mask_dict[key] = True
            elif obs[key].dtype == tf.uint8:
                pad_mask_dict[key] = True
            else:
                raise ValueError(
                    f"Unsupported image dtype: found {key} with dtype {obs[key].dtype}"
                )
        elif "depth" in key:
            if obs[key].dtype == tf.string:
                if tf.strings.length(obs[key]) == 0:
                    # this is a padding image
                    obs[key] = tf.zeros((1, 1), dtype=tf.float32)
                    pad_mask_dict[key] = False
                else:
                    obs[key] = tf.io.decode_image(
                        obs[key], expand_animations=False, dtype=tf.float32
                    )[..., 0]
                    pad_mask_dict[key] = True
            elif obs[key].dtype == tf.float32:
                pad_mask_dict[key] = True
            else:
                raise ValueError(
                    f"Unsupported depth dtype: found {key} with dtype {obs[key].dtype}"
                )

    frame["observation"] = obs
    frame["observation"]["pad_mask_dict"] = pad_mask_dict
    return frame


def _augment(frame, augment_kwargs):
    """Augments images, skipping padding images. Augments all images in a trajectory identically."""
    obs = frame["observation"]
    seed = [frame["_traj_index"], frame["_traj_index"]]
    for key in obs:
        if "image" in key:
            if obs["pad_mask_dict"][key]:
                obs[key] = dl.transforms.augment_image(
                    obs[key], **augment_kwargs, seed=seed
                )
    frame["observation"] = obs
    return frame


def apply_common_transforms(
    dataset: dl.DLataset,
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
    max_action: Optional[float] = None,
    max_proprio: Optional[float] = None,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """Common transforms shared between all datasets. Takes and returns a dataset of trajectories. Includes
    the most CPU-intensive operations (image decoding, resizing, augmentation, and chunking).

    Args:
        dataset (dl.DLataset): The dataset to transform.
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

    dataset = dataset.frame_map(_decode_images, num_parallel_calls)

    if resize_size:
        dataset = dataset.frame_map(
            partial(dl.transforms.resize_images, size=resize_size),
            num_parallel_calls,
        )
        dataset = dataset.frame_map(
            partial(dl.transforms.resize_depth_images, size=resize_size),
            num_parallel_calls,
        )

    if train:
        # augments the entire trajectory with the same seed
        dataset = dataset.frame_map(
            partial(_augment, augment_kwargs=image_augment_kwargs), num_parallel_calls
        )

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

    if train and task_augmentation_strategy is not None:
        dataset = dataset.map(
            partial(
                getattr(task_augmentation, task_augmentation_strategy),
                **task_augmentation_kwargs,
            ),
            num_parallel_calls,
        )

    # chunks actions and observations
    dataset = dataset.map(
        partial(_chunk_act_obs, window_size=window_size), num_parallel_calls
    )

    return dataset


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
    ram_budget: Optional[int] = None,
    action_proprio_normalization_type: Optional[str] = None,
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
        ram_budget (int, optional): limits the RAM used by tf.data.AUTOTUNE, unit: GB, forwarded to AutotuneOptions.
        action_proprio_normalization_type (Optional[str], optional): The type of normalization to perform on the action,
            proprio, or both. Can be "normal" (mean 0, std 1) or "bounds" (normalized to [-1, 1]).
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

        # add timestep info
        traj["observation"]["timestep"] = tf.range(traj_len) + 1

        return traj

    dataset = dataset.map(restructure, num_parallel_calls)

    # tries to load from cache, otherwise computes on the fly
    dataset_statistics = get_dataset_statistics(
        builder, state_obs_keys, restructure, RLDS_TRAJECTORY_MAP_TRANSFORMS[name]
    )

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
    transform_kwargs: dict,
    train: bool,
) -> dl.DLataset:
    """Creates a single dataset from kwargs.

    Args:
        dataset_kwargs: kwargs passed to `make_dataset_from_rlds` that are dataset-specific.
        transform_kwargs: kwargs passed to 'apply_common_transforms'.
        train: whether this is a training or validation dataset.
    """
    dataset_kwargs = copy.deepcopy(dataset_kwargs)
    transform_kwargs = copy.deepcopy(transform_kwargs)

    # SPECIAL CASE: if `num_parallel_calls` is not in `transform_kwargs`, use
    # same value as in `dataset_kwargs`
    if "num_parallel_calls" not in transform_kwargs:
        transform_kwargs["num_parallel_calls"] = dataset_kwargs.get(
            "num_parallel_calls", tf.data.AUTOTUNE
        )

    dataset, dataset_statistics = make_dataset_from_rlds(**dataset_kwargs, train=train)
    dataset = apply_common_transforms(dataset, **transform_kwargs, train=train)

    # save for later
    dataset.dataset_statistics = dataset_statistics
    return dataset


def make_interleaved_dataset(
    common_dataset_kwargs: dict,
    dataset_kwargs_list: List[dict],
    transform_kwargs: dict,
    train: bool,
    sample_weights: Optional[List[float]] = None,
    balance_weights: bool = True,
    shuffle_buffer_size: int = 10000,
) -> dl.DLataset:
    """Creates an interleaved dataset from list of dataset kwargs.

    Args:
        common_dataset_kwargs: shared arguments for `make_dataset_from_rlds` that are common to all datasets.
            Will override kwargs from `dataset_kwargs_list`.
        dataset_kwargs_list: list of kwargs, each element is passed to `make_dataset_from_rlds` for a single dataset.
        transform_kwargs: kwargs passed to 'apply_common_transforms'.
        train: whether this is a training or validation dataset.
        sample_weights: sampling weights for each dataset in list. If None, defaults to uniform.
        balance_weights: whether to rebalance sampling weights by number of transitions in each dataset.
        shuffle_buffer_size: size of the dataset shuffle buffer for interleaved dataset.
    """
    common_dataset_kwargs = copy.deepcopy(common_dataset_kwargs)
    dataset_kwargs_list = copy.deepcopy(dataset_kwargs_list)
    transform_kwargs = copy.deepcopy(transform_kwargs)
    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)
    assert len(sample_weights) == len(dataset_kwargs_list)

    datasets = []
    dataset_sizes = []
    for dataset_kwargs in dataset_kwargs_list:
        dataset_kwargs.update(**common_dataset_kwargs)
        dataset, dataset_statistics = make_dataset_from_rlds(
            **dataset_kwargs, train=train
        )
        dataset_sizes.append(dataset_statistics["num_transitions"])
        datasets.append(dataset.repeat())

    if balance_weights:
        sample_weights = np.array(sample_weights) * np.array(dataset_sizes)
    sample_weights = np.array(sample_weights) / np.sum(sample_weights)
    pprint_data_mixture(dataset_kwargs_list, sample_weights)

    # interleave datasets at the trajectory level with sampling weights
    # (doing it this way saves memory compared to interleaving at the step level)
    dataset = dl.DLataset.sample_from_datasets(datasets, sample_weights)

    # SPECIAL CASE: if `num_parallel_calls` is not in `transform_kwargs`, use
    # same value as in `dataset_kwargs`
    if "num_parallel_calls" not in transform_kwargs:
        transform_kwargs["num_parallel_calls"] = dataset_kwargs.get(
            "num_parallel_calls", tf.data.AUTOTUNE
        )

    # apply common transforms like augmentation, chunking etc on interleaved episode dataset
    dataset = apply_common_transforms(
        dataset,
        **transform_kwargs,
        train=train,
    )

    dataset = dataset.flatten(
        num_parallel_calls=transform_kwargs["num_parallel_calls"]
    ).shuffle(shuffle_buffer_size)
    return dataset
