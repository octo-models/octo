import hashlib
import json
import logging
from functools import partial
from typing import Dict, List, Optional, Sequence, Union

import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
from tensorflow_datasets.core.dataset_builder import DatasetBuilder

from orca.data.dataset_transforms import RLDS_TRAJECTORY_MAP_TRANSFORMS
from orca.data.utils import bc_goal_relabeling


def get_action_proprio_stats(
    builder: DatasetBuilder, dataset: tf.data.Dataset
) -> Dict[str, Dict[str, List[float]]]:
    # get statistics file path --> embed unique hash that catches if dataset info changed
    data_info_hash = hashlib.sha256(str(builder.info).encode("utf-8")).hexdigest()
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
            actions.append(episode["actions"].numpy())
            proprios.append(episode["observations"]["proprio"].numpy())
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
        "action": "actions",
        "proprio": "observations/proprio",
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


def _chunk_act_obs(traj, act_horizon, obs_horizon):
    """Chunks actions and observations into the given horizons.

    The "actions" and "observations" keys are each given a new axis (at index 1) of size `act_horizon` and
    `obs_horizon`, respectively. The actions are chunked into the future while the observations are chunked into the
    past.
    """
    traj_len = tf.shape(traj["actions"])[0]
    if act_horizon is not None:
        chunk_indices = tf.broadcast_to(
            tf.range(act_horizon), [traj_len, act_horizon]
        ) + tf.broadcast_to(tf.range(traj_len)[:, None], [traj_len, act_horizon])
        # pads by repeating the last action
        chunk_indices = tf.minimum(chunk_indices, traj_len - 1)
        traj["actions"] = tf.gather(traj["actions"], chunk_indices)
    if obs_horizon is not None:
        chunk_indices = tf.broadcast_to(
            tf.range(-obs_horizon + 1, 1), [traj_len, obs_horizon]
        ) + tf.broadcast_to(tf.range(traj_len)[:, None], [traj_len, obs_horizon])
        # pads by repeating the first observation
        chunk_indices = tf.maximum(chunk_indices, 0)
        traj["observations"] = tf.nest.map_structure(
            lambda x: tf.gather(x, chunk_indices), traj["observations"]
        )
    return traj


def apply_common_transforms(
    dataset: tf.data.Dataset,
    *,
    train: bool,
    goal_relabeling_strategy: Optional[str] = None,
    goal_relabeling_kwargs: dict = {},
    augment_kwargs: dict = {},
    act_horizon: Optional[int] = None,
    obs_horizon: Optional[int] = None,
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
        augment_kwargs (dict, optional): Keyword arguments to pass to the augmentation function. See
            `dlimp.augmentations.augment_image` for documentation.
        act_horizon (Optional[int], optional): The future horizon to chunk actions, or None for no chunking.
        obs_horizon (Optional[int], optional): The past horizon to chunk observations, or None for no chunking.
        skip_unlabeled (bool, optional): Whether to skip trajectories with no language labels.
        action_proprio_metadata (Optional[dict], optional): A dictionary containing metadata about the action and
            proprio statistics. If None, no normalization is performed.
        action_proprio_normalization_type (Optional[str], optional): The type of normalization to perform on the action,
            proprio, or both. Can be "normal" (mean 0, std 1) or "bounds" (normalized to [-1, 1]).
    """
    if skip_unlabeled:
        dataset = dataset.filter(lambda x: tf.math.reduce_any(x["language"] != ""))

    if action_proprio_metadata is not None:
        dataset = dataset.map(
            partial(
                _normalize_action_and_proprio,
                metadata=action_proprio_metadata,
                normalization_type=action_proprio_normalization_type,
            )
        )

    # decodes string keys with name "image"
    dataset = dataset.frame_map(dl.transforms.decode_images)

    if train:
        # augments the entire trajectory with the same seed
        dataset = dataset.frame_map(
            partial(
                dl.transforms.augment,
                augment_kwargs=augment_kwargs,
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

    # possibly chunks actions and observations
    dataset = dataset.map(
        partial(_chunk_act_obs, act_horizon=act_horizon, obs_horizon=obs_horizon)
    )

    return dataset


def make_dataset(
    name: str,
    data_dir: str,
    train: bool,
    image_obs_keys: Union[str, List[str]] = [],
    depth_obs_keys: Union[str, List[str]] = [],
    state_obs_keys: Union[str, List[str]] = [],
    **kwargs,
) -> tf.data.Dataset:
    """Creates a dataset from the RLDS format.

    Args:
        name (str): The name of the RLDS dataset (usually "name" or "name:version").
        data_dir (str): The path to the data directory.
        train (bool): Whether to use the training or validation set.
        image_obs_keys (str, List[str], optional): List of image observation keys to be decoded. Mapped to "image_XXX".
        depth_obs_keys (str, List[str], optional): List of depth observation keys to be decoded. Mapped to "depth_XXX".
        state_obs_keys (str, List[str], optional): List of low-dim observation keys to be decoded.
          Get concatenated and mapped to "proprio".
        **kwargs: Additional keyword arguments to pass to `apply_common_transforms`.
    """
    builder = tfds.builder(name, data_dir=data_dir)
    if "val" not in builder.info.splits:
        split = "train[:95%]" if train else "train[95%:]"
    else:
        split = "train" if train else "val"

    dataset = dl.DLataset.from_rlds(builder, split=split)

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

        # extracts RGB images, depth images and proprio based on provided keys
        # TODO(karl): avoid two naming structures here --> always use RLDS default keys
        traj["observations"] = {}
        for i, key in enumerate(image_obs_keys):
            traj["observations"][f"image_{i}"] = traj["observation"][key]
        for i, key in enumerate(depth_obs_keys):
            traj["observations"][f"depth_{i}"] = traj["observation"][key]
        if state_obs_keys:
            traj["observations"]["proprio"] = tf.concat(
                [
                    tf.cast(traj["observation"][key], tf.float32)
                    for key in state_obs_keys
                ],
                axis=-1,
            )

        traj["language"] = traj.pop("language_instruction")
        traj["actions"] = tf.cast(traj.pop("action"), tf.float32)
        traj["terminals"] = traj.pop("is_terminal")
        traj["truncates"] = tf.math.logical_and(
            traj.pop("is_last"), tf.math.logical_not(traj["terminals"])
        )
        del traj["observation"]
        return traj

    dataset = dataset.map(restructure)

    action_proprio_metadata = get_action_proprio_stats(builder, dataset)

    dataset = apply_common_transforms(
        dataset, train=train, action_proprio_metadata=action_proprio_metadata, **kwargs
    )

    return dataset
