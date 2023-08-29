from functools import partial
from typing import Optional

import dlimp as dl
import tensorflow as tf
import tensorflow_datasets as tfds

from orca.data.bridge import bridge_dataset as bridge
from orca.data.rlds import rlds_dataset as rlds
from orca.data.rlds.rlds_dataset_transforms import RLDS_TRAJECTORY_MAP_TRANSFORMS
from orca.data.utils import bc_goal_relabeling


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


def _clip_image_preprocess(image):
    # this should be exactly the same as HF's CLIPProcessor
    image = tf.image.resize(image, (224, 224), method="bicubic")
    image = image / 255.0
    image = (image - [0.48145466, 0.4578275, 0.40821073]) / [
        0.26862954,
        0.26130258,
        0.27577711,
    ]
    return image


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
    use_clip_image_preprocessing: bool = False,
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
            proprio, or both. Can be "normal" (mean 0, std 1) or "bounds" (normalized to [0, 1]).
        use_clip_image_preprocessing (bool, optional): Whether to perform the CLIP image preprocessing step.
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

    if use_clip_image_preprocessing:
        dataset = dataset.map(
            partial(
                dl.transforms.selective_tree_map,
                match="image",
                map_fn=_clip_image_preprocess,
            )
        )

    # adds the "goals" key
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


def make_bridge_dataset(
    data_path: str,
    train: bool,
    relabel_actions: bool = True,
    **kwargs,
) -> tf.data.Dataset:
    """Creates a dataset from the BridgeData format.

    Args:
        data_path (str): The path to the data directory (must contain "train" and "val" subdirectories).
        train (bool): Whether to use the training or validation set.
        relabel_actions (bool, optional): Whether to relabel the actions using the reached proprio. Defaults to True.
        **kwargs: Additional keyword arguments to pass to `apply_common_transforms`.
    """
    dataset = dl.DLataset.from_tfrecords(
        f"{data_path}/{'train' if train else 'val'}"
    ).map(dl.transforms.unflatten_dict)

    def restructure(traj):
        traj["observations"] = traj.pop("obs")
        traj["observations"] = {
            "image": traj["observations"]["images0"],  # always take images0 for now
            "proprio": tf.cast(traj["observations"]["state"], tf.float32),
        }
        traj["language"] = traj.pop("lang")
        traj["actions"] = tf.cast(traj["actions"], tf.float32)
        traj["actions"] = tf.concat(
            [
                traj["actions"][:, :6],
                bridge.binarize_gripper_actions(traj["actions"][:, -1])[:, None],
            ],
            axis=1,
        )
        return traj

    dataset = dataset.map(restructure)

    # bridgedata has one more observation than action in each trajectory; we pad with a zero action during
    # preprocessing, so we must discard the last timestep. if relabeling, this happens in relabel_actions (since
    # relabeling uses the last observation); otherwise, we do it manually.
    if relabel_actions:
        dataset = dataset.map(bridge.relabel_actions)
    else:
        dataset = dataset.map(lambda x: tf.nest.map_structure(lambda y: y[:-1], x))

    dataset = apply_common_transforms(dataset, train=train, **kwargs)

    return dataset


def make_rlds_dataset(
    name: str,
    data_dir: str,
    train: bool,
    image_obs_key: str = "image",
    state_obs_key: str = "state",
    **kwargs,
) -> tf.data.Dataset:
    """Creates a dataset from the RLDS format.

    Args:
        name (str): The name of the RLDS dataset (usually "name" or "name:version").
        data_dir (str): The path to the data directory.
        train (bool): Whether to use the training or validation set.
        image_obs_key (str, optional): The key to use for the image observation. Defaults to "image".
        state_obs_key (str, optional): The key to use for the state observation. Defaults to "state".
        **kwargs: Additional keyword arguments to pass to `apply_common_transforms`.
    """
    builder = tfds.builder(name, data_dir=data_dir)
    if "val" not in builder.info.splits:
        split = "train[:95%]" if train else "train[95%:]"
    else:
        split = "train" if train else "val"

    dataset = dl.DLataset.from_rlds(builder, split=split)

    def restructure(traj):
        # apply any dataset-specific transforms
        if name in RLDS_TRAJECTORY_MAP_TRANSFORMS:
            traj = RLDS_TRAJECTORY_MAP_TRANSFORMS[name](traj)

        # restructure RLDS dataset to match BridgeData format. extracts only 2 keys from the "observation" sub-dict: one
        # image (based on image_obs_key) and some sort of proprio (based on state_obs_key)
        traj["observations"] = {
            "image": traj["observation"][image_obs_key],
            "proprio": tf.cast(traj["observation"][state_obs_key], tf.float32),
        }
        del traj["observation"]
        traj["language"] = traj.pop("language_instruction")
        traj["actions"] = tf.cast(traj.pop("action"), tf.float32)
        traj["terminals"] = traj.pop("is_terminal")
        traj["truncates"] = tf.math.logical_and(
            traj.pop("is_last"), tf.math.logical_not(traj["terminals"])
        )

        return traj

    dataset = dataset.map(restructure)

    action_proprio_metadata = rlds.get_action_proprio_stats(builder, dataset)

    dataset = apply_common_transforms(
        dataset, train=train, action_proprio_metadata=action_proprio_metadata, **kwargs
    )

    return dataset
