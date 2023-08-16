"""
tf.data.Dataset based dataloader for the BridgeData format, meaning
TFRecords with one trajectory per example. See the BridgeDataset class
below for more details.

Written by Kevin Black (kvablack@berkeley.edu).
"""

import fnmatch
from typing import Iterable, Iterator, List, Optional, Union

import jax
import numpy as np
import tensorflow as tf
from absl import logging
from flax.core import FrozenDict
from orca.data.text_processing import TextProcessor
from orca.data.tf_augmentations import augment
from orca.data.tf_goal_relabeling import GOAL_RELABELING_FUNCTIONS


def glob_to_path_list(
    glob_strs: Union[str, List[str]], prefix: str = "", exclude: Iterable[str] = ()
):
    """Converts a glob string or list of glob strings to a list of paths."""
    if isinstance(glob_strs, str):
        glob_strs = [glob_strs]
    path_list = []
    for glob_str in glob_strs:
        paths = tf.io.gfile.glob(f"{prefix}/{glob_str}")
        filtered_paths = []
        for path in paths:
            if not any(fnmatch.fnmatch(path, e) for e in exclude):
                filtered_paths.append(path)
            else:
                logging.info(f"Excluding {path}")
        assert len(filtered_paths) > 0, f"{glob_str} came up empty"
        path_list += filtered_paths
    return path_list


@tf.function(jit_compile=True)
def _binarize_gripper_actions(actions):
    """Converts gripper actions from continous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near
    1.0) or fully closed (near 0.0). As it transitions between the two, it
    sometimes passes through a few intermediate values. We relabel those
    intermediate values based on the state that is reached _after_ those
    intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we
    give up on binarizing and relabel that chunk of intermediate values as
    the last action in the trajectory.

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


class BridgeDataset:
    """
    Fast parallel tf.data.Dataset-based dataloader for a dataset in the
    BridgeData format. This format consists of TFRecords where each example
    is one trajectory. See `PROTO_TYPE_SPEC` below for the expected format
    for each example in more detail. See `_process_trajectory` below for
    the output format.

    Includes goal relabeling, image augmentations, and sampling from multiple
    datasets with different weights. Goal relabeling uses a 0/-1 reward scheme:
    0 when the next_obs is labeled as the goal, -1 otherwise.

    Args:
        data_paths: List of paths to the data files. If a list of list of paths
            is provided, the data will be sampled from each sub-list according
            to "sample_weights".
        seed: Random seed.
        action_proprio_metadata: Dictionary containing metadata of the actions and proprio.
            If provided, actions and proprio will be normalized.
        normalization_type: The type of normalization to apply to the actions
            and proprio.
        relabel_actions: Whether to relabel the actions with reached states
            (based on proprioception). Also binarizes gripper actions.
        goal_relabeling_strategy: Goal relabeling strategy. See
            `orca.data.tf_goal_relabeling` for more details.
        goal_relabeling_kwargs: Keyword arguments for goal relabeling. See
            `orca.data.tf_goal_relabeling` for more details.
        sample_weights: If data_paths is a list of list of paths, this is a
            list of weights with which to sample from each sub-list.
        batch_size: Batch size.
        shuffle_buffer_size: Size of the shuffle buffer. It is split between
            sub-datasets by `sample_weights`.
        prefetch_num_batches: Number of batches to prefetch.
        cache: Whether to cache the dataset in memory.
        train: Whether this dataset is intended for training
            (if set to `False`, will disable shuffling and augmentations).
        augment: Whether to apply image augmentations.
        augment_next_obs_goal_differently: Whether to use different random seeds
            for augmenting the obs, next_obs, and goal image.
        augment_kwargs: Keyword arguments for image augmentations. See
            `orca.data.tf_augmentations.augment` for more details.
        act_pred_horizon: Number of consecutive actions that will be predicted.
        obs_horizon: Number of consecutive observations that will be conditioned on.
    """

    def __init__(
        self,
        data_paths: List[Union[str, List[str]]],
        seed: int,
        action_proprio_metadata: Optional[dict] = None,
        normalization_type: Optional[str] = "normal",
        relabel_actions: bool = True,
        goal_relabeling_strategy: str = "uniform",
        goal_relabeling_kwargs: dict = {},
        sample_weights: Optional[List[float]] = None,
        batch_size: int = 256,
        shuffle_buffer_size: int = 10000,
        prefetch_num_batches: int = 5,
        cache: bool = False,
        train: bool = True,
        augment: bool = False,
        augment_next_obs_goal_differently: bool = False,
        augment_kwargs: dict = {},
        act_pred_horizon: Optional[int] = None,
        obs_horizon: Optional[int] = None,
        string_fields: Optional[str] = ["language"],
        text_processor: Optional[TextProcessor] = None,
        image_processor: Optional[str] = "default",
        image_shape: Optional[List[int]] = (256, 256, 3),
        skip_unlabeled: bool = False,
        **kwargs,
    ):
        logging.warning("Extra kwargs passed to BridgeDataset: %s", kwargs)
        if isinstance(data_paths[0], str):
            data_paths = [data_paths]
        if sample_weights is None:
            # default to uniform distribution over sub-lists
            sample_weights = [1 / len(data_paths)] * len(data_paths)
        assert len(data_paths) == len(sample_weights)
        assert np.isclose(sum(sample_weights), 1.0)

        self.relabel_actions = relabel_actions
        self.action_proprio_metadata = action_proprio_metadata
        self.normalization_type = normalization_type
        self.goal_relabeling_strategy = goal_relabeling_strategy
        self.goal_relabeling_kwargs = goal_relabeling_kwargs
        self.cache = cache
        self.augment_kwargs = augment_kwargs
        self.augment_next_obs_goal_differently = augment_next_obs_goal_differently
        self.act_pred_horizon = act_pred_horizon
        self.obs_horizon = obs_horizon
        self.is_train = train
        self.text_processor = text_processor
        self.string_fields = string_fields
        self.load_language = self.text_processor is not None
        self.image_processor = image_processor
        self.image_shape = image_shape

        if self.load_language:
            self.PROTO_TYPE_SPEC["language"] = tf.string

        # construct a dataset for each sub-list of paths
        datasets = []
        for sub_data_paths in data_paths:
            datasets.append(self._construct_tf_dataset(sub_data_paths, seed))

        if train:
            # shuffle and repeat each sub-dataset, allocating the shuffle buffer
            # by sample_weights
            for i in range(len(datasets)):
                datasets[i] = (
                    datasets[i]
                    .shuffle(int(shuffle_buffer_size * sample_weights[i]), seed + i)
                    .repeat()
                )

        # for validation, we want to be able to iterate through the entire dataset;
        # for training, we want to make sure that no sub-dataset is ever exhausted
        # or the sampling ratios will be off. this should never happen because of the
        # repeat() above, but `stop_on_empty_dataset` is a safeguard
        dataset = tf.data.Dataset.sample_from_datasets(
            datasets, sample_weights, seed=seed, stop_on_empty_dataset=train
        )

        if skip_unlabeled:
            dataset = dataset.filter(
                lambda x: tf.math.reduce_any(x["goals"]["language"] != "")
            )

        if train and augment:
            # apply augmentations, using a sequence of integers as seeds.
            # this was the only way I found to avoid a memory leak in tf.random.Generator
            dataset = dataset.enumerate(start=seed)
            dataset = dataset.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(
            batch_size,
            num_parallel_calls=tf.data.AUTOTUNE,
            drop_remainder=True,
            deterministic=not train,
        )

        dataset = dataset.map(
            self._process_image_fields, num_parallel_calls=tf.data.AUTOTUNE
        )

        # always prefetch at the end of the pipeline
        dataset = dataset.prefetch(prefetch_num_batches)

        self.tf_dataset = dataset

    def _construct_tf_dataset(self, paths: List[str], seed: int) -> tf.data.Dataset:
        # construct base tf dataset of trajectories
        dataset = self._construct_base_dataset(paths, seed)

        # yields trajectories
        dataset = dataset.map(
            self._process_actions, num_parallel_calls=tf.data.AUTOTUNE
        )

        # yields trajectories
        dataset = dataset.map(self._chunk_act_obs, num_parallel_calls=tf.data.AUTOTUNE)

        # cache before add_goals because add_goals introduces randomness
        if self.cache:
            dataset = dataset.cache()

        # yields trajectories
        dataset = dataset.map(self._add_goals, num_parallel_calls=tf.data.AUTOTUNE)

        # unbatch to yield individual transitions
        dataset = dataset.unbatch()

        return dataset

    def _construct_base_dataset(self, paths: List[str], seed: int) -> tf.data.Dataset:
        """
        Constructs a tf.data.Dataset from a list of paths.
        The dataset yields a dictionary of tensors for each trajectory.
        """
        # shuffle again using the dataset API so the files are read in a
        # different order every epoch
        dataset = tf.data.Dataset.from_tensor_slices(paths).shuffle(len(paths), seed)

        # yields raw serialized examples
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=tf.data.AUTOTUNE)

        # yields trajectories
        dataset = dataset.map(self._decode_example, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    # the expected type spec for the serialized examples
    PROTO_TYPE_SPEC = {
        "observations/images0": tf.uint8,
        "observations/state": tf.float32,
        "next_observations/images0": tf.uint8,
        "next_observations/state": tf.float32,
        "actions": tf.float32,
        "terminals": tf.bool,
        "truncates": tf.bool,
    }

    def _decode_example(self, example_proto):
        # decode the example proto according to PROTO_TYPE_SPEC
        features = {
            key: tf.io.FixedLenFeature([], tf.string)
            for key in self.PROTO_TYPE_SPEC.keys()
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        parsed_tensors = {
            key: tf.io.parse_tensor(parsed_features[key], dtype)
            for key, dtype in self.PROTO_TYPE_SPEC.items()
        }
        # restructure the dictionary into the downstream format
        return {
            "observations": {
                "image": parsed_tensors["observations/images0"],
                "proprio": parsed_tensors["observations/state"],
            },
            "next_observations": {
                "image": parsed_tensors["next_observations/images0"],
                "proprio": parsed_tensors["next_observations/state"],
            },
            **({"language": parsed_tensors["language"]} if self.load_language else {}),
            "actions": parsed_tensors["actions"],
            "terminals": parsed_tensors["terminals"],
            "truncates": parsed_tensors["truncates"],
        }

    def _process_actions(self, traj):
        if self.relabel_actions:
            # relabel the first 6 action dims (xyz position, xyz rotation)
            # using the reached proprio
            movement_actions = (
                traj["next_observations"]["proprio"][:, :6]
                - traj["observations"]["proprio"][:, :6]
            )
            # binarize the gripper action
            continuous_gripper_actions = traj["actions"][:, 6]
            binarized_gripper_actions = _binarize_gripper_actions(
                continuous_gripper_actions
            )

            traj["actions"] = tf.concat(
                [movement_actions, binarized_gripper_actions[:, None]],
                axis=1,
            )

        # normalize actions and proprio
        if self.action_proprio_metadata is not None:
            if self.normalization_type == "normal":
                # normalize to mean 0, std 1
                traj["actions"] = (
                    traj["actions"] - self.action_proprio_metadata["action"]["mean"]
                ) / self.action_proprio_metadata["action"]["std"]
                for key in ["observations", "next_observations"]:
                    traj[key]["proprio"] = (
                        traj[key]["proprio"]
                        - self.action_proprio_metadata["proprio"]["mean"]
                    ) / self.action_proprio_metadata["proprio"]["std"]
            elif self.normalization_type == "bounds":
                # normalize to [0, 1]
                traj["actions"] = (
                    traj["actions"] - self.action_proprio_metadata["action"]["min"]
                ) / (
                    self.action_proprio_metadata["action"]["max"]
                    - self.action_proprio_metadata["action"]["min"]
                )
                # clip to [0, 1]
                traj["actions"] = tf.clip_by_value(traj["actions"], 0, 1)
                for key in ["observations", "next_observations"]:
                    traj[key]["proprio"] = (
                        traj[key]["proprio"]
                        - self.action_proprio_metadata["proprio"]["min"]
                    ) / (
                        self.action_proprio_metadata["proprio"]["max"]
                        - self.action_proprio_metadata["proprio"]["min"]
                    )
                    traj[key]["proprio"] = tf.clip_by_value(traj[key]["proprio"], 0, 1)
            else:
                raise ValueError

        return traj

    def _chunk_act_obs(self, traj):
        traj_len = len(traj["actions"])
        if self.act_pred_horizon is not None:
            chunk_indices = tf.broadcast_to(
                tf.range(self.act_pred_horizon), [traj_len, self.act_pred_horizon]
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None], [traj_len, self.act_pred_horizon]
            )
            # pads by repeating the last action
            chunk_indices = tf.minimum(chunk_indices, traj_len - 1)
            traj["action_chunks"] = tf.gather(traj["actions"], chunk_indices)
        if self.obs_horizon is not None:
            chunk_indices = tf.broadcast_to(
                tf.range(-self.obs_horizon + 1, 1), [traj_len, self.obs_horizon]
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None], [traj_len, self.obs_horizon]
            )
            # pads by repeating the first observation
            chunk_indices = tf.maximum(chunk_indices, 0)
            traj["obs_chunks"] = tf.nest.map_structure(
                lambda x: tf.gather(x, chunk_indices), traj["observations"]
            )
            traj["next_obs_chunks"] = tf.nest.map_structure(
                lambda x: tf.gather(x, chunk_indices), traj["next_observations"]
            )
        return traj

    def _add_goals(self, traj):
        traj = GOAL_RELABELING_FUNCTIONS[self.goal_relabeling_strategy](
            traj, **self.goal_relabeling_kwargs
        )

        if self.load_language:
            lang_idx = tf.random.uniform(
                shape=[], maxval=len(traj["language"]), dtype=tf.int32
            )
            lang = traj["language"][lang_idx]
            traj["goals"]["language"] = tf.broadcast_to(
                lang, tf.shape(traj["terminals"])
            )
            traj.pop("language")

        # after goal relabeling, we can set actions and obs to chunked version
        if "action_chunks" in traj:
            traj["actions"] = traj.pop("action_chunks")
        if "obs_chunks" in traj:
            traj["observations"] = traj.pop("obs_chunks")
            traj["next_observations"] = traj.pop("next_obs_chunks")

        return traj

    def _augment(self, seed, image):
        if self.augment_next_obs_goal_differently:
            sub_seeds = tf.unstack(
                tf.random.stateless_uniform(
                    [3, 2], seed=[seed, seed], minval=None, maxval=None, dtype=tf.int32
                )
            )
        else:
            # use the same seed for obs, next_obs, and goal
            sub_seeds = [[seed, seed]] * 3

        for key, sub_seed in zip(
            ["observations", "next_observations", "goals"], sub_seeds
        ):
            image[key]["image"] = augment(
                image[key]["image"], sub_seed, **self.augment_kwargs
            )
        return image

    def _process_image(self, image):
        if self.image_processor == "default":
            pass
        elif self.image_processor == "clip":
            # this should be exactly the same as HF's CLIPProcessor
            # but it needs to be in tf graph or it's slow
            # for some reason we need to set this shape or it won't work
            image = tf.reshape(image, [-1, *self.image_shape])
            image.set_shape([None, *self.image_shape])
            image = tf.image.resize(image, (224, 224), method="bicubic")
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = image / 255.0
            image = (image - [0.48145466, 0.4578275, 0.40821073]) / [
                0.26862954,
                0.26130258,
                0.27577711,
            ]
            image = tf.reshape(image, [-1, self.obs_horizon, 224, 224, 3])
        return image

    def _process_image_fields(self, batch):
        for key in ["observations", "next_observations", "goals"]:
            batch[key]["image"] = self._process_image(batch[key]["image"])
        return batch

    def _process_strings(self, strings):
        strings = [s.decode("utf-8") for s in strings]
        if self.text_processor:
            return self.text_processor.encode(strings)
        else:
            return strings

    def _process_string_fields(self, batch):
        return jax.tree_util.tree_map_with_path(
            lambda kp, x: self._process_strings(x)
            if kp[-1].key in self.string_fields
            else x,
            batch,
        )

    def get_iterator(self) -> Iterator[FrozenDict]:
        # yield FrozenDicts. this can be bypassed by using
        # `dataset.tf_dataset.as_numpy_iterator()` instead
        iterator = map(FrozenDict, self.tf_dataset.as_numpy_iterator())
        iterator = map(self._process_string_fields, iterator)
        return iterator
