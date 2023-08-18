import fnmatch
from typing import Iterable, Iterator, List, Optional, Union

import jax
import numpy as np
import tensorflow as tf
from absl import logging
from flax.core import FrozenDict

from orca.data.utils.text_processing import TextProcessor
from orca.data.utils.tf_augmentations import augment
from orca.data.utils.tf_goal_relabeling import GOAL_RELABELING_FUNCTIONS


class BaseDataset:
    """
    Fast parallel tf.data.Dataset-based dataloader.

    Includes goal relabeling, image augmentations, and sampling from multiple
    datasets with different weights. Goal relabeling uses a 0/-1 reward scheme:
    0 when the next_obs is labeled as the goal, -1 otherwise.

    Args:
        dataset_names: Single dataset name OR list of dataset names OR list of filepath-lists.
            If more than one element is provided, the data will be sampled from each
            dataset according to "sample_weights".
        seed: Random seed.
        normalization_type: The type of normalization to apply to the actions
            and proprio.
        goal_relabeling_strategy: Goal relabeling strategy. See
            `orca.data.utils.tf_goal_relabeling` for more details.
        goal_relabeling_kwargs: Keyword arguments for goal relabeling. See
            `orca.data.utils.tf_goal_relabeling` for more details.
        sample_weights: If dataset_names has multiple elements, this is a
            list of weights with which to sample from each dataset.
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
            `orca.data.utils.tf_augmentations.augment` for more details.
        act_pred_horizon: Number of consecutive actions that will be predicted.
        obs_horizon: Number of consecutive observations that will be conditioned on.
    """

    def __init__(
        self,
        dataset_names: Union[str, List[str], List[List[str]]],
        seed: int,
        normalization_type: Optional[str] = "normal",
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
        # TODO(karl): make this a single horizon parameter
        act_pred_horizon: Optional[int] = None,
        obs_horizon: Optional[int] = None,
        # TODO(karl): these arguments are not documented
        string_fields: Optional[str] = ["language"],        # TODO(karl): assume that we have only one language string
        text_processor: Optional[TextProcessor] = None,
        image_processor: Optional[str] = "default",
        image_shape: Optional[List[int]] = (256, 256, 3),
        skip_unlabeled: bool = False,
        **kwargs,
    ):
        logging.warning("Extra kwargs passed to Dataset: %s", kwargs)
        if sample_weights is None:
            # default to uniform distribution over sub-lists
            sample_weights = [1 / len(dataset_names)] * len(dataset_names)
        assert len(dataset_names) == len(sample_weights)
        assert np.isclose(sum(sample_weights), 1.0)

        self.normalization_type = normalization_type
        self.action_proprio_metadata = None             # metadata for normalization, maybe computed on the fly
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

        # construct datasets
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        datasets = []
        for dataset_name in dataset_names:
            datasets.append(self._construct_tf_dataset(dataset_name, seed))

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

    def _construct_tf_dataset(self, dataset_name: Union[str, List[str]], seed: int) -> tf.data.Dataset:
        # construct base tf dataset of trajectories
        dataset = self._construct_base_dataset(dataset_name, seed)

        # maybe apply action & proprio normalization
        dataset = dataset.map(self._normalize_action_proprio, num_parallel_calls=tf.data.AUTOTUNE)

        # maybe chunks into snippets
        dataset = dataset.map(self._chunk_act_obs, num_parallel_calls=tf.data.AUTOTUNE)

        # cache before add_goals because add_goals introduces randomness
        if self.cache:
            dataset = dataset.cache()

        # yields trajectories
        dataset = dataset.map(self._add_goals, num_parallel_calls=tf.data.AUTOTUNE)

        # unbatch to yield individual transitions
        dataset = dataset.unbatch()

        return dataset

    def _construct_base_dataset(self, dataset_name: Union[str, List[str]], seed: int) -> tf.data.Dataset:
        """Constructs basic dataset of trajectories."""
        raise NotImplementedError("This should be implemented in child class.")

    def _normalize_action_proprio(self, traj):
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

        # need to tokenize language instructions here already to allow for sharding (str cannot be sharded)
        # can only apply tokenizers after conversion to numpy
        iterator = map(self._process_string_fields, iterator)
        return iterator