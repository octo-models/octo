from functools import partial
import json
import logging
import os
from typing import Any, Optional

import flax
import flax.struct
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np
import orbax.checkpoint
import tensorflow as tf

from orca.data.utils.text_processing import text_processors, TextProcessor
from orca.model import create_model_def
from orca.model.orca_model import OrcaModel
from orca.utils.typing import Config, Data, Params, Sequence

nonpytree_field = partial(flax.struct.field, pytree_node=False)


@flax.struct.dataclass
class ORCAModel:
    """Recommended way of interacting with a pretrained ORCA model.

    Usage (example):
        model = ORCAModel.load_pretrained(checkpoint_dir)

        # Create the task dict
        tasks = model.create_tasks(texts=["go to the red room"])
        # or
        tasks = model.create_tasks(goals={"image_0": goal_images})

        # Run the model (jax.jit for speed)
        policy_fn = jax.jit(model.sample_actions)
        policy_fn(observations, tasks, rng=jax.random.PRNGKey(0))

    """

    model_def: OrcaModel = nonpytree_field()
    params: Params
    text_processor: TextProcessor = nonpytree_field()
    example_batch: Data
    config: Config = nonpytree_field()

    def __call__(self, *args, **kwargs):
        return self.model_def.apply({"params": self.params}, *args, **kwargs)

    @property
    def orca_transformer(self):
        """Syntactic sugar for calling the transformer."""
        return partial(self.__call__, method="run_transformer")

    @property
    def heads(self):
        """Syntactic sugar for calling heads.
        > self.heads["action"].predict_action(transformer_embeddings)
        """
        head_fns = {}
        for head_name in self.model_def.heads:
            head_fns[head_name] = HeadWrapper(
                partial(self.__call__, method="run_head", head_name=head_name)
            )
        return head_fns

    def create_tasks(self, goals: Data = None, texts: Optional[Sequence[str]] = None):
        """Creates tasks dict from goals and texts.

        Args:
            goals: if not None, dict of shape (batch_size, *)
            texts: if not None, list of texts of length batch_size

        Omit images to run the language-conditioned model, and omit texts to run the
        goal-conditioned model.

        """
        assert goals is not None or texts is not None
        tasks = {}
        if goals is not None:
            tasks.update(goals)
        else:
            batch_size = len(texts)
            tasks = jax.tree_map(
                lambda example: jnp.zeros(
                    (batch_size, *example.shape[1:]), dtype=example.dtype
                ),
                self.example_batch["tasks"],
            )

        if texts is None:
            batch_size = jax.tree_util.tree_leaves(goals)[0].shape[0]
            texts = [""] * batch_size
        if self.text_processor is not None:
            tasks["language_instruction"] = self.text_processor.encode(texts)

        _verify_shapes(tasks, self.example_batch["tasks"], starting_dim=1)
        return tasks

    def run_transformer(self, observations, tasks, pad_mask, train=False):
        """Runs the transformer, but does shape checking on the inputs.
        Args:
            observations: dictionary of arrays of shape (batch_size, window_size, *shape).
                Shape must be consistent with self.example_batch["observation"]
            tasks: dict of tasks of shape (batch_size, *shape)
                Shape must be consistent with self.example_batch["tasks"]
            pad_mask: (batch_size, window_size) Boolean mask that is False when the timestep corresponds to padding
            train: whether to run in train mode
            *args, **kwargs: Additional arguments for transformer or model.apply
        """
        _verify_shapes(observations, self.example_batch["observation"], starting_dim=2)
        _verify_shapes(tasks, self.example_batch["tasks"], starting_dim=1)

        return self.orca_transformer(observations, tasks, pad_mask, train=train)

    def sample_actions(self, observations, tasks, pad_mask=None, train=False, **kwargs):
        """Samples actions from the model.

        Recommended to do this inside a jax.jit

        Args:
            observations: dictionary of arrays of shape (batch_size, window_size, *)
            tasks: dict of tasks of shape (batch_size, *)
            pad_mask: (batch_size, window_size) Boolean mask that is False when the timestep corresponds to padding
            train: whether to run in train mode
            **kwargs: kwargs to pass to model.heads["action"].predict_action
        """
        if pad_mask is None:
            pad_mask = observations["pad_mask"]

        transformer_embeddings = self.run_transformer(
            observations, tasks, pad_mask, train=train
        )
        return self.heads["action"].predict_action(
            transformer_embeddings,
            train=train,
            **kwargs,
        )

    @staticmethod
    def load_dataset_statistics(
        checkpoint_path: str, dataset_name: Optional[str] = None
    ):
        if dataset_name is not None:
            statistics_path = f"dataset_statistics_{dataset_name}.json"
        else:
            statistics_path = "dataset_statistics.json"
        statistics_path = tf.io.gfile.join(checkpoint_path, statistics_path)
        with tf.io.gfile.GFile(statistics_path, "r") as f:
            statistics = json.load(f)

        statistics = jax.tree_map(
            lambda x: np.array(x),
            statistics,
            is_leaf=lambda x: not isinstance(x, dict),
        )
        return statistics

    @staticmethod
    def load_config(
        checkpoint_path: str,
    ):
        config_path = tf.io.gfile.join(checkpoint_path, "config.json")
        with tf.io.gfile.GFile(config_path, "r") as f:
            config = json.load(f)
            config = ConfigDict(config)
        return config

    @classmethod
    def load_pretrained(
        cls,
        checkpoint_path: str,
        step: Optional[int] = None,
    ) -> "ORCAModel":
        """Loads a pretrained model from a checkpoint. Important: this method expects the
        params-only checkpoint, not the full TrainState used for resuming training.

        Args:
            checkpoint_path (str): A path to either a directory of checkpoints or a single checkpoint.
            step (int, optional): If multiple checkpoints are present, which one to load. Defaults to the latest.
        """
        config = cls.load_config(checkpoint_path)
        example_batch_path = tf.io.gfile.join(checkpoint_path, "example_batch.msgpack")
        with tf.io.gfile.GFile(example_batch_path, "rb") as f:
            example_batch = flax.serialization.msgpack_restore(f.read())

        logging.debug(
            "Using example batch with structure: %s",
            flax.core.pretty_repr(jax.tree_map(jnp.shape, example_batch)),
        )

        # load params into original model shape
        model_def = create_model_def(
            **config["model"].to_dict(),
        )
        rng = jax.random.PRNGKey(0)
        params_shape = jax.eval_shape(
            partial(model_def.init, train=False),
            rng,
            example_batch["observation"],
            example_batch["tasks"],
            example_batch["observation"]["pad_mask"],
        )["params"]
        all_steps = orbax.checkpoint.utils.checkpoint_steps(checkpoint_path)
        if all_steps:
            if step is not None and step not in all_steps:
                raise ValueError(
                    f"Step {step} not found in checkpoint path {checkpoint_path}."
                )
            # assume this is a path to a directory of checkpoints
            checkpoint_path = orbax.checkpoint.utils.get_save_directory(
                max(all_steps) if step is None else step,
                checkpoint_path,
            )

        params = orbax.checkpoint.PyTreeCheckpointer().restore(
            tf.io.gfile.join(checkpoint_path, "default"), params_shape
        )

        if config["text_processor"] is not None:
            text_processor = text_processors[config["text_processor"]](
                **config.get("text_processor_kwargs", {})
            )
        else:
            text_processor = None

        return cls(
            model_def=model_def,
            params=params,
            text_processor=text_processor,
            example_batch=example_batch,
            config=config.to_dict(),
        )

    @classmethod
    def from_config(
        cls,
        config: Config,
        example_batch: Data,
        text_processor: Optional[Any] = None,
    ):
        """Initializes a model with a fresh set of weights from a given config + example_batch.

        Args:
            config (Dict[str, Any]): Config dict. If None, defaults to checkpoint_path/config.json.
            example_batch (Dict[str, Any]): Example_batch.
            text_processor (Any, optional): Preprocessor for text inputs.
        """
        model_def = create_model_def(
            **config["model"].to_dict(),
        )

        @jax.jit
        def _init():
            return model_def.init(
                jax.random.PRNGKey(0),
                example_batch["observation"],
                example_batch["tasks"],
                example_batch["observation"]["pad_mask"],
                train=False,
            )

        params = _init()["params"]
        if not text_processor and config["text_processor"] is not None:
            text_processor = text_processors[config["text_processor"]](
                **config.get("text_processor_kwargs", {})
            )
        return cls(
            model_def=model_def,
            params=params,
            text_processor=text_processor,
            example_batch=example_batch,
            config=config.to_dict(),
        )


class HeadWrapper:
    """Dummy class to help with the following syntactic sugar.

    > ORCAModel.heads["action"].predict_action(transformer_embeddings)
    """

    def __init__(self, fn):
        self.__call__ = fn

    def __getattr__(self, name):
        return partial(self.__call__, head_method_name=name)


def _verify_shapes(
    pytree,
    example_pytree,
    starting_dim: int = 0,
    strict: bool = False,
    raise_error: bool = True,
    silent=False,
):
    weak_fail, fail = False, False
    pytree_flat = flax.traverse_util.flatten_dict(pytree)
    example_pytree_flat = flax.traverse_util.flatten_dict(example_pytree)

    # Check that all elements are present
    if set(pytree_flat.keys()) != set(example_pytree_flat.keys()):
        if not silent:
            logging.warning(
                "Provided pytree contains extra items: %s",
                set(pytree_flat.keys()) - set(example_pytree_flat.keys()),
            )
            logging.warning(
                "Provided pytree doesn't contain items: %s",
                set(example_pytree_flat.keys()) - set(pytree_flat.keys()),
            )
        weak_fail = True

    mismatched_keys = {
        k: (pytree_flat[k].shape, example_pytree_flat[k].shape)
        for k in pytree_flat
        if k in example_pytree_flat
        and pytree_flat[k].shape[starting_dim:]
        != example_pytree_flat[k].shape[starting_dim:]
    }
    if mismatched_keys:
        if not silent:
            logging.warning(
                "Provided pytree contains mismatched shapes: %s",
                flax.core.pretty_repr(mismatched_keys),
            )
        fail = True

    if raise_error and (fail or (weak_fail and strict)):
        raise AssertionError("Provided pytree does not match example pytree.")

    return weak_fail or fail
