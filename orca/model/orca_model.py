from dataclasses import dataclass
from functools import partial, update_wrapper
import json
import logging
from typing import Any, Optional

import flax
from flax import struct
from flax.training import orbax_utils
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
import tensorflow as tf

from orca.data.utils.text_processing import TextProcessor
from orca.model import create_model_def
from orca.model.orca_module import ORCAModule
from orca.utils.spec import ModuleSpec
from orca.utils.typing import Config, Data, Params, PRNGKey, Sequence


@struct.dataclass
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

    model_def: ORCAModule = struct.field(pytree_node=False)
    params: Params
    text_processor: TextProcessor = struct.field(pytree_node=False)
    example_batch: Data
    config: Config = struct.field(pytree_node=False)
    dataset_statistics: Optional[Data]

    def __call__(self, *args, **kwargs):
        return self.model_def.apply({"params": self.params}, *args, **kwargs)

    @property
    def orca_transformer(self):
        """Syntactic sugar for calling the transformer.

        >>> transformer_outputs = self.orca_transformer(observations, tasks, pad_mask, train=False)
        """
        return partial(self, method="orca_transformer")

    @property
    def heads(self):
        """Syntactic sugar for calling heads.

        >>> self.heads["action"].predict_action(transformer_outputs)
        """
        return {name: HeadWrapper(self, name) for name in self.model_def.heads}

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
                self.example_batch["task"],
            )

        if texts is None:
            batch_size = jax.tree_util.tree_leaves(goals)[0].shape[0]
            texts = [""] * batch_size
        if self.text_processor is not None:
            tasks["language_instruction"] = self.text_processor.encode(texts)

        _verify_shapes(tasks, self.example_batch["task"], starting_dim=1)
        return tasks

    def run_transformer(self, observations, tasks, pad_mask, train=False):
        """Runs the transformer, but does shape checking on the inputs.
        Args:
            observations: dictionary of arrays of shape (batch_size, window_size, *shape).
                Shape must be consistent with self.example_batch["observation"]
            tasks: dict of tasks of shape (batch_size, *shape)
                Shape must be consistent with self.example_batch["task"]
            pad_mask: (batch_size, window_size) Boolean mask that is False when the timestep corresponds to padding
            train: whether to run in train mode
            *args, **kwargs: Additional arguments for transformer or model.apply
        """
        _verify_shapes(observations, self.example_batch["observation"], starting_dim=2)
        _verify_shapes(tasks, self.example_batch["task"], starting_dim=1)

        return self.orca_transformer(observations, tasks, pad_mask, train=train)

    def sample_actions(self, observations, tasks, pad_mask=None, train=False, **kwargs):
        """Samples actions from the model. See `action_heads.py` for more info.

        Recommended to do this inside a jax.jit.

        Args:
            observations: dictionary of arrays of shape (batch_size, window_size, *)
            tasks: dict of tasks of shape (batch_size, *)
            pad_mask: (batch_size, window_size) Boolean mask that is False when the timestep corresponds to padding
            train: whether to run in train mode
            **kwargs: kwargs to pass to model.heads["action"].predict_action
        Returns:
            actions: (*sample_shape, batch_size, pred_horizon, action_dim)
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

    @classmethod
    def load_pretrained(
        cls,
        checkpoint_path: str,
        step: Optional[int] = None,
    ) -> "ORCAModel":
        """Loads a model from a checkpoint that was saved via `save_pretrained`.

        Args:
            checkpoint_path (str): A path to either a directory of checkpoints or a single checkpoint.
            step (int, optional): If multiple checkpoints are present, which one to load. Defaults to the latest.
        """
        # load config
        with tf.io.gfile.GFile(
            tf.io.gfile.join(checkpoint_path, "config.json"), "r"
        ) as f:
            config = json.load(f)

        # load example batch
        with tf.io.gfile.GFile(
            tf.io.gfile.join(checkpoint_path, "example_batch.msgpack"), "rb"
        ) as f:
            example_batch = flax.serialization.msgpack_restore(f.read())
        # shim for migrating from "tasks" to "task"
        if "tasks" in example_batch:
            example_batch["task"] = example_batch.pop("tasks")

        logging.debug(
            "Model was trained with observations: %s",
            flax.core.pretty_repr(
                jax.tree_map(jnp.shape, example_batch["observation"])
            ),
        )
        logging.debug(
            "Model was trained with tasks: %s",
            flax.core.pretty_repr(jax.tree_map(jnp.shape, example_batch["task"])),
        )

        # load dataset statistics
        with tf.io.gfile.GFile(
            tf.io.gfile.join(checkpoint_path, "dataset_statistics.json"), "r"
        ) as f:
            dataset_statistics = json.load(f)
            dataset_statistics = jax.tree_map(
                np.array, dataset_statistics, is_leaf=lambda x: not isinstance(x, dict)
            )

        # create model def (an ORCAModule)
        model_def = create_model_def(**config["model"])
        # infer params shape without actually doing any computation
        params_shape = jax.eval_shape(
            partial(model_def.init, train=False),
            jax.random.PRNGKey(0),
            example_batch["observation"],
            example_batch["task"],
            example_batch["observation"]["pad_mask"],
        )["params"]
        # restore params, checking to make sure the shape matches
        checkpointer = orbax.checkpoint.CheckpointManager(
            checkpoint_path, orbax.checkpoint.PyTreeCheckpointer()
        )
        step = step or checkpointer.latest_step()
        params = checkpointer.restore(step, params_shape)

        if config["text_processor"] is not None:
            text_processor = ModuleSpec.instantiate(config["text_processor"])()
        else:
            text_processor = None

        return cls(
            model_def=model_def,
            params=params,
            text_processor=text_processor,
            example_batch=example_batch,
            config=config,
            dataset_statistics=dataset_statistics,
        )

    def save_pretrained(
        self,
        step: int,
        checkpoint_path: Optional[str] = None,
        checkpoint_manager: Optional[orbax.checkpoint.CheckpointManager] = None,
    ):
        """Saves a model, as well as corresponding metadata needed for `load_pretrained`. Takes either a
        pre-existing checkpoint manager (which already knows where to save the checkpoint) or a path to a
        directory to save the checkpoint to.

        Args:
            step (int): Step number.
            checkpoint_path (str, optional): Path to save the checkpoint.
            checkpoint_manager (optional): Checkpoint manager to save the checkpoint.
            params (optional): Params to save. If None, uses self.params.
        """
        if (checkpoint_path is None) == (checkpoint_manager is None):
            raise ValueError(
                "Must provide exactly one of checkpoint_path or checkpoint_manager."
            )
        if checkpoint_manager is None:
            checkpoint_manager = orbax.checkpoint.CheckpointManager(
                checkpoint_path, orbax.checkpoint.PyTreeCheckpointer()
            )
        if checkpoint_path is None:
            checkpoint_path = str(checkpoint_manager._directory)

        # save params
        checkpoint_manager.save(
            step,
            self.params,
            {"save_args": orbax_utils.save_args_from_target(self.params)},
        )

        if jax.process_index() == 0:
            # save config
            config_path = tf.io.gfile.join(checkpoint_path, "config.json")
            if not tf.io.gfile.exists(config_path):
                with tf.io.gfile.GFile(config_path, "w") as f:
                    json.dump(self.config, f)

            # save example batch
            example_batch_path = tf.io.gfile.join(
                checkpoint_path, "example_batch.msgpack"
            )
            if not tf.io.gfile.exists(example_batch_path):
                with tf.io.gfile.GFile(example_batch_path, "wb") as f:
                    f.write(
                        flax.serialization.msgpack_serialize(
                            jax.tree_map(
                                lambda x: x[:1],
                                multihost_utils.process_allgather(self.example_batch),
                            )
                        )
                    )

            # save dataset statistics
            dataset_statistics_path = tf.io.gfile.join(
                checkpoint_path, "dataset_statistics.json"
            )
            if not tf.io.gfile.exists(dataset_statistics_path):
                with tf.io.gfile.GFile(dataset_statistics_path, "w") as f:
                    json.dump(
                        jax.tree_map(lambda x: x.tolist(), self.dataset_statistics),
                        f,
                    )

    @classmethod
    def from_config(
        cls,
        config: Config,
        example_batch: Data,
        text_processor: Optional[Any] = None,
        verbose: bool = False,
        rng: Optional[PRNGKey] = None,
        dataset_statistics: Optional[Data] = None,
    ):
        """Initializes a model with a fresh set of weights from a given config + example_batch.

        Args:
            config (Dict[str, Any]): Config dict.
            example_batch (Dict[str, Any]): Example batch.
            text_processor (Any, optional): Preprocessor for text inputs.
            verbose (bool, optional): Whether to print out a summary of the model.
            rng (Optional[PRNGKey], optional): RNG key for initializing the model.
            dataset_statistics (Optional[Dict[str, Any]], optional): Dataset statistics.
        """
        model_def = create_model_def(**config["model"])
        rng = rng if rng is not None else jax.random.PRNGKey(0)
        example_batch = jax.tree_map(lambda x: x[:1], example_batch)

        init_args = (
            example_batch["observation"],
            example_batch["task"],
            example_batch["observation"]["pad_mask"],
        )

        if verbose:
            print(
                model_def.tabulate(rng, *init_args, train=False, verbose=True, depth=2)
            )  # Prints out the parameter count of our model, and tokenizer details

        @jax.jit
        def _init(rng):
            return model_def.init(rng, *init_args, train=False)

        params = _init(rng)["params"]

        return cls(
            model_def=model_def,
            params=params,
            text_processor=text_processor,
            example_batch=example_batch,
            config=config,
            dataset_statistics=dataset_statistics,
        )


@dataclass
class HeadWrapper:
    """Dummy class to help with the following syntactic sugar.

    > ORCAModel.heads["action"].predict_action(transformer_outputs)
    """

    model: ORCAModel
    head_name: str

    def __call__(self, *args, **kwargs):
        return self.__getattr__("__call__")(*args, **kwargs)

    def __getattr__(self, method_name):
        def bound_fn(module: ORCAModule, *args, **kwargs):
            return module.run_head(
                *args, head_name=self.head_name, head_method_name=method_name, **kwargs
            )

        return update_wrapper(
            # calls `self.model.__call__`, which binds the params to the ORCAModule, giving `bound_fn` above
            # access to the bound module
            partial(self.model, method=bound_fn),
            self.model.model_def.heads[self.head_name].__getattribute__(method_name),
        )


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
