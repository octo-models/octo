from functools import partial
import json
import logging

import flax
import flax.training.checkpoints as checkpoints
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import optax
import tensorflow as tf

from orca.data.utils.text_processing import text_processors
from orca.model import create_model_def
from orca.model.orca_model import OrcaModel
from orca.utils.train_utils import create_train_state
from orca.utils.typing import Any, Data, Dict, Params, Sequence

nonpytree_field = partial(flax.struct.field, pytree_node=False)


@flax.struct.dataclass
class PretrainedModel:
    """Recommended way of interacting with a pretrained model.

    Usage (example):
        model = PretrainedModel.load_pretrained(checkpoint_path)

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
    text_processor: Any = nonpytree_field()
    example_batch: Data
    config: flax.core.FrozenDict = nonpytree_field()

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

    def create_tasks(self, goals: Dict[str, Data] = None, texts: Sequence[str] = None):
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
            tasks = {
                k: jnp.zeros((batch_size, *v.shape[1:]), dtype=v.dtype)
                for k, v in self.example_batch["tasks"].items()
            }

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

    @classmethod
    def load_pretrained(
        cls,
        checkpoint_path: str,
        config_path: str = None,
        example_batch_path: str = None,
        skip_verification: bool = False,
        step=None,
    ):
        """Loads a pretrained model from a checkpoint.

        Args:
            checkpoint_path: Checkpoint path (can either be a specific checkpoint or directory of all checkpoints)
            config_path: Path to config.json. If None, defaults to checkpoint_path/config.json
            example_batch_path: Path to example_batch.msgpack. If None, defaults to checkpoint_path/example_batch.msgpack
            skip_verification: If True, doesn't check the loaded params have the correct shape. Faster, but more dangerous!
            step: int or None: If multiple checkpoints are present, which one to load. Defaults to the latest.
        """
        if config_path is None:
            config_path = tf.io.gfile.join(checkpoint_path, "config.json")
        if example_batch_path is None:
            example_batch_path = tf.io.gfile.join(
                checkpoint_path, "example_batch.msgpack"
            )

        with tf.io.gfile.GFile(config_path, "r") as f:
            config = json.load(f)
            config = ConfigDict(config)

        model_def = create_model_def(
            **config["model"].to_dict(),
        )
        with tf.io.gfile.GFile(example_batch_path, "rb") as f:
            example_batch = flax.serialization.msgpack_restore(f.read())
            logging.warning(
                "Loaded example batch with structure: %s",
                flax.core.pretty_repr(jax.tree_map(jnp.shape, example_batch)),
            )

        if config["text_processor"] is None:
            text_processor = None
        else:
            text_processor = text_processors[config["text_processor"]](
                **config["text_processor_kwargs"]
            )

        if skip_verification:
            loaded = checkpoints.restore_checkpoint(
                checkpoint_path, target=None, step=step
            )
            return cls(
                model_def=model_def,
                params=loaded["params"],
                text_processor=text_processor,
                example_batch=example_batch,
                config=flax.core.freeze(config.to_dict()),
            )

        # create train_state
        rng = jax.random.PRNGKey(0)
        rng, construct_rng = jax.random.split(rng)

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config["optimizer"]["learning_rate"],
            warmup_steps=config["optimizer"]["warmup_steps"],
            decay_steps=config["optimizer"]["decay_steps"],
            end_value=0.0,
        )

        tx = optax.adam(lr_schedule)
        train_state = create_train_state(
            construct_rng,
            model_def,
            tx,
            init_args=(
                example_batch["observation"],
                example_batch["tasks"],
                example_batch["observation"]["pad_mask"],
            ),
            init_kwargs={"train": False},
        )

        train_state = checkpoints.restore_checkpoint(
            checkpoint_path, train_state, step=step
        )

        return cls(
            model_def=model_def,
            params=train_state.params,
            text_processor=text_processor,
            example_batch=example_batch,
            config=flax.core.freeze(config.to_dict()),
        )


class HeadWrapper:
    """Dummy class to help with the following syntactic sugar.

    > PretrainedModel.heads["action"].predict_action(transformer_embeddings)
    """

    def __init__(self, fn):
        self.__call__ = fn

    def __getattr__(self, name):
        return partial(self.__call__, head_method_name=name)


def _verify_shapes(pytree, example_pytree, starting_dim: int = 0, strict: bool = False):
    weak_fail, fail = False, False
    pytree_flat = flax.traverse_util.flatten_dict(pytree)
    example_pytree_flat = flax.traverse_util.flatten_dict(example_pytree)

    # Check that all elements are present
    if set(pytree_flat.keys()) != set(example_pytree_flat.keys()):
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
        logging.warning(
            "Provided pytree contains mismatched shapes: %s",
            flax.core.pretty_repr(mismatched_keys),
        )
        fail = True

    if fail or (weak_fail and strict):
        raise AssertionError("Provided pytree does not match example pytree.")
