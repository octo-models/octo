from functools import partial
import json
import logging
from typing import Optional

import flax
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
import optax
import orbax.checkpoint
import tensorflow as tf

from orca.model import create_model_def
from orca.model.orca_model import OrcaModel
from orca.utils.train_utils import create_train_state
from orca.utils.typing import Any, Data, Dict, Params, Sequence

nonpytree_field = partial(flax.struct.field, pytree_node=False)


@flax.struct.dataclass
class PretrainedModel:
    """Recommended way of interacting with a pretrained model.

    Usage (example):
        model = PretrainedModel.load_pretrained(checkpoint_dir)

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
    def load_config(
        cls,
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
        config: Optional[Dict[str, Any]] = None,
        example_batch: Optional[Dict[str, Any]] = None,
        text_processor: Optional[Any] = None,
        step: Optional[int] = None,
    ):
        """Loads a pretrained model from a checkpoint. Important: this method expects the
        params-only checkpoint, not the full TrainState used for resuming training.

        This method operates in three steps:
            1. Load original model params with original config & example batch
            2. Create *new* model params with provided config (defaults to original config if config=None)
            3. Copy over all parameters from original model that have same key+shape in new model.

        Args:
            checkpoint_path (str): A path to either a directory of checkpoints or a single checkpoint.
            config (Dict[str, Any], optional): Config dict. If None, defaults to checkpoint_path/config.json.
            example_batch (Dict[str, Any], optional): Example_batch. If None,
                defaults to checkpoint_path/example_batch.msgpack.
            text_processor (Any, optional): Preprocessor for text inputs.
            step (int, optional): If multiple checkpoints are present, which one to load. Defaults to the latest.
        """
        orig_config = cls.load_config(checkpoint_path)
        if config is None:
            config = orig_config

        orig_example_batch_path = tf.io.gfile.join(
            checkpoint_path, "example_batch.msgpack"
        )
        with tf.io.gfile.GFile(orig_example_batch_path, "rb") as f:
            orig_example_batch = flax.serialization.msgpack_restore(f.read())
        if example_batch is not None:
            _verify_shapes(
                example_batch, orig_example_batch, starting_dim=1, raise_error=False
            )
        else:
            example_batch = orig_example_batch
        logging.info(
            "Using example batch with structure: %s",
            flax.core.pretty_repr(jax.tree_map(jnp.shape, example_batch)),
        )

        # load params into original model shape
        orig_model_def = create_model_def(
            **orig_config["model"].to_dict(),
        )
        rng = jax.random.PRNGKey(0)
        orig_params_shape = jax.eval_shape(
            partial(orig_model_def.init, train=False),
            rng,
            orig_example_batch["observation"],
            orig_example_batch["tasks"],
            orig_example_batch["observation"]["pad_mask"],
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

        orig_params = orbax.checkpoint.PyTreeCheckpointer().restore(
            tf.io.gfile.join(checkpoint_path, "default"), orig_params_shape
        )

        # create new model, then copy params from original model into new model
        model_def = create_model_def(
            **config["model"].to_dict(),
        )

        @jax.jit
        def _init():
            return model_def.init(
                rng,
                example_batch["observation"],
                example_batch["tasks"],
                example_batch["observation"]["pad_mask"],
                train=False,
            )

        params = _init()["params"]
        params = cls._merge_pretrained_params(
            target_params=params, pretrained_params=orig_params
        )

        return cls(
            model_def=model_def,
            params=params,
            example_batch=example_batch,
            text_processor=text_processor,
            config=flax.core.freeze(config.to_dict()),
        )

    @staticmethod
    def _merge_pretrained_params(target_params, pretrained_params):
        """Copies pre-trained params for every param that has corresponding key + shape."""
        flat_target_params = flax.traverse_util.flatten_dict(target_params)
        flat_pretrained_params = flax.traverse_util.flatten_dict(pretrained_params)
        keys_to_update = [
            k
            for k in flat_target_params
            if k in flat_pretrained_params
            and flat_target_params[k].shape == flat_pretrained_params[k].shape
        ]
        missing_keys = [
            k for k in flat_target_params if k not in flat_pretrained_params
        ]
        shape_mismatch_keys = [
            k
            for k in flat_target_params
            if k in flat_pretrained_params
            and flat_target_params[k].shape != flat_pretrained_params[k].shape
        ]

        for key in keys_to_update:
            logging.debug(f"Param copied from pre-trained: {'.'.join(key)}")
        if missing_keys or shape_mismatch_keys:
            logging.info(
                "########## Parameters skipped during model loading: ##########"
            )
            for key in missing_keys:
                logging.info(
                    f"Param missing in pre-trained model, skipping: {'.'.join(key)}"
                )
            for key in shape_mismatch_keys:
                logging.info(
                    f"Param with differing shape in pre-trained model, skipping: {'.'.join(key)}"
                )

        flat_target_params = flax.core.copy(
            flat_target_params, {k: flat_pretrained_params[k] for k in keys_to_update}
        )
        target_params = flax.traverse_util.unflatten_dict(flat_target_params)
        return flax.core.freeze(target_params)


class HeadWrapper:
    """Dummy class to help with the following syntactic sugar.

    > PretrainedModel.heads["action"].predict_action(transformer_embeddings)
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
):
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

    if raise_error and (fail or (weak_fail and strict)):
        raise AssertionError("Provided pytree does not match example pytree.")
