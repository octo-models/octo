from functools import partial
import json
import logging
from typing import Any, Optional, Tuple

import flax
from flax import struct
from flax.training import orbax_utils
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
import orbax.checkpoint
import tensorflow as tf

from octo.data.utils.text_processing import TextProcessor
from octo.model.components.action_heads import ActionHead
from octo.model.octo_module import OctoModule
from octo.utils.spec import ModuleSpec
from octo.utils.typing import Config, Data, Params, PRNGKey, Sequence


@struct.dataclass
class OctoModel:
    """Recommended way of interacting with Octo models.

    Usage for inference:

        >>> model = OctoModel.load_pretrained(checkpoint_dir)
        >>> tasks = model.create_tasks(texts=["go to the red room"])
        >>> # or tasks = model.create_tasks(goals={"image_primary": goal_images})
        >>> actions = model.sample_actions(observations, tasks, rng=jax.random.PRNGKey(0))
        >>> # Note: these are normalized actions (processed to mean 0 and std 1). To get the raw actions,
            # un-normalize them using model.dataset_statistics

    Usage for finetuning:

        >>> model = OctoModel.load_pretrained(checkpoint_dir)
        >>> train_state = octo.utils.train_utils.TrainState.create(
            rng=jax.random.PRNGKey(0),
            model=model,
            tx=optax.adamw(...)
        )
        >>> # access params through train_state.model.params
        >>> train_state, metrics = your_update_function(train_state, batch)
        >>> # when it's time to save (note that this only saves the model parameters,
        >>> # not the full optimizer state)
        >>> train_state.model.save_pretrained(step, save_dir)

    Usage for pretraining:

        >>> model = OctoModel.from_config(
                config,
                example_batch,
                text_processor
            )  # initializes params
        >>> # Continue as in finetuning example

    See full usage examples in train.py and finetune.py.

    """

    module: OctoModule = struct.field(pytree_node=False)
    text_processor: TextProcessor = struct.field(pytree_node=False)
    config: Config = struct.field(pytree_node=False)
    params: Params
    example_batch: Data
    dataset_statistics: Optional[Data]

    def create_tasks(
        self, goals: Optional[Data] = None, texts: Optional[Sequence[str]] = None
    ):
        """Creates tasks dict from goals and texts.

        Args:
            goals: if not None, dict of arrays with shape (batch_size, *)
            texts: if not None, list of texts of length batch_size

        Omit images to run the language-conditioned model, and omit texts to run the
        goal-conditioned model.
        """
        assert goals is not None or texts is not None
        tasks = {"pad_mask_dict": {}}
        if goals is not None:
            tasks.update(goals)
            tasks["pad_mask_dict"].update(
                {k: np.ones(v.shape[:1], dtype=bool) for k, v in goals.items()}
            )
        else:
            batch_size = len(texts)
            tasks.update(
                {
                    k: np.zeros((batch_size, *v.shape[1:]), dtype=v.dtype)
                    for k, v in self.example_batch["task"].items()
                    if k not in ("pad_mask_dict", "language_instruction")
                }
            )
            tasks["pad_mask_dict"].update(
                {
                    k: np.zeros(batch_size, dtype=bool)
                    for k in tasks.keys()
                    if k != "pad_mask_dict"
                }
            )

        if texts is not None:
            assert self.text_processor is not None
            tasks["language_instruction"] = texts
            tasks["pad_mask_dict"]["language_instruction"] = np.ones(
                len(texts), dtype=bool
            )
        else:
            batch_size = jax.tree_leaves(goals)[0].shape[0]
            tasks["language_instruction"] = [""] * batch_size
            tasks["pad_mask_dict"]["language_instruction"] = np.zeros(
                batch_size, dtype=bool
            )

        if self.text_processor is not None:
            tasks["language_instruction"] = self.text_processor.encode(
                tasks["language_instruction"]
            )
        else:
            del tasks["language_instruction"]

        _verify_shapes(tasks, "tasks", self.example_batch["task"], starting_dim=1)
        return tasks

    @partial(jax.jit, static_argnames=("train",))
    def run_transformer(
        self, observations: Data, tasks: Data, pad_mask: ArrayLike, train: bool = False
    ):
        """Runs the transformer, but does shape checking on the inputs.

        Args:
            observations: dictionary of arrays of shape (batch_size, window_size, *shape).
                Shape must be consistent with self.example_batch["observation"]
            tasks: dict of tasks of shape (batch_size, *shape)
                Shape must be consistent with self.example_batch["task"]
            pad_mask: (batch_size, window_size) Boolean mask that is False when the timestep corresponds to padding
            train: whether to run in train mode
        """
        _verify_shapes(
            observations,
            "observations",
            self.example_batch["observation"],
            starting_dim=2,
        )
        _verify_shapes(tasks, "tasks", self.example_batch["task"], starting_dim=1)

        return self.module.apply(
            {"params": self.params},
            observations,
            tasks,
            pad_mask,
            train=train,
            method="octo_transformer",
        )

    @partial(jax.jit, static_argnames=("train", "sample_shape", "argmax"))
    def sample_actions(
        self,
        observations: Data,
        tasks: Data,
        pad_mask: Optional[ArrayLike] = None,
        train: bool = False,
        argmax: bool = False,
        sample_shape: Tuple[int, ...] = (),
        rng: Optional[PRNGKey] = None,
        temperature: float = 1.0,
    ):
        """Samples actions from the model. See `action_heads.py` for more info.

        Args:
            observations: dictionary of arrays of shape (batch_size, window_size, *)
            tasks: dict of tasks of shape (batch_size, *)
            pad_mask: (batch_size, window_size) Boolean mask that is False when the timestep corresponds to padding
            train: whether to run in train mode
            ...see `action_heads.py` for the rest of the kwargs.
        Returns:
            actions: (*sample_shape, batch_size, pred_horizon, action_dim)
        """
        if pad_mask is None:
            pad_mask = observations["pad_mask"]

        transformer_outputs = self.run_transformer(
            observations, tasks, pad_mask, train=train
        )
        action_head: ActionHead = self.module.bind({"params": self.params}).heads[
            "action"
        ]
        return action_head.predict_action(
            transformer_outputs,
            train=train,
            argmax=argmax,
            sample_shape=sample_shape,
            rng=rng,
            temperature=temperature,
        )

    @classmethod
    def load_pretrained(
        cls,
        checkpoint_path: str,
        step: Optional[int] = None,
    ) -> "OctoModel":
        """Loads a model from a checkpoint that was saved via `save_pretrained`.

        Args:
            checkpoint_path (str): A path to either a directory of checkpoints or a single checkpoint.
            step (int, optional): If multiple checkpoints are present, which one to load. Defaults to the latest.
        """
        if checkpoint_path.startswith("hf://"):
            if step:
                raise ValueError(
                    "You can't set config['pretrained_step'] when loading from HuggingFace."
                )
            checkpoint_path = _download_from_huggingface(
                checkpoint_path.removeprefix("hf://")
            )

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

        # create model def (an OctoModule)
        module = OctoModule.create(**config["model"])
        # infer params shape without actually doing any computation
        params_shape = jax.eval_shape(
            partial(module.init, train=False),
            jax.random.PRNGKey(0),
            example_batch["observation"],
            example_batch["task"],
            example_batch["observation"]["pad_mask"],
        )["params"]
        # restore params, checking to make sure the shape matches
        checkpointer = orbax.checkpoint.CheckpointManager(
            checkpoint_path, orbax.checkpoint.PyTreeCheckpointer()
        )
        step = step if step is not None else checkpointer.latest_step()
        params = checkpointer.restore(step, params_shape)

        if config["text_processor"] is not None:
            text_processor = ModuleSpec.instantiate(config["text_processor"])()
        else:
            text_processor = None

        return cls(
            module=module,
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
                    f.write(flax.serialization.msgpack_serialize(self.example_batch))

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
            config (Dict[str, Any]): Config dict. The only required key is "model", but other configuration
                may be saved for posterity.
            example_batch (Dict[str, Any]): Example batch.
            text_processor (Any, optional): Preprocessor for text inputs.
            verbose (bool, optional): Whether to print out a summary of the model.
            rng (Optional[PRNGKey], optional): RNG key for initializing the model.
            dataset_statistics (Optional[Dict[str, Any]], optional): Dataset statistics.
        """
        module = OctoModule.create(**config["model"])
        rng = rng if rng is not None else jax.random.PRNGKey(0)
        example_batch = multihost_utils.process_allgather(example_batch)
        example_batch = jax.tree_map(lambda x: x[:1], example_batch)

        init_args = (
            example_batch["observation"],
            example_batch["task"],
            example_batch["observation"]["pad_mask"],
        )

        if verbose:
            print(
                module.tabulate(rng, *init_args, train=False, verbose=True, depth=2)
            )  # Prints out the parameter count of our model, and tokenizer details

        @jax.jit
        def _init(rng):
            return module.init(rng, *init_args, train=False)

        params = _init(rng)["params"]

        return cls(
            module=module,
            params=params,
            text_processor=text_processor,
            example_batch=example_batch,
            config=config,
            dataset_statistics=dataset_statistics,
        )

    def get_pretty_spec(self):
        """Brief summary of the model's expected inputs and outputs."""
        # TODO: generalize this to print out proprio when it is being tokenized
        window_size = self.example_batch["observation"]["pad_mask"].shape[1]

        observation_space = {
            k: ("batch", "history_window", *v.shape[2:])
            for k, v in self.example_batch["observation"].items()
            if k.startswith("image")
        }
        task_space = {
            k: ("batch", *v.shape[1:])
            for k, v in self.example_batch["task"].items()
            if k.startswith("image")
        }
        if self.text_processor is not None:
            task_space["language_instruction"] = jax.tree_map(
                lambda arr: ("batch", *arr.shape[1:]),
                self.example_batch["task"]["language_instruction"],
            )

        try:
            action_head = self.module.heads["action"]
            action_head_repr = str(action_head.__class__)
            action_dim, pred_horizon = action_head.action_dim, action_head.pred_horizon
        except:
            action_head_repr, action_dim, pred_horizon = "", None, None

        return SPEC_TEMPLATE.format(
            window_size=window_size,
            observation_space=flax.core.pretty_repr(observation_space),
            task_space=flax.core.pretty_repr(task_space),
            action_head_repr=action_head_repr,
            action_dim=action_dim,
            pred_horizon=pred_horizon,
        )


def _verify_shapes(
    pytree,
    name: str,
    example_pytree,
    starting_dim: int = 0,
    strict: bool = False,
    raise_error: bool = True,
    silent: bool = False,
):
    weak_fail, fail = False, False
    pytree_flat = flax.traverse_util.flatten_dict(pytree)
    example_pytree_flat = flax.traverse_util.flatten_dict(example_pytree)

    # Check that all elements are present
    if set(pytree_flat.keys()) != set(example_pytree_flat.keys()):
        if not silent:
            extra = set(pytree_flat.keys()) - set(example_pytree_flat.keys())
            if extra:
                logging.warning(
                    "'%s' contains extra items compared to example_batch: %s",
                    name,
                    {"/".join(x) for x in extra},
                )
            missing = set(example_pytree_flat.keys()) - set(pytree_flat.keys())
            if missing:
                logging.warning(
                    "'%s' is missing items compared to example_batch: %s",
                    name,
                    {"/".join(x) for x in missing},
                )
        weak_fail = True

    mismatched_keys = {
        k: f"{pytree_flat[k].shape} != {example_pytree_flat[k].shape}"
        for k in pytree_flat
        if k in example_pytree_flat
        and pytree_flat[k].shape[starting_dim:]
        != example_pytree_flat[k].shape[starting_dim:]
    }
    if mismatched_keys:
        if not silent:
            logging.error(
                "'%s' contains mismatched shapes compared to example_batch: %s",
                name,
                flax.core.pretty_repr(
                    {"/".join(k): v for k, v in mismatched_keys.items()}
                ),
            )
        fail = True

    if raise_error and (fail or (weak_fail and strict)):
        raise AssertionError(f"{name} does not match example batch.")

    return weak_fail or fail


def _download_from_huggingface(huggingface_repo_id: str):
    import huggingface_hub

    folder = huggingface_hub.snapshot_download(huggingface_repo_id)
    return folder
    return folder


SPEC_TEMPLATE = """
This model is trained with a window size of {window_size}, predicting {action_dim} dimensional actions {pred_horizon} steps into the future.
Observations and tasks conform to the following spec:

Observations: {observation_space}
Tasks: {task_space}

At inference, you may pass in any subset of these observation and task keys, with a history window up to {window_size} timesteps.
"""
