from dataclasses import dataclass
from functools import partial
import logging
import os

import flax
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
import tensorflow as tf
import tqdm

from orca.data.dataset import make_interleaved_dataset, make_single_dataset
from orca.utils.train_utils import batched_apply
from orca.utils.typing import Any, Batch, Callable, Data, Dict, Optional, Sequence
from orca.utils.visualization_lib import Visualizer


class Callback:
    def __call__(self, train_state, step):
        raise NotImplementedError


def create_validation_dataset(
    dataset_kwargs, traj_transform_kwargs, frame_transform_kwargs, train=False
):
    """Creates a tf Dataset for validation and visualization purposes.

    Overwrites default parameters with more conservative options, to ensure stable memory and CPU consumption
    """
    return make_single_dataset(
        dataset_kwargs={
            **dataset_kwargs,
            "num_parallel_reads": 4,
            "num_parallel_calls": 4,
            "shuffle": False,
        },
        traj_transform_kwargs={
            **traj_transform_kwargs,
            "num_parallel_calls": 4,
        },
        frame_transform_kwargs=frame_transform_kwargs,
        train=train,
        frame_transform_threads=16,
    )


@dataclass
class SaveCallback(Callback):
    save_dir: Optional[str]

    def __post_init__(self):
        if self.save_dir is not None and jax.process_index() == 0:
            tf.io.gfile.makedirs(self.save_dir)
            logging.info(f"Created {self.save_dir}")
            # make checkpointers
            # only keep latest full TrainState
            self.state_checkpointer = orbax.checkpoint.CheckpointManager(
                tf.io.gfile.join(self.save_dir, "state"),
                orbax.checkpoint.PyTreeCheckpointer(),
                options=orbax.checkpoint.CheckpointManagerOptions(
                    max_to_keep=1,
                ),
            )
            # keep every params checkpoint
            self.params_checkpointer = orbax.checkpoint.CheckpointManager(
                self.save_dir,
                orbax.checkpoint.PyTreeCheckpointer(),
            )

    def __call__(self, train_state, step):
        if self.save_dir is not None and jax.process_index() == 0:
            self.params_checkpointer.save(
                step,
                train_state.params,
                {"save_args": orbax_utils.save_args_from_target(train_state.params)},
            )
            self.state_checkpointer.save(
                step,
                train_state,
                {"save_args": orbax_utils.save_args_from_target(train_state)},
            )
        return {}

    def open(self, fname, mode):
        if self.save_dir is not None and jax.process_index() == 0:
            logging.info(f"Saving to {tf.io.gfile.join(self.save_dir, fname)}")
            return tf.io.gfile.GFile(tf.io.gfile.join(self.save_dir, fname), mode)
        else:
            return open(os.devnull, mode)


def remove_text(tasks: Data, zero_text_encoding: Data):
    """Replaces language encoding inside task dict with that of the empty string

    zero_text_encoding = jax.tree_map(lambda x: x[0], text_processor.encode([""]))

    """
    if "language_instruction" in tasks:
        new_language = jax.tree_map(
            lambda x, example: jnp.broadcast_to(example[None], x.shape),
            tasks["language_instruction"],
            zero_text_encoding,
        )
        tasks = flax.core.copy(tasks, {"language_instruction": new_language})
    return tasks


def remove_images(tasks: Data):
    """Replaces images inside task dict with zero (black) images."""
    new_images = {k: jnp.zeros_like(v) for k, v in tasks.items() if "image" in k}
    return flax.core.copy(tasks, new_images)


@dataclass
class ValidationCallback(Callback):
    loss_fn: Callable
    process_batch_fn: Callable[[Batch], Batch]
    text_processor: Any
    val_dataset_kwargs_list: Sequence[Dict]
    dataset_kwargs: Dict
    val_shuffle_buffer_size: int
    num_val_batches: int
    modes_to_evaluate: Sequence[str] = ("text_conditioned", "image_conditioned")
    train: bool = False

    def __post_init__(self):
        if self.text_processor is not None:
            self.zero_text = jax.tree_map(
                lambda x: x[0], self.text_processor.encode("")
            )
        self.val_iterators = {}
        for single_dataset_kwargs in self.val_dataset_kwargs_list:
            val_dataset = create_validation_dataset(
                single_dataset_kwargs,
                self.dataset_kwargs["traj_transform_kwargs"],
                self.dataset_kwargs["frame_transform_kwargs"],
                train=train,
            )
            val_iterator = (
                val_dataset.unbatch()
                .shuffle(self.val_shuffle_buffer_size)
                .repeat()
                .batch(self.dataset_kwargs["batch_size"])
                .iterator(prefetch=0)
            )
            val_iterator = map(self.process_batch_fn, val_iterator)
            self.val_iterators[single_dataset_kwargs["name"]] = val_iterator

        @jax.jit
        def eval_step(state, batch):
            loss_fn_partial = partial(
                self.loss_fn,
                params=state.params,
                state=state,
                rng=state.rng,
                train=False,
            )
            all_tasks = {}

            if "base" in self.modes_to_evaluate:
                all_tasks["base"] = batch["tasks"]
            if "image_conditioned" in self.modes_to_evaluate:
                all_tasks["text_conditioned"] = remove_images(batch["tasks"])
            if "text_conditioned" in self.modes_to_evaluate:
                all_tasks["image_conditioned"] = remove_text(
                    batch["tasks"], self.zero_text
                )
            if "unconditioned" in self.modes_to_evaluate:
                all_tasks["unconditioned"] = remove_text(
                    remove_images(batch["tasks"]), self.zero_text
                )
            return {
                k: loss_fn_partial(batch=flax.core.copy(batch, {"tasks": tasks}))[1]
                for k, tasks in all_tasks.items()
            }

        self.eval_step = eval_step

    def __call__(self, train_state, step):
        wandb_metrics = {}
        for name, val_data_iter in self.val_iterators.items():
            metrics = []
            for _, batch in tqdm.tqdm(
                zip(range(self.num_val_batches), val_data_iter),
                total=self.num_val_batches,
                desc=name,
            ):
                metrics.append(self.eval_step(train_state, batch))
            metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
            wandb_metrics[f"validation_{name}"] = metrics
        return wandb_metrics


@dataclass
class VisualizationCallback(Callback):
    text_processor: Any
    val_dataset_kwargs_list: Sequence[Dict]
    dataset_kwargs: Dict
    eval_batch_size: int
    trajs_for_metrics: int
    trajs_for_viz: int
    samples_per_state: int
    modes_to_evaluate: str = ("text_conditioned", "image_conditioned")
    train: bool = False

    def __post_init__(self):
        self.zero_text = jax.tree_map(lambda x: x[0], self.text_processor.encode(""))

        self.visualizers = {}
        for single_dataset_kwargs in self.val_dataset_kwargs_list:
            val_dataset = create_validation_dataset(
                single_dataset_kwargs,
                self.dataset_kwargs["traj_transform_kwargs"],
                self.dataset_kwargs["frame_transform_kwargs"],
                train=train,
            )
            self.visualizers[single_dataset_kwargs["name"]] = Visualizer(
                val_dataset, text_processor=self.text_processor, freeze_trajs=False
            )

        @partial(jax.jit, static_argnames="policy_mode")
        def get_policy_sampled_actions(state, observations, tasks, policy_mode=None):
            # only use first horizon timesteps as input to predict_action

            if policy_mode == "text_conditioned":
                tasks = remove_images(tasks)
            elif policy_mode == "image_conditioned":
                tasks = remove_text(tasks, self.zero_text)
            elif policy_mode == "unconditioned":
                tasks = remove_text(remove_images(tasks), self.zero_text)

            def get_actions(model, observations, tasks, train):
                transformer_embeddings = model.orca_transformer(
                    observations,
                    tasks,
                    observations["pad_mask"],
                    train=train,
                )

                actions = model.heads["action"].predict_action(
                    transformer_embeddings,
                    train=train,
                    argmax=False,
                    sample_shape=(self.samples_per_state,),
                    rng=state.rng,
                )
                return actions

            actions = state.apply_fn(
                {"params": state.params},
                observations,
                tasks,
                train=False,
                method=get_actions,
                rngs={"dropout": state.rng},
            )  # We could also have used run_head here, but this is easier to read

            # actions is (NUM_ACTIONS_FOR_VIS, batch_size, pred_horizon, action_dim)
            # where actions[:, :, i] predicts the action at timestep "window_size + i"
            actions = actions[..., 0, :]

            # viz expects (batch_size, n_samples, action_dim)
            actions = jnp.moveaxis(actions, 0, 1)
            return actions

        self.get_policy_sampled_actions = get_policy_sampled_actions

    def __call__(self, train_state, step):
        wandb_metrics = {}
        modal_policy_fns = {
            k: batched_apply(
                partial(self.get_policy_sampled_actions, train_state, policy_mode=k),
                self.eval_batch_size,
            )
            for k in self.modes_to_evaluate
        }

        for name, visualizer in self.visualizers.items():
            for mode, policy_fn in modal_policy_fns.items():
                if self.trajs_for_metrics > 0:
                    raw_infos = visualizer.raw_evaluations(
                        policy_fn, max_trajs=self.trajs_for_metrics
                    )
                    metrics = visualizer.metrics_for_wandb(raw_infos)
                    wandb_metrics[f"offline_metrics_{name}/{mode}"] = metrics
                if self.trajs_for_viz > 0:
                    images = visualizer.visualize_for_wandb(
                        policy_fn, max_trajs=self.trajs_for_viz
                    )
                    wandb_metrics[f"visualizations_{name}/{mode}"] = images
        return wandb_metrics
