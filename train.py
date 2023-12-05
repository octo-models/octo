# WARNING: importing tensorflow too late can silence important logging (╯°□°)╯︵ ┻━┻
import tensorflow as tf

# isort: split

import datetime
from functools import partial
import json
import os
import os.path as osp
import subprocess

from absl import app, flags, logging
import flax
from flax.training import orbax_utils
from flax.traverse_util import flatten_dict
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_flags
import numpy as np
import optax
import orbax.checkpoint
import tqdm
import wandb

import orca
from orca.data.dataset import make_interleaved_dataset, make_single_dataset
from orca.data.oxe.oxe_dataset_mixes import make_oxe_dataset_kwargs_and_weights, mixes
from orca.data.utils.text_processing import text_processors
from orca.model import create_model_def, OrcaModel
from orca.model.components.hf_weight_loaders import weights_loaders
from orca.utils import jax_utils
from orca.utils.train_utils import (
    batched_apply,
    create_optimizer,
    create_train_state,
    filter_eval_datasets,
    format_name_with_config,
    Timer,
)
from orca.utils.visualization_lib import RolloutVisualizer, Visualizer

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")

config_dir = os.path.join(os.path.dirname(__file__), "configs")
config_flags.DEFINE_config_file(
    "config",
    os.path.join(config_dir, "config.py:transformer_bc"),
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

NUM_ACTIONS_FOR_VIS = 8


def main(_):
    jax_utils.initialize_compilation_cache()

    assert FLAGS.config.dataset_kwargs.batch_size % jax.device_count() == 0

    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(jax.devices(), axis_names="batch")
    # replicated sharding -- does not shard arrays
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    # data-parallel sharding -- shards arrays along the first axis
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))

    def shard(batch):
        return multihost_utils.host_local_array_to_global_array(
            batch, mesh, PartitionSpec("batch")
        )

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    tf.random.set_seed(FLAGS.config.seed + jax.process_index())

    # set up wandb and logging
    if FLAGS.config.get("wandb_resume_id", None) is None:
        name = format_name_with_config(
            FLAGS.name,
            FLAGS.config.to_dict(),
        )
        wandb_id = "{name}_{time}".format(
            name=name,
            time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        wandb_id = jax_utils.host_broadcast_str(wandb_id)
        if jax.process_index() == 0:
            wandb.init(
                config=FLAGS.config.to_dict(),
                id=wandb_id,
                name=name,
                mode="disabled" if FLAGS.debug else None,
                **FLAGS.config.wandb,
            )

        if FLAGS.config.save_dir is not None:
            save_dir = tf.io.gfile.join(
                FLAGS.config.save_dir,
                FLAGS.config.wandb.project,
                FLAGS.config.wandb.group or "",
                wandb_id,
            )
            logging.info("Saving to %s", save_dir)
            if jax.process_index() == 0:
                tf.io.gfile.makedirs(save_dir)
                wandb.config.update(dict(save_dir=save_dir), allow_val_change=True)
                with tf.io.gfile.GFile(
                    os.path.join(save_dir, "config.json"), "w"
                ) as config_file:
                    config_file.write(FLAGS.config.to_json_best_effort())
        else:
            save_dir = None
            logging.info("save_dir not passed in, not saving checkpoints")
    else:
        # resume previous run
        wandb_run = wandb.Api().run(FLAGS.config.wandb_resume_id)
        wandb.init(
            project=wandb_run.project,
            id=wandb_run.id,
            entity=wandb_run.entity,
            resume="must",
        )
        save_dir = wandb_run.config["save_dir"]

    if save_dir is not None:
        # make checkpointers
        # only keep latest full TrainState
        state_checkpointer = orbax.checkpoint.CheckpointManager(
            tf.io.gfile.join(save_dir, "state"),
            orbax.checkpoint.PyTreeCheckpointer(),
            options=orbax.checkpoint.CheckpointManagerOptions(
                max_to_keep=1,
            ),
        )
        # keep every params checkpoint
        params_checkpointer = orbax.checkpoint.CheckpointManager(
            save_dir,
            orbax.checkpoint.PyTreeCheckpointer(),
        )

    if jax.process_index() == 0:
        codebase_directory = osp.abspath(osp.join(osp.dirname(orca.__file__), ".."))
        wandb.run.log_code(codebase_directory)  # TODO: replace w/ codesave_library?

    # set up text tokenization (this needs to happen after batching but before sharding)
    if FLAGS.config.text_processor is None:
        text_processor = None
    else:
        text_processor = text_processors[FLAGS.config.text_processor](
            **FLAGS.config.text_processor_kwargs
        )
        zero_text = jax.tree_map(lambda x: x[0], text_processor.encode([""]))

    def process_text(batch):
        if text_processor is None:
            batch["tasks"].pop("language_instruction")
        else:
            batch["tasks"]["language_instruction"] = text_processor.encode(
                [s.decode("utf-8") for s in batch["tasks"]["language_instruction"]]
            )
        del batch["dataset_name"]
        return batch

    # load datasets
    if "oxe_kwargs" in FLAGS.config.dataset_kwargs:
        # create dataset_kwargs_list from oxe_kwargs
        oxe_kwargs = FLAGS.config.dataset_kwargs["oxe_kwargs"].to_dict()
        del FLAGS.config.dataset_kwargs["oxe_kwargs"]
        oxe_kwargs["data_mix"] = mixes[oxe_kwargs["data_mix"]]
        (
            dataset_kwargs_list,
            dataset_sampling_weights,
        ) = make_oxe_dataset_kwargs_and_weights(**oxe_kwargs)
        FLAGS.config.dataset_kwargs["dataset_kwargs_list"] = dataset_kwargs_list
        FLAGS.config.dataset_kwargs["sample_weights"] = dataset_sampling_weights

    # override each element of dataset_kwargs_list with common_dataset_kwargs
    if "common_dataset_kwargs" in FLAGS.config.dataset_kwargs:
        FLAGS.config.dataset_kwargs["dataset_kwargs_list"] = [
            {**kwargs, **FLAGS.config.dataset_kwargs["common_dataset_kwargs"]}
            for kwargs in FLAGS.config.dataset_kwargs["dataset_kwargs_list"]
        ]
        del FLAGS.config.dataset_kwargs["common_dataset_kwargs"]

    train_data = make_interleaved_dataset(**FLAGS.config.dataset_kwargs, train=True)

    # save dataset statistics
    if save_dir is not None and jax.process_index() == 0:
        for dataset_kwargs, dataset_statistics in zip(
            FLAGS.config.dataset_kwargs["dataset_kwargs_list"],
            train_data.dataset_statistics,
        ):
            with tf.io.gfile.GFile(
                os.path.join(
                    save_dir, f"dataset_statistics_{dataset_kwargs['name']}.json"
                ),
                "w",
            ) as f:
                json.dump(
                    jax.tree_map(lambda x: x.tolist(), dataset_statistics),
                    f,
                )

    # create validation datasets and visualizers
    val_datas = []
    visualizers = []
    val_datasets_kwargs, val_datasets_sample_weights = filter_eval_datasets(
        FLAGS.config.dataset_kwargs["dataset_kwargs_list"],
        FLAGS.config.dataset_kwargs["sample_weights"],
        FLAGS.config.eval_datasets,
    )
    for dataset_kwargs in val_datasets_kwargs:
        val_dataset = make_single_dataset(
            dataset_kwargs={
                **dataset_kwargs,
                "num_parallel_reads": 4,
                "num_parallel_calls": 4,
                "shuffle": False,
            },
            traj_transform_kwargs={
                **FLAGS.config.dataset_kwargs["traj_transform_kwargs"],
                "num_parallel_calls": 4,
            },
            frame_transform_kwargs=FLAGS.config.dataset_kwargs[
                "frame_transform_kwargs"
            ],
            train=False,
            frame_transform_threads=16,
        )
        val_datas.append(
            val_dataset.unbatch()
            .shuffle(FLAGS.config.val_shuffle_buffer_size)
            .repeat()
            .batch(FLAGS.config.dataset_kwargs.batch_size)
        )
        visualizers.append(
            Visualizer(val_dataset, text_processor=text_processor, freeze_trajs=False)
        )

    train_data_iter = map(
        shard,
        map(
            process_text,
            train_data.iterator(prefetch=FLAGS.config.prefetch_num_batches),
        ),
    )
    val_data_iters = [
        map(shard, map(process_text, val_data.iterator(prefetch=0)))
        for val_data in val_datas
    ]

    # optionally build visualizers for sim env evals
    if FLAGS.config.get("rollout_envs", None):
        rollout_visualizers = []
        for env_name, visualizer_kwargs in FLAGS.config["rollout_envs"]:
            input_kwargs = dict(
                env_name=env_name,
                history_length=FLAGS.config["window_size"],
                action_chunk=FLAGS.config["model"]["heads"]["action"]["kwargs"].get(
                    "pred_horizon", 1
                ),
                text_processor=text_processor,
            )
            input_kwargs.update(visualizer_kwargs)
            rollout_visualizers.append(RolloutVisualizer(**input_kwargs))
    else:
        rollout_visualizers = None

    example_batch = next(train_data_iter)
    logging.info(f"Batch size: {example_batch['action'].shape[0]}")
    logging.info(f"Number of devices: {jax.device_count()}")
    logging.info(
        f"Batch size per device: {example_batch['action'].shape[0] // jax.device_count()}"
    )

    # set up model, optimizer, loss
    model_def = create_model_def(
        **FLAGS.config.model.to_dict(),
    )

    # pretrained weights to load
    pretrained_loader_kwargs = FLAGS.config.pretrained_loader_kwargs or [
        dict() for _ in FLAGS.config.pretrained_loaders
    ]
    assert len(pretrained_loader_kwargs) == len(
        FLAGS.config.pretrained_loaders
    ), "supply one kwarg dict for each loader!"
    pretrained_loaders = [
        partial(weights_loaders[w], **kwargs)
        for w, kwargs in zip(FLAGS.config.pretrained_loaders, pretrained_loader_kwargs)
    ]

    # ensure construct rng is same on every host
    construct_rng = jax.random.PRNGKey(FLAGS.config.seed)
    model_init_args = (
        example_batch["observation"],
        example_batch["tasks"],
        example_batch["observation"]["pad_mask"],
    )
    print(
        model_def.tabulate(
            construct_rng,
            *model_init_args,
            train=False,
            verbose=True,
            depth=2,
        )
    )  # Prints out the parameter count of our model

    params_shape = jax.eval_shape(
        partial(model_def.init, train=False),
        construct_rng,
        *model_init_args,
    )["params"]
    tx, lr_callable, param_norm_callable = create_optimizer(
        params_shape,
        FLAGS.config.optimizer.to_dict(),
    )
    train_state = create_train_state(
        construct_rng,
        model_def,
        tx,
        init_args=model_init_args,
        init_kwargs=dict(train=False),
        pretrained_loaders=pretrained_loaders,
    )

    example_batch = multihost_utils.process_allgather(example_batch)
    if save_dir is not None and jax.process_index() == 0:
        # Saving example batch for future checkpoint loading
        with tf.io.gfile.GFile(
            os.path.join(save_dir, "example_batch.msgpack"), "wb"
        ) as f:
            f.write(
                flax.serialization.msgpack_serialize(
                    jax.tree_map(lambda x: x[:1], example_batch)
                )
            )

        example_batch_spec = jax.tree_map(
            lambda arr: (arr.shape, str(arr.dtype)), example_batch
        )
        wandb.config.update(
            dict(example_batch_spec=example_batch_spec), allow_val_change=True
        )

        # Save the git hash
        process = subprocess.Popen(
            ["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE
        )
        git_head_hash = process.communicate()[0].strip()
        with tf.io.gfile.GFile(os.path.join(save_dir, "git_hash.txt"), "wb") as f:
            f.write(git_head_hash)

    if FLAGS.config.get("wandb_resume_id", None) is not None:
        train_state = state_checkpointer.restore(
            state_checkpointer.latest_step(), items=train_state
        )
        checkpoint_step = int(train_state.step)
        logging.info("Restored checkpoint from %s", save_dir)
        if FLAGS.config.start_step is not None:
            start_step = FLAGS.config.start_step  # start_step overrides checkpoint
        else:
            start_step = checkpoint_step
        logging.info("Starting training from step %d", start_step)
    else:
        start_step = FLAGS.config.start_step or 0

    # replicate train state across devices
    train_state = jax_utils.replicate(train_state)

    def loss_fn(params, state, batch, rng, train=True):
        def get_loss(model: OrcaModel, observations, tasks, actions, train):
            transformer_embeddings = model.orca_transformer(
                observations, tasks, observations["pad_mask"], train=train
            )
            action_loss, action_metrics = model.heads["action"].loss(
                transformer_embeddings,  # Action head knows to pull out the action readout_key
                actions,
                pad_mask=observations["pad_mask"],
                train=train,
            )

            return action_loss, action_metrics

        return state.apply_fn(
            {"params": params},
            batch["observation"],
            batch["tasks"],
            batch["action"],
            train=train,
            rngs={"dropout": rng},
            method=get_loss,
        )

    @partial(
        jax.jit,
        # state is replicated, batch is data-parallel
        in_shardings=(replicated_sharding, dp_sharding),
        out_shardings=(replicated_sharding, replicated_sharding),
        # allows jax to modify `state` in-place, saving a lot of memory
        donate_argnums=0,
    )
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, state, batch, dropout_rng, train=True
        )
        grad_norm = optax.global_norm(grads)
        updates, _ = state.tx.update(grads, state.opt_state, state.params)
        update_norm = optax.global_norm(updates)
        info.update(
            {
                "grad_norm": grad_norm,
                "update_norm": update_norm,
                "param_norm": param_norm_callable(state.params),
                "learning_rate": lr_callable(state.step),
            }
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    def remove_text(tasks):
        if text_processor is not None:
            new_language = jax.tree_map(
                lambda x, example: jnp.broadcast_to(example[None], x.shape),
                tasks["language_instruction"],
                zero_text,
            )
            tasks = flax.core.copy(tasks, {"language_instruction": new_language})
        return tasks

    def remove_images(tasks):
        new_images = {k: jnp.zeros_like(v) for k, v in tasks.items() if "image" in k}
        return flax.core.copy(tasks, new_images)

    @partial(
        jax.jit,
        # state is replicated, batch is data-parallel
        in_shardings=(replicated_sharding, dp_sharding),
        out_shardings=replicated_sharding,
    )
    def eval_step(state, batch):
        loss_fn_partial = partial(
            loss_fn, state.params, state, rng=state.rng, train=False
        )
        all_tasks = {"base": batch["tasks"]}
        if text_processor is not None:
            all_tasks["text_conditioned"] = remove_images(batch["tasks"])
            all_tasks["image_conditioned"] = remove_text(batch["tasks"])
            all_tasks["unconditioned"] = remove_text(remove_images(batch["tasks"]))
        return {
            k: loss_fn_partial(flax.core.copy(batch, {"tasks": tasks}))[1]
            for k, tasks in all_tasks.items()
        }

    @partial(jax.jit, static_argnames="policy_mode")
    def get_policy_sampled_actions(state, observations, tasks, policy_mode=None):
        # only use first horizon timesteps as input to predict_action

        if policy_mode == "text_conditioned":
            tasks = remove_images(tasks)
        elif policy_mode == "image_conditioned":
            tasks = remove_text(tasks)
        elif policy_mode == "unconditioned":
            tasks = remove_text(remove_images(tasks))

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
                sample_shape=(NUM_ACTIONS_FOR_VIS,),
                rng=state.rng,
            )
            return actions

        # actions is (NUM_ACTIONS_FOR_VIS, batch_size, pred_horizon, action_dim)
        # where actions[:, :, i] predicts the action at timestep "window_size + i"
        actions = state.apply_fn(
            {"params": state.params},
            observations,
            tasks,
            train=False,
            method=get_actions,
            rngs={"dropout": state.rng},
        )  # We could also have used run_head here, but this is easier to read

        # viz expects (batch_size, n_samples, pred_horizon, action_dim)
        actions = jnp.moveaxis(actions, 0, 1)
        return actions

    def wandb_log(info, step):
        if jax.process_index() == 0:
            wandb.log(flatten_dict(info, sep="/"), step=step)

    timer = Timer()
    for i in tqdm.tqdm(
        range(start_step, int(FLAGS.config.num_steps)),
        total=int(FLAGS.config.num_steps),
        initial=start_step,
        dynamic_ncols=True,
    ):
        timer.tick("total")

        with timer("dataset"):
            batch = next(train_data_iter)

        with timer("train"):
            train_state, update_info = train_step(train_state, batch)

        if (i + 1) % FLAGS.config.save_interval == 0 and save_dir is not None:
            params_checkpointer.save(
                i + 1,
                train_state.params,
                {"save_args": orbax_utils.save_args_from_target(train_state.params)},
            )
            state_checkpointer.save(
                i + 1,
                train_state,
                {"save_args": orbax_utils.save_args_from_target(train_state)},
            )

        if (i + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Evaluating...")
            timer.tick("val")
            per_dataset_metrics = []
            for data_kwargs, val_data_iter in zip(val_datasets_kwargs, val_data_iters):
                metrics = []
                for _, batch in tqdm.tqdm(
                    zip(range(FLAGS.config.num_val_batches), val_data_iter),
                    total=FLAGS.config.num_val_batches,
                    desc=data_kwargs["name"],
                ):
                    metrics.append(eval_step(train_state, batch))
                metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
                wandb_log({f"validation_{data_kwargs['name']}": metrics}, step=i)
                per_dataset_metrics.append(metrics)

            # log weighted aggregate metrics
            val_datasets_sample_weights = (
                val_datasets_sample_weights
                if val_datasets_sample_weights is not None
                else [1.0] * len(per_dataset_metrics)
            )
            val_datasets_sample_weights = val_datasets_sample_weights / np.sum(
                val_datasets_sample_weights
            )  # normalize to sum to 1
            agg_metrics = jax.tree_map(
                lambda *xs: np.sum(xs),
                *[
                    jax.tree_map(lambda x: x * weight, metric)
                    for metric, weight in zip(
                        per_dataset_metrics, val_datasets_sample_weights
                    )
                ],
            )
            wandb_log({"validation_aggregate": agg_metrics}, step=i)
            timer.tock("val")

        if (i + 1) % FLAGS.config.viz_interval == 0:
            logging.info("Visualizing...")
            timer.tick("visualize")

            if text_processor is not None:
                modes_to_evaluate = [
                    "text_conditioned",
                    "image_conditioned",
                    "unconditioned",
                ]
            else:
                modes_to_evaluate = ["image_conditioned"]

            modal_policy_fns = {
                k: batched_apply(
                    partial(get_policy_sampled_actions, train_state, policy_mode=k),
                    FLAGS.config.eval_batch_size,
                )
                for k in modes_to_evaluate
            }

            for data_kwargs, visualizer in zip(val_datasets_kwargs, visualizers):
                for mode, policy_fn in modal_policy_fns.items():
                    raw_infos = visualizer.raw_evaluations(
                        policy_fn, max_trajs=FLAGS.config.trajs_for_metrics
                    )
                    metrics = visualizer.metrics_for_wandb(raw_infos)
                    images = visualizer.visualize_for_wandb(
                        policy_fn, max_trajs=FLAGS.config.trajs_for_viz
                    )
                    wandb_log(
                        {
                            f"offline_metrics_{data_kwargs['name']}/{mode}": metrics,
                            f"visualizations_{data_kwargs['name']}/{mode}": images,
                        },
                        step=i,
                    )

            if rollout_visualizers:
                for rollout_visualizer in rollout_visualizers:
                    for mode, policy_fn in modal_policy_fns.items():
                        logging.info("Running rollouts...")
                        rollout_infos = rollout_visualizer.run_rollouts(
                            policy_fn, n_rollouts=FLAGS.config.trajs_for_rollouts
                        )
                        wandb_log(
                            {
                                f"rollouts_{rollout_visualizer.env_name}"
                                f"_chunk{rollout_visualizer.action_chunk}/{mode}": rollout_infos,
                            },
                            step=i,
                        )

            timer.tock("visualize")

        timer.tock("total")

        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_log(
                {"training": update_info, "timer": timer.get_average_times()}, step=i
            )


if __name__ == "__main__":
    app.run(main)
