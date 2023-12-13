# WARNING: importing tensorflow too late can silence important logging (╯°□°)╯︵ ┻━┻
import tensorflow as tf

# isort: split

import datetime
from functools import partial
import os
import os.path as osp

from absl import app, flags, logging
from flax.traverse_util import flatten_dict
import jax
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_flags
import optax
import tqdm
import wandb

import octo
from octo.data.dataset import make_interleaved_dataset
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.model.octo_model import OctoModel
from octo.utils import jax_utils
from octo.utils.spec import ModuleSpec
from octo.utils.train_callbacks import (
    RolloutVisualizationCallback,
    SaveCallback,
    ValidationCallback,
    VisualizationCallback,
)
from octo.utils.train_utils import (
    create_optimizer,
    filter_eval_datasets,
    format_name_with_config,
    process_text,
    Timer,
    TrainState,
)
from octo.utils.typing import Data

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

    # make sure each process loads different data
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
        else:
            save_dir = None
            logging.info("save_dir not passed in, not saving checkpoints")
    else:
        # resume previous run
        wandb_run = wandb.Api().run(FLAGS.config.wandb_resume_id)
        if jax.process_index() == 0:
            wandb.init(
                project=wandb_run.project,
                id=wandb_run.id,
                entity=wandb_run.entity,
                resume="must",
            )
        save_dir = wandb_run.config["save_dir"]
        logging.info("Resuming run %s", FLAGS.config.wandb_resume_id)
    save_callback = SaveCallback(save_dir)

    if jax.process_index() == 0:
        codebase_directory = osp.abspath(osp.join(osp.dirname(octo.__file__), ".."))
        wandb.run.log_code(codebase_directory)

    # set up text tokenization (this needs to happen after batching but before sharding)
    if FLAGS.config.text_processor is None:
        text_processor = None
    else:
        text_processor = ModuleSpec.instantiate(FLAGS.config.text_processor)()

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    # load datasets
    if "oxe_kwargs" in FLAGS.config.dataset_kwargs:
        # create dataset_kwargs_list from oxe_kwargs
        (
            FLAGS.config.dataset_kwargs["dataset_kwargs_list"],
            FLAGS.config.dataset_kwargs["sample_weights"],
        ) = make_oxe_dataset_kwargs_and_weights(
            **FLAGS.config.dataset_kwargs["oxe_kwargs"]
        )
        del FLAGS.config.dataset_kwargs["oxe_kwargs"]

    train_data = make_interleaved_dataset(**FLAGS.config.dataset_kwargs, train=True)

    # consolidate dataset statistics into one big dict
    dataset_statistics = {
        dataset_kwargs["name"]: statistics
        for dataset_kwargs, statistics in zip(
            FLAGS.config.dataset_kwargs["dataset_kwargs_list"],
            train_data.dataset_statistics,
        )
    }

    train_data_iter = map(
        shard,
        map(
            process_batch,
            train_data.iterator(prefetch=FLAGS.config.prefetch_num_batches),
        ),
    )

    example_batch = next(train_data_iter)
    logging.info(f"Batch size: {example_batch['action'].shape[0]}")
    logging.info(f"Number of devices: {jax.device_count()}")
    logging.info(
        f"Batch size per device: {example_batch['action'].shape[0] // jax.device_count()}"
    )

    # set up model and initialize weights
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, init_rng = jax.random.split(rng)
    model = OctoModel.from_config(
        FLAGS.config.to_dict(),
        example_batch,
        text_processor,
        verbose=True,
        rng=init_rng,
        dataset_statistics=dataset_statistics,
    )

    # create optimizer
    tx, lr_callable, param_norm_callable = create_optimizer(
        model.params,
        **FLAGS.config.optimizer.to_dict(),
    )

    # Load pretrained weights (e.g. text encoder) if necessary
    for loader in FLAGS.config.pretrained_loaders:
        if not callable(loader):  # Means that it is a ModuleSpec
            loader = ModuleSpec.instantiate(loader)
        model = model.replace(params=loader(model.params))

    # create train state
    train_state = TrainState.create(rng, model, tx)

    if FLAGS.config.get("wandb_resume_id", None) is not None:
        train_state = save_callback.state_checkpointer.restore(
            save_callback.state_checkpointer.latest_step(), items=train_state
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
    train_state = train_state.replace(step=start_step)

    # replicate train state across devices
    train_state = jax_utils.replicate(train_state)

    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,  # action head knows to pull out the "action" readout_key
            batch["action"],
            pad_mask=batch["observation"]["pad_mask"],
            train=train,
        )
        return action_loss, action_metrics

    @partial(
        jax.jit,
        # state is replicated, batch is data-parallel
        in_shardings=(replicated_sharding, dp_sharding),
        out_shardings=(replicated_sharding, replicated_sharding),
        # allows jax to modify `state` in-place, saving a lot of memory
        donate_argnums=0,
    )
    def train_step(state: TrainState, batch: Data):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        grad_norm = optax.global_norm(grads)
        updates, _ = state.tx.update(grads, state.opt_state, state.model.params)
        update_norm = optax.global_norm(updates)
        info.update(
            {
                "grad_norm": grad_norm,
                "update_norm": update_norm,
                "param_norm": param_norm_callable(state.model.params),
                "learning_rate": lr_callable(state.step),
            }
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    val_datasets_kwargs_list, _ = filter_eval_datasets(
        FLAGS.config.dataset_kwargs["dataset_kwargs_list"],
        FLAGS.config.dataset_kwargs["sample_weights"],
        FLAGS.config.eval_datasets,
    )
    val_callback = ValidationCallback(
        loss_fn=loss_fn,
        process_batch_fn=lambda batch: shard(process_batch(batch)),
        text_processor=text_processor,
        val_dataset_kwargs_list=val_datasets_kwargs_list,
        dataset_kwargs=FLAGS.config.dataset_kwargs,
        **FLAGS.config.val_kwargs.to_dict(),
    )
    viz_callback = VisualizationCallback(
        text_processor=text_processor,
        val_dataset_kwargs_list=val_datasets_kwargs_list,
        dataset_kwargs=FLAGS.config.dataset_kwargs,
        **FLAGS.config.viz_kwargs.to_dict(),
    )
    if "rollout_kwargs" in FLAGS.config:
        rollout_callback = RolloutVisualizationCallback(
            text_processor=text_processor,
            history_length=FLAGS.config["window_size"],
            model_pred_horizon=FLAGS.config["model"]["heads"]["action"]["kwargs"].get(
                "pred_horizon", 1
            ),
            **FLAGS.config.rollout_kwargs.to_dict(),
        )
    else:
        rollout_callback = None

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

        if (i + 1) % FLAGS.config.save_interval == 0:
            save_callback(train_state, i + 1)

        if (i + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Evaluating...")
            with timer("eval"):
                val_metrics = val_callback(train_state, i + 1)
                wandb_log(val_metrics, step=i + 1)

        if (i + 1) % FLAGS.config.viz_interval == 0:
            logging.info("Visualizing...")
            with timer("visualize"):
                viz_metrics = viz_callback(train_state, i + 1)
                wandb_log(viz_metrics, step=i + 1)

            if rollout_callback is not None:
                with timer("rollout"):
                    rollout_metrics = rollout_callback(train_state, i + 1)
                    wandb_log(rollout_metrics, step=i + 1)

        timer.tock("total")
        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_log(
                {"training": update_info, "timer": timer.get_average_times()},
                step=i + 1,
            )


if __name__ == "__main__":
    app.run(main)
