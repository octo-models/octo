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
from orca.data.dataset import make_interleaved_dataset
from orca.data.oxe.oxe_dataset_mixes import make_oxe_dataset_kwargs_and_weights, mixes
from orca.data.utils.text_processing import text_processors
from orca.model import create_model_def
from orca.model.components.hf_weight_loaders import weights_loaders
from orca.utils import jax_utils
from orca.utils.train_callbacks import (
    SaveCallback,
    ValidationCallback,
    VisualizationCallback,
)
from orca.utils.train_utils import (
    create_optimizer,
    create_train_state,
    filter_eval_datasets,
    format_name_with_config,
    process_text,
    Timer,
)

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
            save_callback = SaveCallback(save_dir)
            if jax.process_index() == 0:
                wandb.config.update(dict(save_dir=save_dir), allow_val_change=True)
                with save_callback.open("config.json", "w") as config_file:
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
        save_callback = SaveCallback(save_dir)
        logging.info("Resuming run %s", FLAGS.config.wandb_resume_id)

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

    def process_batch(batch):
        batch = process_text(batch, text_processor)
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
            fname = f"dataset_statistics_{dataset_kwargs['name']}.json"
            with save_callback.open(fname, "w") as f:
                json.dump(
                    jax.tree_map(lambda x: x.tolist(), dataset_statistics),
                    f,
                )

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
    if jax.process_index() == 0:
        # Saving example batch for future checkpoint loading
        example_batch_spec = jax.tree_map(
            lambda arr: (arr.shape, str(arr.dtype)), example_batch
        )
        wandb.config.update(
            dict(example_batch_spec=example_batch_spec), allow_val_change=True
        )

        if save_dir is not None:
            with save_callback.open("example_batch.msgpack", "wb") as f:
                f.write(
                    flax.serialization.msgpack_serialize(
                        jax.tree_map(lambda x: x[:1], example_batch)
                    )
                )
            try:
                process = subprocess.Popen(
                    ["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE
                )
                git_head_hash = process.communicate()[0].strip()
                with save_callback.open("git_hash.txt", "wb") as f:
                    f.write(git_head_hash)
            except Exception as e:
                logging.warning("Failed to save git hash: %s", e)

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

    # replicate train state across devices
    train_state = jax_utils.replicate(train_state)

    def loss_fn(params, state, batch, rng, train=True):
        model = model_def.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = model.orca_transformer(
            batch["observation"],
            batch["tasks"],
            batch["observation"]["pad_mask"],
            train=train,
        )
        action_loss, action_metrics = model.heads["action"].loss(
            transformer_embeddings,  # Action head knows to pull out the action readout_key
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
            save_callback(train_state, i + 1)

        if (i + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Evaluating...")
            with timer("eval"):
                wandb_metrics = val_callback(train_state, i + 1)
                wandb_log(wandb_metrics, step=i + 1)

        if (i + 1) % FLAGS.config.viz_interval == 0:
            logging.info("Visualizing...")
            with timer("visualize"):
                wandb_metrics = viz_callback(train_state, i + 1)
                wandb_log(wandb_metrics, step=i + 1)

        timer.tock("total")
        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_log(
                {"training": update_info, "timer": timer.get_average_times()},
                step=i + 1,
            )


if __name__ == "__main__":
    app.run(main)
