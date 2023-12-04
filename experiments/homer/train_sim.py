import datetime
from functools import partial
import os

from absl import app, flags, logging
from flax.training import checkpoints
from flax.traverse_util import flatten_dict
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from ml_collections import config_flags
import numpy as np
import optax
import tensorflow as tf
import tqdm
import wandb

from orca.model import create_model_def
from orca.model.components.hf_weight_loaders import weights_loaders
from orca.utils.jax_utils import initialize_compilation_cache
from orca.utils.train_utils import (
    create_train_state,
    format_name_with_config,
    Timer,
)
from orca.utils.visualization_lib import Visualizer
from orca.utils.eval_utils import sample_actions

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

from sim.evaluation import evaluate_gc, supply_rng
from sim.utils import make_mujoco_gc_env, load_recorded_video
from sim.dataset import make_sim_dataset

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

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


def main(_):
    initialize_compilation_cache()

    assert FLAGS.config.batch_size % jax.device_count() == 0

    devices = jax.local_devices()
    num_devices = len(devices)
    assert FLAGS.config.batch_size % num_devices == 0

    # create a 1D mesh with a single axis named "batch"
    mesh = Mesh(jax.devices(), axis_names="batch")
    # replicated sharding -- does not shard arrays
    replicated_sharding = NamedSharding(mesh, PartitionSpec())
    # data-parallel sharding -- shards arrays along the first axis
    dp_sharding = NamedSharding(mesh, PartitionSpec("batch"))

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # set up wandb and logging
    name = format_name_with_config(FLAGS.name, FLAGS.config.to_dict())
    wandb_id = "{name}_{time}".format(
        name=name, time=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
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
        wandb.config.update(dict(save_dir=save_dir), allow_val_change=True)
        logging.info("Saving to %s", save_dir)
        tf.io.gfile.makedirs(save_dir)
        with tf.io.gfile.GFile(
            os.path.join(save_dir, "config.json"), "w"
        ) as config_file:
            config_file.write(FLAGS.config.to_json_best_effort())
    else:
        save_dir = None
        logging.info("save_dir not passed in, not saving checkpoints")

    # load action metadata
    with tf.io.gfile.GFile(
        tf.io.gfile.join(FLAGS.config.dataset_kwargs.data_path, "train/metadata.npy"),
        "rb",
    ) as f:
        action_proprio_metadata = np.load(f, allow_pickle=True).item()

    # load eval goals
    with tf.io.gfile.GFile(
        tf.io.gfile.join(FLAGS.config.dataset_kwargs.data_path, "val/eval_goals.npy"),
        "rb",
    ) as f:
        eval_goals = np.load(f, allow_pickle=True).item()

    horizon = (
        FLAGS.config.dataset_kwargs.window_size
        - FLAGS.config.model.policy_kwargs.pred_horizon
        + 1
    )

    # create sim environment
    eval_env = make_mujoco_gc_env(
        env_name=FLAGS.config.env_name,
        max_episode_steps=FLAGS.config.max_episode_steps,
        action_proprio_metadata=action_proprio_metadata,
        normalization_type=FLAGS.config.dataset_kwargs.action_proprio_normalization_type,
        save_video=FLAGS.config.save_video,
        save_video_dir=tf.io.gfile.join(save_dir, "videos"),
        save_video_prefix="eval",
        goals=eval_goals,
        horizon=horizon,
        pred_horizon=FLAGS.config.model.policy_kwargs.pred_horizon,
        exec_horizon=FLAGS.config.exec_horizon,
    )

    # load datasets
    logging.info(f"Loading data from {FLAGS.config.dataset_kwargs.data_path}")
    train_data = (
        make_sim_dataset(
            **FLAGS.config.dataset_kwargs,
            action_proprio_metadata=action_proprio_metadata,
            train=True,
        )
        .unbatch()
        .shuffle(FLAGS.config.shuffle_buffer_size)
        .repeat()
        .batch(FLAGS.config.batch_size)
    )
    val_data = (
        make_sim_dataset(
            **FLAGS.config.dataset_kwargs,
            action_proprio_metadata=action_proprio_metadata,
            train=False,
        )
        .unbatch()
        .shuffle(FLAGS.config.shuffle_buffer_size)
        .repeat()
        .batch(FLAGS.config.batch_size)
    )
    train_data_iter = train_data.iterator()
    val_data_iter = val_data.iterator()

    example_batch = next(train_data_iter)
    logging.info(f"Batch size: {example_batch['action'].shape[0]}")
    logging.info(f"Number of devices: {num_devices}")
    logging.info(
        f"Batch size per device: {example_batch['action'].shape[0] // num_devices}"
    )

    # truncate batch size for faster init
    example_batch = jax.tree_map(lambda x: x[:1], example_batch)

    # set up model, optimizer, loss
    model_def = create_model_def(
        action_dim=example_batch["action"].shape[-1],
        window_size=example_batch["observation"]["image_0"].shape[1],
        **FLAGS.config.model.to_dict(),
    )

    # pretrained weights to load
    pretrained_loaders = [weights_loaders[w] for w in FLAGS.config.pretrained_loaders]

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=FLAGS.config.optimizer.learning_rate,
        warmup_steps=FLAGS.config.optimizer.warmup_steps,
        decay_steps=FLAGS.config.optimizer.decay_steps,
        end_value=0.0,
    )
    tx = optax.adam(lr_schedule)
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, construct_rng = jax.random.split(rng)

    train_state = create_train_state(
        construct_rng,
        model_def,
        tx,
        init_args=(
            example_batch["observation"],
            example_batch["tasks"],
            example_batch["action"],
        ),
        pretrained_loaders=pretrained_loaders,
    )
    if FLAGS.config.resume_path is not None:
        train_state = checkpoints.restore_checkpoint(
            FLAGS.config.resume_path, target=train_state
        )
        checkpoint_step = int(train_state.step)
        logging.info("Restored checkpoint from %s", FLAGS.config.resume_path)
        if FLAGS.config.start_step is not None:
            start_step = FLAGS.config.start_step  # start_step overrides checkpoint
        else:
            start_step = checkpoint_step
        logging.info("Starting training from step %d", start_step)
    else:
        start_step = FLAGS.config.start_step or 0

    def loss_fn(params, state, batch, rng, train=True):
        info = state.apply_fn(
            {"params": params},
            batch["observation"],
            batch["tasks"],
            batch["action"],
            train=train,
            rngs={"dropout": rng},
        )
        return info["loss"], info

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
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    @partial(
        jax.jit,
        # state is replicated, batch is data-parallel
        in_shardings=(replicated_sharding, dp_sharding),
        out_shardings=replicated_sharding,
    )
    def eval_step(state, batch):
        loss, info = loss_fn(state.params, state, batch, state.rng, train=False)
        return info

    def wandb_log(info, step):
        wandb.log(flatten_dict(info, sep="/"), step=step)

    timer = Timer()
    for i in tqdm.tqdm(range(int(FLAGS.config.num_steps))):
        timer.tick("total")

        with timer("dataset"):
            batch = next(train_data_iter)

        with timer("train"):
            train_state, update_info = train_step(train_state, batch)

        if (i + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Validation...")
            timer.tick("val")
            metrics = []
            for _, batch in zip(range(FLAGS.config.num_val_batches), val_data_iter):
                metrics.append(eval_step(train_state, batch))
            metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
            wandb_log({"validation": metrics}, step=i)
            timer.tock("val")

            rng, policy_key = jax.random.split(rng)
            policy_fn = supply_rng(
                partial(
                    sample_actions,
                    state=train_state,
                    argmax=FLAGS.config.deterministic_eval,
                ),
                rng=policy_key,
            )

            logging.info("Evaluating...")
            timer.tick("evaluation")
            eval_env.goal_sampler = eval_goals
            eval_env.start_recording(
                FLAGS.config.num_episodes_per_video, FLAGS.config.num_episodes_per_row
            )
            eval_info = evaluate_gc(
                policy_fn,
                eval_env,
                num_episodes=FLAGS.config.eval_episodes,
            )
            wandb_log({f"evaluation": eval_info}, step=i)
            if FLAGS.config.save_video:
                eval_video = load_recorded_video(video_path=eval_env.current_save_path)
                wandb_log({"evaluation/video": eval_video}, step=i)
            timer.tock("evaluation")

        if (i + 1) % FLAGS.config.save_interval == 0 and save_dir is not None:
            logging.info("Saving checkpoint...")
            checkpoint_path = checkpoints.save_checkpoint(
                save_dir, train_state, step=i + 1, keep=1e6
            )
            logging.info("Saved checkpoint to %s", checkpoint_path)

        timer.tock("total")

        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_log(
                {"training": update_info, "timer": timer.get_average_times()}, step=i
            )


if __name__ == "__main__":
    app.run(main)
