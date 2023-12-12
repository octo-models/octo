"""
This script demonstrates how to finetune ORCA to a new observation space (single camera + proprio)
and new action space (bimanual) using a simulated ALOHA cube handover dataset (https://tonyzhaozh.github.io/aloha/).
"""
import json

from absl import app, flags, logging
import flax
import jax
import optax
import tensorflow as tf
import tqdm
import wandb

from orca.data.dataset import make_single_dataset
from orca.data.oxe.oxe_dataset_configs import ActionEncoding, StateEncoding
from orca.model.components.action_heads import L1ActionHead
from orca.model.components.tokenizers import LowdimObsTokenizer
from orca.model.orca_model import ORCAModel
from orca.utils.jax_utils import initialize_compilation_cache
from orca.utils.spec import ModuleSpec
from orca.utils.train_callbacks import SaveCallback
from orca.utils.train_utils import (
    freeze_weights,
    merge_params,
    process_text,
    TrainState,
)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "pretrained_path", None, "Path to pre-trained ORCA checkpoint directory."
)
flags.DEFINE_string("data_dir", None, "Path to finetuning dataset, in RLDS format.")
flags.DEFINE_string("save_dir", None, "Directory for saving finetuning checkpoints.")
flags.DEFINE_bool(
    "freeze_transformer",
    False,
    "Whether pre-trained transformer weights should be frozen.",
)


def main(_):
    initialize_compilation_cache()
    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    # setup wandb for logging
    wandb.init(name="finetune_aloha", project="orca")

    # load pre-trained model
    logging.info("Loading pre-trained model...")
    pretrained_model = ORCAModel.load_pretrained(FLAGS.pretrained_path)

    # make finetuning dataset
    # apply Gaussian normalization, load chunks of 50 actions since we'll train with action chunking
    # delete goal images in the data loader since we will train a language-conditioned-only policy
    # TODO: directly load this from raw data to make it less opaque?
    logging.info("Loading finetuning dataset...")
    dataset = make_single_dataset(
        dataset_kwargs=dict(
            name="aloha_sim_cube_scripted_dataset",
            data_dir=FLAGS.data_dir,
            image_obs_keys=["top"],
            state_obs_keys=["state"],
            state_encoding=StateEncoding.JOINT_BIMANUAL,
            action_encoding=ActionEncoding.JOINT_POS_BIMANUAL,
            action_proprio_normalization_type="normal",
        ),
        traj_transform_kwargs=dict(
            window_size=1,
            additional_action_window_size=49,  # so we get 50 actions for our action chunk
            goal_relabeling_strategy="no_image_conditioning",  # train only language-conditioned policy
            action_encoding=ActionEncoding.JOINT_POS_BIMANUAL,
        ),
        frame_transform_kwargs=dict(
            resize_size=(256, 256),
            task_augment_strategy="delete_task_conditioning",
            task_augment_kwargs=dict(
                delete_key_groups_probs=[
                    (["image_.*"], 1.0)
                ],  # delete goal images in task definition
            ),
        ),
        train=True,
    )
    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(100000)  # can reduce this if RAM consumption too high
        .batch(128)
        .iterator()
    )

    # run text tokenizer over batch (this needs to happen before training / sharding) + delete unused keys
    text_processor = pretrained_model.text_processor

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    train_data_iter = map(process_batch, train_data_iter)
    example_batch = next(train_data_iter)

    # load pre-training config and modify --> remove wrist cam, add proprio input, change action head
    # following Zhao et al. we use "action chunks" of length 50 and L1 loss for ALOHA
    config = ORCAModel.load_config(FLAGS.pretrained_path)
    del config["model"]["observation_tokenizers"]["wrist"]
    ###
    config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
        LowdimObsTokenizer,
        n_bins=256,
        bin_type="normal",
        low=-2.0,
        high=2.0,
        obs_keys=["proprio"],
    )
    # Fully override the old action head with a new one (for smaller changes, you can use update_module_config)
    config["model"]["heads"]["action"] = ModuleSpec.create(
        L1ActionHead,
        pred_horizon=50,
        action_dim=14,
        readout_key="obs",  # TODO: switch this to "action" once we add readout keys
    )

    # initialize weights for modified ORCA model, then merge in all applicable pre-trained weights
    # new position encodings for proprio inputs & weights for new action head will remain "from scratch"
    logging.info("Updating model for new observation & action space...")
    model = ORCAModel.from_config(config, example_batch)
    merged_params = merge_params(model.params, pretrained_model.params)
    # can perform any additional parameter surgery here...
    # ...
    model = model.replace(params=merged_params)
    del pretrained_model

    # create optimizer & train_state, optionally freeze keys for pre-trained transformer
    # train_state bundles parameters & optimizers
    model_def = model.model_def
    learning_rate = optax.join_schedules(
        optax.linear_schedule(0, 3e-5, 100), optax.constant_schedule(3e-5)
    )
    tx = optax.adamw(learning_rate)
    frozen_keys = model.config["optimizer"]["frozen_keys"]
    if FLAGS.freeze_transformer:
        frozen_keys.append("BlockTransformer_0")
    tx = freeze_weights(tx, model.params, frozen_keys)
    train_state = TrainState.create(
        apply_fn=model_def.apply,
        params=model.params,
        tx=tx,
        rng=jax.random.PRNGKey(1234),
    )

    # define loss function and train step
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

    @jax.jit
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, state, batch, dropout_rng, train=True
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    # save all info for loading of finetuned model
    # (config, normalization stats, example batch)
    save_callback = SaveCallback(FLAGS.save_dir)
    with save_callback.open("config.json", "w") as config_file:
        config_file.write(config.to_json_best_effort())
    with save_callback.open("dataset_statistics.json", "w") as f:
        stats = jax.tree_map(lambda x: x.tolist(), dataset.dataset_statistics)
        json.dump(stats, f)
    with save_callback.open("example_batch.msgpack", "wb") as f:
        f.write(flax.serialization.msgpack_serialize(example_batch))

    # run finetuning loop
    logging.info("Starting finetuning...")
    for i in tqdm.tqdm(range(2000), total=2000, dynamic_ncols=True):
        batch = next(train_data_iter)
        train_state, update_info = train_step(train_state, batch)
        if (i + 1) % 100 == 0:
            update_info = jax.device_get(update_info)
            wandb.log(
                flax.traverse_util.flatten_dict({"training": update_info}, sep="/"),
                step=i,
            )
        if (i + 1) % 500 == 0:
            # save checkpoint
            save_callback(train_state, i)


if __name__ == "__main__":
    app.run(main)
