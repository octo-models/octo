import numpy as np
import wandb
from flax.training import checkpoints
import jax
from orca.model import create_model_def
from orca.train_utils import create_train_state
import optax

act_pred_horizon = 1
obs_horizon = 1
im_height = 180
im_width = 320


def load_checkpoint(path, wandb_run_name):
    # load information from wandb
    api = wandb.Api()
    run = api.run(wandb_run_name)

    if act_pred_horizon is not None:
        example_actions = np.zeros((1, act_pred_horizon, 7), dtype=np.float32)
    else:
        example_actions = np.zeros((1, 7), dtype=np.float32)

    if obs_horizon is not None:
        example_obs = {
            "image": np.zeros((1, obs_horizon, im_height, im_width, 3), dtype=np.uint8)
        }
    else:
        example_obs = {"image": np.zeros((1, im_height, im_width, 3), dtype=np.uint8)}

    example_batch = {
        "observations": example_obs,
        "tasks": {"image": np.zeros((1, im_height, im_width, 3), dtype=np.uint8)},
        "actions": example_actions,
    }

    # create train_state from wandb config
    rng = jax.random.PRNGKey(0)
    rng, construct_rng = jax.random.split(rng)

    model_def = create_model_def(
        action_dim=example_batch["actions"].shape[-1],
        time_sequence_length=example_batch["observations"]["image"].shape[1],
        **run.config["model"],
    )

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=run.config["optimizer"]["learning_rate"],
        warmup_steps=run.config["optimizer"]["warmup_steps"],
        decay_steps=run.config["optimizer"]["decay_steps"],
        end_value=0.0,
    )
    tx = optax.adam(lr_schedule)
    train_state = create_train_state(
        construct_rng,
        model_def,
        tx,
        init_args=(
            example_batch["observations"],
            example_batch["tasks"],
            example_batch["actions"],
        ),
    )

    # hydrate train_state with parameters from checkpoint
    # breakpoint()
    train_state = checkpoints.restore_checkpoint(ckpt_dir=path, target=train_state)

    # load action metadata from wandb
    action_metadata = run.config["action"]
    action_mean = np.array(action_metadata["mean"])
    action_std = np.array(action_metadata["std"])

    return train_state, action_mean, action_std


train_state, action_mean, action_std = load_checkpoint(
    "/home/homer/checkpoint_10000", "orca/r2d2_wrist_cam_only_20230824_153023"
)
print(train_state)
