import os
import sys
import traceback
import numpy as np
import wandb
from absl import app, flags, logging
from flax.training import checkpoints
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import time
from collections import deque
from datetime import datetime
import jax
import matplotlib.pyplot as plt
import tensorflow as tf
from orca.model import create_model_def
from functools import partial
from orca.train_utils import create_train_state
import optax

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_multi_string("checkpoint_path", None, "Path to checkpoint", required=True)
flags.DEFINE_multi_string("wandb_run_name", None, "Name of wandb run", required=True)
flags.DEFINE_string("video_save_path", None, "Path to save video")
flags.DEFINE_string(
    "goal_image_path",
    None,
    "Path to a single goal image",
)
flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_bool("blocking", False, "Use the blocking controller")
flags.DEFINE_spaceseplist("goal_eep", None, "Goal position")
flags.DEFINE_spaceseplist("initial_eep", None, "Initial position")
flags.DEFINE_integer("obs_horizon", None, "Observation history length")
flags.DEFINE_integer("act_exec_horizon", 1, "Action sequence length")
flags.DEFINE_integer("act_pred_horizon", None, "Action sequence length")
flags.DEFINE_bool("deterministic", True, "Whether to sample action deterministically")

STEP_DURATION = 0.2
NO_PITCH_ROLL = False
NO_YAW = False
STICKY_GRIPPER_NUM_STEPS = 1

FIXED_STD = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

WORKSPACE_BOUNDS = np.array([[0.1, -0.15, -0.1, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]])

def unnormalize_action(action, mean, std):
    return action * std + mean

def stack_obs(obs):
    dict_list = {k: [dic[k] for dic in obs] for k in obs[0]}
    return jax.tree_map(
        lambda x: np.stack(x), dict_list, is_leaf=lambda x: type(x) == list
    )


def load_checkpoint(path, wandb_run_name):
    # load information from wandb
    api = wandb.Api()
    run = api.run(wandb_run_name)

    if FLAGS.act_pred_horizon is not None:
        example_actions = np.zeros((1, FLAGS.act_pred_horizon, 7), dtype=np.float32)
    else:
        example_actions = np.zeros((1, 7), dtype=np.float32)

    if FLAGS.obs_horizon is not None:
        # fix hardcoded sizes
        example_obs = {
            "image": np.zeros(
                (1, FLAGS.obs_horizon, 180, 320, 3), dtype=np.uint8
            ),
        }
    else:
        example_obs = {
            "image": np.zeros((1, FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8)
        }

    example_batch = {
        "observations": example_obs,
        "goals": {
            "image": np.zeros((1, 180, 320, 3), dtype=np.uint8)
        },
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
    import ipdb; ipdb.set_trace()
    tx = optax.adam(lr_schedule)
    train_state = create_train_state(
        construct_rng,
        model_def,
        tx,
        init_args=(
            example_batch["observations"],
            example_batch["goals"],
            example_batch["actions"],
        ),
    )

    # hydrate train_state with parameters from checkpoint
    train_state = checkpoints.restore_checkpoint(path, train_state)

    # load action metadata from wandb
    action_metadata = run.config['action']
    action_mean = np.array(action_metadata["mean"])
    action_std = np.array(action_metadata["std"])

    return train_state, action_mean, action_std

@partial(jax.jit, static_argnames="argmax")
def sample_actions(observations, goals, state, rng, argmax=False, temperature=1.0):
    observations = jax.tree_map(lambda x: x[None], observations)
    goals = jax.tree_map(lambda x: x[None], goals)
    actions = state.apply_fn(
        {"params": state.params},
        observations,
        goals,
        train=False,
        argmax=argmax,
        rng=rng,
        temperature=temperature,
        method="predict_action",
    )
    return actions[0]

def main(_):
    assert len(FLAGS.checkpoint_path) == len(FLAGS.wandb_run_name)

    # policies is a dict from run_name to (train_state, action_mean, action_std)
    policies = {}
    for checkpoint_path, wandb_run_name in zip(
        FLAGS.checkpoint_path, FLAGS.wandb_run_name
    ):
        assert tf.io.gfile.exists(checkpoint_path), checkpoint_path
        train_state, action_mean, action_std = load_checkpoint(
            checkpoint_path, wandb_run_name
        )

        checkpoint_num = int(checkpoint_path.split("_")[-1])
        run_name = wandb_run_name.split("/")[-1]
        policies[f"{run_name}-{checkpoint_num}"] = (train_state, action_mean, action_std)


    # test getting actions
    dummy_goal = dict(image=np.ones((180, 320, 3), dtype=np.uint8))
    dummy_obs  = dict(image=np.ones((FLAGS.obs_horizon, 180, 320, 3), dtype=np.uint8))
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    actions = np.array(
                        sample_actions(
                            dummy_obs, dummy_goal, train_state, rng=key, argmax=FLAGS.deterministic
                        )
                    )
    print(actions)
    import ipdb; ipdb.set_trace()

    if FLAGS.initial_eep is not None:
        assert isinstance(FLAGS.initial_eep, list)
        initial_eep = [float(e) for e in FLAGS.initial_eep]
        start_state = np.concatenate([initial_eep, [0, 0, 0, 1]])
    else:
        start_state = None

    # set up environment
    env_params = {
        "fix_zangle": 0.1,
        "move_duration": 0.2,
        "adaptive_wait": True,
        "move_to_rand_start_freq": 1,
        "override_workspace_boundaries": WORKSPACE_BOUNDS,
        "action_clipping": "xyz",
        "catch_environment_except": False,
        "start_state": start_state,
        "return_full_image": False,
        "camera_topics": [IMTopic("/D435/color/image_raw", flip=True)],
    }
    env = BridgeDataRailRLPrivateWidowX(env_params, fixed_image_size=FLAGS.im_size)

    # load image goal
    image_goal = None
    if FLAGS.goal_image_path is not None:
        image_goal = np.array(Image.open(FLAGS.goal_image_path))

    # goal sampling loop
    while True:
        # ask for new goal
        if image_goal is None:
            print("Taking a new goal...")
            ch = "y"
        else:
            ch = input("Taking a new goal? [y/n]")
        if ch == "y":
            if FLAGS.goal_eep is not None:
                assert isinstance(FLAGS.goal_eep, list)
                goal_eep = [float(e) for e in FLAGS.goal_eep]
            else:
                low_bound = WORKSPACE_BOUNDS[0][:3] + 0.03
                high_bound = WORKSPACE_BOUNDS[1][:3] - 0.03
                goal_eep = np.random.uniform(low_bound, high_bound)
            env._controller.open_gripper(True)
            try:
                env._controller.move_to_state(goal_eep, 0, duration=1.5)
                env._reset_previous_qpos()
            except Exception as e:
                continue
            input("Press [Enter] when ready for taking the goal image. ")
            obs = env._get_obs()
            image_goal = (
                obs["image"].reshape(3, FLAGS.im_size, FLAGS.im_size).transpose(1, 2, 0)
                * 255
            ).astype(np.uint8)

        # ask for which policy to use
        if len(policies) == 1:
            policy_idx = 0
            input("Press [Enter] to start.")
        else:
            print("policies:")
            for i, name in enumerate(policies.keys()):
                print(f"{i}) {name}")
            policy_idx = int(input("select policy: "))

        policy_name = list(policies.keys())[policy_idx]
        train_state, action_mean, action_std = policies[policy_name]
        try:
            env.reset()
            env.start()
        except Exception as e:
            continue

        # move to initial position
        try:
            if FLAGS.initial_eep is not None:
                assert isinstance(FLAGS.initial_eep, list)
                initial_eep = [float(e) for e in FLAGS.initial_eep]
                env._controller.move_to_state(initial_eep, 0, duration=1.5)
                env._reset_previous_qpos()
        except Exception as e:
            continue

        # do rollout
        rng = jax.random.PRNGKey(0)
        obs = env._get_obs()
        last_tstep = time.time()
        images = []
        t = 0
        if FLAGS.obs_horizon is not None:
            obs_hist = deque(maxlen=FLAGS.obs_horizon)
        # keep track of our own gripper state to implement sticky gripper
        is_gripper_closed = False
        num_consecutive_gripper_change_actions = 0
        try:
            while t < FLAGS.num_timesteps:
                if time.time() > last_tstep + STEP_DURATION or FLAGS.blocking:
                    image_obs = (
                        obs["image"]
                        .reshape(3, FLAGS.im_size, FLAGS.im_size)
                        .transpose(1, 2, 0)
                        * 255
                    ).astype(np.uint8)
                    obs = {"image": image_obs, "proprio": obs["state"]}
                    goal_obs = {
                        "image": image_goal,
                    }
                    if FLAGS.obs_horizon is not None:
                        if len(obs_hist) == 0:
                            obs_hist.extend([obs] * FLAGS.obs_horizon)
                        else:
                            obs_hist.append(obs)
                        obs = stack_obs(obs_hist)

                    last_tstep = time.time()

                    rng, key = jax.random.split(rng)
                    actions = np.array(
                        sample_actions(
                            obs, goal_obs, train_state, rng=key, argmax=FLAGS.deterministic
                        )
                    )
                    if len(actions.shape) == 1:
                        actions = actions[None]
                    for i in range(FLAGS.act_exec_horizon):
                        action = actions[i]
                        action = unnormalize_action(action, action_mean, action_std)
                        action += np.random.normal(0, FIXED_STD)

                        # sticky gripper logic
                        if (action[-1] < 0.5) != is_gripper_closed:
                            num_consecutive_gripper_change_actions += 1
                        else:
                            num_consecutive_gripper_change_actions = 0

                        if (
                            num_consecutive_gripper_change_actions
                            >= STICKY_GRIPPER_NUM_STEPS
                        ):
                            is_gripper_closed = not is_gripper_closed
                            num_consecutive_gripper_change_actions = 0

                        action[-1] = 0.0 if is_gripper_closed else 1.0

                        # remove degrees of freedom
                        if NO_PITCH_ROLL:
                            action[3] = 0
                            action[4] = 0
                        if NO_YAW:
                            action[5] = 0

                        # perform environment step
                        obs, rew, done, info = env.step(
                            action, last_tstep + STEP_DURATION, blocking=FLAGS.blocking
                        )

                        # save image
                        image_formatted = np.concatenate(
                            (image_goal, image_obs), axis=0
                        )
                        images.append(Image.fromarray(image_formatted))

                        t += 1
        except Exception as e:
            print(traceback.format_exc(), file=sys.stderr)

        # save video
        if FLAGS.video_save_path is not None:
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                FLAGS.video_save_path,
                f"{curr_time}_{policy_name}_sticky_{STICKY_GRIPPER_NUM_STEPS}.gif",
            )
            print(f"Saving Video at {save_path}")
            images[0].save(
                save_path,
                format="GIF",
                append_images=images[1:],
                save_all=True,
                duration=200,
                loop=0,
            )


if __name__ == "__main__":
    app.run(main)
