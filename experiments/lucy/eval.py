#!/usr/bin/env python3

from datetime import datetime
from functools import partial
import json
import os
from pathlib import Path, PurePath
import time
import pickle

from absl import app, flags, logging
import click
import cv2
import flax
import imageio
from PIL import Image
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from einops import rearrange
import wandb

# aloha
try:
    import sys; sys.path.append(os.path.join(os.getcwd(), 'aloha_pro/aloha_scripts/'))
    from aloha_pro.aloha_scripts.real_env import make_real_env
    from aloha_pro.aloha_scripts.robot_utils import move_grippers
except:
    print("Skipping real env import...")
from aloha_pro.aloha_scripts.constants import DT, PUPPET_GRIPPER_JOINT_OPEN
from aloha_pro.aloha_scripts.visualize_episodes import save_videos
from aloha_wrapper import AlohaGymEnv
from aloha_pro.aloha_scripts.sim_env import make_sim_env, sample_box_pose, sample_insertion_pose, BOX_POSE

from orca.utils.gym_wrappers import HistoryWrapper, RHCWrapper, UnnormalizeActionProprio
from orca.utils.pretrained_utils import ORCAModel

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    "checkpoint_weights_path", None, "Path to checkpoint", required=True
)
flags.DEFINE_multi_integer("checkpoint_step", None, "Checkpoint step", required=True)
flags.DEFINE_bool("add_jaxrlm_baseline", False, "Also compare to jaxrl_m baseline")


flags.DEFINE_string(
    "checkpoint_cache_dir",
    "/tmp/",
    "Where to cache checkpoints downloaded from GCS",
)
flags.DEFINE_string(
    "modality", "", "Either 'g', 'goal', 'l', 'language' (leave empty to prompt when running)"
)

flags.DEFINE_integer("im_size", None, "Image size", required=True)
flags.DEFINE_string("video_save_path", None, "Path to save video")
flags.DEFINE_integer("num_timesteps", 500, "num timesteps")
flags.DEFINE_bool("blocking", False, "Use the blocking controller")
flags.DEFINE_spaceseplist("goal_eep", [0.3, 0.0, 0.15], "Goal position")
flags.DEFINE_spaceseplist("initial_eep", [0.3, 0.0, 0.15], "Initial position")
flags.DEFINE_integer("horizon", 1, "Observation history length")
flags.DEFINE_integer("pred_horizon", 1, "Length of action sequence from model")
flags.DEFINE_integer("exec_horizon", 1, "Length of action sequence to execute")
flags.DEFINE_bool("deterministic", False, "Whether to sample action deterministically")
flags.DEFINE_float("temperature", 1.0, "Temperature for sampling actions")
flags.DEFINE_string("ip", "localhost", "IP address of the robot")
flags.DEFINE_integer("port", 5556, "Port of the robot")

# show image flag
flags.DEFINE_bool("show_image", False, "Show image")

# sim flags
flags.DEFINE_bool("is_sim", False, "Is simulation env")
flags.DEFINE_string("task_name", "sim_transfer_cube_scripted", "Task name")
flags.DEFINE_string("wandb_name", None, "Wandb log name")

##############################################################################

STEP_DURATION_MESSAGE = """
Bridge data was collected with non-blocking control and a step duration of 0.2s.
However, we relabel the actions to make it look like the data was collected with blocking control and we evaluate with blocking control.
We also use a step duration of 0.4s to reduce the jerkiness of the policy.
Be sure to change the step duration back to 0.2 if evaluating with non-blocking control.
"""
STEP_DURATION = 0.4
STICKY_GRIPPER_NUM_STEPS = 1
WORKSPACE_BOUNDS = [[0.1, -0.15, -0.01, -1.57, 0], [0.45, 0.25, 0.25, 1.57, 0]]
CAMERA_TOPICS = [{"name": "/blue/image_raw"}]
ENV_PARAMS = {
    "camera_topics": CAMERA_TOPICS,
    "override_workspace_boundaries": WORKSPACE_BOUNDS,
    "move_duration": STEP_DURATION,
}

##############################################################################


def maybe_download_checkpoint_from_gcs(cloud_path, step, save_path):
    if not cloud_path.startswith("gs://"):
        return cloud_path, step  # Actually on the local filesystem

    checkpoint_path = tf.io.gfile.join(cloud_path, f"{step}")
    norm_path = tf.io.gfile.join(cloud_path, "dataset_statistics*")
    config_path = tf.io.gfile.join(cloud_path, "config.json*")
    example_batch_path = tf.io.gfile.join(cloud_path, "example_batch.msgpack*")

    run_name = Path(cloud_path).name
    save_path = os.path.join(save_path, run_name)

    target_checkpoint_path = os.path.join(save_path, f"{step}")
    if os.path.exists(target_checkpoint_path):
        logging.warning(
            "Checkpoint already exists at %s, skipping download", target_checkpoint_path
        )
        return save_path, step
    os.makedirs(save_path, exist_ok=True)
    logging.warning("Downloading checkpoint and metadata to %s", save_path)

    os.system(f"sudo gsutil cp -r {checkpoint_path} {save_path}/")
    os.system(f"sudo gsutil cp {norm_path} {save_path}/")
    os.system(f"sudo gsutil cp {config_path} {save_path}/")
    os.system(f"sudo gsutil cp {example_batch_path} {save_path}/")

    return save_path, step


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped


@partial(jax.jit, static_argnames="argmax")
def sample_actions(
    pretrained_model: ORCAModel,
    observations,
    tasks,
    rng,
    argmax=False,
    temperature=1.0,
):

    # add batch dim to observations
    observations = jax.tree_map(lambda x: x[None], observations)
    logging.warning(
        "observations: %s", flax.core.pretty_repr(jax.tree_map(jnp.shape, observations))
    )
    logging.warning("tasks: %s", flax.core.pretty_repr(jax.tree_map(jnp.shape, tasks)))
    actions = pretrained_model.sample_actions(
        observations,
        tasks,
        rng=rng,
        argmax=argmax,
        temperature=temperature,
    )
    # remove batch dim
    return actions[0]


def load_checkpoint(weights_path, step):
    model = ORCAModel.load_pretrained(weights_path, step=int(step))

    policy_fn = supply_rng(
        partial(
            sample_actions,
            model,
            # argmax=FLAGS.deterministic, # Python version issue
            argmax=True,
            temperature=FLAGS.temperature,
        ),
    )
    return (policy_fn, model)

def main(_):
    assert len(FLAGS.checkpoint_weights_path) == len(FLAGS.checkpoint_step)
    # policies is a dict from run_name to policy function
    policies = {}
    for (checkpoint_weights_path, checkpoint_step,) in zip(
        FLAGS.checkpoint_weights_path,
        FLAGS.checkpoint_step,
    ):
        checkpoint_weights_path, checkpoint_step = maybe_download_checkpoint_from_gcs(
            checkpoint_weights_path,
            checkpoint_step,
            FLAGS.checkpoint_cache_dir,
        )
        assert tf.io.gfile.exists(checkpoint_weights_path), checkpoint_weights_path
        run_name = checkpoint_weights_path.rpartition("/")[2]
        policies[f"{run_name}-{checkpoint_step}"] = load_checkpoint(
            checkpoint_weights_path,
            checkpoint_step,
        )

    # ask for which policy to use
    if len(policies) == 1:
        policy_idx = 0
        print("Using default policy 0: ", list(policies.keys())[policy_idx])
    else:
        print("policies:")
        for i, name in enumerate(policies.keys()):
            print(f"{i}) {name}")
        policy_idx = click.prompt("Select policy", type=int)

    policy_name = list(policies.keys())[policy_idx]
    policy_fn, model = policies[policy_name]
    model: ORCAModel  # type hinting

    # set up environment
    if FLAGS.is_sim:
        env = make_sim_env(task_name=FLAGS.task_name)
        camera_names = ['top']
        env_max_reward = env.task.max_reward
        episode_returns = []
        highest_rewards = []
    else:
        env = make_real_env(init_node=True)
        from interbotix_xs_modules.arm import InterbotixManipulatorXS
        master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                        robot_name=f'master_left', init_node=False)
        master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                        robot_name=f'master_right', init_node=False)

        camera_names = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']

    # load normalization statistics
    metadata_path = os.path.join(
        checkpoint_weights_path, "dataset_statistics_aloha_sim_cube_scripted_dataset.json"
    )
    with open(metadata_path, "r") as f:
        norm_statistics = json.load(f)

    # wrap environment for history conditioning, action chunking and action/proprio norm/denorm
    env = AlohaGymEnv(env, camera_names)
    env = HistoryWrapper(env, FLAGS.horizon)
    env = RHCWrapper(env, FLAGS.exec_horizon)
    env = UnnormalizeActionProprio(env, norm_statistics, normalization_type="normal")

    query_frequency = FLAGS.exec_horizon # chunk size
    max_timesteps = FLAGS.num_timesteps // query_frequency
    num_rollouts = 50

    wandb_id = "{name}_{task}_chunk{chunk}_{time}".format(
        name=policy_name,
        task=FLAGS.task_name,
        chunk=FLAGS.exec_horizon,
        time=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    wandb.init(
        id=wandb_id,
        name=FLAGS.wandb_name,
        project="aloha_eval"
    )

    n_existing_rollouts = len([f for f in os.listdir(FLAGS.video_save_path) if f.startswith('video')])
    print(f'{n_existing_rollouts=}')

    for rollout_id in range(num_rollouts):
        if FLAGS.is_sim:
            ### set task
            if 'sim_transfer_cube' in FLAGS.task_name:
                BOX_POSE[0] = sample_box_pose() # used in sim reset
            elif 'sim_insertion' in FLAGS.task_name:
                BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset
            rewards = []

        obs, info = env.reset()
        image_list = [] # for visualization

        for t in range(max_timesteps):
            if t > 0:
                image_list.extend(info['images'])

            # query policy
            actions = policy_fn(obs, tasks={})
            target_qpos = np.array(actions)

            obs, reward, done, trunc, info = env.step(target_qpos)
            if FLAGS.is_sim:
                rewards.extend(info["rewards"])

        if FLAGS.is_sim:
            rewards = np.array(rewards)
            episode_return = np.sum(rewards[rewards!=None])
            episode_returns.append(episode_return)
            episode_highest_reward = np.max(rewards)
            highest_rewards.append(episode_highest_reward)
            print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, '
                  f'{env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')
        else:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right],
                          [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open

        print(f'Finished rollout {rollout_id}')
        if rollout_id < 3:
            # construct video, resize
            imgs = [np.array(Image.fromarray(img).resize(int(320*len(camera_names)), 240)) for img in image_list]
            video = np.stack(imgs)
            wandb.log({
                f"{policy_name}/rollout_{rollout_id}": wandb.Video(video.transpose(0, 3, 1, 2)[::2], fps=25)
            })

    if FLAGS.is_sim:
        success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
        avg_return = np.mean(episode_returns)
        summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
        for r in range(env_max_reward+1):
            more_or_equal_r = (np.array(highest_rewards) >= r).sum()
            more_or_equal_r_rate = more_or_equal_r / num_rollouts
            summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

        print(summary_str)

        wandb.log({
            f"{policy_name}/success_rate": success_rate,
            f"{policy_name}/average_return": avg_return,
        })

if __name__ == "__main__":
    app.run(main)
