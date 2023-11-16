#!/usr/bin/env python3

from datetime import datetime
from functools import partial
import json
import os
from pathlib import Path, PurePath
import time

from absl import app, flags, logging
import click
import cv2
import flax
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

# bridge_data_robot imports
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs, WidowXStatus
from widowx_wrapper import convert_obs, state_to_eep, wait_for_obs, WidowXGym

from orca.utils.gym_wrappers import HistoryWrapper, RHCWrapper, TemporalEnsembleWrapper
from orca.utils.pretrained_utils import PretrainedModel

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
flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
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
    norm_path = tf.io.gfile.join(cloud_path, "action_proprio*")
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

    os.system(f"gsutil cp -r {checkpoint_path} {save_path}/")
    os.system(f"gsutil cp {norm_path} {save_path}/")
    os.system(f"gsutil cp {config_path} {save_path}/")
    os.system(f"gsutil cp {example_batch_path} {save_path}/")

    return save_path, step


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped


@partial(jax.jit, static_argnames="argmax")
def sample_actions(
    pretrained_model: PretrainedModel,
    observations,
    tasks,
    mean,
    std,
    rng,
    argmax=False,
    temperature=1.0,
):
    # add batch dim to observations
    observations = jax.tree_map(lambda x: x[None], observations)
    # tasks = jax.tree_map(lambda x: x[None], tasks)
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
    return actions[0] * std + mean


def load_jaxrlm_checkpoint(
    weights_path="/mount/harddrive/homer/bridgev2_packaged/bridgev2policies/gcbc_256/checkpoint_300000/",
    config_path="/mount/harddrive/homer/bridgev2_packaged/bridgev2policies/gcbc_256/gcbc_256_config.json",
    code_path="/mount/harddrive/homer/bridgev2_packaged/bridgev2policies/bridge_data_v2.zip",
):
    from codesave import UniqueCodebase

    with UniqueCodebase(code_path) as cs:
        pretrained_utils = cs.import_module("jaxrl_m.pretrained_utils")
        loaded = pretrained_utils.load_checkpoint(
            weights_path, config_path, im_size=256
        )
        # loaded contains: {
        # "agent": jaxrlm Agent,
        # "policy_fn": callable taking in observation and goal inputs and outputs **unnormalized** actions,
        # "normalization_stats": {"action": {"mean": [7], "std": [7]}}
        # "obs_horizon": int
        # }

    class Dummy:
        def create_tasks(self, goals):
            return goals.copy()

    def new_policy_fn(observations, goals):
        observations = {"image": observations["image_0"]}
        goals = {"image": goals["image_0"]}
        return loaded["policy_fn"](observations, goals)

    return new_policy_fn, Dummy()


def load_checkpoint(weights_path, step):
    model = PretrainedModel.load_pretrained(weights_path, step=int(step))
    metadata_path = os.path.join(
        weights_path, "action_proprio_metadata_bridge_dataset.json"
    )
    with open(metadata_path, "r") as f:
        action_proprio_metadata = json.load(f)
    action_mean = jnp.array(action_proprio_metadata["action"]["mean"])
    action_std = jnp.array(action_proprio_metadata["action"]["std"])

    policy_fn = supply_rng(
        partial(
            sample_actions,
            model,
            argmax=FLAGS.deterministic,
            mean=action_mean,
            std=action_std,
            temperature=FLAGS.temperature,
        ),
    )
    return (policy_fn, model)


def main(_):
    assert len(FLAGS.checkpoint_weights_path) == len(FLAGS.checkpoint_step)
    FLAGS.modality = FLAGS.modality[:1]
    assert FLAGS.modality in ["g", "l", ""]
    if not FLAGS.blocking:
        assert STEP_DURATION == 0.2, STEP_DURATION_MESSAGE

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
    if FLAGS.add_jaxrlm_baseline:
        policies["jaxrl_gcbc"] = load_jaxrlm_checkpoint()

    if FLAGS.initial_eep is not None:
        assert isinstance(FLAGS.initial_eep, list)
        initial_eep = [float(e) for e in FLAGS.initial_eep]
        start_state = np.concatenate([initial_eep, [0, 0, 0, 1]])
    else:
        start_state = None

    # set up environment
    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(ENV_PARAMS)
    env_params["state_state"] = list(start_state)
    widowx_client = WidowXClient(host=FLAGS.ip, port=FLAGS.port)
    widowx_client.init(env_params, image_size=FLAGS.im_size)
    env = WidowXGym(
        widowx_client, FLAGS.im_size, FLAGS.blocking, STICKY_GRIPPER_NUM_STEPS
    )
    env = HistoryWrapper(env, FLAGS.horizon)
    # env = TemporalEnsembleWrapper(env, FLAGS.pred_horizon)
    env = RHCWrapper(env, FLAGS.pred_horizon, FLAGS.exec_horizon)

    goal_image = jnp.zeros((FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8)
    goal_instruction = ""

    # goal sampling loop
    while True:
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
        model: PretrainedModel  # type hinting

        modality = FLAGS.modality
        if not modality:
            modality = click.prompt(
                "Language or goal image?", type=click.Choice(["l", "g"])
            )

        if modality == "g":
            if click.confirm("Take a new goal?", default=True):
                assert isinstance(FLAGS.goal_eep, list)
                _eep = [float(e) for e in FLAGS.goal_eep]
                goal_eep = state_to_eep(_eep, 0)
                widowx_client.move_gripper(1.0)  # open gripper

                move_status = None
                while move_status != WidowXStatus.SUCCESS:
                    move_status = widowx_client.move(goal_eep, duration=1.5)

                input("Press [Enter] when ready for taking the goal image. ")
                obs = wait_for_obs(widowx_client)
                goal = jax.tree_map(lambda x: x[None], convert_obs(obs, FLAGS.im_size))

            task = model.create_tasks(goals=goal)
            goal_image = goal["image_0"][0]
            goal_instruction = ""
        elif modality == "l":
            print("Current instruction: ", goal_instruction)
            if click.confirm("Take a new instruction?", default=True):
                text = input("Instruction?")

            task = model.create_tasks(texts=[text])
            goal_instruction = text
            goal_image = jnp.zeros_like(goal_image)
        else:
            raise NotImplementedError()

        # reset env
        widowx_client.reset()
        time.sleep(2.5)
        obs, _ = env.reset()

        input("Press [Enter] to start.")

        # do rollout
        last_tstep = time.time()
        images = []
        goals = []
        t = 0
        while t < FLAGS.num_timesteps:
            if time.time() > last_tstep + STEP_DURATION or FLAGS.blocking:
                last_tstep = time.time()

                # save images
                images.append(obs["image_0"][-1])
                goals.append(goal_image)

                if FLAGS.show_image:
                    bgr_img = cv2.cvtColor(obs["full_image"][-1], cv2.COLOR_RGB2BGR)
                    cv2.imshow("img_view", bgr_img)
                    cv2.waitKey(10)

                # get action
                action = np.array(policy_fn(obs, task))

                # perform environment step
                obs, _, _, truncated, _ = env.step(action)

                t += 1

                if truncated:
                    break

        # save video
        if FLAGS.video_save_path is not None:
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                FLAGS.video_save_path,
                f"{curr_time}_{policy_name}_sticky_{STICKY_GRIPPER_NUM_STEPS}.mp4",
            )
            video = np.concatenate([np.stack(goals), np.stack(images)], axis=1)
            imageio.mimsave(save_path, video, fps=1.0 / STEP_DURATION * 3)


if __name__ == "__main__":
    app.run(main)
