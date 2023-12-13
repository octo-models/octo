"""
This script shows how to evaluate a finetuned ORCA model on a real WidowX robot.
To reproduce the robot setup, follow the instructions at https://rail-berkeley.github.io/bridgedata/
To install the robot controller, please follow the instructions here: https://github.com/rail-berkeley/bridge_data_robot
Even if you don't plan to run on a WidowX robot, this script demonstrates the general layout of a robot eval loop.
"""

from datetime import datetime
from functools import partial
import os
import time

from absl import app, flags, logging
import click
import cv2
from envs.widowx_env import convert_obs, state_to_eep, wait_for_obs, WidowXGym
import flax
import imageio
import jax
import jax.numpy as jnp
import numpy as np
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs, WidowXStatus

from orca.model.orca_model import ORCAModel
from orca.utils.gym_wrappers import HistoryWrapper, RHCWrapper, UnnormalizeActionProprio
from orca.utils.gym_wrappers import TemporalEnsembleWrapper  # noqa: F401

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "checkpoint_weights_path", None, "Path to checkpoint", required=True
)
flags.DEFINE_integer("checkpoint_step", None, "Checkpoint step", required=True)

# custom to bridge_data_robot
flags.DEFINE_string("ip", "localhost", "IP address of the robot")
flags.DEFINE_integer("port", 5556, "Port of the robot")
flags.DEFINE_spaceseplist("goal_eep", [0.3, 0.0, 0.15], "Goal position")
flags.DEFINE_spaceseplist("initial_eep", [0.3, 0.0, 0.15], "Initial position")
flags.DEFINE_bool("blocking", False, "Use the blocking controller")

flags.DEFINE_string(
    "modality",
    "",
    "Either 'g', 'goal', 'l', 'language' (leave empty to prompt when running)",
)
flags.DEFINE_integer("im_size", None, "Image size", required=True)
flags.DEFINE_string("video_save_path", None, "Path to save video")
flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_integer("horizon", 1, "Observation history length")
flags.DEFINE_integer("pred_horizon", 1, "Length of action sequence from model")
flags.DEFINE_integer("exec_horizon", 1, "Length of action sequence to execute")
flags.DEFINE_float("temperature", 1.0, "Temperature for sampling actions")
flags.DEFINE_bool("deterministic", False, "Whether to sample action deterministically")

# show image flag
flags.DEFINE_bool("show_image", False, "Show image")

##############################################################################

STEP_DURATION_MESSAGE = """
Bridge data was collected with non-blocking control and a step duration of 0.2s.
However, we relabel the actions to make it look like the data was collected with
blocking control and we evaluate with blocking control.
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


def main(_):
    # set up the widowx client
    if FLAGS.initial_eep is not None:
        assert isinstance(FLAGS.initial_eep, list)
        initial_eep = [float(e) for e in FLAGS.initial_eep]
        start_state = np.concatenate([initial_eep, [0, 0, 0, 1]])
    else:
        start_state = None

    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params.update(ENV_PARAMS)
    env_params["state_state"] = list(start_state)
    widowx_client = WidowXClient(host=FLAGS.ip, port=FLAGS.port)
    widowx_client.init(env_params, image_size=FLAGS.im_size)
    env = WidowXGym(
        widowx_client, FLAGS.im_size, FLAGS.blocking, STICKY_GRIPPER_NUM_STEPS
    )
    if not FLAGS.blocking:
        assert STEP_DURATION == 0.2, STEP_DURATION_MESSAGE

    # load models
    model = ORCAModel.load_pretrained(
        FLAGS.checkpoint_weights_path,
        FLAGS.checkpoint_step,
    )

    # wrap the robot environment
    env = UnnormalizeActionProprio(
        env, model.dataset_statistics, normalization_type="normal"
    )
    env = HistoryWrapper(env, FLAGS.horizon)
    # env = TemporalEnsembleWrapper(env, FLAGS.pred_horizon)
    env = RHCWrapper(env, FLAGS.exec_horizon)

    # create policy functions
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
            "observations: %s",
            flax.core.pretty_repr(jax.tree_map(jnp.shape, observations)),
        )
        logging.warning(
            "tasks: %s", flax.core.pretty_repr(jax.tree_map(jnp.shape, tasks))
        )
        actions = pretrained_model.sample_actions(
            observations,
            tasks,
            rng=rng,
            argmax=argmax,
            temperature=temperature,
        )
        # remove batch dim
        return actions[0]

    policy_fn = partial(
        sample_actions,
        model,
        argmax=FLAGS.deterministic,
        temperature=FLAGS.temperature,
    )

    goal_image = jnp.zeros((FLAGS.im_size, FLAGS.im_size, 3), dtype=np.uint8)
    goal_instruction = ""

    modality = FLAGS.modality[:1]
    if modality not in ["g", "l", ""]:
        modality = ""

    # goal sampling loop
    while True:
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
                obs = convert_obs(obs, FLAGS.im_size)
                goal = jax.tree_map(lambda x: x[None], obs)

            task = model.create_tasks(goals=goal)
            goal_image = goal["image_primary"][0]
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
        obs, _ = env.reset()
        time.sleep(2.0)

        input("Press [Enter] to start.")

        # do rollout
        last_tstep = time.time()
        images = []
        goals = []
        t = 0
        while t < FLAGS.num_timesteps:
            if time.time() > last_tstep + STEP_DURATION:
                last_tstep = time.time()

                # save images
                images.append(obs["image_primary"][-1])
                goals.append(goal_image)

                if FLAGS.show_image:
                    bgr_img = cv2.cvtColor(obs["image_primary"][-1], cv2.COLOR_RGB2BGR)
                    cv2.imshow("img_view", bgr_img)
                    cv2.waitKey(20)

                # get action
                forward_pass_time = time.time()
                action = np.array(policy_fn(obs, task), dtype=np.float64)
                print("forward pass time: ", time.time() - forward_pass_time)

                # perform environment step
                start_time = time.time()
                obs, _, _, truncated, _ = env.step(action)
                print("step time: ", time.time() - start_time)

                t += 1

                if truncated:
                    break

        # save video
        if FLAGS.video_save_path is not None:
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                FLAGS.video_save_path,
                f"{curr_time}.mp4",
            )
            video = np.concatenate([np.stack(goals), np.stack(images)], axis=1)
            imageio.mimsave(save_path, video, fps=1.0 / STEP_DURATION * 3)


if __name__ == "__main__":
    app.run(main)
