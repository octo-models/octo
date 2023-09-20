import os
import sys
import traceback
import numpy as np
# import wandb
from absl import app, flags, logging
from flax.training import checkpoints
from PIL import Image

import cv2
from collections import deque
from datetime import datetime
import jax
import tensorflow as tf
from orca.model import create_model_def
from functools import partial
from orca.train_utils import create_train_state
import optax
import pickle as pkl


from r2d2.controllers.oculus_controller import VRPolicy
from r2d2.robot_env import RobotEnv
from r2d2.user_interface.data_collector import DataCollecter
from r2d2.user_interface.gui import RobotGUI


np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

CHECKPOINT_PATH = '/home/sirius/work/checkpoint_10000'
GOAL_IMAGE_PATH = '/home/sirius/work/checkpoint_10000/goal_hand_easy.png'
OBS_HORIZON = 1
DETERMINISTIC = True
ADIM = 7
IM_HEIGHT, IM_WIDTH = 180, 320


def resize_img(img):
    return cv2.resize(img, (IM_WIDTH, IM_HEIGHT), interpolation=cv2.INTER_AREA)

def load_img(path):
    return cv2.imread(path)


def unnormalize_action(action, mean, std):
    return action * std + mean

def stack_obs(obs):
    dict_list = {k: [dic[k] for dic in obs] for k in obs[0]}
    return jax.tree_map(
        lambda x: np.stack(x), dict_list, is_leaf=lambda x: type(x) == list
    )


def load_checkpoint(path):
    # our R2D2 machine cannot connect to internet when robot is on
    # so we pickle the wandb config dict and load offline
    with open(path + '/cfg.pkl', 'rb') as f:
        cfg = pkl.load(f)

    example_actions = np.zeros((1, ADIM), dtype=np.float32)
    example_obs = {
        "image": np.zeros(
            (1, OBS_HORIZON, IM_HEIGHT, IM_WIDTH, 3), dtype=np.uint8
        ),
    }
    example_batch = {
        "observations": example_obs,
        "goals": {
            "image": np.zeros((1, IM_HEIGHT, IM_WIDTH, 3), dtype=np.uint8)
        },
        "actions": example_actions,
    }

    # create train_state from wandb config
    rng = jax.random.PRNGKey(0)
    rng, construct_rng = jax.random.split(rng)

    model_def = create_model_def(
        action_dim=example_batch["actions"].shape[-1],
        time_sequence_length=example_batch["observations"]["image"].shape[1],
        **cfg["model"],
    )

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=cfg["optimizer"]["learning_rate"],
        warmup_steps=cfg["optimizer"]["warmup_steps"],
        decay_steps=cfg["optimizer"]["decay_steps"],
        end_value=0.0,
    )
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
    action_metadata = cfg['action']
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


class OrcaPolicy:
    def __init__(self, train_state, action_mean, action_std, goal_image):
        self.train_state = train_state
        self.action_mean = action_mean
        self.action_std = action_std
        self.goal = dict(image=goal_image)

        rng = jax.random.PRNGKey(0)
        self.rng, self.key = jax.random.split(rng)

    def forward(self, observation):
        image = resize_img(observation['image']['16291792_left'][:,:,:3])[None].copy()
        obs  = dict(image=image)
        actions = np.array(
                    sample_actions(
                        obs, self.goal, self.train_state, rng=self.key, argmax=DETERMINISTIC
                    )
                )
        return np.clip(unnormalize_action(actions, self.action_mean, self.action_std), -1, 1)



# get goal image
goal_image = resize_img(load_img(GOAL_IMAGE_PATH))

# load model and create policy
train_state, action_mean, action_std = load_checkpoint(CHECKPOINT_PATH)
policy = OrcaPolicy(train_state, action_mean, action_std, goal_image)


env = RobotEnv()
controller = VRPolicy()

data_col = DataCollecter(env = env, controller=controller, policy=policy, save_data=False)
RobotGUI(robot=data_col)
