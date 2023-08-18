from typing import Callable, Union

import gym
import numpy as np
from scipy.spatial.transform import Rotation


def convert_obs(obs):
    return {
        "image": obs["pixels"].astype(np.uint8),
        "proprio": np.concatenate(
            [
                obs["end_effector_pos"],
                Rotation.from_quat(obs["end_effector_quat"]).as_euler("xyz"),
                obs["right_finger_qpos"],
                obs["left_finger_qpos"],
            ]
        ),
    }


def filter_info_keys(info):
    keep_keys = [
        "place_success",
    ]
    return {k: v for k, v in info.items() if k in keep_keys}


class GCMujocoWrapper(gym.Wrapper):
    """
    Goal-conditioned wrapper for Mujoco sim environments.

    When reset is called, a new goal is sampled from the goal_sampler. The
    goal_sampler can either be a set of evaluation goals or a function that
    returns a goal (e.g an affordance model).
    """

    def __init__(
        self,
        env: gym.Env,
        goal_sampler: Union[np.ndarray, Callable],
    ):
        super().__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=np.zeros((128, 128, 3)),
                    high=255 * np.ones((128, 128, 3)),
                    dtype=np.uint8,
                ),
                "proprio": gym.spaces.Box(
                    low=np.zeros((8,)),
                    high=np.ones((8,)),
                    dtype=np.uint8,
                ),
            }
        )
        self.current_goal = None
        self.goal_sampler = goal_sampler

    def step(self, *args):
        obs, reward, done, trunc, info = self.env.step(*args)
        info = filter_info_keys(info)
        return (
            convert_obs(obs),
            reward,
            done,
            trunc,
            {"goal": self.current_goal, **info},
        )

    def reset(self, **kwargs):
        if not callable(self.goal_sampler):
            idx = np.random.randint(len(self.goal_sampler["observations"]["image"]))
            goal_image = self.goal_sampler["observations"]["image"][idx]
            original_object_positions = self.goal_sampler["infos"]["initial_positions"][
                idx
            ]
            # original_object_quats = self.goal_sampler["infos"]["initial_quats"][idx]
            target_position = self.goal_sampler["infos"]["target_position"][idx]
            object_names = self.goal_sampler["infos"]["object_names"][idx]
            target_object = self.goal_sampler["infos"]["target_object"][idx]
            self.env.task.change_props(object_names)
            self.env.task.init_prop_poses = original_object_positions
            self.env.task.target_pos = target_position
            self.env.target_obj = target_object
            obs, info = self.env.reset()
            obs = convert_obs(obs)
        else:
            obs, info = self.env.reset()
            obs = convert_obs(obs)
            goal_image = self.goal_sampler(obs)

        goal = {"image": goal_image}

        self.current_goal = goal

        info = filter_info_keys(info)
        return obs, {"goal": goal, **info}
