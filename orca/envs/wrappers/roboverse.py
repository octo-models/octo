from typing import Callable, Union

import gym
import numpy as np


def convert_obs(obs, img_dim):
    return {
        "image": (obs["image"].reshape(img_dim, img_dim, 3) * 255).astype(np.uint8),
        "proprio": obs["state"].astype(np.float32),
    }


class RoboverseWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=np.zeros((env.observation_img_dim, env.observation_img_dim, 3)),
                    high=255
                    * np.ones((env.observation_img_dim, env.observation_img_dim, 3)),
                    dtype=np.uint8,
                ),
                "proprio": gym.spaces.Box(
                    low=np.zeros((10,)),
                    high=np.ones((10,)),
                    dtype=np.uint8,
                ),
            }
        )

    def step(self, *args):
        obs, reward, done, info = self.env.step(*args)
        return (
            convert_obs(obs, self.env.observation_img_dim),
            reward,
            False,
            done,
            info,
        )

    def seed(self, seed):
        pass

    def render(self, *args, **kwargs):
        return self.env.render_obs()

    def reset(self, **kwargs):
        obs = convert_obs(self.env.reset(), self.env.observation_img_dim)
        return obs, self.env.get_info()


class GCRoboverseWrapper(gym.Wrapper):
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
                    low=np.zeros((env.observation_img_dim, env.observation_img_dim, 3)),
                    high=255
                    * np.ones((env.observation_img_dim, env.observation_img_dim, 3)),
                    dtype=np.uint8,
                ),
                "proprio": gym.spaces.Box(
                    low=np.zeros((10,)),
                    high=np.ones((10,)),
                    dtype=np.uint8,
                ),
            }
        )
        self.current_goal = None
        self.goal_sampler = goal_sampler

    def step(self, *args):
        obs, reward, done, info = self.env.step(*args)
        return (
            convert_obs(obs, self.env.observation_img_dim),
            reward,
            False,
            done,
            {"goal": self.current_goal, **info},
        )

    def seed(self, seed):
        pass

    def render(self, *args, **kwargs):
        return self.env.render_obs()

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
            obs = self.env.reset(
                original_object_positions=original_object_positions,
                # original_object_quats=original_object_quats,
                target_position=target_position,
                object_names=object_names,
                target_object=target_object,
            )
            obs = convert_obs(obs, self.env.observation_img_dim)
        else:
            obs = self.env.reset()
            obs = convert_obs(obs, self.env.observation_img_dim)
            goal_image = self.goal_sampler(obs)

        goal = {"image": goal_image}

        self.current_goal = goal

        return obs, {"goal": goal, **self.env.get_info()}
