import gym
import numpy as np
from widowx_envs.widowx_env_service import WidowXClient
import time

def wait_for_obs(widowx_client):
    obs = widowx_client.get_observation()
    while obs is None:
        print("Waiting for observations...")
        obs = widowx_client.get_observation()
        time.sleep(1)
    return obs


class WidowXGym(gym.Env):
    """
    A Gym environment for the WidowX controller provided by:
    https://github.com/rail-berkeley/bridge_data_robot
    Needed to use Gym wrappers.
    """

    def __init__(
        self, widowx_client: WidowXClient, im_size: int = 256, blocking: bool = True
    ):
        self.widowx_client = widowx_client
        self.im_size = im_size
        self.blocking = blocking
        self.observation_space = gym.spaces.Dict(
            {
                "image_0": gym.spaces.Box(
                    low=np.zeros((im_size, im_size, 3)),
                    high=255 * np.ones((im_size, im_size, 3)),
                    dtype=np.int32,
                ),
                "proprio": gym.spaces.Box(
                    low=np.ones((7,)) * -1, high=np.ones((7,)), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.zeros((7,)), high=np.ones((7,)), dtype=np.float32
        )

    def step(self, action):
        self.widowx_client.step_action(action, blocking=self.blocking)

        raw_obs = self.widowx_client.get_observation()

        truncated = False
        if raw_obs is None:
            # this indicates a loss of connection with the server
            # due to an exception in the last step so end the trajectory
            truncated = True

        image_obs = (
            raw_obs["image"].reshape(3, self.im_size, self.im_size).transpose(1, 2, 0)
            * 255
        ).astype(np.uint8)
        # TODO: proprio from robot env doesn't match training proprio,
        # need to add transformation somewhere (probably in PretrainedModel class)
        # obs = {"image_0": image_obs, "proprio": obs["state"]}
        obs = {"image_0": image_obs, "full_image": raw_obs["full_image"]}

        return obs, 0, False, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        raw_obs = wait_for_obs(self.widowx_client)

        image_obs = (
            raw_obs["image"].reshape(3, self.im_size, self.im_size).transpose(1, 2, 0)
            * 255
        ).astype(np.uint8)
        # TODO: proprio from robot env doesn't match training proprio,
        # need to add transformation somewhere (probably in PretrainedModel class)
        # obs = {"image_0": image_obs, "proprio": obs["state"]}
        obs = {"image_0": image_obs, "full_image": raw_obs["full_image"]}

        return obs, {}
