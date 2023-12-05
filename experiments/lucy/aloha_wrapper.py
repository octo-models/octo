from typing import List
import copy
import cv2
from einops import rearrange
import gym
import jax
import jax.numpy as jnp
import numpy as np
import dlimp as dl

from experiments.lucy.aloha_pro.aloha_scripts.utils import crop_resize
from experiments.lucy.aloha_pro.aloha_scripts.sim_env import BOX_POSE, sample_box_pose
from experiments.lucy.aloha_pro.aloha_scripts.sim_env import make_sim_env


class AlohaGymEnv(gym.Env):
    def __init__(
            self,
            env: gym.Env,
            camera_names: List[str],
            im_size: int = 256,
            seed: int = 1234):
        self._env = env
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"image_{i}": gym.spaces.Box(
                        low=np.zeros((im_size, im_size, 3)),
                        high=255 * np.ones((im_size, im_size, 3)),
                        dtype=np.uint8,
                    ) for i in range(len(camera_names))
                },
                "proprio": gym.spaces.Box(
                    low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
        )
        self.camera_names = camera_names
        self._im_size = im_size
        self._rng = np.random.default_rng(seed)

    def step(self, action):
        ts = self._env.step(action)
        obs, images = self.get_obs(ts)
        reward = ts.reward
        info = {"images": images}

        if reward == self._env.task.max_reward:
            self._episode_is_success = 1

        return obs, reward, False, False, info

    def reset(self, **kwargs):
        x_range = [0.0, 0.2]
        y_range = [0.4, 0.6]
        z_range = [0.05, 0.05]
        ranges = np.vstack([x_range, y_range, z_range])
        cube_position = self._rng.uniform(ranges[:, 0], ranges[:, 1])
        cube_quat = np.array([1, 0, 0, 0])
        BOX_POSE[0] = np.concatenate([cube_position, cube_quat])

        ts = self._env.reset(**kwargs)
        obs, images = self.get_obs(ts)
        info = {"images": images}
        self._goal_obs = copy.deepcopy(obs)   # HACK

        self._episode_is_success = 0

        return obs, info

    def get_obs(self, ts):
        curr_obs = {}
        vis_images = []

        for i, cam_name in enumerate(self.camera_names):
            curr_image = ts.observation['images'][cam_name]

            # Check for 'cam_high' and crop
            if cam_name == 'cam_high':
                curr_image = crop_resize(curr_image)

            #curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2RGB)
            vis_images.append(copy.deepcopy(curr_image))
            curr_image = jnp.array(curr_image) # XXX: / 255. ?
            curr_obs[f"image_{i}"] = curr_image
        curr_obs = dl.transforms.resize_images(
            curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size))

        qpos_numpy = np.array(ts.observation['qpos'])
        qpos = jnp.array(qpos_numpy)
        curr_obs['proprio'] = qpos

        return curr_obs, np.concatenate(vis_images, axis=-2)

    def get_task(self):
        assert self._goal_obs, "Need to run reset() before!"
        return {
            "language_instruction": ["pick up the cube and hand it over".encode()],
            **jax.tree_map(lambda x: x[None] * 0, self._goal_obs),
        }

    def get_episode_metrics(self):
        return {
            "success_rate": self._episode_is_success,
        }


# register gym environments
gym.register(
    'aloha-sim-cube-v0',
    entry_point=lambda: AlohaGymEnv(make_sim_env("sim_transfer_cube"), camera_names=["top"]),
)
