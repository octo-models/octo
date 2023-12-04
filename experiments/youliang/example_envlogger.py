import os
import time
import cv2
import argparse
import numpy as np
from orca.sim.widowx_sim_env import WidowXSimEnv

# env logger
# https://github.com/rail-berkeley/oxe_envlogger
from oxe_envlogger.envlogger import OXEEnvLogger
import tensorflow_datasets as tfds
import tensorflow as tf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--im_size", type=int, default=256)
    args = parser.parse_args()

    env = WidowXSimEnv(image_size=args.im_size)

    # this describes custom data to be logged which is not in the observation
    # or action space of the environment
    step_metadata_info={
        'language_instruction': tfds.features.Text(
            doc="verbose language instructions")
    }

    # This will wrap the environment with the logger to log the data
    # as tfds dataset
    env = OXEEnvLogger(env,
                       "widowx",
                       directory=os.path.expanduser("~/logs"),
                       max_episodes_per_file=10,
                       step_metadata_info=step_metadata_info,
                       )
    time.sleep(1)

    # set all custom metadata before calling reset() and step()
    env.set_step_metadata({"language_instruction": "hello world"})

    obs = env.reset()
    done, trunc = False, False

    for i in range(30):
        print("step ", i)
        if done or trunc:
            print("New round, reset")
            obs, _ = env.reset()

        if i % 2 == 0:
            action = np.array([0.01, 0.01, 0, 0, 0, 0, 0], dtype=np.float64)
        else:
            action = np.array([-0.01, -0.01, 0, 0, 0, 0, 0], dtype=np.float64)

        # log the custom step metadata before the step() call
        env.set_step_metadata({"language_instruction": f"hello world {i}"})
        obs, _, done, trunc, _ = env.step(action)

        # Just to visualize the image from widowx
        bgr_img = cv2.cvtColor(obs["image_0"], cv2.COLOR_RGB2BGR)
        cv2.imshow("img_view", bgr_img)
        cv2.waitKey(10)

    # This is important to explicitly quit the env to store the current episode
    del env
    print("Done")
