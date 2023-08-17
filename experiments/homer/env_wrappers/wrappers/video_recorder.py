import os
from typing import List, Optional

import gym
import imageio
import numpy as np
import tensorflow as tf

# Take from
# https://github.com/denisyarats/pytorch_sac/


def compose_frames(
    all_frames: List[np.ndarray],
    num_videos_per_row: int,
    margin: int = 4,
):
    num_episodes = len(all_frames)

    if num_videos_per_row is None:
        num_videos_per_row = num_episodes

    t = 0
    end_of_all_epidoes = False
    frames_to_save = []
    while not end_of_all_epidoes:
        frames_t = []

        for i in range(num_episodes):
            # If the episode is shorter, repeat the last frame.
            t_ = min(t, len(all_frames[i]) - 1)
            frame_i_t = all_frames[i][t_]

            # Add the lines.
            frame_i_t = np.pad(
                frame_i_t,
                [[margin, margin], [margin, margin], [0, 0]],
                "constant",
                constant_values=0,
            )

            frames_t.append(frame_i_t)

        # Arrange the videos based on num_videos_per_row.
        frame_t = None
        while len(frames_t) >= num_videos_per_row:
            frames_t_this_row = frames_t[:num_videos_per_row]
            frames_t = frames_t[num_videos_per_row:]

            frame_t_this_row = np.concatenate(frames_t_this_row, axis=1)
            if frame_t is None:
                frame_t = frame_t_this_row
            else:
                frame_t = np.concatenate([frame_t, frame_t_this_row], axis=0)

        frames_to_save.append(frame_t)
        t += 1
        end_of_all_epidoes = all([len(all_frames[i]) <= t for i in range(num_episodes)])

    return frames_to_save


class VideoRecorder(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        save_folder: str = "",
        save_prefix: str = None,
        height: int = 128,
        width: int = 128,
        fps: int = 30,
        camera_id: int = 0,
        goal_conditioned: bool = False,
    ):
        super().__init__(env)

        self.save_folder = save_folder
        self.save_prefix = save_prefix
        self.height = height
        self.width = width
        self.fps = fps
        self.camera_id = camera_id
        self.frames = []
        self.goal_conditioned = goal_conditioned

        if not tf.io.gfile.exists(save_folder):
            tf.io.gfile.makedirs(save_folder)

        self.num_record_episodes = -1

        self.num_videos = 0

        # self.all_save_paths = None
        self.current_save_path = None

    def start_recording(self, num_episodes: int = None, num_videos_per_row: int = None):
        if num_videos_per_row is not None and num_episodes is not None:
            assert num_episodes >= num_videos_per_row

        self.num_record_episodes = num_episodes
        self.num_videos_per_row = num_videos_per_row

        # self.all_save_paths = []
        self.all_frames = []

    def stop_recording(self):
        self.num_record_episodes = None

    def step(self, action: np.ndarray):  # NOQA

        if self.num_record_episodes is None or self.num_record_episodes == 0:
            observation, reward, terminated, truncated, info = self.env.step(action)

        elif self.num_record_episodes > 0:
            frame = self.env.render(
                height=self.height, width=self.width, camera_id=self.camera_id
            )

            if frame is None:
                try:
                    frame = self.sim.render(
                        width=self.width, height=self.height, mode="offscreen"
                    )
                    frame = np.flipud(frame)
                except Exception:
                    raise NotImplementedError("Rendering is not implemented.")

            self.frames.append(frame.astype(np.uint8))

            observation, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                if self.goal_conditioned:
                    frames = [
                        np.concatenate([self.env.current_goal["image"], frame], axis=0)
                        for frame in self.frames
                    ]
                else:
                    frames = self.frames

                self.all_frames.append(frames)

                if self.num_record_episodes > 0:
                    self.num_record_episodes -= 1

                if self.num_record_episodes is None:
                    # Plot one episode per file.
                    frames_to_save = frames
                    should_save = True
                elif self.num_record_episodes == 0:
                    # Plot all episodes in one file.
                    frames_to_save = compose_frames(
                        self.all_frames, self.num_videos_per_row
                    )
                    should_save = True
                else:
                    should_save = False

                if should_save:
                    filename = "%08d.mp4" % (self.num_videos)
                    if self.save_prefix is not None and self.save_prefix != "":
                        filename = f"{self.save_prefix}_{filename}"
                    self.current_save_path = tf.io.gfile.join(
                        self.save_folder, filename
                    )

                    with tf.io.gfile.GFile(self.current_save_path, "wb") as f:
                        imageio.mimsave(f, frames_to_save, "MP4", fps=self.fps)

                    self.num_videos += 1

                self.frames = []

        else:
            raise ValueError("Do not forget to call start_recording.")

        return observation, reward, terminated, truncated, info
