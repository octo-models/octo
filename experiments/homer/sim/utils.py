from typing import Callable, Union

import gym
import numpy as np

import mujoco_manipulation
import tensorflow as tf
from .wrappers.dmcgym import DMCGYM
from .wrappers.mujoco import GCMujocoWrapper
from octo.utils.gym_wrappers import (
    UnnormalizeActionProprio,
    HistoryWrapper,
    RHCWrapper,
    TemporalEnsembleWrapper,
)
from .wrappers.video_recorder import VideoRecorder
import wandb
import imageio


def make_mujoco_gc_env(
    env_name: str,
    max_episode_steps: int,
    action_proprio_metadata: dict,
    normalization_type: str,
    save_video: bool,
    save_video_dir: str,
    save_video_prefix: str,
    goals: Union[np.ndarray, Callable],
    horizon: int,
    pred_horizon: int,
    exec_horizon: int,
):
    env = mujoco_manipulation.load(env_name)
    env = DMCGYM(env)
    env = GCMujocoWrapper(env, goals)
    env = UnnormalizeActionProprio(env, action_proprio_metadata, normalization_type)
    env = HistoryWrapper(env, horizon, pred_horizon)
    env = TemporalEnsembleWrapper(env, horizon, pred_horizon)
    # env = RHCWrapper(env, horizon, pred_horizon, exec_horizon)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    if save_video:
        env = VideoRecorder(
            env,
            save_folder=save_video_dir,
            save_prefix=save_video_prefix,
            goal_conditioned=True,
        )

    env.reset()

    return env


PROTO_TYPE_SPEC = {
    "observations/images0": tf.uint8,
    "observations/state": tf.float32,
    "next_observations/images0": tf.uint8,
    "next_observations/state": tf.float32,
    "actions": tf.float32,
    "terminals": tf.bool,
    "truncates": tf.bool,
    "info/place_success": tf.bool,
    "info/target_object": tf.uint8,
    "info/object_positions": tf.float32,
    "info/target_position": tf.float32,
    "info/object_names": tf.string,
}


def _decode_example(example_proto):
    # decode the example proto according to PROTO_TYPE_SPEC
    features = {
        key: tf.io.FixedLenFeature([], tf.string) for key in PROTO_TYPE_SPEC.keys()
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    parsed_tensors = {
        key: tf.io.parse_tensor(parsed_features[key], dtype)
        for key, dtype in PROTO_TYPE_SPEC.items()
    }

    return {
        "observations": {
            "image": parsed_tensors["observations/images0"],
            "proprio": parsed_tensors["observations/state"],
        },
        "next_observations": {
            "image": parsed_tensors["next_observations/images0"],
            "proprio": parsed_tensors["next_observations/state"],
        },
        "actions": parsed_tensors["actions"],
        "terminals": parsed_tensors["terminals"],
        "truncates": parsed_tensors["truncates"],
        "infos": {
            "place_success": parsed_tensors["info/place_success"],
            "object_positions": parsed_tensors["info/object_positions"],
            "target_position": parsed_tensors["info/target_position"],
            "target_object": parsed_tensors["info/target_object"],
            "object_names": parsed_tensors["info/object_names"],
        },
    }


def load_tf_dataset(data_path):
    """Load a sim dataset in TFRecord format."""
    data_paths = tf.io.gfile.glob(tf.io.gfile.join(data_path, "*.tfrecord"))

    # shuffle again using the dataset API so the files are read in a
    # different order every epoch
    dataset = tf.data.Dataset.from_tensor_slices(data_paths)

    # yields raw serialized examples
    dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=tf.data.AUTOTUNE)

    # yields trajectories
    dataset = dataset.map(_decode_example, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset


def load_recorded_video(
    video_path: str,
):
    with tf.io.gfile.GFile(video_path, "rb") as f:
        video = np.array(imageio.mimread(f, "MP4")).transpose((0, 3, 1, 2))
        assert video.shape[1] == 3, "Numpy array should be (T, C, H, W)"

    return wandb.Video(video, fps=20)
