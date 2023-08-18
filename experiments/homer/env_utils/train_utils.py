from collections import defaultdict

import imageio
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import wandb
from flax.core import frozen_dict
from orca.data.bridge_dataset import BridgeDataset


def concat_batches(offline_batch, online_batch, axis=1):
    batch = defaultdict(list)

    if type(offline_batch) != dict:
        offline_batch = offline_batch.unfreeze()

    if type(online_batch) != dict:
        online_batch = online_batch.unfreeze()

    for k, v in offline_batch.items():
        if type(v) is dict:
            batch[k] = concat_batches(offline_batch[k], online_batch[k], axis=axis)
        else:
            batch[k] = jnp.concatenate((offline_batch[k], online_batch[k]), axis=axis)

    return frozen_dict.freeze(batch)


def load_recorded_video(
    video_path: str,
):
    with tf.io.gfile.GFile(video_path, "rb") as f:
        video = np.array(imageio.mimread(f, "MP4")).transpose((0, 3, 1, 2))
        assert video.shape[1] == 3, "Numpy array should be (T, C, H, W)"

    return wandb.Video(video, fps=20)
