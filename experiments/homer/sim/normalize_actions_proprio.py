"""
Creates a metadata.npy file with the mean and standard deviation of the actions
in a dataset. Saves the file to the same folder as the data. This file is read
during training and used to normalize the actions in the data loader.
"""

import numpy as np
import tensorflow as tf
from absl import app, flags
from tqdm import tqdm
from collections import defaultdict

from utils import load_tf_dataset

FLAGS = flags.FLAGS
flags.DEFINE_string("data_path", None, "Location of dataset", required=True)


def main(_):
    dataset = load_tf_dataset(FLAGS.data_path)
    actions = []
    proprio = []
    for f in tqdm(iter(dataset)):
        actions.append(f["actions"][:])
        proprio.append(f["observations"]["proprio"][:])
    actions = np.concatenate(actions)
    proprio = np.concatenate(proprio)
    metadata = defaultdict(dict)
    for key, value in [("action", actions), ("proprio", proprio)]:
        metadata[key]["mean"] = np.mean(value, axis=0)
        metadata[key]["std"] = np.std(value, axis=0)

    # sim action/proprio bounds
    metadata["action"]["min"] = np.array(
        [-1, -1, -1, -2 * np.pi, -2 * np.pi, -2 * np.pi, 0]
    )
    metadata["action"]["max"] = np.array(
        [1, 1, 1, 2 * np.pi, 2 * np.pi, 2 * np.pi, 255]
    )
    metadata["action"]["min"] = np.array([-1, -1, -1, np.pi, np.pi, np.pi, 0])
    metadata["action"]["max"] = np.array([1, 1, 1, np.pi, np.pi, np.pi, 255])

    with tf.io.gfile.GFile(
        tf.io.gfile.join(FLAGS.data_path, "metadata.npy"), "wb"
    ) as f:
        np.save(f, metadata)


if __name__ == "__main__":
    app.run(main)
