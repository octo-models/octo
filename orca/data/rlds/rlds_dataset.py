import hashlib
import json
import logging
from typing import Dict, List

import numpy as np
import tensorflow as tf
import tqdm
from tensorflow_datasets.core.dataset_builder import DatasetBuilder


def get_action_proprio_stats(
    builder: DatasetBuilder, dataset: tf.data.Dataset
) -> Dict[str, Dict[str, List[float]]]:
    # get statistics file path --> embed unique hash that catches if dataset info changed
    data_info_hash = hashlib.sha256(str(builder.info).encode("utf-8")).hexdigest()
    path = tf.io.gfile.join(
        builder.info.data_dir, f"action_proprio_stats_{data_info_hash}.json"
    )

    # check if stats already exist and load, otherwise compute
    if tf.io.gfile.exists(path):
        logging.info(f"Loading existing statistics for normalization from {path}.")
        with tf.io.gfile.GFile(path, "r") as f:
            metadata = json.load(f)
    else:
        logging.info("Computing action/proprio statistics for normalization...")
        actions = []
        proprios = []
        for episode in tqdm.tqdm(dataset.take(1000)):
            actions.append(episode["actions"].numpy())
            proprios.append(episode["observations"]["proprio"].numpy())
        actions = np.concatenate(actions)
        proprios = np.concatenate(proprios)
        metadata = {
            "action": {
                "mean": [float(e) for e in actions.mean(0)],
                "std": [float(e) for e in actions.std(0)],
                "max": [float(e) for e in actions.max(0)],
                "min": [float(e) for e in actions.min(0)],
            },
            "proprio": {
                "mean": [float(e) for e in proprios.mean(0)],
                "std": [float(e) for e in proprios.std(0)],
                "max": [float(e) for e in proprios.max(0)],
                "min": [float(e) for e in proprios.min(0)],
            },
        }
        del actions
        del proprios
        with tf.io.gfile.GFile(path, "w") as f:
            json.dump(metadata, f)
        logging.info("Done!")

    return {
        k: {k2: tf.convert_to_tensor(v2, dtype=tf.float32) for k2, v2 in v.items()}
        for k, v in metadata.items()
    }
