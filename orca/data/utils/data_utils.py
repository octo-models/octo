from enum import Enum
import hashlib
import inspect
import json
import logging
from typing import Any, Dict, List, Sequence, Union

import dlimp as dl
from dlimp.transforms import selective_tree_map
import numpy as np
import tensorflow as tf
from tensorflow_datasets.core.dataset_builder import DatasetBuilder
import tqdm


class StateEncoding(Enum):
    """Defines supported proprio state encoding schemes for different datasets."""

    NONE = -1  # no state provided
    POS_EULER = 1  # EEF XYZ + roll-pitch-yaw + 1 x pad + gripper open/close
    POS_QUAT = 2  # EEF XYZ + quaternion + gripper open/close
    JOINT = 3  # 7 x joint angles (padding added if fewer) + gripper open/close


class ActionEncoding(Enum):
    """Defines supported action encoding schemes for different datasets."""

    EEF_POS = 1  # EEF delta XYZ + roll-pitch-yaw + gripper open/close
    JOINT_POS = 2  # 7 x joint delta position + gripper open/close


def pprint_data_mixture(
    dataset_kwargs_list: List[Dict[str, Any]], dataset_weights: List[int]
) -> None:
    print(
        "\n######################################################################################"
    )
    print(
        f"# Loading the following {len(dataset_kwargs_list)} datasets (incl. sampling weight):{'': >24} #"
    )
    for dataset_kwargs, weight in zip(dataset_kwargs_list, dataset_weights):
        pad = 80 - len(dataset_kwargs["name"])
        print(f"# {dataset_kwargs['name']}: {weight:=>{pad}f} #")
    print(
        "######################################################################################\n"
    )


def maybe_decode_depth_images(
    x: Dict[str, Any], match: Union[str, Sequence[str]] = "depth"
) -> Dict[str, Any]:
    """Can operate on nested dicts. Decodes any leaves that have `match` anywhere in their path."""
    if isinstance(match, str):
        match = [match]

    return selective_tree_map(
        x,
        lambda keypath, value: any([s in keypath for s in match])
        and value.dtype == tf.string,
        lambda e: tf.io.decode_image(e, expand_animations=False, dtype=tf.float32)[
            ..., 0
        ],
    )


def get_action_proprio_stats(
    builder: DatasetBuilder,
    dataset: tf.data.Dataset,
    proprio_keys: List[str],
    transform_fcn: Any,
) -> Dict[str, Dict[str, List[float]]]:
    # get statistics file path --> embed unique hash that catches if dataset info / keys / transform changed
    transform_str = inspect.getsource(transform_fcn) if transform_fcn else ""
    data_info_hash = hashlib.sha256(
        (str(builder.info) + str(proprio_keys) + str(transform_str)).encode("utf-8")
    ).hexdigest()
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
            actions.append(episode["action"].numpy())
            proprios.append(episode["observation"]["proprio"].numpy())
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


def normalize_action_and_proprio(traj, metadata, normalization_type):
    # maps keys of `metadata` to corresponding keys in `traj`
    keys_to_normalize = {
        "action": "action",
        "proprio": "observation/proprio",
    }
    if normalization_type == "normal":
        # normalize to mean 0, std 1
        for key, traj_key in keys_to_normalize.items():
            traj = dl.transforms.selective_tree_map(
                traj,
                match=traj_key,
                map_fn=lambda x: (x - metadata[key]["mean"]) / metadata[key]["std"],
            )
        return traj

    if normalization_type == "bounds":
        # normalize to [-1, 1]
        for key, traj_key in keys_to_normalize.items():
            traj = dl.transforms.selective_tree_map(
                traj,
                match=traj_key,
                map_fn=lambda x: tf.clip_by_value(
                    2
                    * (x - metadata[key]["min"])
                    / (metadata[key]["max"] - metadata[key]["min"])
                    - 1,
                    -1,
                    1,
                ),
            )
        return traj

    raise ValueError(f"Unknown normalization type {normalization_type}")
