import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm

from orca.data.dataset import BaseDataset
from orca.data.rlds.rlds_dataset_transforms import RLDS_TRAJECTORY_MAP_TRANSFORMS


def _get_splits(tfds_dataset_splits):
    """Use val split from dataset if defined, otherwise use parts of train split."""
    if "val" in tfds_dataset_splits:
        return {"train": "train", "val": "val"}
    else:
        # use last 5% of training split as validation split
        return {"train": "train[:95%]", "val": "train[95%:]"}


class RLDSDataset(BaseDataset):
    """Fast parallel tf.data.Dataset-based dataloader for RLDS datasets.

    Args:
        image_obs_key: Key used to extract image from raw dataset observation.
        tfds_data_dir: Optional. Directory to load tf_datasets from. Defaults to ~/tensorflow_datasets
    """

    def __init__(
        self,
        *args,
        image_obs_key: str = "image",
        state_obs_key: str = "state",
        tfds_data_dir: Optional[str] = None,
        **kwargs,
    ):
        self._image_obs_key = image_obs_key
        self._state_obs_key = state_obs_key
        self._tfds_data_dir = tfds_data_dir
        super().__init__(*args, **kwargs)

    def _construct_base_dataset(self, dataset_name: str, seed: int) -> tf.data.Dataset:
        # load raw dataset of trajectories
        # skips decoding to get list of episode steps instead of RLDS default of steps as a tf.dataset
        builder = tfds.builder(dataset_name, data_dir=self._tfds_data_dir)
        dataset = builder.as_dataset(
            split=_get_splits(builder.info.splits)["train" if self.is_train else "val"],
            decoders={"steps": tfds.decode.SkipDecoding()},
            shuffle_files=self.is_train,
        )

        def _decode_trajectory(episode: Dict[str, Any]) -> Dict[str, Any]:
            # manually decode all features since we skipped decoding during dataset construction
            steps = episode["steps"]
            for key in steps:
                if key == "observation":
                    # only decode parts of observation we need for improved data loading speed
                    steps["observation"]["image"] = builder.info.features["steps"][
                        "observation"
                    ][self._image_obs_key].decode_batch_example(
                        steps["observation"][self._image_obs_key]
                    )
                    steps["observation"]["state"] = builder.info.features["steps"][
                        "observation"
                    ][self._state_obs_key].decode_batch_example(
                        steps["observation"][self._state_obs_key])
                else:
                    steps[key] = builder.info.features["steps"][
                        key
                    ].decode_batch_example(steps[key])
            return steps

        def _to_transition_trajectories(trajectory: Dict[str, Any]) -> Dict[str, Any]:
            # return transition dataset in convention of Bridge dataset
            padded_img = tf.concat(
                (
                    trajectory["observation"]["image"],
                    trajectory["observation"]["image"][-1:],
                ),
                axis=0,
            )
            padded_state = tf.concat(
                (
                    trajectory["observation"]["state"],
                    trajectory["observation"]["state"][-1:],
                ),
                axis=0,
            )
            return {
                "observations": {
                    "image": padded_img[:-1],
                    "proprio": padded_state[:-1],
                },
                "next_observations": {
                    "image": padded_img[1:],
                    "proprio": padded_state[1:],
                },
                **({"language": trajectory["language_instruction"]} if self.load_language else {}),
                "actions": trajectory["action"],
                "terminals": trajectory["is_terminal"],
                "truncates": tf.math.logical_and(
                    trajectory["is_last"],
                    tf.math.logical_not(trajectory["is_terminal"]),
                ),
            }

        dataset = dataset.map(_decode_trajectory, num_parallel_calls=tf.data.AUTOTUNE)
        if dataset_name in RLDS_TRAJECTORY_MAP_TRANSFORMS:
            # optionally apply transform function to get canonical step representation
            dataset = dataset.map(
                RLDS_TRAJECTORY_MAP_TRANSFORMS[dataset_name],
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        dataset = dataset.map(
            _to_transition_trajectories, num_parallel_calls=tf.data.AUTOTUNE
        )

        # load or compute action metadata for normalization
        self.action_proprio_metadata = self._get_action_proprio_stats(builder, dataset)

        return dataset

    @staticmethod
    def _get_action_proprio_stats(dataset_builder, dataset):
        # get statistics file path --> embed unique hash that catches if dataset info changed
        data_info_hash = hashlib.sha256(
            str(dataset_builder.info).encode("utf-8")
        ).hexdigest()
        path = os.path.join(
            dataset_builder.info.data_dir, f"action_proprio_stats_{data_info_hash}.json"
        )

        # check if stats already exist and load, otherwise compute
        if os.path.exists(path):
            print(f"Loading existing statistics for normalization from {path}.")
            with open(path, "r") as F:
                return json.load(F)
        else:
            print("Computing action/proprio statistics for normalization...")
            actions = []
            proprios = []
            for episode in tqdm.tqdm(dataset.take(1000)):
                actions.append(episode["actions"].numpy())
                proprios.append(episode["observations"]["proprio"].numpy())
            actions = np.concatenate(actions)
            proprios = np.concatenate(proprios)
            action_proprio_metadata = {
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
                }
            }
            del actions
            del proprios
            with open(path, "w") as F:
                json.dump(action_proprio_metadata, F)
            print("Done!")
            return action_proprio_metadata


if __name__ == "__main__":
    from collections import MutableMapping
    ds = RLDSDataset(
        dataset_names="r2_d2_pen",
        image_obs_key="exterior_image_1_left",
        state_obs_key="joint_position",
        tfds_data_dir="/nfs/kun2/datasets/r2d2/tfds",
        shuffle_buffer_size=100,
        seed=0,
        act_pred_horizon=5,
        obs_horizon=2
    )
    sample = next(ds.get_iterator())
    print(sample)
