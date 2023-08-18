import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm

from orca.data.bridge_dataset import BridgeDataset
from orca.data.rlds_data_utils import RLDS_TRAJECTORY_MAP_TRANSFORMS


def _get_splits(tfds_dataset_splits):
    """Use val split from dataset if defined, otherwise use parts of train split."""
    if "val" in tfds_dataset_splits:
        return {"train": "train", "val": "val"}
    else:
        # use last 5% of training split as validation split
        return {"train": "train[:95%]", "val": "train[95%:]"}


class RLDSDataset(BridgeDataset):
    """TF dataset that reads RLDS datasets.

    Args:
        image_obs_key: Key used to extract image from raw dataset observation.
        tfds_data_dir: Optional. Directory to load tf_datasets from. Defaults to ~/tensorflow_datasets
    """

    def __init__(
        self,
        *args,
        image_obs_key: str = "image",
        tfds_data_dir: Optional[str] = None,
        **kwargs,
    ):
        self._image_obs_key = image_obs_key
        self._tfds_data_dir = tfds_data_dir
        super().__init__(*args, **kwargs)

    def _construct_base_dataset(self, paths: List[str], seed: int) -> tf.data.Dataset:
        # can only load single dataset
        assert len(paths) == 1

        # load raw dataset of trajectories
        # skips decoding to get list of episode steps instead of RLDS default of steps as a tf.dataset
        dataset_name = paths[0]
        builder = tfds.builder(dataset_name)
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
                    ]["image"].decode_batch_example(
                        steps["observation"][self._image_obs_key]
                    )
                    steps["observation"]["state"] = builder.info.features["steps"][
                        "observation"
                    ]["state"].decode_batch_example(steps["observation"]["state"])
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

        # load or compute metadata
        self.action_metadata = self._get_action_stats(builder, dataset)

        return dataset

    @staticmethod
    def _get_action_stats(dataset_builder, dataset):
        # get statistics file path --> embed unique hash that catches if dataset info changed
        data_info_hash = hashlib.sha256(
            str(dataset_builder.info).encode("utf-8")
        ).hexdigest()
        path = os.path.join(
            dataset_builder.info.data_dir, f"action_stats_{data_info_hash}.json"
        )

        # check if stats already exist and load, otherwise compute
        if os.path.exists(path):
            print(f"Loading existing action statistics for normalization from {path}.")
            with open(path, "r") as F:
                return json.load(F)
        else:
            print("Computing action statistics for normalization...")
            actions = []
            for episode in tqdm.tqdm(dataset.take(5000)):
                actions.append(episode["actions"].numpy())
            actions = np.concatenate(actions)
            action_metadata = {
                "mean": [float(e) for e in actions.mean(0)],
                "std": [float(e) for e in actions.std(0)],
            }
            del actions
            with open(path, "w") as F:
                json.dump(action_metadata, F)
            print("Done!")
            return action_metadata


if __name__ == "__main__":
    ds = RLDSDataset(
        data_paths=[
            ["stanford_kuka_multimodal_dataset"],
            ["stanford_kuka_multimodal_dataset"],
        ],
        seed=0,
        goal_relabeling_kwargs={"reached_proportion": 0.1},
    )
    sample = next(ds.get_iterator())
    print(sample["observations"]["image"].shape)
