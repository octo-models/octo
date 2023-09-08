from typing import Any, Dict
import tensorflow as tf
import dlimp as dl
from orca.data.utils.bridge_utils import binarize_gripper_actions
from orca.data.dataset import apply_common_transforms


def relabel_actions_bridge(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Relabels the actions in to use the reached proprio instead. Discards the last timestep of the
    trajectory (since we don't have a next state to compute the action.)
    """
    # relabel the first 6 action dims (xyz position, xyz rotation) using the reached proprio
    movement_actions = (
        traj["observations"]["proprio"][1:, :6]
        - traj["observations"]["proprio"][:-1, :6]
    )

    # discard the last timestep of the trajectory
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)

    # recombine to get full actions
    traj_truncated["actions"] = tf.concat(
        [movement_actions, traj["actions"][:-1, -1:]], axis=1
    )

    return traj_truncated


def make_bridge_dataset(
    data_path: str, train: bool, relabel_actions: bool = True, **kwargs
) -> tf.data.Dataset:
    """Creates a dataset from the BridgeData format.

    Args:
        data_path (str): The path to the data directory (must contain "train" and "val" subdirectories).
        train (bool): Whether to use the training or validation set.
        relabel_actions (bool, optional): Whether to relabel the actions using the reached proprio. Defaults to True.
        **kwargs: Additional keyword arguments to pass to `apply_common_transforms`.
    """
    dataset = dl.DLataset.from_tfrecords(
        f"{data_path}/{'train' if train else 'val'}"
    ).map(dl.transforms.unflatten_dict)

    def restructure(traj):
        # traj["observations"] = traj.pop("obs")
        traj["observations"] = {
            "image": traj["observations"]["images0"],  # always take images0 for now
            "proprio": tf.cast(traj["observations"]["state"], tf.float32),
        }
        # traj["language"] = traj.pop("lang")
        traj["actions"] = tf.cast(traj["actions"], tf.float32)
        traj["actions"] = tf.concat(
            [
                traj["actions"][:, :6],
                binarize_gripper_actions(traj["actions"][:, -1])[:, None],
            ],
            axis=1,
        )
        return traj

    dataset = dataset.map(restructure)

    # bridgedata has one more observation than action in each trajectory; we pad with a zero action during
    # preprocessing, so we must discard the last timestep. if relabeling, this happens in relabel_actions (since
    # relabeling uses the last observation); otherwise, we do it manually.
    if relabel_actions:
        dataset = dataset.map(relabel_actions_bridge)
    else:
        dataset = dataset.map(lambda x: tf.nest.map_structure(lambda y: y[:-1], x))

    dataset = apply_common_transforms(dataset, train=train, **kwargs)

    return dataset
