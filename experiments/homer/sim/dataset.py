import tensorflow as tf
import dlimp as dl
from orca.data.dataset import apply_common_transforms

def make_sim_dataset(
    data_path: str,
    train: bool,
    **kwargs,
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
        traj["observation"] = {
            "image_0": traj["observations"]["images0"],  # always take images0 for now
            "proprio": tf.cast(traj["observations"]["state"], tf.float32),
        }
        traj.pop("observations")
        traj["action"] = tf.cast(traj["actions"], tf.float32)
        traj.pop("actions")
        keep_keys = [
            "observation",
            "action",
            "is_terminal",
            "is_last",
            "_traj_index",
        ]
        traj = {k: v for k, v in traj.items() if k in keep_keys}
        return traj

    dataset = dataset.map(restructure)

    dataset = apply_common_transforms(dataset, train=train, **kwargs)

    return dataset
