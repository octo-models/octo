import tensorflow as tf
import dlimp as dl
from functools import partial
from typing import Optional, Tuple
from octo.data.utils import bc_goal_relabeling, task_augmentation
from octo.data.dataset import _normalize_action_and_proprio, _chunk_act_obs


def make_sim_dataset(data_path: str, train: bool, **kwargs) -> tf.data.Dataset:
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
        keep_keys = ["observation", "action", "is_terminal", "is_last", "_traj_index"]
        traj = {k: v for k, v in traj.items() if k in keep_keys}
        return traj

    dataset = dataset.map(restructure)

    dataset = apply_common_transforms(dataset, train=train, **kwargs)

    return dataset


def apply_common_transforms(
    dataset: tf.data.Dataset,
    *,
    train: bool,
    goal_relabeling_strategy: Optional[str] = None,
    goal_relabeling_kwargs: dict = {},
    image_augment_kwargs: dict = {},
    task_augmentation_strategy: Optional[str] = None,
    task_augmentation_kwargs: dict = {},
    window_size: int = 1,
    resize_size: Optional[Tuple[int, int]] = None,
    skip_unlabeled: bool = False,
    action_proprio_metadata: Optional[dict] = None,
    action_proprio_normalization_type: Optional[str] = None,
):
    """Common transforms shared between all datasets.

    Args:
        dataset (tf.data.Dataset): The dataset to transform.
        train (bool): Whether the dataset is for training (affects augmentation).
        goal_relabeling_strategy (Optional[str], optional): The goal relabeling strategy to use, or None for no goal
            relabeling. See `bc_goal_relabeling.py`.
        goal_relabeling_kwargs (dict, optional): Additional keyword arguments to pass to the goal relabeling function.
        image_augment_kwargs (dict, optional): Keyword arguments to pass to the augmentation function. See
            `dlimp.augmentations.augment_image` for documentation.
        task_augmentation_strategy (Optional[str], optional): The task augmentation strategy to use, or None for no task
            augmentation. See `task_augmentation.py`.
        task_augmentation_kwargs (dict, optional): Additional keyword arguments to pass to the task augmentation
        resize_size (tuple, optional): target (height, width) for all RGB and depth images, default to no resize.
        window_size (int, optional): The length of the snippets that trajectories are chunked into.
        skip_unlabeled (bool, optional): Whether to skip trajectories with no language labels.
        action_proprio_metadata (Optional[dict], optional): A dictionary containing metadata about the action and
            proprio statistics. If None, no normalization is performed.
        action_proprio_normalization_type (Optional[str], optional): The type of normalization to perform on the action,
            proprio, or both. Can be "normal" (mean 0, std 1) or "bounds" (normalized to [-1, 1]).
    """
    if skip_unlabeled:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_any(x["language_instruction"] != "")
        )

    if action_proprio_metadata is not None:
        dataset = dataset.map(
            partial(
                _normalize_action_and_proprio,
                metadata=action_proprio_metadata,
                normalization_type=action_proprio_normalization_type,
            )
        )

    # decodes string keys with name "image", resizes "image" and "depth"
    dataset = dataset.frame_map(dl.transforms.decode_images)
    if resize_size:
        dataset = dataset.frame_map(
            partial(dl.transforms.resize_images, size=resize_size)
        )
        dataset = dataset.frame_map(
            partial(dl.transforms.resize_depth_images, size=resize_size)
        )

    if train:
        # augments the entire trajectory with the same seed
        dataset = dataset.frame_map(
            partial(
                dl.transforms.augment,
                augment_kwargs=image_augment_kwargs,
            )
        )

    # adds the "tasks" key
    if goal_relabeling_strategy is not None:
        dataset = dataset.map(
            partial(
                getattr(bc_goal_relabeling, goal_relabeling_strategy),
                **goal_relabeling_kwargs,
            )
        )

    if task_augmentation_strategy is not None:
        dataset = dataset.map(
            partial(
                getattr(task_augmentation, task_augmentation_strategy),
                **task_augmentation_kwargs,
            )
        )

    # chunks actions and observations
    dataset = dataset.map(partial(_chunk_act_obs, window_size=window_size))

    return dataset
