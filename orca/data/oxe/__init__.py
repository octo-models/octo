import copy
import logging
from typing import Any, Dict, List, Sequence, Tuple, Union

from orca.data.oxe.oxe_dataset_configs import ActionEncoding, OXE_DATASET_CONFIGS
from orca.data.oxe.oxe_dataset_mixes import OXE_NAMED_MIXES
from orca.data.oxe.oxe_standardization_transforms import OXE_STANDARDIZATION_TRANSFORMS


def make_oxe_dataset_kwargs_and_weights(
    data_mix: Union[str, Sequence[Tuple[str, float]]],
    data_dir: str,
    deduplicate: bool = True,
    load_camera_views: Sequence[str] = ("primary",),
    load_depth: bool = True,
    load_proprio: bool = True,
) -> Tuple[Dict[str, Any], List[float]]:
    """
    Generates dataset kwargs for a given dataset mix from the Open X-Embodiment dataset.

    Args:
         data_mix: List of (dataset name, sampling weight) tuples, or a string specifying a pre-defined mix to
            load from `OXE_NAMED_MIXES`.
         data_dir: Base data directory that gets registered in each dataset.
         deduplicate: If True, discards any duplicate dataset entries based on dataset name.
         load_camera_views: Which views to load from each dataset. See the top of `oxe_dataset_configs.py`
            for available views.
         load_depth: If True, loads corresponding depth channels for each RGB channel.
         load_proprio: If True, loads proprioceptive information.
    Returns:
        Tuple of (dataset_kwargs_list, sampling weights).
    """
    if isinstance(data_mix, str):
        data_mix = OXE_NAMED_MIXES[data_mix]

    if deduplicate:
        filtered_datasets, included_dataset_names = [], []
        for name, weight in data_mix:
            if name not in included_dataset_names:
                filtered_datasets.append((name, weight))
                included_dataset_names.append(name)
            else:
                logging.warning(f"Skipping duplicate: {(name, weight)}.")
        data_mix = filtered_datasets

    data_kwargs_list, weights = [], []
    for name, weight in data_mix:
        dataset_kwargs = copy.deepcopy(OXE_DATASET_CONFIGS[name])
        if dataset_kwargs["action_encoding"] is not ActionEncoding.EEF_POS:
            logging.warning(
                f"Skipping {name} since only EEF pose delta action encoding "
                f"is supported."
            )
            continue

        # adjust loaded features in kwargs
        dataset_kwargs["image_obs_keys"] = [
            dataset_kwargs["image_obs_keys"][k] for k in load_camera_views
        ]

        if not any([e is not None for e in dataset_kwargs["image_obs_keys"]]):
            logging.warning(f"Skipping {name} since no image input was loaded from it.")
            continue

        dataset_kwargs["depth_obs_keys"] = [
            dataset_kwargs["depth_obs_keys"][k] for k in load_camera_views
        ]

        if not load_depth:
            dataset_kwargs.pop("depth_obs_keys")
        if not load_proprio:
            dataset_kwargs.pop("state_obs_keys")

        del dataset_kwargs["state_encoding"]
        del dataset_kwargs["action_encoding"]

        # get standardization transform
        dataset_kwargs["standardize_fn"] = OXE_STANDARDIZATION_TRANSFORMS[name]

        # add dataset to list
        data_kwargs_list.append({"name": name, "data_dir": data_dir, **dataset_kwargs})
        weights.append(weight)

    return data_kwargs_list, weights
