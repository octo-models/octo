"""Defines dataset mixtures and weights for the Open X-Embodiment Datasets."""
import copy
import logging
from typing import Any, Dict, List, Sequence, Tuple, Union

from orca.data.oxe.oxe_dataset_configs import ActionEncoding, OXE_DATASET_CONFIGS

BRIDGE_MIX = [
    ("bridge_dataset", 1.0),
]

RT_X_MIX = [
    ("fractal20220817_data", 0.54087122203),
    ("kuka", 0.8341046294),
    ("bridge_dataset", 1.0),
    ("taco_play", 2.0),
    ("jaco_play", 2.0),
    ("berkeley_cable_routing", 3.0),
    ("roboturk", 1.0),
    ("nyu_door_opening_surprising_effectiveness", 5.0),
    ("viola", 2.0),
    ("berkeley_autolab_ur5", 1.0),
    ("toto", 1.0),
]


OXE_FRANKA_MIX = [
    ("taco_play", 1.0),
    ("berkeley_cable_routing", 1.0),
    ("viola", 1.0),
    ("toto", 1.0),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 1.0),
    ("austin_buds_dataset_converted_externally_to_rlds", 3.0),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
    ("maniskill_dataset_converted_externally_to_rlds", 0.1),
    ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
    ("cmu_franka_exploration_dataset_converted_externally_to_rlds", 5.0),
    ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
    ("berkeley_rpt_converted_externally_to_rlds", 1.0),
    ("kaist_nonprehensile_converted_externally_to_rlds", 3.0),
    ("stanford_robocook_converted_externally_to_rlds", 1.0),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
    ("utaustin_mutex", 1.0),
    # ("cmu_playing_with_food", 1.0),
    ("cmu_play_fusion", 1.0),
]


OXE_MAGIC_SOUP = [
    ("fractal20220817_data", 0.54087122203),
    ("kuka", 0.8341046294),
    ("bridge_dataset", 1.0),
    ("taco_play", 2.0),
    ("jaco_play", 1.0),
    ("berkeley_cable_routing", 1.0),
    ("roboturk", 2.0),
    ("nyu_door_opening_surprising_effectiveness", 1.0),
    ("viola", 2.0),
    ("berkeley_autolab_ur5", 2.0),
    ("toto", 1.0),
    ("language_table", 0.1),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
    ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
    ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
    ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
    ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
    ("bc_z", 0.2),
    ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
    # ("uiuc_d3field", 1.0),  --> somehow raw data is broken
    ("utaustin_mutex", 1.0),
    ("berkeley_fanuc_manipulation", 2.0),
    ("cmu_stretch", 1.0),
]


OXE_FULL_MIX = [
    ("fractal20220817_data", 1.0),
    ("kuka", 1.0),
    ("bridge_dataset", 1),
    ("taco_play", 1.0),
    ("jaco_play", 1.0),
    ("berkeley_cable_routing", 1.0),
    ("roboturk", 1.0),
    ("nyu_door_opening_surprising_effectiveness", 1.0),
    ("viola", 1.0),
    ("berkeley_autolab_ur5", 1.0),
    ("toto", 1.0),
    ("language_table", 1.0),
    ("columbia_cairlab_pusht_real", 1.0),
    ("stanford_kuka_multimodal_dataset_converted_externally_to_rlds", 1.0),
    ("nyu_rot_dataset_converted_externally_to_rlds", 1.0),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 1.0),
    ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 1.0),
    ("maniskill_dataset_converted_externally_to_rlds", 1.0),
    ("furniture_bench_dataset_converted_externally_to_rlds", 1.0),
    ("cmu_franka_exploration_dataset_converted_externally_to_rlds", 1.0),
    ("ucsd_kitchen_dataset_converted_externally_to_rlds", 1.0),
    ("ucsd_pick_and_place_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
    ("bc_z", 1.0),
    ("utokyo_pr2_opening_fridge_converted_externally_to_rlds", 1.0),
    ("utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds", 1.0),
    ("utokyo_xarm_pick_and_place_converted_externally_to_rlds", 1.0),
    ("utokyo_xarm_bimanual_converted_externally_to_rlds", 1.0),
    ("robo_net", 1.0),
    ("berkeley_mvp_converted_externally_to_rlds", 1.0),
    ("berkeley_rpt_converted_externally_to_rlds", 1.0),
    ("kaist_nonprehensile_converted_externally_to_rlds", 1.0),
    ("stanford_mask_vit_converted_externally_to_rlds", 1.0),
    ("tokyo_u_lsmo_converted_externally_to_rlds", 1.0),
    ("dlr_sara_pour_converted_externally_to_rlds", 1.0),
    ("dlr_sara_grid_clamp_converted_externally_to_rlds", 1.0),
    ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
    ("asu_table_top_converted_externally_to_rlds", 1.0),
    ("stanford_robocook_converted_externally_to_rlds", 1.0),
    ("imperialcollege_sawyer_wrist_cam", 1.0),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
    ("uiuc_d3field", 1.0),
    ("utaustin_mutex", 1.0),
    ("berkeley_fanuc_manipulation", 1.0),
    ("cmu_playing_with_food", 1.0),
    ("cmu_play_fusion", 1.0),
    ("cmu_stretch", 1.0),
    ("berkeley_gnm_recon", 1.0),
    ("berkeley_gnm_cory_hall", 1.0),
    ("berkeley_gnm_sac_son", 1.0),
]

OXE_NAMED_MIXES = {
    "bridge": BRIDGE_MIX,
    "rtx": RT_X_MIX,
    "rtx_franka": RT_X_MIX + OXE_FRANKA_MIX,
    "oxe_magic_soup": OXE_MAGIC_SOUP,
}


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
            load from `OXE_NAMED_MIXES` above.
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
        for dataset, weight in data_mix:
            if dataset not in included_dataset_names:
                filtered_datasets.append((dataset, weight))
                included_dataset_names.append(dataset)
            else:
                logging.warning(f"Skipping duplicate: {(dataset, weight)}.")
        data_mix = filtered_datasets

    data_kwargs_list, weights = [], []
    for dataset, weight in data_mix:
        dataset_kwargs = copy.deepcopy(OXE_DATASET_CONFIGS[dataset])
        if dataset_kwargs["action_encoding"] is not ActionEncoding.EEF_POS:
            logging.warning(
                f"Skipping {dataset} since only EEF pose delta action encoding "
                f"is supported."
            )
            continue

        # adjust loaded features in kwargs
        dataset_kwargs["image_obs_keys"] = [
            dataset_kwargs["image_obs_keys"][k] for k in load_camera_views
        ]

        if not any([e is not None for e in dataset_kwargs["image_obs_keys"]]):
            logging.warning(
                f"Skipping {dataset} since no image input was loaded from it."
            )
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

        # add dataset to list
        data_kwargs_list.append(
            {"name": dataset, "data_dir": data_dir, **dataset_kwargs}
        )
        weights.append(weight)

    return data_kwargs_list, weights
