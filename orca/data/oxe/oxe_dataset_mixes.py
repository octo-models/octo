"""Defines dataset mixtures and weights for the Open X-Embodiment Datasets."""
import copy
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow_datasets as tfds
import tqdm

from orca.data.oxe import oxe_dataset_configs
from orca.utils.typing import ActionEncoding

RT_X_MIX = [
    ("fractal20220817_data", 1.0),
    # ('kuka', 1.),     --> contains a lot of failure data too (as opposed to RT-X train mix)
    ("bridge_dataset", 1.0),
    ("taco_play", 1.0),
    ("jaco_play", 1.0),
    ("berkeley_cable_routing", 1.0),
    ("roboturk", 1.0),
    ("nyu_door_opening_surprising_effectiveness", 1.0),
    ("viola", 1.0),
    ("berkeley_autolab_ur5", 1.0),
    ("toto", 1.0),
]


OXE_FRANKA_MIX = [
    ("taco_play", 1.0),
    ("berkeley_cable_routing", 1.0),
    ("viola", 1.0),
    ("toto", 1.0),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 1.0),
    ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 1.0),
    ("maniskill_dataset_converted_externally_to_rlds", 1.0),
    ("furniture_bench_dataset_converted_externally_to_rlds", 1.0),
    ("cmu_franka_exploration_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
    ("berkeley_rpt_converted_externally_to_rlds", 1.0),
    ("kaist_nonprehensile_converted_externally_to_rlds", 1.0),
    ("stanford_robocook_converted_externally_to_rlds", 1.0),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
    ("utaustin_mutex", 1.0),
    ("cmu_food_manipulation", 1.0),
    ("cmu_play_fusion", 1.0),
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
    ("cmu_play_fusion", 1.0),
    ("cmu_stretch", 1.0),
    ("berkeley_gnm_recon", 1.0),
    ("berkeley_gnm_cory_hall", 1.0),
    ("berkeley_gnm_sac_son", 1.0),
]


def make_oxe_dataset_kwargs_and_weights(
    data_mix: List[Tuple[str, float]],
    data_dir: str,
    balance_sampling_ratios: bool = True,
    n_cameras: int = 3,
    load_depth: bool = True,
    load_proprio: bool = True,
) -> Tuple[Dict[str, Any], List[float]]:
    """
    Generates dataset kwargs for a given dataset mix from the Open X-Embodiment dataset.

    Args:
         data_mix: List of (dataset name, sampling weight) tuples.
         data_dir: Base data directory that gets registered in each dataset.
         balance_sampling_ratios: If True, multiplies sampling weights by weights that compensate for the number
            of episodes in each dataset.
         n_cameras: Number of RGB input channels to load.
         load_depth: If True, loads corresponding depth channels for each RGB channel.
         load_proprio: If True, loads proprioceptive information.
    Returns:
        Tuple of (dataset_kwargs_list, sampling weights).
    """
    data_kwargs_list, weights = [], []
    for dataset, weight in data_mix:
        dataset_kwargs = copy.deepcopy(oxe_dataset_configs.OXE_DATASET_KWARGS[dataset])
        if dataset_kwargs["action_encoding"] is not ActionEncoding.EEF_POS:
            print(
                f"Skipping {dataset} since only EEF pose delta action encoding "
                f"is supported."
            )
            continue

        # adjust loaded features in kwargs
        dataset_kwargs["image_obs_keys"] = dataset_kwargs["image_obs_keys"][:n_cameras]
        dataset_kwargs["depth_obs_keys"] = dataset_kwargs["depth_obs_keys"][:n_cameras]
        if not load_depth:
            dataset_kwargs.pop("depth_obs_keys")
        if not load_proprio:
            dataset_kwargs.pop("state_obs_keys")

        # add dataset to list
        data_kwargs_list.append(
            {"name": dataset, "data_dir": data_dir, **dataset_kwargs}
        )
        weights.append(weight)

    if balance_sampling_ratios:
        # compute number of samples in each dataset
        print(
            "Balancing dataset sampling ratios based on #episodes, computing sampling weights..."
        )
        n_samples_per_dataset = []
        for data_kwargs in tqdm.tqdm(data_kwargs_list):
            builder = tfds.builder(data_kwargs["name"], data_dir=data_dir)
            n_samples_per_dataset.append(builder.info.splits["train"].num_examples)
        n_samples_per_dataset = np.array(n_samples_per_dataset)
        weights = list(
            np.array(weights) * n_samples_per_dataset / n_samples_per_dataset.sum()
        )
        print("... Done!")

    return data_kwargs_list, weights


if __name__ == "__main__":
    from orca.data.dataset import make_interleaved_dataset

    base_data_config = dict(
        window_size=4,
        image_augment_kwargs=dict(
            random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
            random_brightness=[0.2],
            random_contrast=[0.8, 1.2],
            random_saturation=[0.8, 1.2],
            random_hue=[0.1],
            augment_order=[
                "random_resized_crop",
                "random_brightness",
                "random_contrast",
                "random_saturation",
                "random_hue",
            ],
        ),
        goal_relabeling_strategy="uniform",
        action_proprio_normalization_type="normal",
        resize_size=(256, 256),
    )
    data_kwargs_list, weights = make_oxe_dataset_kwargs_and_weights(
        data_mix=RT_X_MIX,
        data_dir="gs://rail-orca-central1",
        balance_sampling_ratios=True,
        n_cameras=1,
        load_depth=False,
    )
    ds = make_interleaved_dataset(
        base_data_config, data_kwargs_list, train=True, sample_weights=weights
    )
    print(ds.element_spec)
