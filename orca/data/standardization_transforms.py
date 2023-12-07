"""Episode transforms for different RLDS datasets to canonical dataset definition."""
from typing import Any, Dict

import tensorflow as tf

import orca.data.bridge.bridge_utils as bridge
from orca.data.oxe import oxe_dataset_transforms as ox
from orca.data.utils.data_utils import binarize_gripper_actions


def r2_d2_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["action"] = tf.concat(
        (
            trajectory["action_dict"]["cartesian_velocity"],
            trajectory["action_dict"]["gripper_velocity"],
        ),
        axis=-1,
    )
    return trajectory


def fmb_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["state"],
            trajectory["observation"]["gripper_state"][..., None],
        ),
        axis=-1,
    )
    return trajectory


def bridge_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory = bridge.relabel_actions(trajectory)
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][
        :, -1:
    ]
    return trajectory


def aloha_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory


RLDS_STANDARDIZATION_TRANSFORMS = {
    "r2_d2": r2_d2_dataset_transform,
    "r2_d2_pen_cmu_rgb": r2_d2_dataset_transform,
    "r2_d2_play_cmu_rgb": r2_d2_dataset_transform,
    "r2_d2_pen": r2_d2_dataset_transform,
    "fmb_dataset": fmb_dataset_transform,
    "bridge_dataset": bridge_dataset_transform,
    "aloha_screwdriver_dataset": aloha_dataset_transform,
    "aloha_sim_cube_scripted_dataset": aloha_dataset_transform,
    # Open X-Embodiment Datasets
    "fractal20220817_data": ox.rt1_dataset_transform,
    "kuka": ox.kuka_dataset_transform,
    "taco_play": ox.taco_play_dataset_transform,
    "jaco_play": ox.jaco_play_dataset_transform,
    "berkeley_cable_routing": ox.berkeley_cable_routing_dataset_transform,
    "roboturk": ox.roboturk_dataset_transform,
    "nyu_door_opening_surprising_effectiveness": ox.nyu_door_opening_dataset_transform,
    "viola": ox.viola_dataset_transform,
    "berkeley_autolab_ur5": ox.berkeley_autolab_ur5_dataset_transform,
    "toto": ox.toto_dataset_transform,
    "language_table": ox.language_table_dataset_transform,
    "columbia_cairlab_pusht_real": ox.pusht_dataset_transform,
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": ox.stanford_kuka_multimodal_dataset_transform,
    "nyu_rot_dataset_converted_externally_to_rlds": ox.nyu_rot_dataset_transform,
    "stanford_hydra_dataset_converted_externally_to_rlds": ox.stanford_hydra_dataset_transform,
    "austin_buds_dataset_converted_externally_to_rlds": ox.austin_buds_dataset_transform,
    "nyu_franka_play_dataset_converted_externally_to_rlds": ox.nyu_franka_play_dataset_transform,
    "maniskill_dataset_converted_externally_to_rlds": ox.maniskill_dataset_transform,
    "furniture_bench_dataset_converted_externally_to_rlds": ox.furniture_bench_dataset_transform,
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": ox.cmu_franka_exploration_dataset_transform,
    "ucsd_kitchen_dataset_converted_externally_to_rlds": ox.ucsd_kitchen_dataset_transform,
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": ox.ucsd_pick_place_dataset_transform,
    "austin_sailor_dataset_converted_externally_to_rlds": ox.austin_sailor_dataset_transform,
    "austin_sirius_dataset_converted_externally_to_rlds": ox.austin_sirius_dataset_transform,
    "bc_z": ox.bc_z_dataset_transform,
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": ox.tokyo_pr2_opening_fridge_dataset_transform,
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": ox.tokyo_pr2_tabletop_manipulation_dataset_transform,
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": ox.utokyo_xarm_pick_place_dataset_transform,
    "utokyo_xarm_bimanual_converted_externally_to_rlds": ox.utokyo_xarm_bimanual_dataset_transform,
    "robo_net": ox.robo_net_dataset_transform,
    "berkeley_mvp_converted_externally_to_rlds": ox.berkeley_mvp_dataset_transform,
    "berkeley_rpt_converted_externally_to_rlds": ox.berkeley_rpt_dataset_transform,
    "kaist_nonprehensile_converted_externally_to_rlds": ox.kaist_nonprehensible_dataset_transform,
    "stanford_mask_vit_converted_externally_to_rlds": ox.stanford_mask_vit_dataset_transform,
    "tokyo_u_lsmo_converted_externally_to_rlds": ox.tokyo_lsmo_dataset_transform,
    "dlr_sara_pour_converted_externally_to_rlds": ox.dlr_sara_pour_dataset_transform,
    "dlr_sara_grid_clamp_converted_externally_to_rlds": ox.dlr_sara_grid_clamp_dataset_transform,
    "dlr_edan_shared_control_converted_externally_to_rlds": ox.dlr_edan_shared_control_dataset_transform,
    "asu_table_top_converted_externally_to_rlds": ox.asu_table_top_dataset_transform,
    "stanford_robocook_converted_externally_to_rlds": ox.robocook_dataset_transform,
    "imperialcollege_sawyer_wrist_cam": ox.imperial_wristcam_dataset_transform,
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": ox.iamlab_pick_insert_dataset_transform,
    "uiuc_d3field": ox.uiuc_d3field_dataset_transform,
    "utaustin_mutex": ox.utaustin_mutex_dataset_transform,
    "berkeley_fanuc_manipulation": ox.berkeley_fanuc_dataset_transform,
    "cmu_playing_with_food": ox.cmu_playing_with_food_dataset_transform,
    "cmu_play_fusion": ox.playfusion_dataset_transform,
    "cmu_stretch": ox.cmu_stretch_dataset_transform,
    "berkeley_gnm_recon": ox.gnm_dataset_transform,
    "berkeley_gnm_cory_hall": ox.gnm_dataset_transform,
    "berkeley_gnm_sac_son": ox.gnm_dataset_transform,
}
