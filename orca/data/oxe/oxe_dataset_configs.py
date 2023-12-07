"""Dataset kwargs for Open X-Embodiment datasets.

Target configuration:
    image_obs_keys:
        2x external RGB
        1x wrist/ego RGB
    depth_obs_keys:
        2x external depth
        1x wrist depth
    state_obs_keys:     # 8-dim, changes based on used StateEncoding
        StateEncoding.POS_EULER: EEF XYZ + roll-pitch-yaw + 1 x pad + gripper open/close
        StateEncoding.POS_QUAT: EEF XYZ + quaternion + gripper open/close
        StateEncoding.JOINT: 7 x joint angles (padding added if fewer) + gripper open/close
    state_encoding: Type of state encoding used -- see above
    action_encoding: Type of action encoding used, e.g. EEF position vs joint position control
"""
from enum import IntEnum


class StateEncoding(IntEnum):
    """Defines supported proprio state encoding schemes for different datasets."""

    NONE = -1  # no state provided
    POS_EULER = 1  # EEF XYZ + roll-pitch-yaw + 1 x pad + gripper open/close
    POS_QUAT = 2  # EEF XYZ + quaternion + gripper open/close
    JOINT = 3  # 7 x joint angles (padding added if fewer) + gripper open/close
    JOINT_BIMANUAL = 4  # 2 x [6 x joint angles + gripper open/close]


class ActionEncoding(IntEnum):
    """Defines supported action encoding schemes for different datasets."""

    EEF_POS = 1  # EEF delta XYZ + roll-pitch-yaw + gripper open/close
    JOINT_POS = 2  # 7 x joint delta position + gripper open/close
    JOINT_POS_BIMANUAL = 3  # 2 x [6 x joint pos + gripper]


OXE_DATASET_KWARGS = {
    "fractal20220817_data": {
        "image_obs_keys": ["image", None, None],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": ["base_pose_tool_reached", "gripper_closed"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "kuka": {
        "image_obs_keys": ["image", None, None],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "clip_function_input/base_pose_tool_reached",
            "gripper_closed",
        ],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "bridge_dataset": {
        "image_obs_keys": ["image_0", "image_1", None],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "EEF_state",
            None,
            "gripper_state",
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "taco_play": {
        "image_obs_keys": [
            "rgb_static",
            None,
            "rgb_gripper",
        ],
        "depth_obs_keys": [
            "depth_static",
            None,
            "depth_gripper",
        ],
        "state_obs_keys": ["state_eef", None, "state_gripper"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "jaco_play": {
        "image_obs_keys": [
            "image",
            None,
            "image_wrist",
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "state_eef",
            None,
            "state_gripper",
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_cable_routing": {
        "image_obs_keys": [
            "image",
            "top_image",
            "wrist45_image",
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "robot_state",
            None,
        ],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "roboturk": {
        "image_obs_keys": [
            "front_rgb",
            None,
            None,
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [None] * 8,
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "nyu_door_opening_surprising_effectiveness": {
        "image_obs_keys": [
            None,
            None,
            "image",
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [None] * 8,
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "viola": {
        "image_obs_keys": [
            "agentview_rgb",
            None,
            "eye_in_hand_rgb",
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "joint_states",
            "gripper_states",
        ],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_autolab_ur5": {
        "image_obs_keys": [
            "image",
            None,
            "hand_image",
        ],
        "depth_obs_keys": ["depth", None, None],
        "state_obs_keys": [
            "state",
        ],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "toto": {
        "image_obs_keys": [
            "image",
            None,
            None,
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": ["state", None],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "language_table": {
        "image_obs_keys": [
            "rgb",
            None,
            None,
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "effector_translation",
            *([None] * 6),
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "columbia_cairlab_pusht_real": {
        "image_obs_keys": [
            "image",
            None,
            "wrist_image",
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "robot_state",
            *([None] * 6),
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image",
            None,
            None,
        ],
        "depth_obs_keys": ["depth_image", None, None],
        "state_obs_keys": ["ee_position", "ee_orientation", None],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "nyu_rot_dataset_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image",
            None,
            None,
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "eef_state",
            None,
            "gripper_state",
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "stanford_hydra_dataset_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image",
            None,
            "wrist_image",
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "eef_state",
            None,
            "gripper_state",
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "austin_buds_dataset_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image",
            None,
            "wrist_image",
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "state",
        ],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "nyu_franka_play_dataset_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image",
            "image_additional_view",
            None,
        ],
        "depth_obs_keys": ["depth", "depth_additional_view", None],
        "state_obs_keys": ["eef_state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "maniskill_dataset_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image",
            None,
            "wrist_image",
        ],
        "depth_obs_keys": [
            "depth",
            None,
            "wrist_depth",
        ],
        "state_obs_keys": [
            "tcp_pose",
            "gripper_state",
        ],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "furniture_bench_dataset_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image",
            None,
            "wrist_image",
        ],
        "depth_obs_keys": [
            None,
            None,
            None,
        ],
        "state_obs_keys": [
            "state",
        ],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": {
        "image_obs_keys": [
            "highres_image",
            None,
            None,
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [None] * 8,
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "ucsd_kitchen_dataset_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image",
            None,
            None,
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "joint_state",
            None,
        ],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image",
            None,
            None,
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "eef_state",
            None,
            "gripper_state",
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "austin_sailor_dataset_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image",
            None,
            "wrist_image",
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "state",
        ],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "austin_sirius_dataset_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image",
            None,
            "wrist_image",
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "state",
        ],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "bc_z": {
        "image_obs_keys": [
            "image",
            None,
            None,
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "present/xyz",
            "present/axis_angle",
            None,
            "present/sensed_close",
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image",
            None,
            None,
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "eef_state",
            None,
            "gripper_state",
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image",
            None,
            None,
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "eef_state",
            None,
            "gripper_state",
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image",
            "image2",
            "hand_image",
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "end_effector_pose",
            None,
            None,
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utokyo_xarm_bimanual_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image",
            None,
            None,
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "pose_r",
            None,
            None,
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "robo_net": {
        "image_obs_keys": [
            "image",
            "image1",
            None,
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "eef_state",
            None,
            "gripper_state",
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_mvp_converted_externally_to_rlds": {
        "image_obs_keys": [
            None,
            None,
            "hand_image",
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "pose",
            "gripper",
        ],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.JOINT_POS,
    },
    "berkeley_rpt_converted_externally_to_rlds": {
        "image_obs_keys": [
            None,
            None,
            "hand_image",
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "joint_pos",
            "gripper",
        ],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.JOINT_POS,
    },
    "kaist_nonprehensile_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image",
            None,
            None,
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "state",
            None,
        ],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "stanford_mask_vit_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image",
            None,
            None,
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "eef_state",
            None,
            "gripper_state",
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "tokyo_u_lsmo_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image",
            None,
            None,
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "eef_state",
            None,
            "gripper_state",
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "dlr_sara_pour_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image",
            None,
            None,
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "state",
            None,
            None,
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "dlr_sara_grid_clamp_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image",
            None,
            None,
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "state",
            None,
            None,
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "dlr_edan_shared_control_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image",
            None,
            None,
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "state",
            None,
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "asu_table_top_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image",
            None,
            None,
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "eef_state",
            None,
            "gripper_state",
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "stanford_robocook_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image_1",
            "image_2",
            None,
        ],
        "depth_obs_keys": ["depth_1", "depth_2", None],
        "state_obs_keys": [
            "eef_state",
            None,
            "gripper_state",
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "imperialcollege_sawyer_wrist_cam": {
        "image_obs_keys": [
            "image",
            None,
            "wrist_image",
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            *([None] * 7),
            "state",
        ],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
        "image_obs_keys": [
            "image",
            None,
            "wrist_image",
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "joint_state",
            "gripper_state",
        ],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "uiuc_d3field": {
        "image_obs_keys": [
            "image_1",
            "image_2",
            None,
        ],
        "depth_obs_keys": ["depth_1", "depth_2", None],
        "state_obs_keys": [None] * 8,
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utaustin_mutex": {
        "image_obs_keys": [
            "image",
            None,
            "wrist_image",
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "state",
        ],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_fanuc_manipulation": {
        "image_obs_keys": [
            "image",
            None,
            "wrist_image",
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": ["joint_state", None, "gripper_state"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "cmu_playing_with_food": {
        "image_obs_keys": [
            "image",
            None,
            "finger_vision_1",
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "state",
            None,
            None,
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "cmu_play_fusion": {
        "image_obs_keys": [
            "image",
            None,
            None,
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "state",
        ],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "cmu_stretch": {
        "image_obs_keys": [
            "image",
            None,
            None,
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "eef_state",
            None,
            "gripper_state",
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_gnm_recon": {
        "image_obs_keys": [
            None,
            None,
            "image",
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "state",
            None,
            None,
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_gnm_cory_hall": {
        "image_obs_keys": [
            None,
            None,
            "image",
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "state",
            None,
            None,
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_gnm_sac_son": {
        "image_obs_keys": [
            None,
            None,
            "image",
        ],
        "depth_obs_keys": [None, None, None],
        "state_obs_keys": [
            "state",
            None,
            None,
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
}
