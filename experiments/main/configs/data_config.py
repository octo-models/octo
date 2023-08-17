import ml_collections

PNP_TASKS = [
    "bridge_data_v1/berkeley/toysink1_room8052/put_pan_from_sink_into_drying_rack/",
    "bridge_data_v1/berkeley/toysink1_room8052/put_pan_from_drying_rack_into_sink/",
    "bridge_data_v1/berkeley/toysink1_room8052/put_pan_on_stove_from_sink/",
    "bridge_data_v1/berkeley/toysink1_room8052/put_spoon_into_pan/",
    "bridge_data_v1/berkeley/toysink1_room8052/put_pan_from_stove_to_sink/",
    "bridge_data_v1/berkeley/toysink1_room8052/put_eggplant_into_pan/",
    "bridge_data_v1/berkeley/realkitchen1_counter/put_spoon_on_plate/",
    "bridge_data_v1/berkeley/realkitchen1_counter/pick_up_sponge_and_wipe_plate/",
    "bridge_data_v1/berkeley/realkitchen1_dishwasher/pick_up_any_cup/",
    "bridge_data_v1/berkeley/realkitchen1_dishwasher/pick_up_green_mug/",
    "bridge_data_v1/berkeley/realkitchen1_dishwasher/pick_up_glass_cup/",
    "bridge_data_v1/berkeley/toysink2_bww/put_carrot_on_plate/",
    "bridge_data_v1/berkeley/toysink2_bww/put_spoon_in_pot/",
    "bridge_data_v1/berkeley/toysink2_bww/put_knife_on_cutting_board/",
    "bridge_data_v1/berkeley/toysink2_bww/put_cup_from_counter_or_drying_rack_into_sink/",
    "bridge_data_v1/berkeley/toysink2_bww/put_eggplant_into_pot_or_pan/",
    "bridge_data_v1/berkeley/toykitchen4/put_banana_in_pot_or_pan/",
    "bridge_data_v1/berkeley/toykitchen4/put_lid_on_pot_or_pan/",
    "bridge_data_v1/berkeley/toykitchen4/put_pear_on_plate/",
    "bridge_data_v1/berkeley/toykitchen4/put_carrot_in_bowl/",
    "bridge_data_v1/berkeley/toykitchen4/put_sushi_in_pot_or_pan/",
    "bridge_data_v1/berkeley/toykitchen4/put_detergent_in_sink/",
    "bridge_data_v1/berkeley/laundry_machine/take_clothes_out_of_laundry_machine/",
    "bridge_data_v1/berkeley/laundry_machine/put_clothes_in_laundry_machine/",
    "bridge_data_v1/berkeley/toysink3_bww/put_cup_into_pot_or_pan/",
    "bridge_data_v1/berkeley/toysink3_bww/put_lid_on_pot_or_pan/",
    "bridge_data_v1/berkeley/toysink3_bww/put_cup_from_anywhere_into_sink/",
    "bridge_data_v1/berkeley/toysink3_bww/put_knife_in_pot_or_pan/",
    "bridge_data_v1/berkeley/toysink3_bww/put_green_squash_into_pot_or_pan/",
    "bridge_data_v1/berkeley/toysink3_bww/take_lid_off_pot_or_pan/",
    "bridge_data_v1/berkeley/toysink3_bww/put_pot_or_pan_from_sink_into_drying_rack/",
    "bridge_data_v1/berkeley/toysink3_bww/put_brush_into_pot_or_pan/",
    "bridge_data_v1/berkeley/toysink3_bww/put_detergent_from_sink_into_drying_rack/",
    "bridge_data_v1/berkeley/toykitchen1/put_sushi_on_plate/",
    "bridge_data_v1/berkeley/toykitchen1/put_pan_in_sink/",
    "bridge_data_v1/berkeley/toykitchen1/put_broccoli_in_bowl/",
    "bridge_data_v1/berkeley/toykitchen1/put_pot_on_stove_which_is_near_stove_distractors/",
    "bridge_data_v1/berkeley/toykitchen1/put_corn_into_bowl/",
    "bridge_data_v1/berkeley/toykitchen1/take_can_out_of_pan/",
    "bridge_data_v1/berkeley/toykitchen1/put_fork_from_basket_to_tray/",
    "bridge_data_v1/berkeley/toykitchen1/put_eggplant_on_plate/",
    "bridge_data_v1/berkeley/toykitchen1/put_lid_on_pot_or_pan/",
    "bridge_data_v1/berkeley/toykitchen1/put_corn_in_pot_which_is_in_sink_distractors/",
    "bridge_data_v1/berkeley/toykitchen1/put_carrot_on_plate/",
    "bridge_data_v1/berkeley/toykitchen1/take_carrot_off_plate_cardboardfence/",
    "bridge_data_v1/berkeley/toykitchen1/put_sweet_potato_in_pan_which_is_on_stove_distractors/",
    "bridge_data_v1/berkeley/toykitchen1/put_big_spoon_from_basket_to_tray/",
    "bridge_data_v1/berkeley/toykitchen1/take_broccoli_out_of_pan/",
    "bridge_data_v1/berkeley/toykitchen1/put_lid_on_stove/",
    "bridge_data_v1/berkeley/toykitchen1/put_sweet_potato_in_pan_which_is_on_stove/",
    "bridge_data_v1/berkeley/toykitchen1/put_corn_in_pan_which_is_on_stove_distractors/",
    "bridge_data_v1/berkeley/toykitchen1/put_pear_in_bowl/",
    "bridge_data_v1/berkeley/toykitchen1/pick_up_pan_from_stove_distractors/",
    "bridge_data_v1/berkeley/toykitchen1/put_banana_on_plate/",
    "bridge_data_v1/berkeley/toykitchen1/pick_up_pot_from_sink_distractors/",
    "bridge_data_v1/berkeley/toykitchen1/put_sweet_potato_in_pot_which_is_in_sink_distractors/",
    "bridge_data_v1/berkeley/toykitchen1/put_pot_in_sink/",
    "bridge_data_v1/berkeley/toykitchen1/take_sushi_out_of_pan/",
    "bridge_data_v1/berkeley/toykitchen1/put_detergent_in_sink/",
    "bridge_data_v1/berkeley/toykitchen1/take_broccoli_out_of_pan_cardboardfence/",
    "bridge_data_v1/berkeley/toykitchen1/put_broccoli_in_pot_cardboardfence/",
    "bridge_data_v1/berkeley/toykitchen1/take_lid_off_pot_or_pan/",
    "bridge_data_v1/berkeley/toykitchen1/take_carrot_off_plate/",
    "bridge_data_v1/berkeley/toykitchen1/put_eggplant_in_pot_or_pan/",
    "bridge_data_v1/berkeley/toykitchen1/put_carrot_on_plate_cardboardfence/",
    "bridge_data_v1/berkeley/toykitchen1/pick_up_bowl_and_put_in_small4fbox/",
    "bridge_data_v1/berkeley/toykitchen1/put_carrot_on_cutting_board/",
    "bridge_data_v1/berkeley/toykitchen1/put_red_bottle_in_sink/",
    "bridge_data_v1/berkeley/toykitchen1/put_pepper_in_pan/",
    "bridge_data_v1/berkeley/toykitchen1/put_broccoli_in_pot_or_pan/",
    "bridge_data_v1/berkeley/toykitchen1/put_knife_on_cutting_board/",
    "bridge_data_v1/berkeley/toykitchen1/put_small_spoon_from_basket_to_tray/",
    "bridge_data_v1/berkeley/toykitchen1/put_corn_in_pan_which-is_on_stove_distractors/",
    "bridge_data_v1/berkeley/toykitchen1/put_pepper_in_pot_or_pan/",
    "bridge_data_v1/berkeley/toykitchen1/put_green_squash_in_pot_or_pan/",
    "bridge_data_v1/berkeley/tabletop_dark_wood/put_spatula_on_cutting_board/",
    "bridge_data_v1/berkeley/tabletop_dark_wood/put_banana_in_colander/",
    "bridge_data_v1/berkeley/tabletop_dark_wood/take_banana_out_of_colander/",
    "bridge_data_v1/berkeley/toykitchen6/take_cup_off_plate/",
    "bridge_data_v1/berkeley/toykitchen6/put_spatula_on_plate_sink/",
    "bridge_data_v1/berkeley/toykitchen6/take_spoon_out_of_bowl_sink/",
    "bridge_data_v1/berkeley/toykitchen6/put_beet_in_pot_sink/",
    "bridge_data_v1/berkeley/toykitchen6/put_corn_in_bowl_sink/",
    "bridge_data_v1/berkeley/toykitchen6/put_blueberries_on_plate_sink/",
    "bridge_data_v1/berkeley/toykitchen6/take_corn_out_of_bowl_sink/",
    "bridge_data_v1/berkeley/toykitchen6/put_cup_on_plate/",
    "bridge_data_v1/berkeley/toykitchen6/take_blueberries_off_plate_sink/",
    "bridge_data_v1/berkeley/toykitchen6/take_beet_from_pot_sink/",
    "bridge_data_v1/berkeley/toykitchen6/take_spatula_off_plate_sink/",
    "bridge_data_v1/berkeley/toykitchen6/put_spoon_in_bowl_sink/",
    "bridge_data_v1/berkeley/tabletop_white/put_sushi_on_plate/",
    "bridge_data_v1/berkeley/tabletop_white/take_sushi_off_plate/",
    "bridge_data_v1/berkeley/tabletop_light_wood/put_cucumber_in_cup/",
    "bridge_data_v1/berkeley/tabletop_light_wood/take_cucumber_out_of_cup/",
    "bridge_data_v1/berkeley/toykitchen2/take_bowl_off_plate_cardboard_fence/",
    "bridge_data_v1/berkeley/toykitchen2/put_bowl_on_plate/",
    "bridge_data_v1/berkeley/toykitchen2/take_bowl_off_plate/",
    "bridge_data_v1/berkeley/toykitchen2/take_sushi_out_of_pot_cardboard_fence/",
    "bridge_data_v1/berkeley/toykitchen2/take_lid_off_pot_cardboardfence/",
    "bridge_data_v1/berkeley/toykitchen2/put_potato_in_pot_cardboard_fence/",
    "bridge_data_v1/berkeley/toykitchen2/take_carrot_out_of_pot_cardboard_fence/",
    "bridge_data_v1/berkeley/toykitchen2/put_carrot_in_pot_cardboard_fence/",
    "bridge_data_v1/berkeley/toykitchen2/put_knife_on_cutting_board_cardboard_fence/",
    "bridge_data_v1/berkeley/toykitchen2/put_cap_on_container/",
    "bridge_data_v1/berkeley/toykitchen2/put_sushi_in_pot_cardboard_fence/",
    "bridge_data_v1/berkeley/toykitchen2/put_bowl_on_plate_cardboard_fence/",
    "bridge_data_v1/berkeley/toykitchen2/put_lid_on_pot_cardboardfence/",
    "bridge_data_v1/berkeley/toykitchen2/put_banana_in_pot_cardboard_fence/",
    "bridge_data_v1/berkeley/toykitchen2/put_pear_in_bowl_cardboardfence/",
    "bridge_data_v1/berkeley/toykitchen2/put_knife_in_pot_cardboard_fence/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_spatula_in_pan/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_pot_or_pan_on_stove/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_potato_in_pot_or_pan/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_pear_in_bowl/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_knife_on_cutting_board/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_pot_or_pan_in_sink/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_strawberry_in_pot/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_lemon_on_plate/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_corn_on_plate/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_sushi_on_plate/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_can_in_pot/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_potato_on_plate/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/lift_bowl/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_carrot_in_pot_or_pan/",
    "bridge_data_v1/berkeley/toykitchen2_room8052/put_sweet_potato_in_pot/",
    "bridge_data_v1/berkeley/tool_chest/pick_up_blue_pen_and_put_into_drawer/",
    "bridge_data_v1/berkeley/tool_chest/pick_up_red_srewdriver/",
    "bridge_data_v1/berkeley/tool_chest/pick_up_box_cutter_and_put_into_drawer/",
    "bridge_data_v1/berkeley/tool_chest/pick_up_violet_Allen_key/",
    "bridge_data_v1/berkeley/tool_chest/pick_up_bit_holder/",
    "bridge_data_v1/berkeley/tool_chest/pick_up_scissors_and_put_into_drawer/",
    "bridge_data_v1/berkeley/tool_chest/pick_up_glue_and_put_into_drawer/",
]

ACT_MEAN = [
    1.9296819e-04,
    1.3667766e-04,
    -1.4583133e-04,
    -1.8390431e-04,
    -3.0808983e-04,
    2.7425270e-04,
    5.9716219e-01,
]

ACT_STD = [
    0.00912848,
    0.0127196,
    0.01229497,
    0.02606696,
    0.02875283,
    0.07807977,
    0.48710242,
]


def get_config(config_string):
    possible_structures = {
        "all": ml_collections.ConfigDict(
            {
                "include": [
                    [
                        "icra/?*/?*/?*",
                        "flap/?*/?*/?*",
                        "bridge_data_v1/berkeley/?*/?*",
                        "rss/?*/?*/?*",
                        "bridge_data_v2/?*/?*/?*",
                        "scripted/?*",
                    ]
                ],
                "exclude": [],
                "sample_weights": None,
                "action_proprio_metadata": {
                    "action": {
                        "mean": ACT_MEAN,
                        "std": ACT_STD,
                    },
                    "proprio": {
                        "mean": ACT_MEAN,
                        "std": ACT_STD,
                    },
                },
            }
        ),
        "test": ml_collections.ConfigDict(
            {
                "include": [
                    [
                        "rss/?*/?*/?*",
                    ]
                ],
                "exclude": [],
                "sample_weights": None,
                "action_proprio_metadata": {
                    "action": {
                        "mean": ACT_MEAN,
                        "std": ACT_STD,
                    },
                    "proprio": {
                        "mean": ACT_MEAN,
                        "std": ACT_STD,
                    },
                },
            }
        ),
        "all_except_scripted": ml_collections.ConfigDict(
            {
                "include": [
                    [
                        "icra/?*/?*/?*",
                        "icra_validation/?*/?*/?*",
                        "flap/?*/?*/?*",
                        "bridge_data_v1/berkeley/?*/?*",
                        "rss/?*/?*/?*",
                        "bridge_data_v2/?*/?*/?*",
                    ]
                ],
                "exclude": [],
                "sample_weights": None,
                "action_metadata": {
                    "mean": ACT_MEAN,
                    "std": ACT_STD,
                },
            }
        ),
        "franka": ml_collections.ConfigDict(
            {
                "include": [["?*"]],
                "exclude": [],
                "sample_weights": None,
                "action_metadata": {
                    "mean": [
                        5.2401489e-01,
                        -6.7343891e-02,
                        2.5386891e-01,
                        2.6513453e00,
                        -8.4149389e-04,
                        1.2696550e-02,
                        2.9238686e-01,
                    ],
                    "std": [
                        0.08792825,
                        0.08270102,
                        0.11227315,
                        1.5259572,
                        0.09435784,
                        0.16661045,
                        0.41294536,
                    ],
                },
            }
        ),
        "all_exclude_toykitchen7": ml_collections.ConfigDict(
            {
                "include": [
                    [
                        "icra/?*/?*/?*",
                        "icra_validation/?*/?*/?*",
                        "flap/?*/?*/?*",
                        "bridge_data_v1/berkeley/?*/?*",
                        "rss/?*/?*/?*",
                        "bridge_data_v2/?*/?*/?*",
                        "scripted/?*",
                    ]
                ],
                "exclude": [
                    "*toykitchen7*",
                    "*tabletop_dark_wood*",
                    "*icra_validation/toykitchen_fixed_cam_offline_validation/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree_combo/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree_push_sweep/tabletop*",
                ],
                "sample_weights": None,
                "action_metadata": {
                    "mean": ACT_MEAN,
                    "std": ACT_STD,
                },
            }
        ),
        "all_finetune": ml_collections.ConfigDict(
            {
                "include": [
                    [
                        "icra/*/*/*",
                        "icra_validation/?*/?*/?*",
                        "flap/?*/?*/?*",
                        "bridge_data_v1/berkeley/?*/?*",
                        "rss/toykitchen2/?*/?*",
                        "rss/toykitchen6/?*/?*",
                    ],
                    ["rss/toykitchen7/pnp_sweep_target_fixed/?*"],
                ],
                "exclude": [
                    "*tabletop_dark_wood*",
                    "*icra_validation/toykitchen_fixed_cam_offline_validation/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree_combo/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree_push_sweep/tabletop*",
                ],
                "sample_weights": [0.9, 0.1],
                "action_metadata": {
                    "mean": ACT_MEAN,
                    "std": ACT_STD,
                },
            }
        ),
        "all_finetune_autonomous": ml_collections.ConfigDict(
            {
                "include": [
                    [
                        "icra/?*/?*/?*",
                        "icra_validation/?*/?*/?*",
                        "flap/?*/?*/?*",
                        "bridge_data_v1/berkeley/?*/?*",
                        "rss/?*/?*/?*",
                        "bridge_data_v2/?*/?*/?*",
                        "scripted/?*",
                    ],
                    ["learned/toykitchen7/pnp_sweep_v2"],
                ],
                "exclude": [
                    "*rss/toykitchen7*",
                    "*tabletop_dark_wood*",
                    "*icra_validation/toykitchen_fixed_cam_offline_validation/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree_combo/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree_push_sweep/tabletop*",
                    "*sweep_12-03*",
                ],
                "sample_weights": [0.9, 0.1],
                "action_metadata": {
                    "mean": ACT_MEAN,
                    "std": ACT_STD,
                },
            }
        ),
        "all_finetune_autonomous_oracle": ml_collections.ConfigDict(
            {
                "include": [
                    [
                        "icra/*/*/*",
                        "icra_validation/?*/?*/?*",
                        "flap/?*/?*/?*",
                        "bridge_data_v1/berkeley/?*/?*",
                        "rss/toykitchen2/?*/?*",
                        "rss/toykitchen6/?*/?*",
                    ],
                    [
                        "finetuning/ours_2_22/?*",
                        "rss/toykitchen7/pnp_sweep_target_fixed/?*",
                    ],
                ],
                "exclude": [
                    "*tabletop_dark_wood*",
                    "*icra_validation/toykitchen_fixed_cam_offline_validation/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree_combo/tabletop*",
                    "*icra/toykitchen_fixed_cam_resetfree_push_sweep/tabletop*",
                ],
                "sample_weights": [0.9, 0.1],
                "action_metadata": {
                    "mean": ACT_MEAN,
                    "std": ACT_STD,
                },
            }
        ),
    }
    return possible_structures[config_string]
