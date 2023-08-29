import ml_collections

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
                    "action": {"mean": ACT_MEAN, "std": ACT_STD},
                    "proprio": {"mean": ACT_MEAN, "std": ACT_STD},
                },
            }
        ),
        "test": ml_collections.ConfigDict(
            {
                "include": [["rss/?*/?*/?*"]],
                "exclude": [],
                "sample_weights": None,
                "action_proprio_metadata": {
                    "action": {"mean": ACT_MEAN, "std": ACT_STD},
                    "proprio": {"mean": ACT_MEAN, "std": ACT_STD},
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
                "action_metadata": {"mean": ACT_MEAN, "std": ACT_STD},
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
                "action_metadata": {"mean": ACT_MEAN, "std": ACT_STD},
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
                "action_metadata": {"mean": ACT_MEAN, "std": ACT_STD},
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
                "action_metadata": {"mean": ACT_MEAN, "std": ACT_STD},
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
                "action_metadata": {"mean": ACT_MEAN, "std": ACT_STD},
            }
        ),
    }
    return possible_structures[config_string]
