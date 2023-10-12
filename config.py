from copy import deepcopy

from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder


def update_config(config, **kwargs):
    new_config = deepcopy(config)
    for key, value in kwargs.items():
        if key in config:
            if isinstance(config[key], dict) or isinstance(config[key], ConfigDict):
                new_config[key] = update_config(config[key], **value)
            else:
                new_config[key] = value
        else:
            new_config[key] = value
    return ConfigDict(new_config)


def get_config(config_string):
    base_wandb_config = dict(
        project="orca", group=placeholder(str), entity=placeholder(str)
    )

    base_config = dict(
        batch_size=256,
        shuffle_buffer_size=1000,
        num_val_batches=8,
        num_steps=int(2e6),
        start_step=placeholder(int),
        log_interval=100,
        eval_interval=5000,
        save_interval=5000,
        save_dir=placeholder(str),
        resume_path=placeholder(str),
        seed=42,
        text_processor=None,
        text_processor_kwargs=dict(),
        pretrained_weights=[],
        wandb=base_wandb_config,
    )

    # params that need to be specified multiple places
    normalization_type = "normal"

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
        action_proprio_normalization_type=normalization_type,
    )

    base_bridge_data_config = {
        "common_kwargs": base_data_config,
        "data_kwargs_list": [
            {
                "name": "bridge_dataset",
                "data_dir": "/nfs/kun2/datasets/tfds",
                "image_obs_keys": ["image_0"],
                "state_obs_keys": ["state"],
            },
        ],
    }

    base_optimizer_config = dict(
        learning_rate=3e-4, warmup_steps=2000, decay_steps=int(2e6)
    )

    base_model_config = dict(
        policy_kwargs=dict(
            num_layers=4,
            mlp_dim=1024,
            vocab_size=256,
            num_heads=8,
            dropout_rate=0.1,
            normalization_type=normalization_type,
            pred_horizon=1,
        )
    )

    base_tokenizer_kwargs = dict(
        encoder="resnetv1-34-bridge",
        encoder_kwargs=dict(
            pooling_method="none", add_spatial_coordinates=True, act="swish"
        ),
        task_stack_keys=[
            "image_.*"
        ],  # by default, early fuse goal images into visual encoder
    )

    base_task_augmentation_kwargs = dict(
        task_augmentation_strategy="drop_keys_independent",
        task_augmentation_kwargs=dict(
            drop_key_groups_probs=[(["image_0"], 0.5), (["language_instruction"], 0.5)],
            allow_drop_all=False,
        ),
    )

    possible_structures = {
        "gc_bridge": ConfigDict(
            dict(
                model=update_config(
                    base_model_config,
                    observation_tokenizers=[
                        (
                            "image_tokenizer",
                            {"num_tokens": 64, **base_tokenizer_kwargs},
                        ),
                    ],
                    task_tokenizers=[],
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=base_bridge_data_config,
                **base_config,
            )
        ),
        "gc_r2d2": ConfigDict(
            dict(
                model=update_config(
                    base_model_config,
                    observation_tokenizers=[
                        (
                            "image_tokenizer",
                            {"num_tokens": 60, **base_tokenizer_kwargs},
                        ),
                    ],
                    task_tokenizers=[],
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs={
                    "common_kwargs": base_data_config,
                    "data_kwargs_list": [
                        {
                            "name": "r2_d2_pen",
                            "data_dir": "/nfs/kun2/datasets/r2d2/tfds",
                            "image_obs_keys": [
                                "exterior_image_1_left",
                                "exterior_image_2_left",
                                "wrist_image_left",
                            ],
                            "state_obs_keys": ["joint_position"],
                        },
                    ],
                },
                **base_config,
            )
        ),
        "gc_bridge_r2d2": ConfigDict(
            dict(
                model=update_config(
                    base_model_config,
                    observation_tokenizers=[
                        (
                            "image_tokenizer",
                            {"num_tokens": 60, **base_tokenizer_kwargs},
                        ),
                    ],
                    task_tokenizers=[],
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs={
                    "common_kwargs": update_config(
                        base_data_config,
                        resize_size=(180, 320),
                    ),
                    "data_kwargs_list": [
                        {
                            "name": "r2_d2_pen",
                            "data_dir": "/nfs/kun2/datasets/r2d2/tfds",
                            "image_obs_keys": [
                                "exterior_image_1_left",
                                "exterior_image_2_left",
                                "wrist_image_left",
                            ],
                            "state_obs_keys": ["joint_position"],
                        },
                        {
                            "name": "bridge_dataset",
                            "data_dir": "/nfs/kun2/datasets/tfds",
                            "image_obs_keys": ["image_0", None, None],
                            "state_obs_keys": ["state"],
                        },
                    ],
                },
                **base_config,
            )
        ),
        "lc_film_bridge": ConfigDict(
            dict(
                model=update_config(
                    base_model_config,
                    observation_tokenizers=[
                        (
                            "image_tokenizer",
                            update_config(
                                base_tokenizer_kwargs,
                                num_tokens=64,
                                task_stack_keys=[],
                                task_film_keys=["language_instruction"],
                            ),
                        ),
                    ],
                    task_tokenizers=[],
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=base_bridge_data_config,
                **update_config(
                    base_config,
                    text_processor="muse_embedding",
                ),
            )
        ),
        "lc_bridge": ConfigDict(
            dict(
                model=update_config(
                    base_model_config,
                    observation_tokenizers=[
                        (
                            "image_tokenizer",
                            update_config(
                                base_tokenizer_kwargs,
                                num_tokens=64,
                                task_stack_keys=[],
                            ),
                        ),
                    ],
                    task_tokenizers=[
                        ("language_tokenizer", {"num_tokens": 1}),
                    ],
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=base_bridge_data_config,
                **update_config(
                    base_config,
                    text_processor="muse_embedding",
                ),
            )
        ),
        "lc_distilbert_bridge": ConfigDict(
            dict(
                model=update_config(
                    base_model_config,
                    observation_tokenizers=[
                        (
                            "image_tokenizer",
                            update_config(
                                base_tokenizer_kwargs,
                                num_tokens=64,
                                task_stack_keys=[],
                            ),
                        ),
                    ],
                    task_tokenizers=[
                        (
                            "language_tokenizer",
                            {
                                "num_tokens": 64,
                                "projection_dim": 512,
                                "encoder": "distilbert-base-uncased",
                            },
                        ),
                    ],
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=base_bridge_data_config,
                **update_config(
                    base_config,
                    text_processor="hf_tokenizer",
                    text_processor_kwargs=dict(
                        tokenizer_name="distilbert-base-uncased",
                        encode_with_model=False,
                    ),
                    pretrained_weights=["distilbert"],
                ),
            )
        ),
        "multimodal_independent_bridge": ConfigDict(
            dict(
                model=update_config(
                    base_model_config,
                    observation_tokenizers=[
                        (
                            "image_tokenizer",
                            update_config(
                                base_tokenizer_kwargs,
                                num_tokens=64,
                                task_stack_keys=["image_.*"],
                                task_film_keys=["language_instruction"],
                            ),
                        ),
                    ],
                    task_tokenizers=[],
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=update_config(
                    base_bridge_data_config,
                    common_kwargs=dict(
                        **base_task_augmentation_kwargs,
                    ),
                ),
                **update_config(
                    base_config,
                    text_processor="muse_embedding",
                ),
            )
        ),
        "multimodal_switch_bridge": ConfigDict(
            dict(
                model=update_config(
                    base_model_config,
                    observation_tokenizers=[
                        (
                            "image_tokenizer",
                            update_config(
                                base_tokenizer_kwargs,
                                num_tokens=64,
                                task_stack_keys=["image_.*"],
                                task_film_keys=["language_instruction"],
                            ),
                        ),
                    ],
                    task_tokenizers=[],
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=update_config(
                    base_bridge_data_config,
                    common_kwargs=dict(
                        task_augmentation_strategy="switch_keys",
                        task_augmentation_kwargs=dict(
                            switch_key_groups_probs=[
                                (["image_0"], 0.5),
                                (["language_instruction"], 0.5),
                            ],
                        ),
                    ),
                ),
                **update_config(
                    base_config,
                    text_processor="muse_embedding",
                ),
            )
        ),
        "ci_debug_dataset": ConfigDict(
            dict(
                model=dict(
                    policy_kwargs=dict(
                        num_layers=1,
                        mlp_dim=128,
                        vocab_size=256,
                        num_heads=1,
                        dropout_rate=0.1,
                        normalization_type=normalization_type,
                        pred_horizon=1,
                    ),
                    observation_tokenizers=[
                        (
                            "image_tokenizer",
                            dict(
                                num_tokens=64,
                                encoder="resnetv1-18-bridge",
                                encoder_kwargs=dict(
                                    pooling_method="none",
                                    add_spatial_coordinates=True,
                                    act="swish",
                                ),
                                task_stack_keys=[
                                    "image_.*"
                                ],  # by default, early fuse goal images into visual encoder
                            ),
                        ),
                    ],
                    task_tokenizers=[],
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs={
                    "common_kwargs": update_config(
                        base_data_config,
                        resize_size=(256, 256),
                    ),
                    "data_kwargs_list": [
                        {
                            "name": "bridge_dataset",
                            "data_dir": "./datasets/debug_dataset",
                            "image_obs_keys": ["image_0"],
                            "state_obs_keys": ["state"],
                        },
                    ],
                },
                **update_config(base_config, batch_size=2, num_steps=20),
            )
        ),
    }

    return possible_structures[config_string]
