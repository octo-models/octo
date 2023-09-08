from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder
from copy import deepcopy


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
        log_interval=100,
        eval_interval=5000,
        save_interval=5000,
        save_dir=placeholder(str),
        resume_path=placeholder(str),
        seed=42,
        text_processor="muse_embedding",
        text_processor_kwargs=dict(),
        pretrained_weights=[],
        wandb=base_wandb_config,
    )

    base_sim_config = dict(
        batch_size=4,
        num_steps=int(2e6),
        log_interval=100,
        eval_interval=1,
        save_interval=int(2e6),
        save_dir="/mnt2/homer/jaxrl_log",
        resume_path=None,
        seed=42,
        env_name="franka_shoe_pick_and_place",
        save_video=True,
        max_episode_steps=55,
        deterministic_eval=True,
        num_episodes_per_video=8,
        num_episodes_per_row=4,
        eval_episodes=20,
        num_val_batches=8,
        pretrained_weights=[],
        wandb=base_wandb_config,
        shuffle_buffer_size=25000,
        action_exec_horizon=2,
    )

    # params that need to be specified multiple places
    normalization_type = "normal"

    base_data_config = dict(
        name="bridge_dataset",
        data_dir="/nfs/kun2/datasets/tfds",
        image_obs_keys=["image_0"],
        state_obs_keys=["state"],
        horizon=1,
        augment_kwargs=dict(
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

    base_sim_data_config = dict(
        data_path="/mnt2/homer/datasets/mujoco_sim/franka_shoe_pick_and_place_2K_20230709-201001",
        horizon=8,
        augment_kwargs=dict(
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
            action_pred_horizon=4,
            attend_prev_actions=False,
        )
    )

    base_encoder_kwargs = dict(
        encoder="resnetv1-34-bridge",
        encoder_kwargs=dict(
            pooling_method="none", add_spatial_coordinates=True, act="swish"
        ),
    )

    base_sim_encoder_kwargs = dict(
        encoder="resnetv1-18-bridge",
        encoder_kwargs=dict(
            pooling_method="none", add_spatial_coordinates=True, act="swish"
        ),
    )

    possible_structures = {
        "sim_transformer_bc": ConfigDict(
            dict(
                agent="transformer_bc",
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={
                        "obs-tokenizer": {"num_tokens": 16, **base_sim_encoder_kwargs}
                    },
                    task_tokenizer_kwargs={
                        "goal-obs-tokenizer": {
                            "num_tokens": 16,
                            **base_sim_encoder_kwargs,
                        }
                    },
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=base_sim_data_config,
                **base_sim_config,
            )
        ),
        "transformer_bc_bridge": ConfigDict(
            dict(
                agent="transformer_bc",
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={
                        "obs-tokenizer": {"num_tokens": 64, **base_encoder_kwargs}
                    },
                    task_tokenizer_kwargs={
                        "goal-obs-tokenizer": {"num_tokens": 64, **base_encoder_kwargs}
                    },
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=update_config(
                    base_data_config,
                    name="bridge_dataset",
                    data_dir="/nfs/kun2/datasets/tfds",
                    image_obs_keys=["image_0"],
                    state_obs_keys=["state"],
                ),
                **base_config,
            )
        ),
        "transformer_bc_r2d2": ConfigDict(
            dict(
                agent="transformer_bc",
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={
                        "obs-tokenizer": {"num_tokens": 60, **base_encoder_kwargs}
                    },
                    task_tokenizer_kwargs={
                        "goal-obs-tokenizer": {"num_tokens": 60, **base_encoder_kwargs}
                    },
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=update_config(
                    base_data_config,
                    name="r2_d2_pen",
                    data_dir="/nfs/kun2/datasets/r2d2/tfds",
                    image_obs_keys=[
                        "exterior_image_1_left",
                        "exterior_image_2_left",
                        "wrist_image_left",
                    ],
                    state_obs_keys=["joint_position"],
                ),
                **base_config,
            )
        ),
        "transformer_bc_film_lang": ConfigDict(
            dict(
                agent="transformer_bc",
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={
                        "obs-film-language-tokenizer": {
                            "num_tokens": 64,
                            **base_encoder_kwargs,
                        }
                    },
                    task_tokenizer_kwargs={},
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=base_data_config,
                **base_config,
            )
        ),
        "transformer_bc_lang": ConfigDict(
            dict(
                agent="transformer_bc",
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={"obs-tokenizer": {"num_tokens": 64}},
                    task_tokenizer_kwargs={"language-tokenizer": {"num_tokens": 16}},
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=base_data_config,
                **base_config,
            )
        ),
        "transformer_bc_distilbert": ConfigDict(
            dict(
                agent="transformer_bc",
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={
                        "obs-tokenizer": {"num_tokens": 64, **base_encoder_kwargs}
                    },
                    task_tokenizer_kwargs={
                        "language-tokenizer": {
                            "num_tokens": 64,
                            "projection_dim": 512,
                            "encoder": "distilbert-base-uncased",
                        }
                    },
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=base_data_config,
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
    }

    return possible_structures[config_string]
