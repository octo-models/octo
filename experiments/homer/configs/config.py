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
        project="octo", group=placeholder(str), entity=placeholder(str)
    )

    base_config = dict(
        batch_size=256,
        num_steps=int(2e6),
        start_step=placeholder(int),
        log_interval=1000,
        eval_interval=5000,
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
        pretrained_loaders=[],
        pretrained_loader_kwargs=[],
        wandb=base_wandb_config,
        shuffle_buffer_size=25000,
        exec_horizon=2,
    )

    # params that need to be specified multiple places
    normalization_type = "normal"

    base_data_config = dict(
        data_path="/mnt2/homer/datasets/mujoco_sim/franka_shoe_pick_and_place_2K_20230709-201001",
        window_size=8,
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
            pred_horizon=4,
        )
    )

    base_tokenizer_kwargs = dict(
        encoder="resnetv1-18-bridge",
        encoder_kwargs=dict(
            pooling_method="none", add_spatial_coordinates=True, act="swish"
        ),
        task_stack_keys=[
            "image_.*"
        ],  # by default, early fuse goal images into visual encoder
    )

    possible_structures = {
        "sim": ConfigDict(
            dict(
                model=update_config(
                    base_model_config,
                    observation_tokenizers=[
                        (
                            "image_tokenizer",
                            {"num_tokens": 16, **base_tokenizer_kwargs},
                        ),
                    ],
                    task_tokenizers=[],
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=base_data_config,
                **base_config,
            )
        ),
    }

    return possible_structures[config_string]
