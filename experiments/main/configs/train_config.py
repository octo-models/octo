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

    base_real_config = dict(
        batch_size=4,
        num_steps=int(2e6),
        log_interval=100,
        eval_interval=5000,
        save_interval=5000,
        save_dir="/mnt2/homer/jaxrl_log",
        data_path="/nfs/kun2/datasets/r2d2/tfds",
        resume_path=placeholder(str),
        seed=42,
        text_processor=None,
        text_processor_kwargs=dict(),
        pretrained_weights=[],
        wandb=base_wandb_config,
        shuffle_buffer_size=25000,
    )

    # params that need to be specified multiple places
    normalization_type = "normal"

    base_data_config = dict(
        name="r2_d2_pen",
        data_dir="/nfs/kun2/datasets/r2d2/tfds",
        image_obs_key="exterior_image_1_left",
        state_obs_key="joint_position",
        obs_horizon=1,
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
        )
    )

    base_encoder_kwargs = dict(
        encoder="resnetv1-34-bridge",
        encoder_kwargs=dict(
            pooling_method="none", add_spatial_coordinates=True, act="swish"
        ),
    )

    possible_structures = {
        "transformer_bc": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
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
                dataset_kwargs=base_data_config,
                **base_real_config,
            )
        ),
        "transformer_bc_film_lang": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={
                        "obs-film-language-tokenizer": {"num_tokens": 64}
                    },
                    task_tokenizer_kwargs={},
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=base_data_config,
                **base_real_config,
            )
        ),
        "transformer_bc_lang": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={"obs-tokenizer": {"num_tokens": 64}},
                    task_tokenizer_kwargs={"language-tokenizer": {"num_tokens": 16}},
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=base_data_config,
                **base_real_config,
            )
        ),
        "transformer_bc_clip_text": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={"obs-tokenizer": {"num_tokens": 64}},
                    task_tokenizer_kwargs={"clip-text-tokenizer": {"num_tokens": 64}},
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=base_data_config,
                **update_config(
                    base_real_config,
                    text_processor="clip_processor",
                    pretrained_weights=["clip"],
                ),
            )
        ),
        "transformer_bc_clip_vit_and_text": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizer_kwargs={
                        "clip-obs-tokenizer": {"num_tokens": 50}
                    },
                    task_tokenizer_kwargs={"clip-text-tokenizer": {"num_tokens": 64}},
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=update_config(base_data_config, image_processor="clip"),
                **update_config(
                    base_real_config,
                    text_processor="clip_processor",
                    pretrained_weights=["clip"],
                ),
            )
        ),
    }

    return possible_structures[config_string]
