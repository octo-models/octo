from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder, required_placeholder

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


def update_config(_prototype, **kwargs):
    result = dict(_prototype)
    for key, value in kwargs.items():
        if type(result.get(key)) == dict or type(result.get(key)) == ConfigDict:
            if not kwargs[key].get("_overwrite", False):
                value = dict(update_config(_prototype=result[key], **kwargs[key]))
            value.pop("_overwrite", None)
        result[key] = value
    result.pop("_overwrite", None)
    return ConfigDict(result)


def get_config(config_string):
    base_wandb_config = dict(
        project="orca",
        group=placeholder(str),
        entity=placeholder(str),
    )

    base_real_config = dict(
        batch_size=64,
        shuffle_buffer_size=1000,
        num_steps=int(2e6),
        log_interval=100,
        eval_interval=5000,
        save_interval=5000,
        save_dir=placeholder(str),
        data_path=placeholder(str),
        resume_path=placeholder(str),
        seed=42,
        text_processor="muse_embedding",
        text_processor_kwargs=dict(),
        pretrained_weights=[],
        wandb=base_wandb_config,
    )

    # params that need to be specified multiple places
    normalization_type = "normal"

    base_data_config = dict(
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
        action_proprio_metadata={
            "action": {
                "mean": ACT_MEAN,
                "std": ACT_STD,
            },
            "proprio": {
                "mean": ACT_MEAN,
                "std": ACT_STD,
            },
        },
    )
    base_optimizer_config = dict(
        learning_rate=3e-4,
        warmup_steps=2000,
        decay_steps=int(2e6),
    )

    base_model_config = dict(
        policy_kwargs=dict(
            num_layers=4,
            mlp_dim=1024,
            vocab_size=256,
            num_heads=8,
            dropout_rate=0.1,
            action_proprio_normalization_type=normalization_type,
        ),
    )

    possible_structures = {
        "transformer_bc": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizers=["obs-tokenizer"],
                    observation_tokenizer_kwargs={"obs-tokenizer": {}},
                    task_tokenizers=["goal-obs-tokenizer"],
                    task_tokenizer_kwargs={"goal-obs-tokenizer": {}},
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
                    observation_tokenizers=["obs-film-language-tokenizer"],
                    observation_tokenizer_kwargs={
                        "obs-film-language-tokenizer": {"num_tokens": 64}
                    },
                    task_tokenizers=[],
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
                    observation_tokenizers=["obs-tokenizer"],
                    observation_tokenizer_kwargs={"obs-tokenizer": {"num_tokens": 64}},
                    task_tokenizers=["language-tokenizer"],
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
                    observation_tokenizers=["obs-tokenizer"],
                    observation_tokenizer_kwargs={"obs-tokenizer": {"num_tokens": 64}},
                    task_tokenizers=["clip-text-tokenizer"],
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
                    observation_tokenizers=["clip-obs-tokenizer"],
                    observation_tokenizer_kwargs={
                        "clip-obs-tokenizer": {"num_tokens": 50}
                    },
                    task_tokenizers=["clip-text-tokenizer"],
                    task_tokenizer_kwargs={"clip-text-tokenizer": {"num_tokens": 64}},
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=update_config(
                    base_data_config, use_clip_image_preprocessing=True
                ),
                **update_config(
                    base_real_config,
                    text_processor="clip_processor",
                    pretrained_weights=["clip"],
                ),
            )
        ),
    }

    return possible_structures[config_string]
