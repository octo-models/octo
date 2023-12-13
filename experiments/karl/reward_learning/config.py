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
        project="octo", group=placeholder(str), entity=placeholder(str)
    )

    base_config = dict(
        batch_size=256,
        shuffle_buffer_size=10000,
        num_val_batches=8,
        num_steps=int(2e6),
        start_step=placeholder(int),
        log_interval=1000,
        eval_interval=20000,
        save_interval=5000,
        save_dir="gs://karl-central-1",  # placeholder(str),
        resume_path=placeholder(str),
        seed=42,
        text_processor=None,
        text_processor_kwargs=dict(),
        pretrained_loaders=[],
        pretrained_loader_kwargs=[],
        wandb=base_wandb_config,
        eval_datasets=None,
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
                "data_dir": "gs://rail-octo-central1",  # "/nfs/kun2/datasets/tfds",
                "image_obs_keys": ["image_0"],
                "state_obs_keys": ["state"],
            },
        ],
    }

    base_optimizer_config = dict(
        learning_rate=3e-4, warmup_steps=2000, decay_steps=int(2e6)
    )

    base_model_config = dict(
        token_embedding_size=256,
        max_horizon=10,
        readouts=dict(reward=1),
        transformer_kwargs=dict(
            num_layers=4,
            mlp_dim=1024,
            num_attention_heads=8,
            dropout_rate=0.1,
        ),
        heads=dict(
            reward=dict(
                cls_name="temporal_distance_reward_head",
                kwargs=dict(
                    n_bins=100,
                    readout_key="reward",
                ),
            )
        ),
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
        "lc_film_bridge": ConfigDict(
            dict(
                model=update_config(
                    base_model_config,
                    observation_tokenizers=[
                        (
                            "image_tokenizer",
                            update_config(
                                base_tokenizer_kwargs,
                                encoder="resnetv1-34-bridge-film",
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
        "multimodal_switch_bridge": ConfigDict(
            dict(
                model=update_config(
                    base_model_config,
                    observation_tokenizers=[
                        (
                            "image_tokenizer",
                            update_config(
                                base_tokenizer_kwargs,
                                encoder="resnetv1-34-bridge-film",
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
    }

    return possible_structures[config_string]
