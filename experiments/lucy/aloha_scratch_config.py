from copy import deepcopy

from config import update_config
from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder


def get_config(
    transformer_size,
):
    assert transformer_size in ["vanilla", "vit_ti", "vit_s", "vit_b", "vit_l"]

    base_wandb_config = dict(
        project="orca", group=placeholder(str), entity=placeholder(str)
    )

    base_config = dict(
        batch_size=128,
        eval_batch_size=128,
        shuffle_buffer_size=10000,
        val_shuffle_buffer_size=1000,
        num_val_batches=16,
        num_steps=20000, #int(2e6),
        start_step=placeholder(int),
        log_interval=100,
        eval_interval=500, #5000,
        save_interval=500, #5000,
        save_dir=placeholder(str),
        resume_path=placeholder(str),
        seed=42,
        text_processor=None,
        text_processor_kwargs=dict(),
        pretrained_weights=[],
        wandb=base_wandb_config,
        wandb_resume_id=placeholder(str),
        eval_datasets=None,
    )

    # params that need to be specified multiple places
    normalization_type = "normal"

    base_data_config = dict(
        window_size=1,
        additional_action_window_size=49,
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
    )

    base_optimizer_config = dict(
        learning_rate=dict(
            init_value=0.0,
            peak_value=3e-4,
            warmup_steps=2000,
            decay_steps=int(2e6),
            end_value=0.0,
        ),
        weight_decay=0.01,
        clip_gradient=placeholder(float),
    )

    TRANSFORMER_SIZES = {
        "vanilla": dict(
            num_layers=4,
            mlp_dim=1024,
            num_attention_heads=8,
            dropout_rate=0.1,
        ),
        "vit_ti": dict(
            num_layers=12,
            mlp_dim=768,
            num_attention_heads=3,
            dropout_rate=0.0,
            attention_dropout_rate=0.0,
        ),
        "vit_s": dict(
            num_layers=12,
            mlp_dim=1536,
            num_attention_heads=6,
            dropout_rate=0.0,
        ),
        "vit_b": dict(
            num_layers=12,
            mlp_dim=3072,
            num_attention_heads=12,
            dropout_rate=0.0,
        ),
        "vit_l": dict(
            num_layers=24,
            mlp_dim=4096,
            num_attention_heads=16,
            dropout_rate=0.1,
        ),
    }

    TOKEN_DIMS = {
        "vanilla": 256,
        "vit_ti": 192,
        "vit_s": 384,
        "vit_b": 768,
        "vit_l": 1024,
    }

    base_model_config = dict(
        token_embedding_size=TOKEN_DIMS[transformer_size],
        max_horizon=50,
        readouts=dict(action=7),
        transformer_kwargs=TRANSFORMER_SIZES[transformer_size],
        heads=dict(
            action=dict(
                cls_name="mse_action_head",
                kwargs=dict(
                    pred_horizon=50,
                    action_dim=14,
                    vocab_size=256,
                    normalization_type=normalization_type,
                    readout_key="action",
                ),
            )
        ),
    )
    if transformer_size == "vanilla":
        encoder = "resnetv1-18-bridge"
        encoder_kwargs = dict(
            pooling_method="none",
            add_spatial_coordinates=True,
            act="swish",
        )
    else:
        encoder = "small-stem-16"
        encoder_kwargs = dict()

    base_tokenizer_kwargs = dict(
        encoder=encoder,
        encoder_kwargs=encoder_kwargs,
        task_stack_keys=[],
    )

    return ConfigDict(
        dict(
            model=update_config(
                base_model_config,
                observation_tokenizers=[
                    (
                        "image_tokenizer",
                        {
                            "num_tokens": 64,
                            "task_film_keys": [],
                            **base_tokenizer_kwargs,
                        },
                    ),
                    (
                        "lowdim_obs_tokenizer",
                        {
                            "n_bins": 256,
                            "bin_type": normalization_type,
                            "low": -2.,
                            "high": 2.,
                            "obs_keys": ["proprio"],
                        }
                    ),
                ],
                task_tokenizers=[],
            ),
            optimizer=base_optimizer_config,
            dataset_kwargs={
                # common_kwargs override specific kwargs from data_kwargs_list
                "common_kwargs": dict(
                    ram_budget=1,  # limit RAM per dataset
                    num_parallel_reads=8,  # for reading from GCS
                    num_parallel_calls=16,  # for the less CPU-intensive ops in initial dataset construction
                    action_proprio_normalization_type=normalization_type,
                    data_dir="gs://rail-orca-central2",
                    image_obs_keys=[
                        "cam_high",
                        "cam_low",
                        "cam_left_wrist",
                        "cam_right_wrist",
                    ],
                    state_obs_keys=["state"],
                ),
                "data_kwargs_list": [
                    {"name": "aloha_screwdriver_dataset"},
                ],
                "transform_kwargs": update_config(
                    base_data_config,
                    resize_size=(256, 256),
                    num_parallel_calls=16,  # for the most CPU-intensive ops (decoding, resizing, augmenting)
                ),
            },
            **update_config(
                base_config,
                text_processor="muse_embedding",
            ),
        )
    )
