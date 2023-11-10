from copy import deepcopy

from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder
from config import update_config
from orca.data.oxe.oxe_dataset_mixes import *


def get_config(config_string):
    base_wandb_config = dict(
        project="orca", group=placeholder(str), entity=placeholder(str)
    )

    base_config = dict(
        batch_size=2048,
        shuffle_buffer_size=100000,
        num_val_batches=8,
        num_steps=int(2e6),
        start_step=placeholder(int),
        log_interval=100,
        eval_interval=10000,
        save_interval=10000,
        save_dir="gs://karl-central-2",
        resume_path=placeholder(str),
        seed=42,
        text_processor=None,
        text_processor_kwargs=dict(),
        pretrained_weights=[],
        wandb=base_wandb_config,
        #eval_datasets=["bridge_dataset"],
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
        clip_gradient=1.0,
    )

    base_model_config = dict(
        token_embedding_size=256,
        max_horizon=10,
        readouts=dict(action=7),
        transformer_kwargs=dict(
            num_layers=6,
            mlp_dim=2048,
            num_attention_heads=8,
            dropout_rate=0.1,
        ),
        heads=dict(
            action=dict(
                cls_name="token_per_dim_action_head",
                kwargs=dict(
                    pred_horizon=1,
                    action_dim=7,
                    vocab_size=256,
                    normalization_type=normalization_type,
                    readout_key="action",
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

    possible_structures = {
        "transformer_bc_r2d2": ConfigDict(
            dict(
                model=update_config(
                    base_model_config,
                    observation_tokenizers=[
                        (
                            "image_tokenizer",
                            {
                                "num_tokens": 16,
                                **base_tokenizer_kwargs,
                            },
                        ),
                    ],
                    task_tokenizers=[],
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs={
                    # common_kwargs override specific kwargs from data_kwargs_list
                    "common_kwargs": dict(
                        # ram_budget=1,  # limit RAM per dataset
                        num_parallel_reads=8,  # for reading from GCS
                        num_parallel_calls=16,  # for the less CPU-intensive ops in initial dataset construction
                        action_proprio_normalization_type=normalization_type,
                        data_dir="gs://rail-orca-central2",
                        image_obs_keys=[
                            "exterior_image_1_left",
                            "exterior_image_2_left",
                            "wrist_image_left",
                        ],
                        state_obs_keys=["joint_position"],
                    ),
                    "data_kwargs_list": [
                        {"name": "r2_d2"},
                        {"name": "r2_d2_pen_cmu_rgb"},
                        {"name": "r2_d2_play_cmu_rgb"},
                    ],
                    "transform_kwargs": update_config(
                        base_data_config,
                        resize_size=(128, 128),
                        num_parallel_calls=16,  # for the most CPU-intensive ops (decoding, resizing, augmenting)
                    ),
                    "sample_weights": [1.0, 1.0, 1.0],
                },
                **base_config,
            )
        ),
    }

    return possible_structures[config_string]
