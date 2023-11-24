from copy import deepcopy

from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from orca.data.utils.data_utils import ActionEncoding, StateEncoding


def update_config(config, **kwargs):
    updates = ConfigDict(kwargs)
    new_config = deepcopy(config)
    new_config.update(updates)
    return new_config


def get_config():
    # If starting with an ORCA-wrist model, there should be two image keys
    # first image key should be the third-person view
    # and second image key should be the wrist view

    # If starting with an ORCA model, there should be one image key
    # and it should be the third-person view

    FINETUNING_KWARGS = dict(
        name="aloha_screwdriver_dataset",
        data_dir="gs://rail-orca-central2",
        image_obs_keys=[
            "cam_high",
            "cam_low",
            "cam_left_wrist",
            "cam_right_wrist",
        ],
        state_obs_keys=["state"],
        state_encoding=StateEncoding.JOINT_BIMANUAL,
        action_encoding=ActionEncoding.JOINT_POS_BIMANUAL,
        action_proprio_normalization_type="normal",
        # If the default data loading speed is too slow, try these:
        # and "num_parallel_calls" in `transform_kwargs` below
        num_parallel_reads=8,  # for reading from disk / GCS
        num_parallel_calls=16,  # for initial dataset construction
    )

    max_steps = FieldReference(20000)

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        batch_size=1024,
        shuffle_buffer_size=20000,
        num_val_batches=8,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=500,
        save_interval=500,
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(
            project="orca_finetune", group=placeholder(str), entity=placeholder(str)
        ),
        finetuning_dataset=FINETUNING_KWARGS,
        modality=None, #modality,
        optimizer=dict(
            learning_rate=dict(
                init_value=0.0,
                peak_value=3e-4,
                warmup_steps=1000,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=placeholder(float),
        ),
    )

    transform_kwargs = dict(
        window_size=1,
        additional_action_window_size=49,
        resize_size=(256, 256),
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
        goal_relabeling_strategy="uniform", #goal_relabeling_strategy,
        action_encoding=ActionEncoding.JOINT_POS_BIMANUAL,
        # If the default data loading speed is too slow, try these:
        num_parallel_calls=16,  # for the most CPU-intensive ops (decoding, resizing, augmenting)
    )
    config["data_transforms"] = transform_kwargs

    config["overwrite_model_config"] = dict(
        token_embedding_size=384,
        max_horizon=10,
        readouts=dict(action=7),
        transformer_kwargs=dict(
            num_layers=12,
            mlp_dim=1536,
            num_attention_heads=6,
            dropout_rate=0.0,
        ),
        heads=dict(
            action=dict(
                cls_name="mse_action_head",
                kwargs=dict(
                    pred_horizon=50,
                    action_dim=14,
                    vocab_size=256,
                    normalization_type="normal", #normalization_type,
                    readout_key="action",
                ),
            )
        ),
        observation_tokenizers=[
            (
                "image_tokenizer",
                {
                    "num_tokens": 64,
                    "task_film_keys": [],
                    "encoder": "small-stem-16",
                    "encoder_kwargs": {},
                    "task_stack_keys": [],
                },
            ),
            (
                "lowdim_obs_tokenizer",
                {
                    "n_bins": 256,
                    "bin_type": "normal", #normalization_type,
                    "low": -2.,
                    "high": 2.,
                    "obs_keys": ["proprio"],
                }
            ),
        ],
        task_tokenizers=[],
    )
    config['overwrite_example_batch_path'] = (
        "gs://karl-central-2/orca/aloha_scratch_chunk50_vit_ti_updated_20231120_213331/example_batch.msgpack")
    return ConfigDict(config)
