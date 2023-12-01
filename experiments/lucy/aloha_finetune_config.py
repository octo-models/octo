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
        name="aloha_sim_cube_scripted_dataset",
        data_dir="gs://rail-orca-central2",
        image_obs_keys=[
            "top",
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
        batch_size=128,
        shuffle_buffer_size=100000,
        num_val_batches=8,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=500,
        save_interval=500,
        save_dir="gs://karl-central-2",
        seed=42,
        wandb=dict(
            project="orca_finetune", group=placeholder(str), entity=placeholder(str)
        ),
        finetuning_dataset=FINETUNING_KWARGS,
        modality=None,
        optimizer=dict(
            learning_rate=dict(
                init_value=0.0,
                peak_value=3e-5,
                warmup_steps=1000,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.0,
            clip_gradient=placeholder(float),
            frozen_keys=tuple(),
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
        goal_relabeling_strategy="uniform",
        action_encoding=ActionEncoding.JOINT_POS_BIMANUAL,
        # If the default data loading speed is too slow, try these:
        num_parallel_calls=16,  # for the most CPU-intensive ops (decoding, resizing, augmenting)
    )
    config["data_transforms"] = transform_kwargs

    config["config_delete_keys"] = dict(
        model=dict(
            observation_tokenizers=dict(
                wrist=None
            )
        )
    )

    config["update_config"] = dict(
        model=dict(
            heads=dict(
                action=dict(
                    cls_name="mse_action_head",
                    kwargs=dict(
                        pred_horizon=50,
                        action_dim=14,
                        vocab_size=256,
                        normalization_type="normal",
                        readout_key="obs",
                    ),
                )
            ),
            observation_tokenizers={
                "proprio": {
                    "cls_name": "lowdim_obs_tokenizer",
                    "kwargs": dict(
                        n_bins=256,
                        bin_type="normal",
                        low=-2.,
                        high=2.,
                        obs_keys=["proprio"],
                    ),
                },
            },
        )
    )
    return ConfigDict(config)
