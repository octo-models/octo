from copy import deepcopy

from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from orca.data.utils.data_utils import ActionEncoding, StateEncoding


def update_config(config, **kwargs):
    updates = ConfigDict(kwargs)
    new_config = deepcopy(config)
    new_config.update(updates)
    return new_config


def get_config(modality="image_conditioned"):
    assert modality in ["image_conditioned", "language_conditioned", "multimodal"]
    # Fill this in for your own dataset!

    # If starting with an ORCA-wrist model, there should be two image keys
    # first image key should be the third-person view
    # and second image key should be the wrist view

    # If starting with an ORCA model, there should be one image key
    # and it should be the third-person view

    FINETUNING_KWARGS = {
        "name": "bridge_dataset",
        # On v4, this might be "gs://rail-orca-central2/resize_256_256"
        "data_dir": placeholder(str),
        "image_obs_keys": ["image_0", None],
        "state_obs_keys": [
            "EEF_state",
            None,
            "gripper_state",
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "action_proprio_normalization_type": "normal",
        # If the default data loading speed is too slow, try these:
        # and "num_parallel_calls" in `transform_kwargs` below
        # "num_parallel_reads": 8,  # for reading from disk / GCS
        # "num_parallel_calls": 16,  # for initial dataset construction
    }

    max_steps = FieldReference(200000)

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        batch_size=1024,
        shuffle_buffer_size=100000,
        num_val_batches=8,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=5000,
        save_interval=5000,
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(
            project="orca_finetune", group=placeholder(str), entity=placeholder(str)
        ),
        finetuning_dataset=FINETUNING_KWARGS,
        modality=modality,
        optimizer=dict(
            learning_rate=dict(
                init_value=0.0,
                peak_value=3e-4,
                warmup_steps=2000,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=placeholder(float),
        ),
    )

    if modality == "image_conditioned":
        goal_relabeling_strategy = "uniform"
        delete_key_groups_probs = [
            (["language_instruction"], 1.0),
        ]
    elif modality == "language_conditioned":
        goal_relabeling_strategy = "no_image_conditioning"
        delete_key_groups_probs = [
            (["image_.*"], 1.0),
        ]
    elif modality == "multimodal":
        goal_relabeling_strategy = "uniform"
        delete_key_groups_probs = [
            (["image_.*"], 0.5),
            (["language_instruction"], 0.5),
        ]
    else:
        raise ValueError("Invalid modality")

    transform_kwargs = dict(
        window_size=1,
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
        goal_relabeling_strategy=goal_relabeling_strategy,
        task_augmentation_strategy="delete_task_conditioning",
        task_augmentation_kwargs=dict(
            delete_key_groups_probs=delete_key_groups_probs,
        ),
        # If the default data loading speed is too slow, try these:
        # num_parallel_calls=16,  # for the most CPU-intensive ops (decoding, resizing, augmenting)
    )
    config["data_transforms"] = transform_kwargs
    return ConfigDict(config)
