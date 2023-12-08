from config import update_config, wrap
from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from orca.data.utils.data_utils import ActionEncoding, StateEncoding


@wrap
def get_config(
    task="image_conditioned",
    mode="full",
):
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    # Fill this in for your own dataset!

    # There should be two image keys
    # first image key should be the third-person view (None if not used)
    # and second image key should be the wrist view (None if not used)

    FINETUNING_KWARGS = {
        "name": "bridge_dataset",
        # On v4, this might be "gs://rail-orca-central2/resize_256_256"
        "data_dir": "./tests/debug_dataset",
        "image_obs_keys": ["image_0", None],
        "state_obs_keys": ["state", None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        "action_proprio_normalization_type": "normal",
        # If the default data loading speed is too slow, try these:
        # "num_parallel_reads": 8,  # for reading from disk / GCS
        # "num_parallel_calls": 16,  # for initial dataset construction
    }

    if mode == "full":
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("orca_transformer.*",)
    elif mode == "head_mlp_only":
        frozen_keys = (
            "orca_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        )
    elif mode == "frozen_transformer":
        frozen_keys = ("orca_transformer.BlockTransformer_0.*",)
    else:
        raise ValueError("Invalid mode")

    max_steps = FieldReference(20000)
    window_size = FieldReference(default=1)

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        batch_size=256,
        shuffle_buffer_size=10000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=5000,
        save_interval=5000,
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(
            project="orca_finetune", group=placeholder(str), entity=placeholder(str)
        ),
        dataset_kwargs=FINETUNING_KWARGS,
        modality=task,
        finetuning_mode=mode,
        window_size=window_size,
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=3e-4,
                warmup_steps=2000,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=16,
        ),
        viz_kwargs=dict(
            eval_batch_size=128,
            trajs_for_metrics=100,
            trajs_for_viz=8,
            samples_per_state=8,
        ),
    )

    if task == "image_conditioned":
        goal_relabeling_strategy = "uniform"
        delete_key_groups_probs = [
            (["language_instruction"], 1.0),
        ]
    elif task == "language_conditioned":
        goal_relabeling_strategy = "no_image_conditioning"
        delete_key_groups_probs = [
            (["image_.*"], 1.0),
        ]
    elif task == "multimodal":
        goal_relabeling_strategy = "uniform"
        delete_key_groups_probs = [
            (["image_.*"], 0.5),
            (["language_instruction"], 0.5),
        ]
    else:
        raise ValueError("Invalid modality")

    traj_transform_kwargs = dict(
        window_size=window_size,
        additional_action_window_size=0,
        goal_relabeling_strategy=goal_relabeling_strategy,
        # If the default data loading speed is too slow, try these:
        # num_parallel_calls=16,  # for less CPU-intensive ops
    )
    workspace_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    wrist_augment_kwargs = dict(
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    frame_transform_kwargs = dict(
        resize_size=[
            (256, 256),  # workspace (3rd person) camera is at 256x256
            (128, 128),  # wrist camera is at 128x128
        ],
        image_augment_kwargs=[
            workspace_augment_kwargs,
            wrist_augment_kwargs,
        ],
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            delete_key_groups_probs=delete_key_groups_probs,
        ),
    )
    # If the default data loading speed is too slow, try these:
    config[
        "frame_transform_threads"
    ] = 16  # for the most CPU-intensive ops (decoding, resizing, augmenting)

    config["traj_transform_kwargs"] = traj_transform_kwargs
    config["frame_transform_kwargs"] = frame_transform_kwargs
    return ConfigDict(config)
