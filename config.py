from copy import deepcopy
import functools

from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder


def update_config(config, **kwargs):
    updates = ConfigDict(kwargs)
    new_config = deepcopy(config)
    new_config.update(updates)
    return new_config


def wrap(f):
    """Simple wrapper to enable passing config strings to `get_config`

    Usage:

    python train.py --config=config.py:vit_s,multimodal
    python train.py --config=config.py:transformer_size=vit_s
    """

    @functools.wraps(f)
    def wrapped_f(config_string=None):
        if config_string is None:
            return f()
        elements = config_string.split(",")
        args, kwargs = [], {}
        for e in elements:
            if "=" in e:
                k, v = e.split("=")
                kwargs[k] = v
            else:
                args.append(e)
        return f(*args, **kwargs)

    return wrapped_f


@wrap
def get_config(
    transformer_size="vit_s",
    modality="multimodal",
):
    print("Creating config with: ", locals())
    num_steps = FieldReference(default=int(2e6))
    window_size = FieldReference(default=1)
    return ConfigDict(
        dict(
            seed=42,
            num_steps=num_steps,
            save_dir=placeholder(str),
            model=get_model_config(transformer_size),
            window_size=window_size,
            dataset_kwargs=get_dataset_config(modality, window_size),
            optimizer=dict(
                learning_rate=dict(
                    name="rsqrt",
                    init_value=0.0,
                    peak_value=3e-4,
                    warmup_steps=2000,
                    timescale=10000,
                ),
                weight_decay=0.1,
                clip_gradient=1.0,
                frozen_keys=tuple(),
            ),
            prefetch_num_batches=0,
            start_step=placeholder(int),
            log_interval=100,
            eval_interval=5000,
            viz_interval=20000,
            save_interval=10000,
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
            resume_path=placeholder(str),
            text_processor="muse_embedding",
            text_processor_kwargs=dict(),
            pretrained_loaders=tuple(),
            pretrained_loader_kwargs=tuple(),
            wandb=dict(
                project="orca",
                group=placeholder(str),
                entity=placeholder(str),
            ),
            wandb_resume_id=placeholder(str),
            eval_datasets=(
                "bridge_dataset",
                "taco_play",
                "berkeley_cable_routing",
                "berkeley_autolab_ur5",
            ),
        )
    )


def get_dataset_config(modality="multimodal", window_size=1):
    normalization_type = "normal"
    if modality == "multimodal":
        task_augmentation = dict(
            task_augment_strategy="delete_task_conditioning",
            task_augment_kwargs=dict(
                delete_key_groups_probs=[
                    (["image_*"], 0.5),
                    (["language_instruction"], 0.5),
                ],
            ),
        )
    else:
        raise ValueError(f"Unknown modality {modality}")

    return {
        # oxe_kwargs will generate dataset_kwargs_list and sampling weights
        "oxe_kwargs": dict(
            data_mix=placeholder(str),
            # for v4 TPUs: "gs://rail-orca-central2/resize_336_336"
            data_dir=placeholder(str),
            n_third_person_cameras=1,
            n_wrist_cameras=0,
            load_depth=False,
        ),
        # common_dataset_kwargs override specific kwargs from dataset_kwargs_list
        "common_dataset_kwargs": dict(
            action_proprio_normalization_type=normalization_type,
        ),
        "traj_transform_kwargs": dict(
            window_size=window_size,
            additional_action_window_size=0,
            goal_relabeling_strategy="uniform",
            subsample_length=100,
        ),
        "frame_transform_kwargs": dict(
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
            **task_augmentation,
        ),
        "traj_transform_threads": 48,  # shared between all datasets
        "traj_read_threads": 48,  # shared between all datasets
        "frame_transform_threads": 200,  # not shared between datasets
        "shuffle_buffer_size": 100000,  # shared between all datasets
        "batch_size": 1024,
        "balance_weights": True,
    }


def get_transformer_kwargs(transformer_size):
    assert transformer_size in ["dummy", "vanilla", "vit_s", "vit_b", "vit_l"]
    default_params = {
        "attention_dropout_rate": 0.0,
        "add_position_embedding": False,
    }

    TRANSFORMER_SIZES = {
        "dummy": dict(
            num_layers=1,
            mlp_dim=256,
            num_attention_heads=2,
            dropout_rate=0.1,
        ),
        "vanilla": dict(
            num_layers=4,
            mlp_dim=1024,
            num_attention_heads=8,
            dropout_rate=0.1,
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
        "dummy": 256,
        "vanilla": 256,
        "vit_s": 384,
        "vit_b": 768,
        "vit_l": 1024,
    }
    return dict(
        token_embedding_size=TOKEN_DIMS[transformer_size],
        transformer_kwargs={
            **default_params,
            **TRANSFORMER_SIZES[transformer_size],
        },
    )


def get_model_config(transformer_size):
    normalization_type = "normal"
    base_tokenizer_kwargs = dict(
        encoder="small-stem-16",
        encoder_kwargs=dict(use_film=True),
    )

    return {
        **get_transformer_kwargs(transformer_size),
        "proper_pad_mask": True,
        "max_horizon": 10,
        "readouts": dict(),
        "heads": dict(
            action=dict(
                cls_name="mse_action_head",
                kwargs=dict(
                    pred_horizon=1,
                    action_dim=7,
                    readout_key="obs",
                ),
            )
        ),
        "observation_tokenizers": {
            "image": {
                "cls_name": "image_tokenizer",
                "kwargs": dict(
                    num_tokens=256,
                    obs_stack_keys=["image_.*"],
                    task_stack_keys=["image_.*"],
                    task_film_keys=["language_instruction"],
                    **base_tokenizer_kwargs,
                ),
            },
        },
        "task_tokenizers": dict(),
    }
