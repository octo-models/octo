from copy import deepcopy

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
    return ConfigDict(
        dict(
            seed=42,
            num_steps=num_steps,
            save_dir=placeholder(str),
            model=get_model_config(transformer_size),
            dataset_kwargs=get_dataset_config(modality),
            optimizer=dict(
                learning_rate=dict(
                    init_value=0.0,
                    peak_value=3e-4,
                    warmup_steps=2000,
                    decay_steps=num_steps,
                    end_value=0.0,
                ),
                weight_decay=0.1,
                clip_gradient=1.0,
            ),
            batch_size=1024,
            eval_batch_size=128,
            shuffle_buffer_size=100000,
            val_shuffle_buffer_size=1000,
            num_val_batches=16,
            start_step=placeholder(int),
            log_interval=100,
            eval_interval=5000,
            save_interval=5000,
            trajs_for_metrics=100,
            trajs_for_viz=8,
            resume_path=placeholder(str),
            text_processor="muse_embedding",
            text_processor_kwargs=dict(),
            pretrained_loaders=[],
            pretrained_loader_kwargs=[],
            wandb=dict(
                project="orca",
                group=placeholder(str),
                entity=placeholder(str),
            ),
            wandb_resume_id=placeholder(str),
            eval_datasets=[
                "bridge_dataset",
                "taco_play",
                "berkeley_cable_routing",
                "berkeley_autolab_ur5",
            ],
        )
    )


def get_dataset_config(modality="multimodal"):
    normalization_type = "normal"
    if modality == "multimodal":
        task_augmentation = dict(
            task_augmentation_strategy="delete_task_conditioning",
            task_augmentation_kwargs=dict(
                delete_key_groups_probs=[
                    (["image_*"], 0.5),
                    (["language_instruction"], 0.5),
                ],
            ),
        )
    else:
        raise ValueError(f"Unknown modality {modality}")

    return {
        # oxe_kwargs will generate data_kwargs_list and sampling weights
        "oxe_kwargs": dict(
            data_mix=placeholder(str),
            # for v4 TPUs: "gs://rail-orca-central2/resize_336_336"
            data_dir=placeholder(str),
            n_third_person_cameras=1,
            n_wrist_cameras=0,
            load_depth=False,
        ),
        # common_kwargs override specific kwargs from data_kwargs_list
        "common_kwargs": dict(
            ram_budget=1,  # limit RAM per dataset
            num_parallel_reads=8,  # for reading from GCS
            num_parallel_calls=16,  # for the less CPU-intensive ops in initial dataset construction
            action_proprio_normalization_type=normalization_type,
        ),
        "transform_kwargs": dict(
            resize_size=(256, 256),
            num_parallel_calls=32,  # for the most CPU-intensive ops (decoding, resizing, augmenting)
            window_size=1,
            additional_action_window_size=0,
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
            **task_augmentation,
        ),
    }


def get_transformer_kwargs(transformer_size):
    assert transformer_size in ["dummy", "vanilla", "vit_s", "vit_b", "vit_l"]
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
        transformer_kwargs=TRANSFORMER_SIZES[transformer_size],
    )


def get_model_config(transformer_size):
    normalization_type = "normal"
    base_tokenizer_kwargs = dict(
        encoder="small-stem-16",
        encoder_kwargs=dict(use_film=True),
    )

    return {
        **get_transformer_kwargs(transformer_size),
        "max_horizon": 10,
        "readouts": dict(),
        "heads": dict(
            action=dict(
                cls_name="mse_action_head",
                kwargs=dict(
                    pred_horizon=1,
                    action_dim=7,
                    readout_key="obs_0",
                ),
            )
        ),
        "observation_tokenizers": [
            (
                "image_tokenizer",
                {
                    "num_tokens": 256,
                    "obs_stack_keys": ["image_.*"],
                    "task_stack_keys": ["image_.*"],
                    "task_film_keys": ["language_instruction"],
                    **base_tokenizer_kwargs,
                },
            ),
        ],
        "task_tokenizers": [],
    }
