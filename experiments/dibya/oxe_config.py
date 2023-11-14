from copy import deepcopy

from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder

from config import update_config
from orca.data.oxe.oxe_dataset_mixes import *


def get_config(
    config_string="nonexistentdataset,main,multimodal,vanilla",
):
    mixture_type, observation_mode, task_modes, transformer_size = config_string.split(
        ","
    )
    HELP_STRING = """
    This config takes in 4 parameters which must be comma-separated. Use as
    python train.py --config=oxe_config.py:bridge,main,multimodal,vanilla

    The first option specifies which dataset to train on (["bridge", "rtx", "oxe"])
    The second option specifies which camera angles are used (["main", "main_and_wrist"])
    The third option specifies how tasks are specified (["gc", "multimodal"])
    The fourth option specifies the size of the model (["vanilla", "vit_s", "vit_b"])
    """

    assert mixture_type in ["bridge", "rtx", "oxe"], HELP_STRING
    assert observation_mode in ["main", "main_and_wrist"], HELP_STRING
    assert task_modes in ["gc", "multimodal"], HELP_STRING
    assert transformer_size in ["vanilla", "vit_s", "vit_b"], HELP_STRING

    base_wandb_config = dict(
        project="orca", group=placeholder(str), entity=placeholder(str)
    )

    base_config = dict(
        batch_size=1024,
        shuffle_buffer_size=100000,
        num_val_batches=16,
        num_steps=int(2e6),
        start_step=placeholder(int),
        log_interval=100,
        eval_interval=5000,
        save_interval=5000,
        save_dir=placeholder(str),
        resume_path=placeholder(str),
        seed=42,
        text_processor=None,
        text_processor_kwargs=dict(),
        pretrained_weights=[],
        wandb=base_wandb_config,
        wandb_resume_id=placeholder(str),
        eval_datasets=[
            "bridge_dataset",
            "fractal20220817_data",
        ],
    )

    # params that need to be specified multiple places
    normalization_type = "normal"

    base_data_config = dict(
        window_size=1,
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
    }
    TOKEN_DIMS = {
        "vanilla": 256,
        "vit_s": 384,
        "vit_b": 768,
    }

    base_model_config = dict(
        token_embedding_size=TOKEN_DIMS[transformer_size],
        max_horizon=10,
        readouts=dict(action=7),
        transformer_kwargs=TRANSFORMER_SIZES[transformer_size],
        heads=dict(
            action=dict(
                cls_name="mse_action_head",
                kwargs=dict(
                    pred_horizon=1,
                    action_dim=7,
                    vocab_size=256,
                    normalization_type=normalization_type,
                    readout_key="obs_0",
                ),
            )
        ),
    )
    if transformer_size == "vanilla":
        encoder = "resnetv1-50-bridge-film"
        encoder_kwargs = dict(
            pooling_method="none",
            add_spatial_coordinates=True,
            act="swish",
            use_film=True,
        )
    else:
        encoder = "small-stem-16-film"
        encoder_kwargs = dict()

    base_tokenizer_kwargs = dict(
        encoder=encoder,
        encoder_kwargs=encoder_kwargs,
        task_stack_keys=[
            "image_.*"
        ],  # by default, early fuse goal images into visual encoder
    )

    MIXES = {
        "bridge": BRIDGE_MIX,
        "rtx": RT_X_MIX,
        "oxe": RT_X_MIX + OXE_FRANKA_MIX,
    }
    MIX = MIXES[mixture_type]

    observation_modes = {
        "main_and_wrist": dict(
            n_third_person_cameras=1, n_wrist_cameras=1, load_depth=False
        ),
        "main": dict(n_third_person_cameras=1, n_wrist_cameras=0, load_depth=False),
    }
    observation_mode_kwargs = observation_modes[observation_mode]

    dataset_kwargs_list, dataset_sampling_weights = make_oxe_dataset_kwargs_and_weights(
        MIX,
        data_dir="gs://rail-orca-central2/resize_336_336",
        **observation_mode_kwargs,
    )

    n_cameras = (
        observation_mode_kwargs["n_third_person_cameras"]
        + observation_mode_kwargs["n_wrist_cameras"]
    )
    gc_drop_keys = [
        ([f"image_{i}" for i in range(n_cameras)], 0.0),
        (["language_instruction"], 1.0),
    ]
    multimodal_drop_keys = [
        ([f"image_{i}" for i in range(n_cameras)], 0.5),
        (["language_instruction"], 0.5),
    ]
    drop_keys = {
        "gc": gc_drop_keys,
        "multimodal": multimodal_drop_keys,
    }[task_modes]

    possible_structures = {
        "multimodal": ConfigDict(
            dict(
                model=update_config(
                    base_model_config,
                    observation_tokenizers=[
                        (
                            "image_tokenizer",
                            {
                                "num_tokens": 64,
                                "task_film_keys": ["language_instruction"],
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
                        ram_budget=1,  # limit RAM per dataset
                        num_parallel_reads=8,  # for reading from GCS
                        num_parallel_calls=16,  # for the less CPU-intensive ops in initial dataset construction
                        action_proprio_normalization_type=normalization_type,
                    ),
                    "data_kwargs_list": dataset_kwargs_list,
                    "transform_kwargs": update_config(
                        base_data_config,
                        resize_size=(256, 256),
                        num_parallel_calls=16,  # for the most CPU-intensive ops (decoding, resizing, augmenting)
                        task_augmentation_strategy="drop_keys_independent",
                        task_augmentation_kwargs=dict(
                            drop_key_groups_probs=drop_keys,
                            allow_drop_all=True,
                        ),
                    ),
                    "sample_weights": dataset_sampling_weights,
                },
                **update_config(
                    base_config,
                    text_processor="muse_embedding",
                ),
            )
        ),
    }

    return possible_structures["multimodal"]
