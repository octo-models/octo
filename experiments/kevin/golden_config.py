from config import get_config as get_base_config
from config import update_config, wrap


def get_config(config_string=None):
    base_config = get_base_config(config_string)

    # Can't delete with update_config
    del base_config["model"]["observation_tokenizers"]
    # Field reference can't be updated with update_config
    base_config["window_size"] = 2
    base_config["num_steps"] = 300000

    # different augmentations for wrist and workspace
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

    del base_config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"]
    del base_config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"]

    base_config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"] = [
        (256, 256),  # workspace (3rd person) camera is at 256x256
        (128, 128),  # wrist camera is at 128x128
    ]
    base_config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"] = [
        workspace_augment_kwargs,
        wrist_augment_kwargs,
    ]

    config = update_config(
        base_config,
        optimizer=dict(
            frozen_keys=("*hf_model*",),
        ),
        dataset_kwargs=dict(
            oxe_kwargs=dict(
                data_mix="oxe_magic_soup",
                data_dir="gs://rail-orca-central2/resize_256_256",
            ),
            batch_size=256,
            shuffle_buffer_size=500000,
            balance_weights=True,
        ),
        model={
            "observation_tokenizers": {
                "workspace": {
                    "cls_name": "image_tokenizer",
                    "kwargs": dict(
                        obs_stack_keys=["image_0"],
                        task_stack_keys=["image_0"],
                        task_film_keys=[],
                        encoder="small-stem-16",
                    ),
                },
                "wrist": {
                    "cls_name": "image_tokenizer",
                    "kwargs": dict(
                        obs_stack_keys=["image_1"],
                        task_stack_keys=["image_1"],
                        task_film_keys=[],
                        encoder="small-stem-16",
                    ),
                },
            },
            "task_tokenizers": {
                "language": {
                    "cls_name": "language_tokenizer",
                    "kwargs": dict(
                        encoder="t5-base",
                        finetune_encoder=False,
                    ),
                },
            },
        },
        text_processor="hf_tokenizer",
        text_processor_kwargs=dict(
            tokenizer_name="t5-base",
            encode_with_model=False,
            tokenizer_kwargs={
                "max_length": 16,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "np",
            },
        ),
        pretrained_loaders=["from_huggingface"],
        pretrained_loader_kwargs=[dict(hf_model="t5-base")],
        eval_datasets=["bridge_dataset"],
    )

    return config
