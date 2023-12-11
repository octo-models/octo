import copy
from copy import deepcopy

from config import get_config as get_base_config
from ml_collections import ConfigDict

from orca.data.utils.text_processing import HFTokenizer
from orca.model.components.hf_weight_loaders import hf_weights_loader
from orca.model.components.tokenizers import ImageTokenizer, LanguageTokenizer
from orca.model.components.vit_encoders import SmallStem16
from orca.spec import ModuleSpec


def update_config(config, **kwargs):
    updates = ConfigDict(kwargs)
    new_config = deepcopy(config)
    new_config.update(updates)
    return new_config


def get_config(config_string=None):
    base_config = get_base_config(config_string)

    # Can't delete with update_config
    del base_config["model"]["observation_tokenizers"]
    # Field reference can't be updated with update_config
    base_config["window_size"] = 2
    base_config["num_steps"] = 300000

    #
    # Changes to the model:
    #

    encoder = ModuleSpec.create(SmallStem16)

    base_config["model"]["observation_tokenizers"] = {
        "workspace": ModuleSpec.create(
            ImageTokenizer,
            obs_stack_keys=["image_0"],
            task_stack_keys=["image_0"],
            task_film_keys=[],
            encoder=encoder,
        ),
        "wrist": ModuleSpec.create(
            ImageTokenizer,
            obs_stack_keys=["image_1"],
            task_stack_keys=["image_1"],
            task_film_keys=[],
            encoder=encoder,
        ),
    }
    base_config["model"]["task_tokenizers"] = {
        "language": ModuleSpec.create(
            LanguageTokenizer,
            encoder="t5-base",
            finetune_encoder=False,
        ),
    }

    #
    # Changes to data-loading
    #

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

    base_config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"] = {
        "primary": (256, 256),  # workspace camera is at 256x256
        "wrist": (128, 128),  # wrist camera is at 128x128
    }
    base_config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"] = {
        "primary": workspace_augment_kwargs,
        "wrist": wrist_augment_kwargs,
    }

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
        text_processor=ModuleSpec.create(
            HFTokenizer,
            tokenizer_name="t5-base",
            encode_with_model=False,
            tokenizer_kwargs={
                "max_length": 16,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "np",
            },
        ),
        pretrained_loaders=(
            ModuleSpec.create(
                hf_weights_loader,
                hf_model="t5-base",
            ),
        ),
        eval_datasets=("bridge_dataset",),
    )

    return config
