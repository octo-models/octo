from copy import deepcopy
import imp
import os

from ml_collections import ConfigDict

get_base_config = imp.load_source(
    "config", os.path.join(os.path.dirname(__file__), "config.py")
).get_config

from octo.data.utils.text_processing import HFTokenizer
from octo.model.components.action_heads import DiffusionActionHead
from octo.model.components.tokenizers import ImageTokenizer, LanguageTokenizer
from octo.model.components.vit_encoders import SmallStem16
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import hf_weights_loader


def update_config(config, **kwargs):
    updates = ConfigDict(kwargs)
    new_config = deepcopy(config)
    new_config.update(updates)
    return new_config


def get_config(config_string=None):
    config = get_base_config(config_string)

    config["window_size"] = 2
    config["num_steps"] = 300000
    config["model"]["observation_tokenizers"] = {
        "primary": ModuleSpec.create(
            ImageTokenizer,
            obs_stack_keys=["image_primary"],
            task_stack_keys=["image_primary"],
            encoder=ModuleSpec.create(SmallStem16),
        ),
        "wrist": ModuleSpec.create(
            ImageTokenizer,
            obs_stack_keys=["image_wrist"],
            task_stack_keys=["image_wrist"],
            encoder=ModuleSpec.create(SmallStem16),
        ),
    }
    config["model"]["task_tokenizers"] = {
        "language": ModuleSpec.create(
            LanguageTokenizer,
            encoder="t5-base",
            finetune_encoder=False,
        ),
    }
    config["model"]["repeat_task_tokens"] = True
    config["model"]["readouts"] = {"action": 1}
    config["model"]["heads"]["action"] = ModuleSpec.create(
        DiffusionActionHead,
        readout_key="readout_action",
        use_map=False,
        pred_horizon=4,
        action_dim=7,
        dropout_rate=0.0,
    )

    # We augment differently for the primary and wrist cameras
    primary_augment_kwargs = dict(
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

    # ML-collections complains if the type of an existing field changes
    # so we delete and re-add the field

    del config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"]
    del config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"]

    config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"] = {
        "primary": (256, 256),  # workspace camera is at 256x256
        "wrist": (128, 128),  # wrist camera is at 128x128
    }
    config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"] = [
        primary_augment_kwargs,
        wrist_augment_kwargs,
    ]

    config = update_config(
        config,
        optimizer=dict(
            frozen_keys=("*hf_model*",),
        ),
        dataset_kwargs=dict(
            oxe_kwargs=dict(
                data_mix="oxe_magic_soup",
                data_dir="gs://rail-octo-central2/resize_256_256",
                load_camera_views=("primary", "wrist"),
                load_depth=False,
            ),
            traj_transform_kwargs=dict(
                future_action_window_size=3,
                task_augment_strategy="delete_and_rephrase",
                task_augment_kwargs=dict(
                    paraphrases_repo="rail-berkeley/OXE_paraphrases",
                    paraphrases_filename="paraphrases_oxe.pkl",
                    rephrase_prob=0.5,
                ),
            ),
            frame_transform_kwargs=dict(
                image_dropout_prob=0.5,
            ),
            batch_size=128,
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
        eval_datasets=["bridge_dataset"],
    )

    return config
