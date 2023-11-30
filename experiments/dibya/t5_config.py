from config import get_config as get_base_config
from config import update_config, wrap


def get_config(config_string=None):
    base_config = get_base_config(config_string)

    config = update_config(
        base_config,
        optimizer=dict(
            frozen_keys=("*hf_model*",),
        ),
        model={
            "observation_tokenizers": [
                (
                    "image_tokenizer",
                    {
                        "num_tokens": 256,
                        "encoder": "small-stem-16",
                        "encoder_kwargs": dict(),
                        "task_stack_keys": [
                            "image_.*"
                        ],  # by default, early fuse goal images into visual encoder
                    },
                ),
            ],
            "task_tokenizers": [
                (
                    "language_tokenizer",
                    {"encoder": "t5-base", "finetune_encoder": False},
                ),
            ],
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
    )

    return config
