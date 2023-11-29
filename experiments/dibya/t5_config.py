from copy import deepcopy

from config import update_config
from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder
from experiments.dibya.oxe_config import get_config as get_base_config

def get_config(
    transformer_size,
):
    base_config = get_base_config(transformer_size)

    config = update_config(base_config, dict(
        model={
            "observation_tokenizers": [
                    (
                        "image_tokenizer",
                        {
                            "num_tokens": 64,
                            "encoder": "small-stem-16",
                            "encoder_kwargs": dict(),
                            "task_stack_keys": [
                                "image_.*"
                            ],  # by default, early fuse goal images into visual encoder
                        },
                    ),
                ],
                "task_tokenizers": [
                    ("language_tokenizer", {"encoder": "t5-base", "finetune_encoder": False}),
                ],
        },
        text_processor="hf_tokenizer",
        text_processor_kwargs=dict(
            tokenizer_name="t5-base",
            encode_with_model=False,
            tokenizer_kwargs= {
                "max_length": 16,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "np",
            },
        ),
        pretrained_weights=["from_huggingface"],
        pretrained_loader_kwargs=[dict(hf_model="t5-base")],
    ))

    return config
