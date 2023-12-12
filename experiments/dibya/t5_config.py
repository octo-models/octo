from scripts.configs.config import get_config as get_base_config
from scripts.configs.config import update_config, wrap


def get_config(config_string=None):
    base_config = get_base_config(config_string)
    del base_config["model"]["observation_tokenizers"]
    config = update_config(
        base_config,
        optimizer=dict(
            frozen_keys=("*hf_model*",),
        ),
        model={
            "observation_tokenizers": {
                "image": {
                    "cls_name": "image_tokenizer",
                    "kwargs": dict(
                        num_tokens=256,
                        obs_stack_keys=["image_.*"],
                        task_stack_keys=["image_.*"],
                        task_film_keys=[],
                        encoder="small-stem-16",
                        encoder_kwargs=dict(),
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
    )

    return config
