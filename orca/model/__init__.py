import logging

from .tokenizers import tokenizers
from .transformer_policy import TransformerPolicy


def create_model_def(
    observation_tokenizer_kwargs,
    task_tokenizer_kwargs,
    action_dim,
    horizon,
    policy_kwargs,
    **kwargs,
):
    if len(kwargs) > 0:
        logging.warn(f"Extra kwargs passed into create_model_def: {kwargs}")
    observation_tokenizer_defs = tuple(
        tokenizers[k](**kwargs) for k, kwargs in observation_tokenizer_kwargs.items()
    )
    task_tokenizer_defs = tuple(
        tokenizers[k](**kwargs) for k, kwargs in task_tokenizer_kwargs.items()
    )
    model_def = TransformerPolicy(
        observation_tokenizers=observation_tokenizer_defs,
        task_tokenizers=task_tokenizer_defs,
        action_dim=action_dim,
        horizon=horizon,
        **policy_kwargs,
    )
    return model_def
