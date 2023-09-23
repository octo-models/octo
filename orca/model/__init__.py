import logging

from .tokenizers import TOKENIZERS
from .transformer_policy import TransformerPolicy


def create_model_def(
    observation_tokenizers,
    task_tokenizers,
    action_dim,
    horizon,
    policy_kwargs,
    **kwargs,
):
    if len(kwargs) > 0:
        logging.warn(f"Extra kwargs passed into create_model_def: {kwargs}")
    observation_tokenizer_defs = tuple(
        tokenizer(**kwargs) for tokenizer, kwargs in observation_tokenizers
    )
    task_tokenizer_defs = tuple(
        tokenizer(**kwargs) for tokenizer, kwargs in task_tokenizers
    )
    model_def = TransformerPolicy(
        observation_tokenizers=observation_tokenizer_defs,
        task_tokenizers=task_tokenizer_defs,
        action_dim=action_dim,
        horizon=horizon,
        **policy_kwargs,
    )
    return model_def
