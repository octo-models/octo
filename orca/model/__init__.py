import logging

from orca.model.components.tokenizers import TOKENIZERS

from .orca_policy import ORCAPolicy


def create_model_def(
    observation_tokenizers,
    task_tokenizers,
    action_dim,
    window_size,
    policy_kwargs,
    **kwargs,
):
    if len(kwargs) > 0:
        logging.warn(f"Extra kwargs passed into create_model_def: {kwargs}")
    observation_tokenizer_defs = tuple(
        TOKENIZERS[tokenizer](**kwargs) for tokenizer, kwargs in observation_tokenizers
    )
    task_tokenizer_defs = tuple(
        TOKENIZERS[tokenizer](**kwargs) for tokenizer, kwargs in task_tokenizers
    )
    model_def = ORCAPolicy(
        observation_tokenizers=observation_tokenizer_defs,
        task_tokenizers=task_tokenizer_defs,
        action_dim=action_dim,
        window_size=window_size,
        **policy_kwargs,
    )
    return model_def
