import logging

from orca.model.components.computation_groups import ComputationPlaceholder
from orca.model.components.heads import HEADS
from orca.model.components.tokenizers import TOKENIZERS

from .orca_model import OrcaModel, ORCATransformer


def create_model_def(
    observation_tokenizers,
    task_tokenizers,
    computation_placeholders,
    max_horizon,
    transformer_kwargs,
    head_kwargs,
    **kwargs,
):
    if len(kwargs) > 0:
        logging.warn(f"Extra kwargs passed into create_model_def: {kwargs}")

    token_embedding_size = transformer_kwargs.get("token_embedding_size")
    observation_tokenizer_defs = tuple(
        TOKENIZERS[tokenizer](**kwargs) for tokenizer, kwargs in observation_tokenizers
    )
    task_tokenizer_defs = tuple(
        TOKENIZERS[tokenizer](**kwargs) for tokenizer, kwargs in task_tokenizers
    )
    computation_placeholder_defs = {
        k: ComputationPlaceholder(
            n_tokens=n_tokens,
            token_embedding_size=token_embedding_size,
            max_horizon=max_horizon,
        )
        for k, n_tokens in computation_placeholders.items()
    }

    head_defs = {
        k: HEADS.get(head_info["name"])(**head_info["kwargs"])
        for k, head_info in head_kwargs.items()
    }

    model_def = ORCATransformer(
        observation_tokenizers=observation_tokenizer_defs,
        task_tokenizers=task_tokenizer_defs,
        max_horizon=max_horizon,
        **transformer_kwargs,
    )

    return OrcaModel(
        orca_transformer=model_def,
        computation_placeholders=computation_placeholder_defs,
        heads=head_defs,
    )
