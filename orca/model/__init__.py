import logging

from orca.model.components.heads import HEADS
from orca.model.components.tokenizers import TOKENIZERS
from orca.model.orca_model import OrcaModel, OrcaTransformer


def create_model_def(
    observation_tokenizers,
    task_tokenizers,
    readouts,
    token_embedding_size,
    max_horizon,
    transformer_kwargs,
    heads,
    **kwargs,
):
    """
    Args:
        observation_tokenizers: list of (tokenizer_name, tokenizer_kwargs) tuples
        task_tokenizers: list of (tokenizer_name, tokenizer_kwargs) tuples
        readouts: dict of {readout_name: n_tokens_for_readout}
        max_horizon: int
        transformer_kwargs: dict of kwargs for Transformer
        head_kwargs: dict of {
            head_name: {
                'cls_name': str, # which head in HEADS
                'kwargs': dict
            }
        }

    """
    if len(kwargs) > 0:
        logging.warn(f"Extra kwargs passed into create_model_def: {kwargs}")

    observation_tokenizer_defs = tuple(
        TOKENIZERS[tokenizer](**kwargs) for tokenizer, kwargs in observation_tokenizers
    )
    task_tokenizer_defs = tuple(
        TOKENIZERS[tokenizer](**kwargs) for tokenizer, kwargs in task_tokenizers
    )

    head_defs = {
        k: HEADS.get(head_info["cls_name"])(**head_info["kwargs"])
        for k, head_info in heads.items()
    }

    model_def = OrcaTransformer(
        observation_tokenizers=observation_tokenizer_defs,
        task_tokenizers=task_tokenizer_defs,
        readouts=readouts,
        max_horizon=max_horizon,
        token_embedding_size=token_embedding_size,
        transformer_kwargs=transformer_kwargs,
    )

    return OrcaModel(
        orca_transformer=model_def,
        heads=head_defs,
    )
