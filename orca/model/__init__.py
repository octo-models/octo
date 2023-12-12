import logging
from typing import Dict

from orca.model.orca_module import ORCAModule, ORCATransformer
from orca.utils.spec import ModuleSpec


def base_orca_model_config():
    return {
        "observation_tokenizers": {},  # Dict[str, ModuleSpec], see orca.model.components.tokenizers for our standard tokenizers
        "task_tokenizers": {},  # Dict[str, ModuleSpec], see orca.model.components.tokenizers for our standard tokenizers
        "heads": {},  # Dict[str, ModuleSpec], see orca.model.components.heads for our standard heads
        "readouts": {},  # Dict[str, int]
        "token_embedding_size": 256,  # int
        "transformer_kwargs": {},  # See orca.model.components.transformer.Transformer for kwargs (basically, scaling)
        "max_horizon": 10,  # Sets the size of positional embeddings, and provides an upper limit on the maximum horizon of the model
    }


def create_model_def(
    observation_tokenizers: Dict[str, ModuleSpec],
    task_tokenizers: Dict[str, ModuleSpec],
    heads: Dict[str, ModuleSpec],
    **kwargs,  # Options for ORCATransformer
) -> ORCAModule:
    """
    Args:
        observation_tokenizers: dict of {tokenizer_name (str): tokenizer_config (ModuleConfig)}
        task_tokenizers: dict of {tokenizer_name (str): tokenizer_config (ModuleConfig)}
        heads: dict of {head_name (str): head_config (ModuleConfig)}

        # Options for ORCATransformer

        readouts: dict of {readout_name (str): n_tokens_for_readout (int)}
        token_embedding_size: int # The latent dimension of the token embeddings
        max_horizon: int # Sets the size of positional embeddings, and provides an upper limit on the maximum horizon of the model
        transformer_kwargs: dict of kwargs for Transformer
    """

    logging.warn("Proper pad mask is set to True by default now.")

    observation_tokenizer_defs = {
        k: ModuleSpec.instantiate(spec)() for k, spec in observation_tokenizers.items()
    }
    task_tokenizer_defs = {
        k: ModuleSpec.instantiate(spec)() for k, spec in task_tokenizers.items()
    }

    head_defs = {k: ModuleSpec.instantiate(spec)() for k, spec in heads.items()}

    model_def = ORCATransformer(
        observation_tokenizers=observation_tokenizer_defs,
        task_tokenizers=task_tokenizer_defs,
        **kwargs,
    )

    return ORCAModule(
        orca_transformer=model_def,
        heads=head_defs,
    )
