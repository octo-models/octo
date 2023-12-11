import logging
from typing import Any, Dict

from orca.config_utils import create_module_from_spec, ModuleSpec
from orca.model.orca_model import OrcaModel, OrcaTransformer


def create_model_def(
    observation_tokenizers: Dict[str, ModuleSpec],
    task_tokenizers: Dict[str, ModuleSpec],
    heads: Dict[str, ModuleSpec],
    **kwargs,  # Options for OrcaTransformer
):
    """
    Args:
        observation_tokenizers: dict of {tokenizer_name (str): tokenizer_config (ModuleConfig)}
        task_tokenizers: dict of {tokenizer_name (str): tokenizer_config (ModuleConfig)}
        heads: dict of {head_name (str): head_config (ModuleConfig)}

        # Options for OrcaTransformer

        readouts: dict of {readout_name (str): n_tokens_for_readout (int)}
        token_embedding_size: int # The latent dimension of the token embeddings
        max_horizon: int # Sets the size of positional embeddings, and provides an upper limit on the maximum horizon of the model
        transformer_kwargs: dict of kwargs for Transformer
    """

    logging.warn("Proper pad mask is set to True by default now.")

    observation_tokenizer_defs = {
        k: create_module_from_spec(
            spec, default_library="orca.model.components.tokenizers"
        )
        for k, spec in observation_tokenizers.items()
    }
    task_tokenizer_defs = {
        k: create_module_from_spec(
            spec, default_library="orca.model.components.tokenizers"
        )
        for k, spec in task_tokenizers.items()
    }

    head_defs = {
        k: create_module_from_spec(spec, default_library="orca.model.components.heads")
        for k, spec in heads.items()
    }

    model_def = OrcaTransformer(
        observation_tokenizers=observation_tokenizer_defs,
        task_tokenizers=task_tokenizer_defs,
        **kwargs,
    )

    return OrcaModel(
        orca_transformer=model_def,
        heads=head_defs,
    )
