"""Utilities for making ORCAModel configs."""
from functools import partial
import importlib
import logging
from typing import Any, Dict, TypedDict, Union


class ModuleConfig(TypedDict):
    """A dict specifying the name of the module class (to find it) and the kwargs to pass to it.

    cls_name (str): Specifies which class the module is (can be an absolute identifier e.g. `orca.model.components.tokenizers:Resnet18`, or a relative identifier e.g. `Resnet18`)
    kwargs (dict): The kwargs to pass to the module
    """

    cls_name: str
    kwargs: Dict[str, Any]


def base_orca_model_config():
    return {
        "observation_tokenizers": {},  # Dict[str, ModuleConfig], see orca.model.components.tokenizers for our standard tokenizers
        "task_tokenizers": {},  # Dict[str, ModuleConfig], see orca.model.components.tokenizers for our standard tokenizers
        "heads": {},  # Dict[str, ModuleConfig], see orca.model.components.heads for our standard heads
        "readouts": {},  # Dict[str, int]
        "token_embedding_size": 256,  # int
        "transformer_kwargs": {},  # See orca.model.components.transformer.Transformer for kwargs (basically, scaling)
        "max_horizon": 10,  # Sets the size of positional embeddings, and provides an upper limit on the maximum horizon of the model
    }


def create_module_from_config(config: ModuleConfig, default_library: str = None):
    """Looks for the referenced object, and calls it with the kwargs provided.

    If cls_name is a full identifier, e.g. "orca.model.components.tokenizers:Resnet18",
    we will import the module, and look for the object in there.

    Otherwise, we will import the default_library, and look for the object there.
    """
    cls = import_from_string(config["cls_name"], default_library=default_library)
    return cls(**config["kwargs"])


def create_module_config(cls_or_cls_name: Union[str, object], **kwargs) -> ModuleConfig:
    """Create a new tokenizer config.

    Args:
        cls_or_cls_name (str or object): The class or import string of the module
        kwargs (dict, optional): The kwargs to pass to the module. Must be JSON serializable.
    """
    cls_name = resolve_cls_name(cls_or_cls_name)
    return dict(cls_name=cls_name, kwargs=kwargs)


def resolve_cls_name(cls_or_cls_name):
    if isinstance(cls_or_cls_name, str):
        return cls_or_cls_name
    else:
        o = cls_or_cls_name
        if hasattr(o, "__module__") and hasattr(o, "__name__"):
            logging.info(f"Converting {o} to {o.__module__}.{o.__name__}")
            return f"{o.__module__}.{o.__name__}"
        else:
            logging.error(
                f"Could not figure out how {o} is imported. Please specify an import string of the form `module.submodule:class_name"
            )
            raise ValueError()


def update_module_config(config: ModuleConfig, cls_or_cls_name=None, **kwargs):
    """Update a module config in-place.

    Args:
        config (dict): Config of a module
        cls_or_cls_name (str or object): The class or import string of the tokenizer
        kwargs (dict, optional): The kwargs to pass to the tokenizer. Defaults to None. Must be JSON serializable.
    """
    if cls_or_cls_name is not None:
        config["cls_name"] = resolve_cls_name(cls_or_cls_name)
    config["kwargs"].update(kwargs)


def import_from_string(import_string: str, default_library: str = None):
    if ":" in import_string:
        library, name = import_string.split(":")
    else:
        library, name = default_library, import_string
    library = importlib.import_module(library)
    return getattr(library, name)


def kwargs_for_common_transformer_sizes(transformer_size):
    """
    Args:
        transformer_size (str): The size of the transformer. One of "dummy", "vanilla", "vit_s", "vit_b", "vit_l", "vit_h"

    Returns dict with elements:
            token_embedding_size (int): The size of the token embeddings
            transformer_kwargs (dict): The kwargs to pass to the transformer

    """
    assert transformer_size in ["dummy", "vanilla", "vit_s", "vit_b", "vit_l", "vit_h"]
    default_params = {
        "attention_dropout_rate": 0.0,
        "add_position_embedding": False,
    }

    TRANSFORMER_SIZES = {
        "dummy": dict(
            num_layers=1,
            mlp_dim=256,
            num_attention_heads=2,
            dropout_rate=0.1,
        ),
        "vanilla": dict(
            num_layers=4,
            mlp_dim=1024,
            num_attention_heads=8,
            dropout_rate=0.1,
        ),
        "vit_s": dict(
            num_layers=12,
            mlp_dim=1536,
            num_attention_heads=6,
            dropout_rate=0.0,
        ),
        "vit_b": dict(
            num_layers=12,
            mlp_dim=3072,
            num_attention_heads=12,
            dropout_rate=0.0,
        ),
        "vit_l": dict(
            num_layers=24,
            mlp_dim=4096,
            num_attention_heads=16,
            dropout_rate=0.1,
        ),
        "vit_h": dict(
            num_layers=32,
            mlp_dim=5120,
            num_attention_heads=16,
            dropout_rate=0.1,
        ),
    }

    TOKEN_DIMS = {
        "dummy": 256,
        "vanilla": 256,
        "vit_s": 384,
        "vit_b": 768,
        "vit_l": 1024,
        "vit_h": 1280,
    }
    return dict(
        token_embedding_size=TOKEN_DIMS[transformer_size],
        transformer_kwargs={
            **default_params,
            **TRANSFORMER_SIZES[transformer_size],
        },
    )
