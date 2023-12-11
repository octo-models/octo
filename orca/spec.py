"""Utilities for making ORCAModel configs."""
from copy import deepcopy
from functools import partial, wraps
import importlib
import logging
from typing import Any, Dict, Tuple, TypedDict, Union

from ml_collections import ConfigDict


class ModuleSpec(TypedDict):
    """A dict specifying the name of a callable and the kwargs to pass to it.

    module (str): The module the callable is located in
    name (str): The name of the callable in the module
    kwargs (dict): The kwargs to pass to the callable
    """

    module: str
    name: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

    @staticmethod
    def create(callable_or_full_name: Union[str, callable], *args, **kwargs) -> "ModuleSpec":  # type: ignore
        """Create a module spec from a class or import string.

        Args:
            callable_or_full_name (str or object): Either the object or a fully qualified import string
                (e.g. "orca.model.components.transformer:Transformer")
        kwargs (dict, optional): Passed into callable.
        """
        if isinstance(callable_or_full_name, str):
            module, name = callable_or_full_name.split(":")
        else:
            module, name = _infer_cls_name(callable_or_full_name)

        return ModuleSpec(module=module, name=name, args=args, kwargs=kwargs)

    @staticmethod
    def instantiate(spec: "ModuleSpec"):  # type: ignore
        cls = _import_from_string(spec["module"], spec["name"])
        return partial(cls, *spec["args"], **spec["kwargs"])


def _infer_cls_name(o: object):
    if hasattr(o, "__module__") and hasattr(o, "__name__"):
        return o.__module__, o.__name__
    else:
        raise ValueError(f"Could not infer identifier for {o}")


def _import_from_string(module_string: str, name: str):
    try:
        module = importlib.import_module(module_string)
        return getattr(module, name)
    except Exception as e:
        raise ValueError(f"Could not import {module_string}:{name}") from e
