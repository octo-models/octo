from functools import partial
import importlib
from typing import Any, Dict, Tuple, TypedDict, Union


class ModuleSpec(TypedDict):
    """A JSON-serializable representation of a function or class with some default args and kwargs to pass to
    it. Useful for specifying a particular class or function in a config file, while keeping it serializable
    and overridable from the command line using ml_collections.

    Usage:

        # Preferred way to create a spec:
        >>> from octo.model.components.transformer import Transformer
        >>> spec = ModuleSpec.create(Transformer, num_layers=3)
        # Same as above using the fully qualified import string:
        >>> spec = ModuleSpec.create("octo.model.components.transformer:Transformer", num_layers=3)

        # Usage:
        >>> ModuleSpec.instantiate(spec) == partial(Transformer, num_layers=3)
        # can pass additional kwargs at instantiation time
        >>> transformer = ModuleSpec.instantiate(spec, num_heads=8)

    Note: ModuleSpec is just an alias for a dictionary (that is strongly typed), not a real class. So from
    your code's perspective, it is just a dictionary.

    module (str): The module the callable is located in
    name (str): The name of the callable in the module
    args (tuple): The args to pass to the callable
    kwargs (dict): The kwargs to pass to the callable
    """

    module: str
    name: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

    @staticmethod
    def create(callable_or_full_name: Union[str, callable], *args, **kwargs) -> "ModuleSpec":  # type: ignore
        """Create a module spec from a callable or import string.

        Args:
            callable_or_full_name (str or object): Either the object itself or a fully qualified import string
                (e.g. "octo.model.components.transformer:Transformer")
        args (tuple, optional): Passed into callable upon instantiation.
        kwargs (dict, optional): Passed into callable upon instantiation.
        """
        if isinstance(callable_or_full_name, str):
            assert callable_or_full_name.count(":") == 1, (
                "If passing in a string, it must be a fully qualified import string "
                "(e.g. 'octo.model.components.transformer:Transformer')"
            )
            module, name = callable_or_full_name.split(":")
        else:
            module, name = _infer_full_name(callable_or_full_name)

        return ModuleSpec(module=module, name=name, args=args, kwargs=kwargs)

    @staticmethod
    def instantiate(spec: "ModuleSpec"):  # type: ignore
        if set(spec.keys()) != {"module", "name", "args", "kwargs"}:
            raise ValueError(
                f"Expected ModuleSpec, but got {spec}. "
                "ModuleSpec must have keys 'module', 'name', 'args', and 'kwargs'."
            )
        cls = _import_from_string(spec["module"], spec["name"])
        return partial(cls, *spec["args"], **spec["kwargs"])


def _infer_full_name(o: object):
    if hasattr(o, "__module__") and hasattr(o, "__name__"):
        return o.__module__, o.__name__
    else:
        raise ValueError(
            f"Could not infer identifier for {o}. "
            "Please pass in a fully qualified import string instead "
            "e.g. 'octo.model.components.transformer:Transformer'"
        )


def _import_from_string(module_string: str, name: str):
    try:
        module = importlib.import_module(module_string)
        return getattr(module, name)
    except Exception as e:
        raise ValueError(f"Could not import {module_string}:{name}") from e
