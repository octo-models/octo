from typing import Any, Callable, Dict, Sequence, Union

import flax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]
Array = Union[np.ndarray, jnp.ndarray, tf.Tensor]
Data = Union[Array, Dict[str, "Data"]]
Batch = Dict[str, Data]
# A method to be passed into TrainState.__call__
ModuleMethod = Union[str, Callable, None]
