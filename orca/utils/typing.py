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
StateEncodingType = int
ActionEncodingType = int


class StateEncoding:
    """Defines supported proprio state encoding schemes for different datasets."""

    NONE = -1  # no state provided
    POS_EULER = 1  # EEF XYZ + roll-pitch-yaw + 1 x pad + gripper open/close
    POS_QUAT = 2  # EEF XYZ + quaternion + gripper open/close
    JOINT = 3  # 7 x joint angles (padding added if fewer) + gripper open/close


class ActionEncoding:
    """Defines supported action encoding schemes for different datasets."""

    EEF_POS = 1  # EEF delta XYZ + roll-pitch-yaw + gripper open/close
    JOINT_POS = 2  # 7 x joint delta position + gripper open/close
