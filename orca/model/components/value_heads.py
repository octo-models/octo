# adapted from https://github.com/google-research/robotics_transformer/blob/master/transformer_network.py
from typing import Any

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from orca.model.components.base_head import BaseTemporalHead
from orca.utils.typing import PRNGKey


class TemporalDistanceValueHead(BaseTemporalHead):
    pass


VALUE_HEADS = {
    "temporal_distance_value_head": TemporalDistanceValueHead,
}
