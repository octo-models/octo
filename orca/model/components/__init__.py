from dataclasses import dataclass
from enum import auto, Enum

from orca.model.components.resnet_v1 import resnetv1_configs

encoders = dict()
encoders.update(resnetv1_configs)


class TokenType(Enum):
    TASK = auto()
    OBS = auto()
    ACTION = auto()


@dataclass
class TokenMetadata:
    """Useful metadata for computing attention masks"""

    kind: TokenType  # either 'task', 'obs', or 'action'
    timestep: int  # What timestep the token belongs to
    extra_metadata: dict = None  # Any extra information useful for attention mask
