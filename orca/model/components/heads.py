from orca.model.components.action_heads import ACTION_HEADS
from orca.model.components.reward_heads import REWARD_HEADS
from orca.model.components.value_heads import VALUE_HEADS

HEADS = {
    **ACTION_HEADS,
    **REWARD_HEADS,
    **VALUE_HEADS,
}
