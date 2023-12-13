# adapted from https://github.com/google-research/robotics_transformer/blob/master/transformer_network.py

from octo.model.components.base_head import BaseTemporalHead


class TemporalDistanceRewardHead(BaseTemporalHead):
    pass


REWARD_HEADS = {
    "temporal_distance_reward_head": TemporalDistanceRewardHead,
}
