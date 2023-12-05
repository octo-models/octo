# adapted from https://github.com/google-research/robotics_transformer/blob/master/transformer_network.py

from orca.model.components.base_head import BaseTemporalHead


class TemporalDistanceValueHead(BaseTemporalHead):
    pass


VALUE_HEADS = {
    "temporal_distance_value_head": TemporalDistanceValueHead,
}
