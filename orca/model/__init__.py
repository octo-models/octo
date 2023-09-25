import logging

from .orca_policy import ORCAPolicy


def create_model_def(
    action_dim,
    window_size,
    policy_kwargs,
    **kwargs,
):
    if len(kwargs) > 0:
        logging.warn(f"Extra kwargs passed into create_model_def: {kwargs}")

    model_def = ORCAPolicy(
        action_dim=action_dim,
        window_size=window_size,
        **policy_kwargs,
    )
    return model_def
