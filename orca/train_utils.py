import logging
import time
from collections import defaultdict

import flax
import jax
from flax.training import train_state
from jax.experimental.compilation_cache import compilation_cache

from orca.typing import PRNGKey


class TrainState(train_state.TrainState):
    rng: PRNGKey


def shard_batch(batch, sharding):
    """Shards a batch across devices along its first dimension.

    Args:
        batch: A pytree of arrays.
        sharding: A jax Sharding object with shape (num_devices,).
    """
    return jax.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


def create_train_state(
    rng, model_def, tx, init_args=(), init_kwargs=dict(), pretrained_loaders=tuple()
):
    """Utility to create a TrainState."""
    init_rng, state_rng = jax.random.split(rng)

    # Initializing the model in a jit avoids running the model on CPU
    @jax.jit
    def init(rng):
        return model_def.init(rng, *init_args, **init_kwargs)

    ev, params = init(init_rng).pop("params")
    assert (
        len(ev) == 0
    ), "Are you forgetting to store some variables in the state? {}".format(ev.keys())

    for loader in pretrained_loaders:
        params = loader(params)

    return TrainState.create(
        apply_fn=model_def.apply,
        params=params,
        tx=tx,
        rng=state_rng,
    )


def format_name_with_config(name, config):
    """Formats a name string with a config dict.

    Formatting keys may be specified as {key} or {full_path_to_key_with_underscores}.

    Example:
        name = "model_{model_type}_{model_size}"
        config = {"model_type": "transformer", "model_size": "small"}
        format_name_with_config(name, config) -> "model_transformer_small"
    """
    config_flat = flax.traverse_util.flatten_dict(config, sep="_")
    config_final = {k.split("_")[-1]: v for k, v in config_flat.items()}
    format_dict = {**config_final, **config_flat}
    return name.format(**format_dict)


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.counts = defaultdict(int)
        self.times = defaultdict(float)
        self.start_times = {}

    def tick(self, key):
        if key in self.start_times:
            raise ValueError(f"Timer is already ticking for key: {key}")
        self.start_times[key] = time.time()

    def tock(self, key):
        if key not in self.start_times:
            raise ValueError(f"Timer is not ticking for key: {key}")
        self.counts[key] += 1
        self.times[key] += time.time() - self.start_times[key]
        del self.start_times[key]

    def get_average_times(self, reset=True):
        ret = {key: self.times[key] / self.counts[key] for key in self.counts}
        if reset:
            self.reset()
        return ret


def initialize_compilation_cache(cache_dir="/tmp/jax_cache"):
    compilation_cache.initialize_cache(cache_dir)

    for logger in [logging.getLogger(name) for name in logging.root.manager.loggerDict]:
        logger.addFilter(
            lambda record: "Not writing persistent cache entry for"
            not in record.getMessage()
        )
