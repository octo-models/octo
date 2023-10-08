from collections import defaultdict
from contextlib import contextmanager
import logging
import time

import flax
from flax.training import train_state
import jax
from jax.experimental.compilation_cache import compilation_cache
import numpy as np

from orca.utils.typing import PRNGKey


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

    ev, params = flax.core.pop(init(init_rng), "params")
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
    """
    Timer utility. Usage:

        timer = Timer()
        with timer("foo"):
            do_something()

        timer.tick("bar")
        do_something_else()
        timer.tock("bar")

        timer.get_average_times() -> {"foo": 0.1, "bar": 0.2}
    """

    def __init__(self):
        self.reset()

    @contextmanager
    def __call__(self, key):
        self.tick(key)
        try:
            yield None
        finally:
            self.tock(key)

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


def batched_apply(fn, batch_size, sharding=None):
    """Turns a function that applies to a fixed batch size into one that applies to a variable batch size.
    Useful for passing variable batch sizes to jit-compiled functions.

    Currently assumes that the first axis is the batch axis **for both inputs and outputs**
    Pass in a sharding object to additionally shard the batch across devices.
    """

    def pad_to_size(arr, size):
        return np.pad(arr, ((0, size - len(arr)), *[(0, 0)] * (arr.ndim - 1)))

    def get_batch_size(tree):
        return next(iter(jax.tree_util.tree_leaves(tree))).shape[0]

    def wrapped_fn(*args, **kwargs):
        input_batch_size = get_batch_size((args, kwargs))
        outputs = []
        for i in range(0, input_batch_size, batch_size):
            step_batch_size = min(batch_size, input_batch_size - i)
            step_args, step_kwargs = jax.tree_map(
                lambda arr: pad_to_size(arr[i : i + batch_size], batch_size),
                (args, kwargs),
            )
            if sharding is not None:
                (step_args, step_kwargs) = shard_batch(
                    (step_args, step_kwargs), sharding
                )
            step_output = fn(*step_args, **step_kwargs)
            outputs.append(
                jax.tree_map(
                    lambda arr: arr[:step_batch_size],
                    step_output,
                )
            )
        return jax.tree_map(lambda *args: np.concatenate(args, axis=0), *outputs)

    return wrapped_fn
