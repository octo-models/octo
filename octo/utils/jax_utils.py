import logging
import os
from typing import Any, Optional, Sequence

import jax
from jax.experimental import multihost_utils
from jax.experimental.compilation_cache import compilation_cache
import jax.numpy as jnp
import numpy as np


def host_broadcast_str(x: str) -> str:
    """Broadcast_one_to_all, but with a string. Strings should all be the same length."""
    multihost_utils.assert_equal(
        len(x), f"String lengths are not equal: got {len(x)} for {jax.process_index()}"
    )
    encoded = np.array([ord(c) for c in x], dtype=np.uint8)
    encoded = multihost_utils.broadcast_one_to_all(encoded)
    return "".join([chr(u) for u in encoded])


def shard_along_axis(x: Any, devices: Sequence[jax.Device], axis: int = 0) -> jax.Array:
    """Shard a PyTree of arrays along a given axis, putting them on device in
    the process. Works in multi-host setting as long as PyTrees are equal on all
    hosts."""
    sharding = jax.sharding.NamedSharding(
        jax.sharding.Mesh(devices, "x"),
        jax.sharding.PartitionSpec(*([None] * axis + ["x"])),
    )
    x = jax.tree_map(jnp.array, x)
    return jax.tree_map(
        lambda arr: jax.make_array_from_callback(
            arr.shape, sharding, lambda index: arr[index]
        ),
        x,
    )


def merge_along_axis(x: Any, axis: int = 0) -> jax.Array:
    """Convert a PyTree of host-local arrays to a global array, concatenating and sharding along
    `axis`."""
    return multihost_utils.host_local_array_to_global_array(
        x,
        jax.sharding.Mesh(jax.devices(), "x"),
        jax.sharding.PartitionSpec(*([None] * axis + ["x"])),
    )


def split_along_axis(x: Any, axis: int = 0) -> jax.Array:
    """Convert a PyTree of global arrays to a host-local array, splitting along `axis`."""
    return multihost_utils.global_array_to_host_local_array(
        x,
        jax.sharding.Mesh(jax.devices(), "x"),
        jax.sharding.PartitionSpec(*([None] * axis + ["x"])),
    )


def replicate(x: Any, devices: Optional[Sequence[jax.Device]] = None) -> jax.Array:
    """Replicate a PyTree of arrays across devices. Works in multi-host setting
    as long as PyTrees are equal on all hosts."""
    if devices is None:
        devices = jax.devices()
    sharding = jax.sharding.PositionalSharding(devices).replicate()
    x = jax.tree_map(jnp.array, x)
    return jax.tree_map(
        lambda arr: jax.make_array_from_callback(
            arr.shape, sharding, lambda index: arr[index]
        ),
        x,
    )


def initialize_compilation_cache(
    cache_dir=os.path.expanduser("~/.jax_compilation_cache"),
):
    """Initializes the Jax persistent compilation cache."""
    compilation_cache.initialize_cache(cache_dir)
    for logger in [logging.getLogger(name) for name in logging.root.manager.loggerDict]:
        logger.addFilter(
            lambda record: "Not writing persistent cache entry for"
            not in record.getMessage()
        )
