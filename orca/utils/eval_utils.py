from functools import partial
import logging
import os
from pathlib import Path, PurePath

import flax
import jax
import jax.numpy as jnp
import tensorflow as tf

from orca.utils.pretrained_utils import PretrainedModel


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped


@partial(jax.jit, static_argnames="argmax")
def sample_actions(
    pretrained_model: PretrainedModel,
    observations,
    tasks,
    rng,
    argmax=False,
    temperature=1.0,
):
    # add batch dim to observations
    observations = jax.tree_map(lambda x: x[None], observations)
    logging.warning(
        "observations: %s", flax.core.pretty_repr(jax.tree_map(jnp.shape, observations))
    )
    logging.warning("tasks: %s", flax.core.pretty_repr(jax.tree_map(jnp.shape, tasks)))
    actions = pretrained_model.sample_actions(
        observations,
        tasks,
        rng=rng,
        argmax=argmax,
        temperature=temperature,
    )
    # remove batch dim
    return actions[0]


def download_checkpoint_from_gcs(cloud_path: str, step: str, save_path: str):
    if not cloud_path.startswith("gs://"):
        return cloud_path, step  # Actually on the local filesystem

    checkpoint_path = tf.io.gfile.join(cloud_path, f"{step}")
    ds_stats_path = tf.io.gfile.join(cloud_path, "dataset_statistics*")
    config_path = tf.io.gfile.join(cloud_path, "config.json*")
    example_batch_path = tf.io.gfile.join(cloud_path, "example_batch.msgpack*")

    run_name = Path(cloud_path).name
    save_path = os.path.join(save_path, run_name)

    target_checkpoint_path = os.path.join(save_path, f"{step}")
    if os.path.exists(target_checkpoint_path):
        logging.warning(
            "Checkpoint already exists at %s, skipping download", target_checkpoint_path
        )
        return save_path, step
    os.makedirs(save_path, exist_ok=True)
    logging.warning("Downloading checkpoint and metadata to %s", save_path)

    os.system(f"gsutil cp -r {checkpoint_path} {save_path}/")
    os.system(f"gsutil cp {ds_stats_path} {save_path}/")
    os.system(f"gsutil cp {config_path} {save_path}/")
    os.system(f"gsutil cp {example_batch_path} {save_path}/")

    return save_path, step


def load_jaxrlm_checkpoint(weights_path: str, config_path: str, code_path: str):
    from codesave import UniqueCodebase

    with UniqueCodebase(code_path) as cs:
        pretrained_utils = cs.import_module("jaxrl_m.pretrained_utils")
        loaded = pretrained_utils.load_checkpoint(
            weights_path, config_path, im_size=256
        )
    # loaded contains: {
    # "agent": jaxrlm Agent,
    # "policy_fn": callable taking in observation and goal inputs and outputs **unnormalized** actions,
    # "normalization_stats": {"action": {"mean": [7], "std": [7]}}
    # "obs_horizon": int
    # }

    class Dummy:
        def create_tasks(self, goals):
            return goals.copy()

    def new_policy_fn(observations, goals):
        observations = {"image": observations["image_0"]}
        goals = {"image": goals["image_0"]}
        return loaded["policy_fn"](observations, goals)

    return new_policy_fn, Dummy()
