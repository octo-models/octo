from collections import defaultdict
from contextlib import contextmanager
from fnmatch import fnmatch
import logging
import time

import flax
from flax.training import train_state
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
from ml_collections import ConfigDict
import numpy as np
import optax

from orca.utils import jax_utils
from orca.utils.typing import PRNGKey


class TrainState(train_state.TrainState):
    rng: PRNGKey


def create_train_state(
    rng,
    model_def,
    tx,
    init_args=(),
    init_kwargs=dict(),
    pretrained_loaders=tuple(),
    init_method=None,
):
    """Utility to create a TrainState."""
    init_rng, state_rng = jax.random.split(rng)

    # Initializing the model in a jit avoids running the model on CPU
    @jax.jit
    def _init():
        return model_def.init(init_rng, *init_args, **init_kwargs, method=init_method)

    init_dict = _init()

    ev, params = flax.core.pop(init_dict, "params")
    assert (
        len(ev) == 0
    ), "Are you forgetting to store some variables in the state? {}".format(ev.keys())

    for loader in pretrained_loaders:
        params = loader(params)

    return TrainState.create(
        apply_fn=model_def.apply,
        params=params.unfreeze(),
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


def batched_apply(fn, batch_size):
    """Turns a function that applies to a fixed batch size into one that applies to a variable batch size.
    Useful for passing variable batch sizes to jit-compiled functions.
    """

    def pad_to_size(arr, size):
        return np.pad(arr, ((0, size - len(arr)), *[(0, 0)] * (arr.ndim - 1)))

    def get_batch_size(tree):
        return next(iter(jax.tree_util.tree_leaves(tree))).shape[0]

    def wrapped_fn(*args, **kwargs):
        input_batch_size = get_batch_size((args, kwargs))
        multihost_utils.assert_equal(
            input_batch_size // batch_size,
            "batched_apply has been called with arguments that would lead to"
            " a different number of iterations on different hosts."
            f" got batch_size={batch_size}, input_batch_size={input_batch_size}"
            f" on host {jax.process_index()}.",
        )
        outputs = []
        for i in range(0, input_batch_size, batch_size):
            step_batch_size = min(batch_size, input_batch_size - i)
            step_args, step_kwargs = jax.tree_map(
                lambda arr: pad_to_size(arr[i : i + batch_size], batch_size),
                (args, kwargs),
            )
            step_args, step_kwargs = jax_utils.merge_along_axis(
                (step_args, step_kwargs)
            )
            step_output = fn(*step_args, **step_kwargs)
            step_output = jax.device_get(jax_utils.split_along_axis(step_output))
            outputs.append(
                jax.tree_map(
                    lambda arr: arr[:step_batch_size],
                    step_output,
                )
            )
        return jax.tree_map(lambda *args: np.concatenate(args, axis=0), *outputs)

    return wrapped_fn


def filter_eval_datasets(dataset_kwargs_list, sample_weights, eval_datasets=None):
    if sample_weights is None:
        sample_weights = [1.0] * len(dataset_kwargs_list)
    if eval_datasets is None:
        return dataset_kwargs_list, sample_weights
    else:
        return list(
            map(
                list,
                zip(
                    *[
                        (dkwargs, weight)
                        for dkwargs, weight in zip(dataset_kwargs_list, sample_weights)
                        if (dkwargs["name"] in eval_datasets)
                    ]
                ),
            )
        )


def create_optimizer(params_or_params_shape, optimizer_kwargs: dict):
    """Creates optimizer for ORCA.

    Optimizer_kwargs are the kwargs for optax.adamw; if the learning rate is a dict,
    it is interpreted as the kwargs for optax.warmup_cosine_decay_schedule. If clip_gradient
    is specified, then gradient clipping is applied.

    Returns:
        tx: an Optax optimizer
        lr_callable: Function that takes the current step and returns the learning rate
    """
    if isinstance(optimizer_kwargs["learning_rate"], dict):
        optimizer_kwargs["learning_rate"] = optax.warmup_cosine_decay_schedule(
            **optimizer_kwargs["learning_rate"]
        )
        lr_callable = optimizer_kwargs["learning_rate"]
    else:
        lr_callable = lambda _: optimizer_kwargs["learning_rate"]

    # Following ViT, timm, MAE: this mask skips weight decay on biases and LayerNorm parameters
    wd_mask = jax.tree_util.tree_map_with_path(
        lambda path, x: "kernel" in jax.tree_util.keystr(path), params_or_params_shape
    )

    clip_gradient = optimizer_kwargs.pop("clip_gradient", None)
    frozen_keys = optimizer_kwargs.pop("frozen_keys", None)

    tx = optax.adamw(mu_dtype=jnp.bfloat16, **optimizer_kwargs, mask=wd_mask)
    if clip_gradient is not None:
        tx = optax.chain(
            optax.clip_by_global_norm(clip_gradient),
            tx,
        )

    if frozen_keys:
        # define trainable and frozen parameter sets
        logging.info(
            f"Freezing parameters that include the following keys: {frozen_keys}."
        )
        partition_optimizers = {
            "trainable": tx,
            "frozen": optax.set_to_zero(),
        }
        # freeze anything that matches fnmatch patterns in `frozen_keys`
        # path is a string of .-separated module names, e.g. ('orca_transformer.BlockTransformer_0...')
        param_partitions = flax.traverse_util.path_aware_map(
            lambda path, v: "frozen"
            if any([fnmatch(".".join(path), key) for key in frozen_keys])
            else "trainable",
            params_or_params_shape,
        )
        tx = optax.multi_transform(partition_optimizers, param_partitions)

        logging.debug("Frozen params:")
        flax.traverse_util.path_aware_map(
            lambda path, opt_status: logging.debug(".".join(path))
            if opt_status == "frozen"
            else None,
            param_partitions,
        )
        total_params = sum(
            jax.tree_util.tree_leaves(
                jax.tree_map(lambda x: x.size, params_or_params_shape)
            )
        )
        trainable_params = sum(
            jax.tree_util.tree_leaves(
                jax.tree_map(
                    lambda x, y: x.size if y == "trainable" else 0,
                    params_or_params_shape,
                    param_partitions,
                )
            )
        )
        logging.info(f"Num trainable params: {trainable_params:,}.")
        logging.info(f"Num frozen params: {total_params - trainable_params:,}.")
        logging.info(
            "To see a detailed list of frozen params, set logging level to DEBUG."
        )

        zero_frozen_params = lambda params: jax.tree_map(
            lambda x, y: x if y == "trainable" else jnp.zeros(()),
            params,
            param_partitions,
        )
        param_norm_callable = lambda params: optax.global_norm(
            zero_frozen_params(params)
        )
    else:
        param_norm_callable = optax.global_norm

    return tx, lr_callable, param_norm_callable


def check_config_diff(new_conf, old_conf, silent=False):
    """Checks for differences between new config and old config dicts."""
    new_conf_flat = flax.traverse_util.flatten_dict(
        new_conf.to_dict() if isinstance(new_conf, ConfigDict) else new_conf
    )
    old_conf_flat = flax.traverse_util.flatten_dict(
        old_conf.to_dict() if isinstance(old_conf, ConfigDict) else old_conf
    )

    # check for missing / new keys
    if set(new_conf_flat.keys()) != set(old_conf_flat.keys()) and not silent:
        logging.info(
            "New config contains extra items: %s",
            set(new_conf_flat.keys()) - set(old_conf_flat.keys()),
        )
        logging.info(
            "New config doesn't contain items: %s",
            set(old_conf_flat.keys()) - set(new_conf_flat.keys()),
        )

    # print differing key values
    mismatched_keys = {
        k: (new_conf_flat[k], old_conf_flat[k])
        for k in new_conf_flat
        if k in old_conf_flat and new_conf_flat[k] != old_conf_flat[k]
    }
    if mismatched_keys and not silent:
        logging.info(
            "New config contains keys with new values: %s",
            flax.core.pretty_repr(mismatched_keys),
        )
    return mismatched_keys or (set(new_conf_flat.keys()) != set(old_conf_flat.keys()))


def merge_params(target_params, pretrained_params):
    """Copies pre-trained params into target_params for every param that has corresponding key + shape."""
    flat_target_params = flax.traverse_util.flatten_dict(target_params)
    flat_pretrained_params = flax.traverse_util.flatten_dict(pretrained_params)
    keys_to_update = [
        k
        for k in flat_target_params
        if k in flat_pretrained_params
        and flat_target_params[k].shape == flat_pretrained_params[k].shape
    ]
    missing_keys = [k for k in flat_target_params if k not in flat_pretrained_params]
    shape_mismatch_keys = [
        k
        for k in flat_target_params
        if k in flat_pretrained_params
        and flat_target_params[k].shape != flat_pretrained_params[k].shape
    ]

    for key in keys_to_update:
        logging.debug(f"Param copied from pre-trained: {'.'.join(key)}")
    if missing_keys or shape_mismatch_keys:
        logging.info("########## Parameters skipped during model loading: ##########")
        for key in missing_keys:
            logging.info(
                f"Param missing in pre-trained model, skipping: {'.'.join(key)}"
            )
        for key in shape_mismatch_keys:
            logging.info(
                f"Param with differing shape in pre-trained model, skipping: {'.'.join(key)}"
            )

    flat_target_params = flax.core.copy(
        flat_target_params, {k: flat_pretrained_params[k] for k in keys_to_update}
    )
    target_params = flax.traverse_util.unflatten_dict(flat_target_params)
    return flax.core.freeze(target_params)


def process_text(batch, text_processor):
    if text_processor is None:
        batch["tasks"].pop("language_instruction")
    else:
        batch["tasks"]["language_instruction"] = text_processor.encode(
            [s.decode("utf-8") for s in batch["tasks"]["language_instruction"]]
        )
    return batch


def remove_text(tasks, zero_text):
    if "language_instruction" in tasks:
        new_language = jax.tree_map(
            lambda x, example: jnp.broadcast_to(example[None], x.shape),
            tasks["language_instruction"],
            zero_text,
        )
        tasks = flax.core.copy(tasks, {"language_instruction": new_language})
    return tasks


def remove_images(tasks):
    new_images = {k: jnp.zeros_like(v) for k, v in tasks.items() if "image" in k}
    return flax.core.copy(tasks, new_images)
