"""
Contains observation-level transforms used in the octo data pipeline. These transforms operate on the
"observation" dictionary, and are applied at a per-frame level.
"""
from typing import Mapping, Optional, Tuple, Union

from absl import logging
import dlimp as dl
import tensorflow as tf


def augment(
    obs: dict, seed: tf.Tensor, augment_kwargs: Union[dict, Mapping[str, dict]]
) -> dict:
    """Augments images, skipping padding images."""
    if not hasattr(augment_kwargs, "items"):
        raise ValueError(
            "augment_kwargs must be a dict with keys corresponding to image names, or a single dict "
            "with an 'augment_order' key."
        )
    image_names = {key[6:] for key in obs if key.startswith("image_")}

    # "augment_order" is required in augment_kwargs, so if it's there, we can assume that the user has passed
    # in a single augmentation dict (otherwise, we assume that the user has passed in a mapping from image
    # name to augmentation dict)
    if "augment_order" in augment_kwargs:
        augment_kwargs = {name: augment_kwargs for name in image_names}

    for i, name in enumerate(image_names):
        if name not in augment_kwargs:
            continue
        kwargs = augment_kwargs[name]
        logging.debug(f"Augmenting image_{name} with kwargs {kwargs}")
        obs[f"image_{name}"] = tf.cond(
            obs["pad_mask_dict"][f"image_{name}"],
            lambda: dl.transforms.augment_image(
                obs[f"image_{name}"],
                **kwargs,
                seed=seed + i,  # augment each image differently
            ),
            lambda: obs[f"image_{name}"],  # skip padding images
        )

    return obs


def image_dropout(
    obs: dict,
    seed: tf.Tensor,
    dropout_prob: float,
    always_keep_key: Optional[str] = None,
) -> dict:
    """Independently drops out image keys, each with probability `dropout_prob`, but always keeps at least one
    image present.
    """
    image_keys = [key for key in obs if key.startswith("image_")]
    if not image_keys:
        return obs
    pad_mask = tf.stack([obs["pad_mask_dict"][key] for key in image_keys])
    # if any non-padding images exist, pick one of them to keep no matter what
    shuffle_seed, seed = tf.unstack(tf.random.split(seed))

    if always_keep_key:
        assert (
            always_keep_key in image_keys
        ), f"Specified always_keep_key {always_keep_key} not present in image_keys: {image_keys} during dropout."
        always_keep_index = tf.constant(
            image_keys.index(always_keep_key), dtype=tf.int64
        )
    else:
        always_keep_index = tf.cond(
            tf.reduce_any(pad_mask),
            # pick a random index from the non-padding images
            lambda: tf.random.experimental.stateless_shuffle(
                tf.where(pad_mask)[:, 0], seed=shuffle_seed
            )[0],
            # all images are padding, so it doesn't matter
            lambda: tf.constant(0, dtype=tf.int64),
        )

    # drop images independently, except for the one at always_keep_index
    rands = tf.random.stateless_uniform([len(image_keys)], seed=seed)
    pad_mask = tf.logical_and(
        pad_mask,
        tf.logical_or(
            tf.range(len(image_keys), dtype=tf.int64) == always_keep_index,
            rands > dropout_prob,
        ),
    )

    # perform the dropout and update pad_mask_dict
    for i, key in enumerate(image_keys):
        obs["pad_mask_dict"][key] = pad_mask[i]
        obs[key] = tf.cond(
            pad_mask[i],
            lambda: obs[key],
            lambda: tf.zeros_like(obs[key]),
        )
    return obs


def decode_and_resize(
    obs: dict,
    resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]],
    depth_resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]],
) -> dict:
    """Decodes images and depth images, and then optionally resizes them."""
    # just gets the part after "image_" or "depth_"
    image_names = {key[6:] for key in obs if key.startswith("image_")}
    depth_names = {key[6:] for key in obs if key.startswith("depth_")}

    if isinstance(resize_size, tuple):
        resize_size = {name: resize_size for name in image_names}
    if isinstance(depth_resize_size, tuple):
        depth_resize_size = {name: depth_resize_size for name in depth_names}

    for name in image_names:
        if name not in resize_size:
            logging.warning(
                f"No resize_size was provided for image_{name}. This will result in 1x1 "
                "padding images, which may cause errors if you mix padding and non-padding images."
            )
        image = obs[f"image_{name}"]
        if image.dtype == tf.string:
            if tf.strings.length(image) == 0:
                # this is a padding image
                image = tf.zeros((*resize_size.get(name, (1, 1)), 3), dtype=tf.uint8)
            else:
                image = tf.io.decode_image(
                    image, expand_animations=False, dtype=tf.uint8
                )
        elif image.dtype != tf.uint8:
            raise ValueError(
                f"Unsupported image dtype: found image_{name} with dtype {image.dtype}"
            )
        if name in resize_size:
            image = dl.transforms.resize_image(image, size=resize_size[name])
        obs[f"image_{name}"] = image

    for name in depth_names:
        if name not in depth_resize_size:
            logging.warning(
                f"No depth_resize_size was provided for depth_{name}. This will result in 1x1 "
                "padding depth images, which may cause errors if you mix padding and non-padding images."
            )
        depth = obs[f"depth_{name}"]
        if depth.dtype == tf.string:
            if tf.strings.length(depth) == 0:
                # this is a padding image
                depth = tf.zeros(
                    (*depth_resize_size.get(name, (1, 1)), 1), dtype=tf.float32
                )
            else:
                depth = tf.io.decode_image(
                    depth, expand_animations=False, dtype=tf.float32
                )[..., 0]
        elif depth.dtype != tf.float32:
            raise ValueError(
                f"Unsupported depth dtype: found depth_{name} with dtype {depth.dtype}"
            )
        if name in depth_resize_size:
            depth = dl.transforms.resize_depth_image(
                depth, size=depth_resize_size[name]
            )
        obs[f"depth_{name}"] = depth

    return obs
