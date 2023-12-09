"""
Contains observation-level transforms used in the orca data pipeline. These transforms operate on the
"observation" dictionary, and are applied at a per-frame level.
"""
import logging
from typing import Mapping

import dlimp as dl
import tensorflow as tf


def decode_images(obs: dict) -> dict:
    """Decodes images and depth images."""
    for key in obs:
        if "image" in key:
            if obs[key].dtype == tf.string:
                if tf.strings.length(obs[key]) == 0:
                    # this is a padding image
                    obs[key] = tf.zeros((1, 1, 3), dtype=tf.uint8)
                else:
                    obs[key] = tf.io.decode_image(
                        obs[key], expand_animations=False, dtype=tf.uint8
                    )
            elif obs[key].dtype == tf.uint8:
                pass
            else:
                raise ValueError(
                    f"Unsupported image dtype: found {key} with dtype {obs[key].dtype}"
                )
        elif "depth" in key:
            if obs[key].dtype == tf.string:
                if tf.strings.length(obs[key]) == 0:
                    # this is a padding image
                    obs[key] = tf.zeros((1, 1), dtype=tf.float32)
                else:
                    obs[key] = tf.io.decode_image(
                        obs[key], expand_animations=False, dtype=tf.float32
                    )[..., 0]
            elif obs[key].dtype == tf.float32:
                pass
            else:
                raise ValueError(
                    f"Unsupported depth dtype: found {key} with dtype {obs[key].dtype}"
                )
    return obs


def augment(obs: dict, seed, augment_kwargs) -> dict:
    """Augments images, skipping padding images."""
    image_names = {key[6:] for key in obs if key.startswith("image_")}

    # "augment_order" is required in augment_kwargs, so if it's there, we can assume that the user has passed
    # in a single augmentation dict (otherwise, we assume that the user has passed in a mapping from image
    # name to augmentation dict)
    if "augment_order" in augment_kwargs:
        augment_kwargs = {name: augment_kwargs for name in image_names}

    for i, (name, kwargs) in enumerate(augment_kwargs.items()):
        logging.debug(f"Augmenting image_{name} with kwargs {kwargs}")
        obs[f"image_{name}"] = tf.cond(
            obs["pad_mask_dict"][f"image_{name}"],
            lambda: dl.transforms.augment_image(
                obs[f"image_{name}"],
                **kwargs,
                seed=seed + i,  # augment each image differently
            ),
            lambda: obs[f"image_{name}"],
        )

    return obs


def resize(obs: dict, resize_size, depth_resize_size) -> dict:
    """Resizes images and depth images."""
    # just gets the part after "image_" or "depth_"
    image_names = {key[6:] for key in obs if key.startswith("image_")}
    depth_names = {key[6:] for key in obs if key.startswith("depth_")}

    if not isinstance(resize_size, Mapping):
        resize_size = {name: resize_size for name in image_names}
    if not isinstance(depth_resize_size, Mapping):
        depth_resize_size = {name: depth_resize_size for name in depth_names}

    for name, size in resize_size.items():
        obs[f"image_{name}"] = dl.transforms.resize_image(
            obs[f"image_{name}"], size=size
        )

    for name, size in depth_resize_size.items():
        obs[f"depth_{name}"] = dl.transforms.resize_depth_image(
            obs[f"depth_{name}"], size=size
        )

    return obs
