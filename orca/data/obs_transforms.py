"""
Contains observation-level transforms used in the orca data pipeline. These transforms operate on the
"observation" dictionary, and are applied at a per-frame level.
"""
import copy
from typing import Sequence

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
    num_image_keys = sum(["image" in key for key in obs])

    if not isinstance(augment_kwargs, Sequence):
        augment_kwargs = [copy.deepcopy(augment_kwargs)] * num_image_keys

    for i in range(num_image_keys):
        if augment_kwargs[i] is not None:
            key = f"image_{i}"
            if obs["pad_mask_dict"][key]:
                obs[key] = dl.transforms.augment_image(
                    obs[key], **augment_kwargs[i], seed=seed + i
                )
    return obs


def resize(obs: dict, resize_size, depth_resize_size) -> dict:
    """Resizes images and depth images."""
    num_image_keys = sum(["image" in key for key in obs])
    num_depth_keys = sum(["depth" in key for key in obs])

    if resize_size is None or isinstance(resize_size[0], int):
        resize_size = [resize_size] * num_image_keys
    if depth_resize_size is None or isinstance(depth_resize_size[0], int):
        depth_resize_size = [depth_resize_size] * num_depth_keys

    for i in range(num_image_keys):
        if resize_size[i] is not None:
            key = f"image_{i}"
            obs[key] = dl.transforms.resize_image(obs[key], size=resize_size[i])

    for i in range(num_depth_keys):
        if depth_resize_size[i] is not None:
            key = f"depth_{i}"
            obs[key] = dl.transforms.resize_depth_image(
                obs[key], size=depth_resize_size[i]
            )
    return obs
