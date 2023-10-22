from abc import ABC, abstractmethod
from typing import Any, Dict

import dlimp as dl
import tensorflow as tf
import tensorflow_datasets as tfds


class TfdsModFunction(ABC):
    @staticmethod
    @abstractmethod
    def mod_features(
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        """
        Modifies the data builder feature dict to reflect feature changes of ModFunction.
        """
        ...

    @staticmethod
    @abstractmethod
    def mod_dataset(ds: tf.data.Dataset) -> tf.data.Dataset:
        """
        Perform arbitrary modifications on the dataset that comply with the modified feature definition.
        """
        ...


def mod_obs_features(features, obs_feature_mod_function):
    """Utility function to only modify keys in observation dict."""
    return tfds.features.FeaturesDict(
        {
            "steps": tfds.features.Dataset(
                {
                    "observation": tfds.features.FeaturesDict(
                        {
                            key: obs_feature_mod_function(
                                key, features["steps"]["observation"][key]
                            )
                            for key in features["steps"]["observation"].keys()
                        }
                    ),
                    **{
                        key: features["steps"][key]
                        for key in features["steps"].keys()
                        if key not in ("observation",)
                    },
                }
            ),
            **{key: features[key] for key in features.keys() if key not in ("steps",)},
        }
    )


class ResizeAndJpegEncode(TfdsModFunction):
    MAX_RES: int = 336

    @staticmethod
    def mod_features(
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        def downsize_and_jpeg(key, feat):
            """Downsizes image features, encodes as jpeg."""
            if len(feat.shape) >= 2 and feat.shape[0] >= 64:  # is image / depth feature
                size_h = min(feat.shape[0], ResizeAndJpegEncode.MAX_RES)
                size_w = min(feat.shape[1], ResizeAndJpegEncode.MAX_RES)
                should_jpeg_encode = (
                    isinstance(feat, tfds.features.Image) and "depth" not in key
                )
                if len(feat.shape) > 2:
                    new_shape = (size_h, size_w, feat.shape[2])
                else:
                    new_shape = (size_h, size_w)

                if isinstance(feat, tfds.features.Image):
                    return tfds.features.Image(
                        shape=new_shape,
                        dtype=feat.dtype,
                        encoding_format="jpeg" if should_jpeg_encode else "png",
                        doc=feat.doc,
                    )
                else:
                    return tfds.features.Tensor(
                        shape=new_shape,
                        dtype=feat.dtype,
                        doc=feat.doc,
                    )

            return feat

        return mod_obs_features(features, downsize_and_jpeg)

    @staticmethod
    def mod_dataset(ds: tf.data.Dataset) -> tf.data.Dataset:
        def resize_image_fn(step):
            # resize images
            for key in step["observation"]:
                if len(step["observation"][key].shape) > 2 and (
                    step["observation"][key].shape[0] > ResizeAndJpegEncode.MAX_RES
                    or step["observation"][key].shape[1] > ResizeAndJpegEncode.MAX_RES
                ):
                    size = (
                        min(
                            step["observation"][key].shape[0],
                            ResizeAndJpegEncode.MAX_RES,
                        ),
                        min(
                            step["observation"][key].shape[1],
                            ResizeAndJpegEncode.MAX_RES,
                        ),
                    )
                    if "depth" in key:
                        step["observation"][key] = tf.cast(
                            dl.utils.resize_depth_image(
                                tf.cast(step["observation"][key], tf.float32), size
                            ),
                            step["observation"][key].dtype,
                        )
                    else:
                        step["observation"][key] = tf.cast(
                            dl.utils.resize_image(step["observation"][key], size),
                            tf.uint8,
                        )
            return step

        def episode_map_fn(episode):
            episode["steps"] = episode["steps"].map(resize_image_fn)
            return episode

        return ds.map(episode_map_fn)


class FilterSuccess(TfdsModFunction):
    @staticmethod
    def mod_features(
        features: tfds.features.FeaturesDict,
    ) -> tfds.features.FeaturesDict:
        return features  # no feature changes

    @staticmethod
    def mod_dataset(ds: tf.data.Dataset) -> tf.data.Dataset:
        return ds.filter(lambda e: e["success"])


TFDS_MOD_FUNCTIONS = {
    "resize_and_jpeg_encode": ResizeAndJpegEncode,
    "filter_success": FilterSuccess,
}
