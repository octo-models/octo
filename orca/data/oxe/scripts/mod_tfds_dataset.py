"""Modifies TFDS dataset with a map function, updates the feature definition and stores new dataset."""
from absl import app, flags
import dlimp as dl
import tensorflow as tf
import tensorflow_datasets as tfds

TARGET_MAX_RES = 336

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset", None, "Dataset name.")
flags.DEFINE_string("data_dir", None, "Directory where source data is stored.")
flags.DEFINE_string("target_dir", None, "Directory where modified data is stored.")


def maybe_mod_obs_feature(feat, key):
    """Downsizes image features, encodes as jpeg."""
    if len(feat.shape) >= 2 and feat.shape[0] >= 64:  # is image / depth feature
        size_h = min(feat.shape[0], TARGET_MAX_RES)
        size_w = min(feat.shape[1], TARGET_MAX_RES)
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


def mod_features(features):
    """Modifies feature dict."""
    return tfds.features.FeaturesDict(
        {
            "steps": tfds.features.Dataset(
                {
                    "observation": tfds.features.FeaturesDict(
                        {
                            key: maybe_mod_obs_feature(
                                features["steps"]["observation"][key], key
                            )
                            for key in features["steps"]["observation"].keys()
                        }
                    ),
                    **{
                        key: features["steps"][key]
                        for key in features["steps"].keys()
                        if key not in ("observation", "language_embedding")
                    },
                }
            ),
            "episode_metadata": features["episode_metadata"],
        }
    )


def mod_dataset(ds):
    """Modifies dataset features."""

    def mod_data(traj):
        def mod_step(step):
            # optionally resize images
            for key in step["observation"]:
                if len(step["observation"][key].shape) > 2 and (
                    step["observation"][key].shape[0] > TARGET_MAX_RES
                    or step["observation"][key].shape[1] > TARGET_MAX_RES
                ):
                    size = (
                        min(step["observation"][key].shape[0], TARGET_MAX_RES),
                        min(step["observation"][key].shape[1], TARGET_MAX_RES),
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
            if "language_embedding" in step:
                step.pop("language_embedding")
            return step

        traj["steps"] = traj["steps"].map(mod_step)
        return traj

    ds = ds.map(mod_data)
    return ds


def main(_):
    builder = tfds.builder(FLAGS.dataset, data_dir=FLAGS.data_dir)
    tfds.dataset_builders.store_as_tfds_dataset(
        name=FLAGS.dataset,
        version=builder.version,
        features=mod_features(builder.info.features),
        split_datasets={
            split: mod_dataset(builder.as_dataset(split=split))
            for split in builder.info.splits
        },
        config=builder.builder_config,
        data_dir=FLAGS.target_dir,
    )


if __name__ == "__main__":
    app.run(main)
