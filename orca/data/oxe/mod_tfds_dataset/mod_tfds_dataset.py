"""Modifies TFDS dataset with a map function, updates the feature definition and stores new dataset."""
from absl import app, flags
import tensorflow as tf
import tensorflow_datasets as tfds

from orca.data.oxe.mod_tfds_dataset.utils.mod_functions import TFDS_MOD_FUNCTIONS

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset", None, "Dataset name.")
flags.DEFINE_string("data_dir", None, "Directory where source data is stored.")
flags.DEFINE_string("target_dir", None, "Directory where modified data is stored.")
flags.DEFINE_list("mods", None, "List of modification functions, applied in order.")
flags.DEFINE_integer("n_workers", 10, "Number of parallel workers for data conversion.")
flags.DEFINE_integer(
    "max_episodes_in_memory",
    100,
    "Number of episodes converted & stored in memory before writing to disk.",
)


def mod_features(features):
    """Modifies feature dict."""
    for mod in FLAGS.mods:
        features = TFDS_MOD_FUNCTIONS[mod].mod_features(features)
    return features


def mod_dataset(ds):
    """Modifies dataset features."""
    for mod in FLAGS.mods:
        ds = TFDS_MOD_FUNCTIONS[mod].mod_dataset(ds)
    return ds


def main(_):
    builder = tfds.builder(FLAGS.dataset, data_dir=FLAGS.data_dir)

    features = mod_features(builder.info.features)
    print("############# Target features: ###############")
    print(features)
    print("##############################################")

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
