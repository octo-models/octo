from typing import Any, Dict, List, Sequence, Union

from dlimp.transforms import selective_tree_map
import tensorflow as tf


def pprint_data_mixture(
    dataset_kwargs_list: List[Dict[str, Any]], dataset_weights: List[int]
) -> None:
    print(
        "\n######################################################################################"
    )
    print(
        f"# Loading the following {len(dataset_kwargs_list)} datasets (incl. sampling weight):{'': >24} #"
    )
    for dataset_kwargs, weight in zip(dataset_kwargs_list, dataset_weights):
        pad = 80 - len(dataset_kwargs["name"])
        print(f"# {dataset_kwargs['name']}: {weight:=>{pad}f} #")
    print(
        "######################################################################################\n"
    )


def maybe_decode_depth_images(
    x: Dict[str, Any], match: Union[str, Sequence[str]] = "depth"
) -> Dict[str, Any]:
    """Can operate on nested dicts. Decodes any leaves that have `match` anywhere in their path."""
    if isinstance(match, str):
        match = [match]

    return selective_tree_map(
        x,
        lambda keypath, value: any([s in keypath for s in match])
        and value.dtype == tf.string
        and value.dtype == tf.string,
        lambda e: tf.cast(tf.io.decode_image(e, expand_animations=False), tf.float32)[
            ..., 0
        ],
    )


def set_ram_budget(dataset, ram_budget):
    """Sets the RAM budget used by tf.data.AUTOTUNE."""
    autotune_options = tf.data.experimental.AutotuneOptions()
    autotune_options.ram_budget = ram_budget * 1024 * 1024 * 1024  # GB --> Bytes

    options = tf.data.Options()
    options.autotune = autotune_options
    return dataset.with_options(options)
