from copy import deepcopy

from scripts.configs.config import get_config as get_base_config
from scripts.configs.config import update_config


def get_config():
    base_config = get_base_config("dummy")
    del base_config["dataset_kwargs"]["oxe_kwargs"]
    config = update_config(
        base_config,
        num_steps=20,
        optimizer=dict(
            learning_rate=dict(
                warmup_steps=1,
            ),
        ),
        batch_size=64,
        shuffle_buffer_size=1000,
        num_val_batches=1,
        log_interval=10,
        eval_interval=10,
        eval_datasets=None,
        trajs_for_metrics=1,
        trajs_for_viz=1,
        dataset_kwargs={
            "data_kwargs_list": [
                {
                    "name": "bridge_dataset",
                    "data_dir": "./datasets/debug_dataset",
                    "image_obs_keys": ["image_0"],
                    "state_obs_keys": ["state"],
                },
            ],  # common_kwargs override specific kwargs from data_kwargs_list
            "common_kwargs": dict(
                ram_budget=1,  # limit RAM per dataset
                num_parallel_reads=1,  # for reading from GCS
                num_parallel_calls=1,  # for the less CPU-intensive ops in initial dataset construction
            ),
            "transform_kwargs": dict(
                num_parallel_calls=1,  # for the most CPU-intensive ops (decoding, resizing, augmenting)
            ),
        },
    )
    return config
