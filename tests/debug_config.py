from copy import deepcopy
import imp
import os

from ml_collections import ConfigDict

get_base_config = imp.load_source(
    "config", os.path.join(os.path.dirname(__file__), "../scripts/configs/config.py")
).get_config


def update_config(config: ConfigDict, **kwargs):
    assert isinstance(config, ConfigDict)
    updates = ConfigDict(kwargs)
    new_config = deepcopy(config)
    new_config.update(updates)
    return new_config


def get_config():
    base_config = get_base_config("dummy")
    del base_config["dataset_kwargs"]["oxe_kwargs"]
    config = update_config(
        base_config,
        num_steps=2,
        optimizer=dict(
            learning_rate=dict(
                warmup_steps=1,
            ),
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=1,
            num_val_batches=2,
        ),
        viz_kwargs=dict(
            eval_batch_size=2,
            trajs_for_metrics=4,
            trajs_for_viz=4,
            samples_per_state=4,
        ),
        log_interval=1,
        eval_interval=2,
        viz_interval=2,
        save_interval=2,
        eval_datasets=None,
        dataset_kwargs={
            "dataset_kwargs_list": [
                {
                    "name": "bridge_dataset",
                    "data_dir": "./tests/debug_dataset",
                    "image_obs_keys": {"primary": "image_0"},
                    "state_obs_keys": ["state"],
                    "language_key": "language_instruction",
                },
            ],
            "frame_transform_kwargs": {
                "resize_size": (128, 128),
                "num_parallel_calls": 4,
            },
            "traj_transform_threads": 1,  # shared between all datasets
            "traj_read_threads": 1,  # shared between all datasets
            "batch_size": 64,
            "sample_weights": None,
            "shuffle_buffer_size": 1000,
            "balance_weights": True,
        },
    )
    return config
