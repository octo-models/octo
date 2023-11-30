from config import get_config as get_base_config
from config import update_config, wrap

from orca.data.utils.data_utils import StateEncoding, ActionEncoding


def get_config(config_string=None):
    base_config = get_base_config(config_string)
    del base_config["dataset_kwargs"]["oxe_kwargs"]
    config = update_config(
        base_config,
        eval_datasets=None,
        batch_size=128,
        eval_interval=500,
        save_interval=500,
        model={
            "observation_tokenizers": {
                "image": {
                    "cls_name": "image_tokenizer",
                    "kwargs": dict(
                        num_tokens=256,
                        obs_stack_keys=["image_.*"],
                        task_stack_keys=[],
                        task_film_keys=[],
                        encoder="small-stem-16",
                        encoder_kwargs=dict(use_film=False),
                    ),
                },
                "proprio": {
                    "cls_name": "lowdim_obs_tokenizer",
                    "kwargs": dict(
                        n_bins=256,
                        bin_type=base_config["dataset_kwargs"]["common_kwargs"][
                            "action_proprio_normalization_type"],
                        low=-2.,
                        high=2.,
                        obs_keys=["proprio"],
                    ),
                },
            },
            "task_tokenizers": {},
            "heads": dict(
                action=dict(
                    cls_name="mse_action_head",
                    kwargs=dict(
                        pred_horizon=50,
                        action_dim=14,
                        readout_key="obs",
                    ),
                )
            ),
        },
        text_processor=None,
        text_processor_kwargs=dict(),
        dataset_kwargs=dict(
            data_kwargs_list=[
                dict(
                    name="aloha_sim_cube_scripted_dataset",
                    data_dir="gs://rail-orca-central2",
                    image_obs_keys=["top"],
                    state_obs_keys=["state"],
                    state_encoding=StateEncoding.JOINT_BIMANUAL,
                    action_encoding=ActionEncoding.JOINT_POS_BIMANUAL,
                )
            ],
            transform_kwargs=dict(
                window_size=1,
                additional_action_window_size=49,
                action_encoding=ActionEncoding.JOINT_POS_BIMANUAL,
            )
        )
    )

    return config
