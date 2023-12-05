from config import get_config as get_base_config
from config import update_config, wrap
from functools import partial

from orca.data.utils.data_utils import StateEncoding, ActionEncoding
from experiments.lucy.aloha_wrapper import AlohaGymEnv


def get_config(config_string=None):
    base_config = get_base_config(config_string)
    del base_config["dataset_kwargs"]["oxe_kwargs"]
    config = update_config(
        base_config,
        eval_datasets=None,
        batch_size=128,
        eval_interval=500,
        save_interval=500,
        trajs_for_rollouts=10,
        shuffle_buffer_size=50000,
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
        ),
        rollout_envs=[
            (
                "aloha-sim-cube-v0",
                dict(
                    max_episode_length=200,
                    action_chunk=50,
                    vis_fps=25,
                    video_subsample_rate=2,
                    norm_statistics="gs://rail-orca-central2/aloha_sim_cube_scripted_dataset/1.0.0/dataset_statistics_707801797899cdd91dcb18bd45463cf73ac935bfd6ac6b62456653e96f120a5f.json",
                )
            ),
            (
                "aloha-sim-cube-v0",
                dict(
                    max_episode_length=200,
                    action_chunk=30,
                    vis_fps=25,
                    video_subsample_rate=2,
                    norm_statistics="gs://rail-orca-central2/aloha_sim_cube_scripted_dataset/1.0.0/dataset_statistics_707801797899cdd91dcb18bd45463cf73ac935bfd6ac6b62456653e96f120a5f.json",
                )
            )
        ],
    )

    return config
