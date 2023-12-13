from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from octo.data.utils.text_processing import MuseEmbedding
from octo.model.components.action_heads import MSEActionHead
from octo.model.components.tokenizers import ImageTokenizer
from octo.model.components.transformer import common_transformer_sizes
from octo.model.components.vit_encoders import SmallStem16
from octo.utils.spec import ModuleSpec


def get_model_config(transformer_size):
    """
    Transformer_size is one of ["dummy", "vanilla", "vit_s", "vit_b", "vit_l", "vit_h"]

    This model stacks all the images from different cameras together, and passes it through
    a small convolutional stem before entering the transformer.

    The action head pools all the observation token embeddings, and passes it through a small MLP
    before predicting the action using a MSE loss.
    """
    token_embedding_size, transformer_kwargs = common_transformer_sizes(
        transformer_size
    )
    return dict(
        observation_tokenizers=dict(
            image=ModuleSpec.create(
                ImageTokenizer,
                num_tokens=256,
                obs_stack_keys=["image_.*"],
                task_stack_keys=["image_.*"],
                task_film_keys=["language_instruction"],
                encoder=ModuleSpec.create(SmallStem16, use_film=True),
            ),
        ),
        task_tokenizers=dict(),
        heads=dict(
            action=ModuleSpec.create(
                MSEActionHead,
                pred_horizon=1,
                action_dim=7,
                readout_key="obs",
            ),
        ),
        readouts=dict(),
        token_embedding_size=token_embedding_size,
        transformer_kwargs=transformer_kwargs,
        max_horizon=10,
    )


def get_config(
    transformer_size="vit_s",
):
    print("Creating config with: ", locals())
    num_steps = FieldReference(default=int(2e6))
    window_size = FieldReference(default=1)
    return ConfigDict(
        dict(
            seed=42,
            num_steps=num_steps,
            save_dir=placeholder(str),
            model=get_model_config(transformer_size),
            window_size=window_size,
            dataset_kwargs=get_dataset_config(window_size),
            optimizer=dict(
                learning_rate=dict(
                    name="rsqrt",
                    init_value=0.0,
                    peak_value=3e-4,
                    warmup_steps=2000,
                    timescale=10000,
                ),
                weight_decay=0.1,
                clip_gradient=1.0,
                frozen_keys=tuple(),
            ),
            prefetch_num_batches=0,
            start_step=placeholder(int),
            log_interval=100,
            eval_interval=5000,
            viz_interval=20000,
            save_interval=10000,
            val_kwargs=dict(
                val_shuffle_buffer_size=1000,
                num_val_batches=16,
            ),
            viz_kwargs=dict(
                eval_batch_size=128,
                trajs_for_metrics=100,
                trajs_for_viz=8,
                samples_per_state=8,
            ),
            resume_path=placeholder(str),
            text_processor=ModuleSpec.create(MuseEmbedding),
            pretrained_loaders=tuple(),
            wandb=dict(
                project="octo",
                group=placeholder(str),
                entity=placeholder(str),
            ),
            wandb_resume_id=placeholder(str),
            eval_datasets=(
                "bridge_dataset",
                "taco_play",
                "berkeley_cable_routing",
                "berkeley_autolab_ur5",
            ),
        )
    )


def get_dataset_config(window_size=1):
    task_augmentation = dict(
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=0.5,
        ),
    )

    return {
        # oxe_kwargs will generate dataset_kwargs_list and sampling weights
        "oxe_kwargs": dict(
            data_mix=placeholder(str),
            data_dir=placeholder(str),
            load_camera_views=("primary", "wrist"),
            load_depth=False,
        ),
        "traj_transform_kwargs": dict(
            window_size=window_size,
            future_action_window_size=0,
            goal_relabeling_strategy="uniform",
            subsample_length=100,
            **task_augmentation,
        ),
        "frame_transform_kwargs": dict(
            resize_size=(256, 256),
            image_augment_kwargs=dict(
                random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                random_brightness=[0.2],
                random_contrast=[0.8, 1.2],
                random_saturation=[0.8, 1.2],
                random_hue=[0.1],
                augment_order=[
                    "random_resized_crop",
                    "random_brightness",
                    "random_contrast",
                    "random_saturation",
                    "random_hue",
                ],
            ),
            num_parallel_calls=200,
        ),
        "traj_transform_threads": 48,  # shared between all datasets
        "traj_read_threads": 48,  # shared between all datasets
        "shuffle_buffer_size": 100000,  # shared between all datasets
        "batch_size": 1024,
        "balance_weights": True,
    }
