MODELS=(
    "transformer_bc_bridge_20230926_060435"
)

CKPTS=(
    "400000"
)

VIDEO_DIR="9-26"

CMD="python experiments/homer/bridge/eval.py \
    --num_timesteps 80 \
    --video_save_path ../../trainingdata/homer/videos/$VIDEO_DIR \
    $(for i in "${!MODELS[@]}"; do echo "--checkpoint_weights_path checkpoints/${MODELS[$i]}/checkpoint_${CKPTS[$i]} "; done) \
    $(for i in "${!MODELS[@]}"; do echo "--checkpoint_config_path checkpoints/${MODELS[$i]}/config.json "; done) \
    $(for i in "${!MODELS[@]}"; do echo "--checkpoint_metadata_path checkpoints/${MODELS[$i]}/action_proprio_metadata_bridge_dataset.json "; done) \
    --im_size 256 \
    --blocking
"

echo $CMD

$CMD --goal_eep "0.3 0.0 0.15" --initial_eep "0.3 0.0 0.15"
