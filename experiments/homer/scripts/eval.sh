NAMES=(
    "gc_bridge_20231016_035512"
    "orca_gcbc_rtx_full_v4_2_20231019_002711"
)

STEPS=(
    "250000"
    "780000"
)

VIDEO_DIR="10-20"

TIMESTEPS="50"

CMD="python experiments/homer/bridge/eval.py \
    --num_timesteps $TIMESTEPS \
    --video_save_path /mount/harddrive/homer/videos/$VIDEO_DIR \
    $(for i in "${!NAMES[@]}"; do echo "--checkpoint_weights_path /mount/harddrive/homer/checkpoints/${NAMES[$i]}/checkpoint_${STEPS[$i]} "; done) \
    $(for i in "${!NAMES[@]}"; do echo "--checkpoint_config_path /mount/harddrive/homer/checkpoints/${NAMES[$i]}/config.json "; done) \
    $(for i in "${!NAMES[@]}"; do echo "--checkpoint_metadata_path /mount/harddrive/homer/checkpoints/${NAMES[$i]}/action_proprio_metadata_bridge_dataset.json "; done) \
    --im_size 256 \
    --blocking
"

echo $CMD

$CMD --goal_eep "0.3 0.0 0.15" --initial_eep "0.3 0.0 0.15"
