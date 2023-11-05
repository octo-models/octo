NAMES=(
    "gc_bridge_match_old_20231026_193653"
)

STEPS=(
    "345000"
)

VIDEO_DIR="11-3"

TIMESTEPS="50"

TEMPERATURE="1.0"

HORIZON="1"

PRED_HORIZON="1"

EXEC_HORIZON="1"

CMD="python experiments/homer/bridge/eval.py \
    --num_timesteps $TIMESTEPS \
    --video_save_path /mount/harddrive/homer/videos/$VIDEO_DIR \
    $(for i in "${!NAMES[@]}"; do echo "--checkpoint_weights_path /mount/harddrive/homer/checkpoints/${NAMES[$i]}/checkpoint_${STEPS[$i]} "; done) \
    $(for i in "${!NAMES[@]}"; do echo "--checkpoint_config_path /mount/harddrive/homer/checkpoints/${NAMES[$i]}/config.json "; done) \
    $(for i in "${!NAMES[@]}"; do echo "--checkpoint_metadata_path /mount/harddrive/homer/checkpoints/${NAMES[$i]}/action_proprio_metadata_bridge_dataset.json "; done) \
    $(for i in "${!NAMES[@]}"; do echo "--checkpoint_example_batch_path /mount/harddrive/homer/checkpoints/${NAMES[$i]}/example_batch.msgpack "; done) \
    --im_size 256 \
    --temperature $TEMPERATURE \
    --horizon $HORIZON \
    --pred_horizon $PRED_HORIZON \
    --exec_horizon $EXEC_HORIZON \
    --blocking
"

echo $CMD

$CMD --goal_eep "0.3 0.0 0.15" --initial_eep "0.3 0.0 0.15"
