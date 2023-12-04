PATHS=(
    "gs://rail-dibya-central2/orca_releases/20231130/orca_small_ws2"
)

STEPS=(
    "120000"
)

CONDITIONING_MODE=""
VIDEO_DIR="11-14"

TIMESTEPS="100"

TEMPERATURE="1.0"

HORIZON="1"

PRED_HORIZON="1"

EXEC_HORIZON="1"

CMD="python experiments/homer/bridge/eval.py \
    --num_timesteps $TIMESTEPS \
    --video_save_path /home/youliang/orca_eval/videos/$VIDEO_DIR \
    $(for i in "${!PATHS[@]}"; do echo "--checkpoint_weights_path ${PATHS[$i]} "; done) \
    $(for i in "${!PATHS[@]}"; do echo "--checkpoint_step ${STEPS[$i]} "; done) \
    --im_size 256 \
    --ip 128.32.175.252 \
    --temperature $TEMPERATURE \
    --horizon $HORIZON \
    --pred_horizon $PRED_HORIZON \
    --exec_horizon $EXEC_HORIZON \
    --blocking \
    --modality $CONDITIONING_MODE \
    --checkpoint_cache_dir /home/youliang/orca_eval/checkpoints/ \
    --show_image
"

echo $CMD

$CMD --goal_eep "0.3 0.0 0.15" --initial_eep "0.3 0.0 0.15"
