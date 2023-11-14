PATHS=(
    "gs://rail-dibya-central2/experiment_output/oxe_sweep/bridge_vits_20231111_165439"
    "gs://rail-dibya-central2/experiment_output/oxe_sweep/bridge_baseline_20231112_025236"
    "gs://rail-dibya-central2/experiment_output/oxe_sweep/bridge_jaxrlm_baseline_20231112_073307"
)

STEPS=(
    "120000"
    "500000"
    "300000"
)

CONDITIONING_MODE="goal"
VIDEO_DIR="11-12"

TIMESTEPS="50"

TEMPERATURE="1.0"

HORIZON="1"

PRED_HORIZON="1"

EXEC_HORIZON="1"

CMD="python experiments/homer/bridge/eval.py \
    --num_timesteps $TIMESTEPS \
    --video_save_path /mount/harddrive/homer/videos/$VIDEO_DIR \
    $(for i in "${!NAMES[@]}"; do echo "--checkpoint_weights_path ${NAMES[$i]} "; done) \
    $(for i in "${!NAMES[@]}"; do echo "--checkpoint_step ${STEPS[$i]} "; done) \
    --im_size 256 \
    --temperature $TEMPERATURE \
    --horizon $HORIZON \
    --pred_horizon $PRED_HORIZON \
    --exec_horizon $EXEC_HORIZON \
    --blocking \
    --modality $CONDITIONING_MODE \
    --checkpoint_cache_dir /mount/harddrive/homer/checkpoints/
"

echo $CMD

$CMD --goal_eep "0.3 0.0 0.15" --initial_eep "0.3 0.0 0.15"
