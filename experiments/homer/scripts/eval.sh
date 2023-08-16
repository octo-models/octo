MODELS=(
    "orca_bridgedata/bc_20230530_082159"
)

CKPTS=(
    "140000"
)

VIDEO_DIR="5-30"

CMD="python experiments/homer/eval_policy.py \
    --num_timesteps 100 \
    --video_save_path ../../trainingdata/homer/videos/$VIDEO_DIR \
    $(for i in "${!MODELS[@]}"; do echo "--checkpoint_path gs://rail-tpus-homer/log/${MODELS[$i]}/checkpoint_${CKPTS[$i]} "; done) \
    $(for i in "${!MODELS[@]}"; do echo "--wandb_run_name widowx-gcrl/${MODELS[$i]} "; done) \
    --blocking \
"

echo $CMD

$CMD --goal_eep "0.3 0.0 0.1" --initial_eep "0.3 0.0 0.1"
