MODELS=(
    "orca/bridge_goal_20230821_042544"
)

CKPTS=(
    "100000"
)

VIDEO_DIR="8-22"

CMD="python experiments/homer/bridge/eval.py \
    --num_timesteps 100 \
    --video_save_path ../../trainingdata/homer/videos/$VIDEO_DIR \
    $(for i in "${!MODELS[@]}"; do echo "--checkpoint_path gs://rail-tpus-homer/log/${MODELS[$i]}/checkpoint_${CKPTS[$i]} "; done) \
    $(for i in "${!MODELS[@]}"; do echo "--wandb_run_name widowx-gcrl/${MODELS[$i]} "; done) \
    --im_size 256 \
"

echo $CMD

$CMD --goal_eep "0.3 0.0 0.15" --initial_eep "0.3 0.0 0.15" --obs_horizon 2 --act_pred_horizon 1
