PATHS=(
    # "gs://karl-central-2/orca/aloha_scratch_chunk50_vit_s_updated_20231120_213607"
    #"gs://karl-central-2/orca/aloha_scratch_chunk50_vit_ti_updated_20231120_213331"
    # "gs://karl-central-2/orca/aloha_scratch_chunk50_vit_ti_d01_20231119_000843"
    # "gs://karl-central-2/orca_finetune/aloha_finetune_naiv_20231124_231333"
    # "gs://karl-central-2/orca_finetune/aloha_finetune_frozen_20231124_231411"
    "gs://karl-central-2/orca_finetune/aloha_sim_scratch_vit_s_20231130_080455"
)

STEPS=(
    "50000"
    #"20000"
)

CONDITIONING_MODE=""

TIMESTEPS="400"

TEMPERATURE="1.0"

HORIZON="1"

PRED_HORIZON="50"

EXEC_HORIZON="50"

CMD="python eval.py \
    --num_timesteps $TIMESTEPS \
    --video_save_path gs://karl-central-2/orca_sim_eval/videos \
    $(for i in "${!PATHS[@]}"; do echo "--checkpoint_weights_path ${PATHS[$i]} "; done) \
    $(for i in "${!PATHS[@]}"; do echo "--checkpoint_step ${STEPS[$i]} "; done) \
    --im_size 256 \
    --temperature $TEMPERATURE \
    --horizon $HORIZON \
    --pred_horizon $PRED_HORIZON \
    --exec_horizon $EXEC_HORIZON \
    --modality $CONDITIONING_MODE \
    --checkpoint_cache_dir /tmp/ \
    --is_sim \
    --task_name sim_transfer_cube_scripted
"

echo $CMD

$CMD
