PATHS=(
   # "gs://rail-dibya-central2/experiment_output/oxe_sweep/bridge_notapod_vits_20231115_144527" # bridge only base
   # "gs://rail-dibya-central2/experiment_output/oxe_sweep/bridge_notapod_vits_20231115_144527" # bridge only base
   # "gs://rail-dibya-central2/experiment_output/oxe_sweep/bridge_notapod_vits_20231115_062209" # base
   # "gs://rail-tpus-homer-v4/log/octo/widowx_cleaver_from_scratch_vit_ti_20231124_210045" # from scratch
   # "gs://rail-tpus-homer-v4/log/octo_finetune/finetune_widowx_cleaver_20231122_040252" # end-to-end finetuning
   # "gs://rail-tpus-homer-v4/log/octo_finetune/finetune_widowx_cleaver_just_head_20231129_184032"
   # "gs://rail-tpus-homer-v4/log/octo_finetune/finetune_widowx_cleaver_just_head_20231129_184032"
   # "gs://rail-tpus-homer-v4/log/octo_finetune/finetune_widowx_cleaver_just_head_20231129_184032"
   # "gs://rail-dibya-central2/octo_releases/20231130/octo_small_ws2"
   # "gs://rail-tpus-homer-v4/log/octo_finetune/finetune_full_easy_20231202_003411"
   # "gs://rail-tpus-homer-v4/log/octo_finetune/finetune_head_only_easy_20231201_233637"
   # "gs://rail-tpus-homer-v4/log/octo_finetune/finetune_head_only_easy_20231201_225553"
   # "gs://rail-tpus-homer-v4/log/octo_finetune/finetune_full_easy_20231201_232840"
   # "gs://rail-tpus-homer/log/octo/from_scratch_easy_20231202_035635"
   # "gs://rail-dibya-central2/experiment_output/main_runs/oxe_magic_soup_vitb_20231203_021440"
   # "gs://rail-dibya-central2/experiment_output/main_runs/oxe_magic_soup_vitb_20231203_021440"
   # "gs://rail-tpus-homer-v4/log/octo_finetune/finetune_full_vitb_20231203_231147"
   # "gs://rail-tpus-homer-v4/log/octo_finetune/finetune_full_vitb_best_bridge_step_20231204_005209"
   # "gs://rail-tpus-homer-v4/log/octo_finetune/finetune_full_vitb_best_bridge_step_no_aug_20231204_010743"
   # "gs://rail-tpus-homer/log/octo/from_scratch_vit_ti_easy_no_aug_20231205_215607"
   # "gs://rail-tpus-homer/log/octo/from_scratch_vit_ti_easy_20231205_195257"
   # "gs://rail-dibya-central2/experiment_output/main_runs/oxe_magic_soup_vitb_nocrop_20231205_022211"
   "gs://rail-dibya-central2/experiment_output/main_runs/oxe_magic_soup_vitb_nocrop_20231205_022211"
)

STEPS=(
   #  "55000"
   #  "120000"
   # "200000"
   # "1900"
   # "140"
   # "700"
   # "800"
   # "8000"
   # "235000"
   # "200"
   # "500"
   # "1400"
   # "100"
   # "4200"
   # "300000"
   # "140000"
   # "80"
   # "100"
   # "100"
   # "4400"
   # "2200"
   # "70000"
   "120000"
)

CONDITIONING_MODE="g"
VIDEO_DIR="12-6"

TIMESTEPS="60"

TEMPERATURE="1.0"

HORIZON="2"

PRED_HORIZON="1"

EXEC_HORIZON="1"

CMD="python examples/widowx_eval/eval.py \
    --num_timesteps $TIMESTEPS \
    --video_save_path /mount/harddrive/homer/videos/$VIDEO_DIR \
    $(for i in "${!PATHS[@]}"; do echo "--checkpoint_weights_path ${PATHS[$i]} "; done) \
    $(for i in "${!PATHS[@]}"; do echo "--checkpoint_step ${STEPS[$i]} "; done) \
    --im_size 256 \
    --temperature $TEMPERATURE \
    --horizon $HORIZON \
    --pred_horizon $PRED_HORIZON \
    --exec_horizon $EXEC_HORIZON \
    --blocking \
    --modality $CONDITIONING_MODE \
    --checkpoint_cache_dir /mount/harddrive/homer/checkpoints/ \
"

echo $CMD

$CMD --goal_eep "0.3 0.0 0.15" --initial_eep "0.3 0.0 0.15"
