# If first argument is bridge

case $1 in
    bridge)
        python experiments/main/train.py --config=experiments/main/configs/train_config.py:transformer_bc_bridge --config.batch_size=64 --config.dataset_kwargs.data_dir=gs://rail-octo-central2 --config.eval_interval=100 --config.wandb.group=debug --config.num_steps=1000
    ;;
    r2d2)
        python experiments/main/train.py --config=experiments/main/configs/train_config.py:transformer_bc_r2d2 --config.batch_size=64 --config.dataset_kwargs.data_dir=gs://rail-octo-central2 --config.eval_interval=100 --config.wandb.group=debug --config.num_steps=1000
    ;;
    *)
        echo "Invalid argument"
    ;;
esac
