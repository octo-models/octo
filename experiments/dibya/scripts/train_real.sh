#!/bin/bash

CONFIG_NAME=${1:-transformer_bc}
DATA_CONFIG_NAME=${2:-all}
echo "Using config $CONFIG_NAME and data config $DATA_CONFIG_NAME"
NAME="test"

CMD="python experiments/main/train.py \
    --config experiments/main/configs/train_config.py:$CONFIG_NAME \
    --bridgedata_config experiments/main/configs/data_config.py:$DATA_CONFIG_NAME \
    --name $NAME \
    --config.data_path=gs://rail-tpus-homer-v4/data_new"
shift 2
echo $CMD "$@"
$CMD "$@"
