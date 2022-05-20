#!/bin/bash

if [[ $# -ne 4 ]]; then
    echo "Usage: bash scripts/run_standalone_train_gpu.sh [DEVICE_ID] [LOGS_CKPT_DIR] [DATA_ROOT] [BACKBONE_PATH]"
exit 1
fi

export CUDA_VISIBLE_DEVICES=$1

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        realpath -m "$PWD/$1"
    fi
}

LOGS_DIR=$(get_real_path "$2")
DATASET_ROOT=$(get_real_path "$3")
CKPT_URL=$(get_real_path "$4")

if [ !  -d "$LOGS_DIR" ]; then
  mkdir "$LOGS_DIR"
  mkdir "$LOGS_DIR/training_configs"
fi

cp ./*.py "$LOGS_DIR"/training_configs
cp ./cfg/*.yaml "$LOGS_DIR"/training_configs
cp -r src "$LOGS_DIR"/training_configs

python train.py \
    --logs_dir="$LOGS_DIR" \
    --dataset_root="$DATASET_ROOT" \
    --ckpt_url="$CKPT_URL" \
    > "$LOGS_DIR"/standalone_train.log 2>&1 &
