#!/bin/bash

# 使用方法: bash run_lora.sh <数据集名称>
# 例如: bash run_lora.sh DC

DATASET=$1

if [ -z "$DATASET" ]; then
    echo "Usage: bash run_lora.sh <DATASET_NAME>"
    exit 1
fi

# 默认设置，可根据显存调整
# 如果显存不够 (如 < 24GB)，添加 --use_4bit 参数
BATCH_SIZE=2
GRAD_ACCUM=1
MAX_LEN=1024

# 根据数据集设置参数
case $DATASET in
    DC)
        NUM_LABELS=4
        K_NUMBER=4
        TH_VALUE=0.85
        ;;
    USTC)
        NUM_LABELS=12
        K_NUMBER=5
        TH_VALUE=0.85
        ;;
    DF)
        NUM_LABELS=60
        K_NUMBER=30
        TH_VALUE=0.85
        ;;
    CSTNet)
        NUM_LABELS=75
        K_NUMBER=20
        TH_VALUE=0.85
        ;;
    AWF)
        NUM_LABELS=200
        K_NUMBER=50
        TH_VALUE=0.85
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        exit 1
        ;;
esac

echo "Running LoRA training for $DATASET..."
echo "Labels: $NUM_LABELS, K: $K_NUMBER, Threshold: $TH_VALUE"

# 1. 准备数据 (假设预处理数据在 preprocessed_data 文件夹)
if [ ! -d "preprocessed_data/$DATASET" ]; then
    echo "Error: Preprocessed data not found in preprocessed_data/$DATASET"
    exit 1
fi

mkdir -p temp_dir
cp -r preprocessed_data/$DATASET/* temp_dir/
echo "Data loaded to temp_dir"

# 2. 运行 Python 脚本
# 如果显存不足，请在最后添加 --use_4bit
python train_lora.py \
    --dataset $DATASET \
    --num_labels $NUM_LABELS \
    --K_number $K_NUMBER \
    --TH_value $TH_VALUE \
    --batch_size $BATCH_SIZE \
    --grad_accum_steps $GRAD_ACCUM \
    --max_len $MAX_LEN \
    --epochs 3 \
    --lr 1e-4 \
    --use_4bit

# 3. 清理
# rm -rf temp_dir