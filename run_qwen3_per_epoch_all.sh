#!/bin/bash

# Qwen3 Per-Epoch Evaluation - 所有数据集
# 在每个epoch后进行评估，追踪性能变化

echo "======================================"
echo "Qwen3 Per-Epoch Evaluation"
echo "所有数据集 - 每个epoch后评估"
echo "======================================"
echo ""

START_TIME=$(date +%s)

# ========== DC数据集 (4类) ==========
echo ""
echo "========== [1/5] DC数据集 =========="
echo "恢复预处理数据..."
cp -r preprocessed_data/DC/* temp_dir/

echo "开始训练和逐epoch评估（使用GPU 1）..."
CUDA_VISIBLE_DEVICES=1 python3 run_Qwen3_per_epoch.py \
    --max_len 1024 \
    --batch_size 4 \
    --epochs 5 \
    --num_labels 4 \
    --K_number 4 \
    --TH_value 0.85 \
    --dataset DC

if [ $? -eq 0 ]; then
    echo "✓ DC数据集完成！"
    echo "  结果: ./epoch_results/DC_qwen3_final.json"
    echo "  报告: ./epoch_results/DC_qwen3_summary.txt"
else
    echo "✗ DC数据集失败！"
    exit 1
fi

# ========== USTC数据集 (12类) ==========
echo ""
echo "========== [2/5] USTC数据集 =========="
echo "恢复预处理数据..."
cp -r preprocessed_data/USTC/* temp_dir/

echo "开始训练和逐epoch评估（使用GPU 1）..."
CUDA_VISIBLE_DEVICES=1 python3 run_Qwen3_per_epoch.py \
    --max_len 1024 \
    --batch_size 4 \
    --epochs 5 \
    --num_labels 12 \
    --K_number 5 \
    --TH_value 0.85 \
    --dataset USTC

if [ $? -eq 0 ]; then
    echo "✓ USTC数据集完成！"
    echo "  结果: ./epoch_results/USTC_qwen3_final.json"
    echo "  报告: ./epoch_results/USTC_qwen3_summary.txt"
else
    echo "✗ USTC数据集失败！"
    exit 1
fi

# ========== DF数据集 (60类) ==========
echo ""
echo "========== [3/5] DF数据集 =========="
echo "恢复预处理数据..."
cp -r preprocessed_data/DF/* temp_dir/

echo "开始训练和逐epoch评估（使用GPU 1）..."
CUDA_VISIBLE_DEVICES=1 python3 run_Qwen3_per_epoch.py \
    --max_len 1024 \
    --batch_size 2 \
    --epochs 5 \
    --num_labels 60 \
    --K_number 30 \
    --TH_value 0.85 \
    --dataset DF

if [ $? -eq 0 ]; then
    echo "✓ DF数据集完成！"
    echo "  结果: ./epoch_results/DF_qwen3_final.json"
    echo "  报告: ./epoch_results/DF_qwen3_summary.txt"
else
    echo "✗ DF数据集失败！"
    exit 1
fi

# ========== CSTNet数据集 (75类) ==========
echo ""
echo "========== [4/5] CSTNet数据集 =========="
echo "恢复预处理数据..."
cp -r preprocessed_data/CSTNet/* temp_dir/

echo "开始训练和逐epoch评估（使用GPU 1）..."
CUDA_VISIBLE_DEVICES=1 python3 run_Qwen3_per_epoch.py \
    --max_len 1024 \
    --batch_size 2 \
    --epochs 5 \
    --num_labels 75 \
    --K_number 20 \
    --TH_value 0.85 \
    --dataset CSTNet

if [ $? -eq 0 ]; then
    echo "✓ CSTNet数据集完成！"
    echo "  结果: ./epoch_results/CSTNet_qwen3_final.json"
    echo "  报告: ./epoch_results/CSTNet_qwen3_summary.txt"
else
    echo "✗ CSTNet数据集失败！"
    exit 1
fi

# ========== AWF数据集 (200类) ==========
echo ""
echo "========== [5/5] AWF数据集 =========="
echo "恢复预处理数据..."
cp -r preprocessed_data/AWF/* temp_dir/

echo "开始训练和逐epoch评估（使用GPU 1）..."
CUDA_VISIBLE_DEVICES=1 python3 run_Qwen3_per_epoch.py \
    --max_len 1024 \
    --batch_size 2 \
    --epochs 5 \
    --num_labels 200 \
    --K_number 50 \
    --TH_value 0.85 \
    --dataset AWF

if [ $? -eq 0 ]; then
    echo "✓ AWF数据集完成！"
    echo "  结果: ./epoch_results/AWF_qwen3_final.json"
    echo "  报告: ./epoch_results/AWF_qwen3_summary.txt"
else
    echo "✗ AWF数据集失败！"
    exit 1
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "======================================"
echo "全部训练和评估完成！"
echo "======================================"
echo "总用时: ${HOURS}小时 ${MINUTES}分钟"
echo ""
echo "所有结果文件："
echo "  JSON结果:"
echo "    - ./epoch_results/DC_qwen3_final.json"
echo "    - ./epoch_results/USTC_qwen3_final.json"
echo "    - ./epoch_results/DF_qwen3_final.json"
echo "    - ./epoch_results/CSTNet_qwen3_final.json"
echo "    - ./epoch_results/AWF_qwen3_final.json"
echo ""
echo "  文本报告:"
echo "    - ./epoch_results/DC_qwen3_summary.txt"
echo "    - ./epoch_results/USTC_qwen3_summary.txt"
echo "    - ./epoch_results/DF_qwen3_summary.txt"
echo "    - ./epoch_results/CSTNet_qwen3_summary.txt"
echo "    - ./epoch_results/AWF_qwen3_summary.txt"
echo ""
echo "  所有epoch的模型检查点在: ./trained_models/"

