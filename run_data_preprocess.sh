#!/bin/bash

# model可以写成LLaMA

# 创建备份目录
mkdir -p preprocessed_data

# DF数据集
echo "预处理DF数据集..."
python3 data_preprocess.py --data_path ./data --dataset DF --model GPT2
cp -r temp_dir preprocessed_data/DF
echo "DF数据已备份到 preprocessed_data/DF"

# AWF数据集
echo "预处理AWF数据集..."
python3 data_preprocess.py --data_path ./data --dataset AWF --model GPT2
cp -r temp_dir preprocessed_data/AWF
echo "AWF数据已备份到 preprocessed_data/AWF"

# DC数据集
echo "预处理DC数据集..."
python3 data_preprocess.py --data_path ./data --dataset DC --model GPT2
cp -r temp_dir preprocessed_data/DC
echo "DC数据已备份到 preprocessed_data/DC"

# USTC数据集
echo "预处理USTC数据集..."
python3 data_preprocess.py --data_path ./data --dataset USTC --model GPT2
cp -r temp_dir preprocessed_data/USTC
echo "USTC数据已备份到 preprocessed_data/USTC"

# CSTNet数据集
echo "预处理CSTNet数据集..."
python3 data_preprocess.py --data_path ./data --dataset CSTNet --model GPT2
cp -r temp_dir preprocessed_data/CSTNet
echo "CSTNet数据已备份到 preprocessed_data/CSTNet"

echo "所有数据预处理完成并已备份！"