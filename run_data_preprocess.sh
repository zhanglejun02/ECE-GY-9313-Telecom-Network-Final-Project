#!/bin/bash

# model can be LLaMA



mkdir -p preprocessed_data

# DF
echo "DF processing..."
python3 data_preprocess.py --data_path ./data --dataset DF --model GPT2
cp -r temp_dir preprocessed_data/DF
echo "DF has been backed up to preprocessed_data/DF"

# AWF
echo "AWF processing..."
python3 data_preprocess.py --data_path ./data --dataset AWF --model GPT2
cp -r temp_dir preprocessed_data/AWF
echo "AWF has been backed up to preprocessed_data/AWF"

# DC
echo "DC processing..."
python3 data_preprocess.py --data_path ./data --dataset DC --model GPT2
cp -r temp_dir preprocessed_data/DC
echo "DC has been backed up to preprocessed_data/DC"

# USTC
echo "USTC processing..."
python3 data_preprocess.py --data_path ./data --dataset USTC --model GPT2
cp -r temp_dir preprocessed_data/USTC
echo "USTC has been backed up to preprocessed_data/USTC"

# CSTNet
echo "CSTNet processing..."
python3 data_preprocess.py --data_path ./data --dataset CSTNet --model GPT2
cp -r temp_dir preprocessed_data/CSTNet
echo "CSTNet has been backed up to preprocessed_data/CSTNet"

echo "Done"