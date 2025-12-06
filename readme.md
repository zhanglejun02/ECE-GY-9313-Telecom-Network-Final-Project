## Final Project


In this repository, we guide you to run this projec. 
A complete run including model downloading, trian and inference, which is time-consuming.
I have provided the train & inference log file for your convenience to check.
[logfile](https://drive.google.com/drive/folders/1Br5Vr6u-cjVKA_K6H1nfySBPS2cL5QnP?usp=drive_link)

**Datasets:** [AWF](https://arxiv.org/abs/1708.06376), [DF](https://arxiv.org/abs/1801.02265), [DC](https://www.semanticscholar.org/paper/Deep-Content%3A-Unveiling-Video-Streaming-Content-Li-Huang/f9feff95bc1d68674d5db426053f417bd2c8786b), [USTC](https://drive.google.com/file/d/1F09zxln9iFg2HWoqc6m4LKFhYK7cDQv_/view), [CSTNet-tls](https://drive.google.com/drive/folders/1JSsYmevkxQFanoKOi_i1ooA6pH3s9sDr)

**Openset methods**
- [K-LND methods](https://github.com/ThiliniDahanayaka/Open-Set-Traffic-Classification)

# Steps

First, clone the git repo and install the requirements.
```
git clone https://github.com/zhanglejun02/ECE-GY-9313-Telecom-Network-Final-Project.git
pip install -r requirements.txt
```
Next, download the dataset and place it in the data directory.
```
gdown https://drive.google.com/uc?id=1-MVfxyHdQeUguBmYrIIw1jhMVSqxXQgO
unzip data.zip 
```

For GPT2 and LLaMA, do data preprocess
Here, the dataset name should be DF, AWF, DC, USTC, or CSTNet, and the model_name should be either GPT2 or LLaMA.
```
python3 data_preprocess.py --data_path ./data --dataset <dataset_name> --model <model_name>
```

### GPT-2 Fine-tunning

To fine-tune the model, run the suitable code for the dataset:
```
python3 train.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 60  --dataset DF
python3 train.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 200  --dataset AWF
python3 train.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 4  --dataset DC
python3 train.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 12  --dataset USTC
python3 train.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 75  --dataset CSTNet
```
To evaluate, run the suitable code for the dataset:
```
python3 evaluate.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 60 --K_number 30 --TH_value 0.8 --dataset DF
python3 evaluate.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 200 --K_number 50 --TH_value 0.9 --dataset AWF
python3 evaluate.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 4 --K_number 4 --TH_value 0.9 --dataset DC
python3 evaluate.py --max_len 1024 --batch_size 12 --epochs 3 --num_labels 12 --K_number 5 --TH_value 0.8 --dataset USTC
python3 evaluate.py --max_len 1024 --batch_size 12 --epochs 5 --num_labels 75 --K_number 20 --TH_value 0.8 --dataset CSTNe
```

### LLaMA Fine-tuning

To fine-tune the LLaMA model and obtain results, run the following commands accordingly.
```
python3 run_LLaMA.py --max_len 1024 --batch_size 6 --epochs 3 --num_labels 60  --dataset DF --K_number 30 --TH_value 0.85
python3 run_LLaMA.py --max_len 1024 --batch_size 6 --epochs 3 --num_labels 200  --dataset AWF --K_number 50 --TH_value 0.85
python3 run_LLaMA.py --max_len 1024 --batch_size 8 --epochs 2 --num_labels 4  --dataset DC --K_number 4 --TH_value 0.85
python3 run_LLaMA.py --max_len 1024 --batch_size 8 --epochs 2 --num_labels 12  --dataset USTC --K_number 5 --TH_value 0.85
python3 run_LLaMA.py --max_len 1024 --batch_size 6 --epochs 2 --num_labels 75  --dataset CSTNet --K_number 20 --TH_value 0.85
python3 run_LLaMA.py --max_len 1024 --batch_size 6 --epochs 2 --num_labels 60  --dataset IoT --K_number 30 --TH_value 0.85
python3 run_LLaMA.py --max_len 1024 --batch_size 8 --epochs 2 --num_labels 10  --dataset ISCX --K_number 5 --TH_value 0.9
```

### Qwen3 Fine-tunning

do data preprocess
```
bash run_data_preprocess.sh
```
download [Qwen3-30B-A3B-Instruct-2507] (https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)

change the QWEN3_MODEL_PATH to your local path in file train_lora.py

Then 
```
bash run_lora.sh <dataset_name>
```
### Mistral Fine-tunning

download [Mistral-Nemo-Instruct-2407] (https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)

change the MODEL_PATH to your local path in file train_mistral_lora.py

change file name to train_mistral_lora.py in run_lora.sh file

Then
```
bash run_lora.sh <dataset_name>
```
### Gemma Fine-tunning

download [Gemma-7b] (https://huggingface.co/google/gemma-7b)

change the QWEN3_MODEL_PATH to your local Gemma Model path in file train_lora.py

Then
```
bash run_lora.sh <dataset_name>
```


