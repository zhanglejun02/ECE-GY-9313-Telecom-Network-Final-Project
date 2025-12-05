

import argparse
import numpy as np 
import pandas as pd
from tqdm.auto import tqdm
import os
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer,
                          get_linear_schedule_with_warmup,
                          BitsAndBytesConfig)
from peft import get_peft_model, LoraConfig, TaskType


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# MODEL_PATH = "/nlp/common_models/zai-org/glm-4-9b" 
# MODEL_PATH = "/nlp/common_models/Qwen/Qwen3-4B-Instruct-2507"
MODEL_PATH = "/maindata/data/shared/ai_story_workspace-dsw/common_models/mistralai/Mistral-Nemo-Instruct-2407"

class DatasetCreator(Dataset):
    def __init__(self, processed_data, train):
        self.data = processed_data
        self.train = train
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        line = self.data.iloc[index]
        if self.train:
            return {'text': line['trace'], 'label': int(line['target'])}
        else:
            return {'text': line['trace'], 'label': 0}

class GLM4_collator(object):
    def __init__(self, tokenizer, max_seq_len=None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    
    def __call__(self, sequences):
        texts = [sequence['text'] for sequence in sequences]
        labels = [int(sequence['label']) for sequence in sequences]
        inputs = self.tokenizer(text=texts,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=self.max_seq_len)
        inputs.update({'labels': torch.tensor(labels)})       
        return inputs

def pre_process(dataset):
    if 'text' not in dataset.columns and 'trace' in dataset.columns:
        dataset['text'] = dataset['trace']
    elif 'trace' not in dataset.columns and 'text' in dataset.columns:
        dataset['trace'] = dataset['text']
    return dataset

def get_labels(file):
    df = pd.read_csv(file)
    return np.array(df['target'])

# ================= K-LND Math Logic =================
def calculate_mean_vectors(NB_CLASSES, model_predictions, y_train):
    Means = {}
    count = [0] * NB_CLASSES
    txt_O = "Mean_{Class1:.0f}"
    for i in range(NB_CLASSES):
        Means[txt_O.format(Class1=i)] = np.array([0.0] * NB_CLASSES)

    for i in range(len(model_predictions)):
        pred_label = np.argmax(model_predictions[i])
        if pred_label == y_train[i]:
            Means[txt_O.format(Class1=y_train[i])] += model_predictions[i]
            count[y_train[i]] += 1

    Mean_Vectors = []
    for i in range(NB_CLASSES):
        if count[i] > 0:
            Means[txt_O.format(Class1=i)] /= count[i]
        Mean_Vectors.append(Means[txt_O.format(Class1=i)])
    return np.array(Mean_Vectors)

def calculate_thresholds(NB_CLASSES, model_predictions, y_valid, Mean_Vectors, K_number, TH_value):
    txt_1 = "Dist_{Class1:.0f}"
    Indexes = [[] for _ in range(NB_CLASSES)]
    Values = {i: [0] * NB_CLASSES for i in range(NB_CLASSES)}

    for i in range(len(model_predictions)):
        if y_valid[i] == np.argmax(model_predictions[i]):
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]] - model_predictions[i])
            for k in range(NB_CLASSES):
                if k != int(y_valid[i]):
                    Values[y_valid[i]][k] += np.linalg.norm(Mean_Vectors[k] - model_predictions[i]) - dist

    for i in range(NB_CLASSES):
        for l in range(min(K_number, NB_CLASSES - 1)):
            Min = min(Values[i])
            idx = Values[i].index(Min)
            Indexes[i].append(idx)
            Values[i][idx] = 1e9
    Indexes = np.array(Indexes)

    # Threshold 1
    Distances = {txt_1.format(Class1=i): [] for i in range(NB_CLASSES)}
    for i in range(len(model_predictions)):
        if y_valid[i] == np.argmax(model_predictions[i]):
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]] - model_predictions[i])
            Distances[txt_1.format(Class1=y_valid[i])].append(dist)
    TH1 = [0] * NB_CLASSES
    for j in range(NB_CLASSES):
        Dist = sorted(Distances[txt_1.format(Class1=j)])
        if len(Dist) > 0:
            TH1[j] = Dist[int(len(Dist) * TH_value)] if int(len(Dist) * TH_value) < len(Dist) else Dist[-1]
        else:
            TH1[j] = 10 if j == 0 else TH1[j-1]
    
    # Threshold 2
    Distances = {txt_1.format(Class1=i): [] for i in range(NB_CLASSES)}
    for i in range(len(model_predictions)):
        if y_valid[i] == np.argmax(model_predictions[i]):
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]] - model_predictions[i])
            Tot = 0
            for k in range(NB_CLASSES):
                if k != int(y_valid[i]) and k in Indexes[y_valid[i]]:
                    Tot += (np.linalg.norm(Mean_Vectors[k] - model_predictions[i]) - dist)
            Distances[txt_1.format(Class1=y_valid[i])].append(Tot)
    TH2 = [0] * NB_CLASSES
    for j in range(NB_CLASSES):
        Dist = sorted(Distances[txt_1.format(Class1=j)])
        if len(Dist) > 0:
            TH2[j] = Dist[int(len(Dist) * (1 - TH_value))] if int(len(Dist) * (1 - TH_value)) < len(Dist) else Dist[-1]
        else:
            TH2[j] = 10 if j == 0 else TH2[j-1]

    # Threshold 3
    Distances = {txt_1.format(Class1=i): [] for i in range(NB_CLASSES)}
    for i in range(len(model_predictions)):
        if y_valid[i] == np.argmax(model_predictions[i]):
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]] - model_predictions[i])
            Tot = 0
            for k in range(NB_CLASSES):
                if k != int(y_valid[i]) and k in Indexes[y_valid[i]]:
                    Tot += np.linalg.norm(Mean_Vectors[k] - model_predictions[i])
            if Tot > 0: Tot = dist / Tot
            Distances[txt_1.format(Class1=y_valid[i])].append(Tot)
    TH3 = [0] * NB_CLASSES
    for j in range(NB_CLASSES):
        Dist = sorted(Distances[txt_1.format(Class1=j)])
        if len(Dist) > 0:
            TH3[j] = Dist[int(len(Dist) * TH_value)] if int(len(Dist) * TH_value) < len(Dist) else Dist[-1]
        else:
            TH3[j] = 10 if j == 0 else TH3[j-1]

    return np.array(TH1), np.array(TH2), np.array(TH3), Indexes

def evaluate_klnd(NB_CLASSES, model_predictions_test, model_predictions_open, y_test, y_open, 
                  Mean_vectors, Indexes, Threasholds_1, Threasholds_2, Threasholds_3):
    results = {}
    
    # Method 1
    p_close_1 = []
    for i in range(len(model_predictions_test)):
        d = np.argmax(model_predictions_test[i])
        if np.linalg.norm(model_predictions_test[i] - Mean_vectors[d]) > Threasholds_1[d]:
            p_close_1.append(NB_CLASSES)
        else:
            p_close_1.append(d)
    p_open_1 = []
    for i in range(len(model_predictions_open)):
        d = np.argmax(model_predictions_open[i])
        if np.linalg.norm(model_predictions_open[i] - Mean_vectors[d]) > Threasholds_1[d]:
            p_open_1.append(NB_CLASSES)
        else:
            p_open_1.append(d)

    # Method 2
    p_close_2 = []
    for i in range(len(model_predictions_test)):
        d = np.argmax(model_predictions_test[i])
        dist = np.linalg.norm(Mean_vectors[d] - model_predictions_test[i])
        Tot = 0
        for k in range(NB_CLASSES):
            if k != d: Tot += np.linalg.norm(Mean_vectors[k] - model_predictions_test[i]) - dist
        if Tot < Threasholds_2[d]: p_close_2.append(NB_CLASSES)
        else: p_close_2.append(d)
    p_open_2 = []
    for i in range(len(model_predictions_open)):
        d = np.argmax(model_predictions_open[i])
        dist = np.linalg.norm(Mean_vectors[d] - model_predictions_open[i])
        Tot = 0
        for k in range(NB_CLASSES):
            if k != int(d) and k in Indexes[d]: Tot += np.linalg.norm(Mean_vectors[k] - model_predictions_open[i]) - dist
        if Tot < Threasholds_2[d]: p_open_2.append(NB_CLASSES)
        else: p_open_2.append(d)

    # Method 3
    p_close_3 = []
    for i in range(len(model_predictions_test)):
        d = np.argmax(model_predictions_test[i])
        dist = np.linalg.norm(Mean_vectors[d] - model_predictions_test[i])
        Tot = 0
        for k in range(NB_CLASSES):
            if k != d: Tot += np.linalg.norm(Mean_vectors[k] - model_predictions_test[i])
        if Tot > 0: Tot = dist / Tot
        if Tot > Threasholds_3[d]: p_close_3.append(NB_CLASSES)
        else: p_close_3.append(d)
    p_open_3 = []
    for i in range(len(model_predictions_open)):
        d = np.argmax(model_predictions_open[i])
        dist = np.linalg.norm(Mean_vectors[d] - model_predictions_open[i])
        Tot = 0
        for k in range(NB_CLASSES):
            if k != int(d) and k in Indexes[d]: Tot += np.linalg.norm(Mean_vectors[k] - model_predictions_open[i])
        if Tot > 0: Tot = dist / Tot
        if Tot > Threasholds_3[d]: p_open_3.append(NB_CLASSES)
        else: p_open_3.append(d)

    results['K-LND1'] = {'closed_acc': accuracy_score(y_test, p_close_1), 'open_acc': accuracy_score(y_open, p_open_1)}
    results['K-LND2'] = {'closed_acc': accuracy_score(y_test, p_close_2), 'open_acc': accuracy_score(y_open, p_open_2)}
    results['K-LND3'] = {'closed_acc': accuracy_score(y_test, p_close_3), 'open_acc': accuracy_score(y_open, p_open_3)}
    
    return results

# ================= Training & Utils (Fixed) =================

def train_epoch(model, dataloader, optimizer, scheduler, device, accumulation_steps=1):
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    optimizer.zero_grad()
    
    # use bfloat16，if the GPU can not use it like(P40, V100)，change to float32 or float16
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    if not torch.cuda.is_bf16_supported():
        print("Warning: BFloat16 not supported, falling back to Float32. This may consume more memory.")
    
    pbar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Mixed precision context
        with torch.amp.autocast('cuda', dtype=dtype):
            outputs = model(**batch)
            loss = outputs.loss / accumulation_steps
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss at step {step}. Skipping.")
            continue

        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        logits = outputs.logits
        predictions += logits.argmax(dim=-1).cpu().numpy().tolist()
        true_labels += batch['labels'].cpu().numpy().tolist()
        
        pbar.set_postfix({'loss': loss.item() * accumulation_steps})
        
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    acc = accuracy_score(true_labels, predictions) if len(predictions) > 0 else 0
    return predictions, true_labels, avg_loss, acc

def predict_and_extract(model, dataloader, device, desc="Predicting"):
    model.eval()
    all_logits = []
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast('cuda', dtype=dtype):
                outputs = model(**batch)
                logits = outputs.logits
            all_logits.append(logits.float().cpu().numpy()) # 
            
    return np.concatenate(all_logits, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8, help="Actual batch size per step")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num_labels", type=int, default=4)
    parser.add_argument("--dataset", type=str, default='DC')
    parser.add_argument("--K_number", type=int, default=4)
    parser.add_argument("--TH_value", type=float, default=0.85)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--use_4bit", action="store_true")
    args = parser.parse_args()

    os.makedirs(f"./lora_models/{args.dataset}", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ================= Load Data =================
    print("Loading datasets...")
    train_df = pd.read_csv('./temp_dir/train.csv')
    valid_df = pd.read_csv('./temp_dir/valid.csv')
    test_df = pd.read_csv('./temp_dir/test.csv')
    open_df = pd.read_csv('./temp_dir/open.csv')

    # ================= Load Model & LoRA =================
    print(f"Loading GLM-4 from {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    
    torch_dtype = torch.bfloat16
    
    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

    # 
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=args.num_labels,
        trust_remote_code=True,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map="auto" if args.use_4bit else {"": device}
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        modules_to_save=["score"]
    )

    model = get_peft_model(model, peft_config)
    
    model.print_trainable_parameters()
    
    for name, param in model.named_parameters():
        if "classifier_head" in name:
            param.data = param.data.to(torch.bfloat16)

    # ================= Prepare DataLoader =================
    collator = GLM4_collator(tokenizer, max_seq_len=args.max_len)
    
    train_ds = DatasetCreator(pre_process(train_df), train=True)
    valid_ds = DatasetCreator(pre_process(valid_df), train=True)
    test_ds = DatasetCreator(pre_process(test_df), train=False)
    open_ds = DatasetCreator(pre_process(open_df), train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    eval_bs = args.batch_size * 2
    valid_loader = DataLoader(valid_ds, batch_size=eval_bs, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False, collate_fn=collator)
    open_loader = DataLoader(open_ds, batch_size=eval_bs, shuffle=False, collate_fn=collator)

    # ================= Optimizer =================
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

    results_history = []

    # ================= Training Loop =================
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        _, _, train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, args.grad_accum_steps
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Check for NaN loss
        if pd.isna(train_loss):
            print("Error: Train loss is NaN. Stopping training.")
            break

        # Save checkpoint
        save_path = f"./lora_models/{args.dataset}/epoch_{epoch+1}"
        model.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

        # ================= K-LND Evaluation =================
        print("Starting K-LND Evaluation...")
        
        train_eval_loader = DataLoader(train_ds, batch_size=eval_bs, shuffle=False, collate_fn=collator)
        
        train_logits = predict_and_extract(model, train_eval_loader, device, "Extracting Train Vectors")
        y_train = get_labels('./temp_dir/train.csv')
        
        valid_logits = predict_and_extract(model, valid_loader, device, "Extracting Valid Vectors")
        y_valid = get_labels('./temp_dir/valid.csv')
        
        test_logits = predict_and_extract(model, test_loader, device, "Extracting Test Vectors")
        y_test = get_labels('./temp_dir/test.csv')
        
        open_logits = predict_and_extract(model, open_loader, device, "Extracting Open Vectors")
        y_open = np.array([args.num_labels] * len(open_logits))

        print("Calculating Statistics...")
        Mean_Vectors = calculate_mean_vectors(args.num_labels, train_logits, y_train)
        
        Th1, Th2, Th3, Indexes = calculate_thresholds(
            args.num_labels, valid_logits, y_valid, Mean_Vectors, args.K_number, args.TH_value
        )
        
        klnd_res = evaluate_klnd(
            args.num_labels, test_logits, open_logits, y_test, y_open,
            Mean_Vectors, Indexes, Th1, Th2, Th3
        )
        
        print(f"\nResults Epoch {epoch+1}:")
        print(json.dumps(klnd_res, indent=2))
        
        results_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'klnd_results': klnd_res
        })
        
        with open(f"./results/{args.dataset}_lora_history.json", "w") as f:
            json.dump(results_history, f, indent=2)

    print("Training Completed.")

if __name__ == "__main__":
    main()