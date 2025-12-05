"""
Qwen3 Per-Epoch Evaluation Script
在每个epoch后都进行完整的K-LND评估，追踪性能变化
"""

import argparse
import numpy as np 
import pandas as pd
from tqdm.auto import tqdm
import os
import random
import gc
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from torch.optim import AdamW
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer,
                          get_linear_schedule_with_warmup)

random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Qwen模型路径
QWEN_MODEL_PATH = "/maindata/data/shared/ai_story_workspace-dsw/common_models/Qwen/Qwen3-4B-Instruct-2507"


class DatasetCreator(Dataset):
    def __init__(self, processed_data, train):
        self.data = processed_data
        self.train = train
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        line = self.data.iloc[index]
        if self.train:
            return {'text': line['trace'], 'label': line['target']}
        else:
            return {'text': line['trace'], 'label': 0}


class Qwen_collator(object):
    def __init__(self, tokenizer, max_seq_len=None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        return
    
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
    dataset['trace'] = dataset['text']
    return dataset


def get_labels(file):
    df = pd.read_csv(file)
    return np.array(df['target'])


def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    predictions_labels = []
    true_labels = []
    total_loss = 0
    
    for batch in tqdm(dataloader, total=len(dataloader), desc="Training"):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss, logits = outputs[:2]
        if loss.dim() > 0:
            loss = loss.mean() 

        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()
    avg_epoch_loss = total_loss / len(dataloader)
    return predictions_labels, true_labels, avg_epoch_loss


def validate(model, dataloader, device):
    model.eval()
    predictions_labels = []
    true_labels = []
    total_loss = 0
    
    for batch in tqdm(dataloader, total=len(dataloader), desc="Validating"):
        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss, logits = outputs[:2]
            if loss.dim() > 0:
                loss = loss.mean() 

            total_loss += loss.item()
            predictions_labels += logits.argmax(axis=-1).flatten().tolist()
    avg_epoch_loss = total_loss / len(dataloader)
    return predictions_labels, true_labels, avg_epoch_loss


def predict(model, dataloader, device, desc="Predicting"):
    model.eval()
    predictions_labels = []
    
    for ind, batch in enumerate(tqdm(dataloader, total=len(dataloader), desc=desc)):
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            _, logits = outputs[:2]
            logits_float = logits.float().to('cpu').numpy()
            if ind == 0:
                predictions_labels = logits_float
            else:
                predictions_labels = np.concatenate((predictions_labels, logits_float), axis=0)
    return predictions_labels


def claculate_mean_vectors(NB_CLASSES, model_predictions, y_train):
    Means = {}
    count = [0] * NB_CLASSES
    txt_O = "Mean_{Class1:.0f}"
    
    for i in range(NB_CLASSES):
        Means[txt_O.format(Class1=i)] = np.array([0] * NB_CLASSES)

    for i in range(len(model_predictions)):
        if (np.argmax(model_predictions[i]) == y_train[i]):
            Means[txt_O.format(Class1=y_train[i])] = Means[txt_O.format(Class1=y_train[i])] + model_predictions[i]
            count[y_train[i]] += 1

    Mean_Vectors = []
    for i in range(NB_CLASSES):
        if count[i] > 0:
            Means[txt_O.format(Class1=i)] = Means[txt_O.format(Class1=i)] / count[i]
        Mean_Vectors.append(Means[txt_O.format(Class1=i)])

    return np.array(Mean_Vectors)


def calculate_thresholds(NB_CLASSES, model_predictions, y_valid, Mean_Vectors, K_number, TH_value):
    txt_1 = "Dist_{Class1:.0f}"
    Distances = {}
    Indexes = []
    Values = {}
    
    for i in range(NB_CLASSES):
        Distances[txt_1.format(Class1=i)] = []
        Indexes.append([])
        Values[i] = [0] * NB_CLASSES

    for i in range(len(model_predictions)):
        if (y_valid[i] == np.argmax(model_predictions[i])):
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]] - model_predictions[i])
            for k in range(NB_CLASSES):
                if k != int(y_valid[i]):
                    Values[y_valid[i]][k] += np.linalg.norm(Mean_Vectors[k] - model_predictions[i]) - dist

    for i in range(NB_CLASSES):
        for l in range(min(K_number, NB_CLASSES - 1)):
            Min = min(Values[i])
            Indexes[i].append(Values[i].index(Min))
            Values[i][Values[i].index(Min)] = 1000000

    Indexes = np.array(Indexes)

    # Threshold 1
    Distances = {}
    for i in range(NB_CLASSES):
        Distances[txt_1.format(Class1=i)] = []

    for i in range(len(model_predictions)):
        if (y_valid[i] == np.argmax(model_predictions[i])):
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]] - model_predictions[i])
            Distances[txt_1.format(Class1=y_valid[i])].append(dist)

    TH = [0] * NB_CLASSES
    for j in range(NB_CLASSES):
        Distances[txt_1.format(Class1=j)].sort()
        Dist = Distances[txt_1.format(Class1=j)]
        try:
            TH[j] = Dist[int(len(Dist) * TH_value)]
        except:
            TH[j] = 10 if j == 0 else TH[j-1]

    Threasholds_1 = np.array(TH)

    # Threshold 2
    Distances = {}
    for i in range(NB_CLASSES):
        Distances[txt_1.format(Class1=i)] = []

    for i in range(len(model_predictions)):
        if (y_valid[i] == np.argmax(model_predictions[i])):
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]] - model_predictions[i])
            Tot = 0
            for k in range(NB_CLASSES):
                if k != int(y_valid[i]) and k in Indexes[y_valid[i]]:
                    Tot += (np.linalg.norm(Mean_Vectors[k] - model_predictions[i]) - dist)
            Distances[txt_1.format(Class1=y_valid[i])].append(Tot)

    TH = [0] * NB_CLASSES
    for j in range(NB_CLASSES):
        Distances[txt_1.format(Class1=j)].sort()
        Dist = Distances[txt_1.format(Class1=j)]
        try:
            TH[j] = Dist[int(len(Dist) * (1 - TH_value))]
        except:
            TH[j] = 10 if j == 0 else TH[j-1]

    Threasholds_2 = np.array(TH)

    # Threshold 3
    Distances = {}
    for i in range(NB_CLASSES):
        Distances[txt_1.format(Class1=i)] = []

    for i in range(len(model_predictions)):
        if (y_valid[i] == np.argmax(model_predictions[i])):
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]] - model_predictions[i])
            Tot = 0
            for k in range(NB_CLASSES):
                if k != int(y_valid[i]) and k in Indexes[y_valid[i]]:
                    Tot += np.linalg.norm(Mean_Vectors[k] - model_predictions[i])
            if Tot > 0:
                Tot = dist / Tot
            Distances[txt_1.format(Class1=y_valid[i])].append(Tot)

    TH = [0] * NB_CLASSES
    for j in range(NB_CLASSES):
        Distances[txt_1.format(Class1=j)].sort()
        Dist = Distances[txt_1.format(Class1=j)]
        try:
            TH[j] = Dist[int(len(Dist) * TH_value)]
        except:
            TH[j] = 10 if j == 0 else TH[j-1]

    Threasholds_3 = np.array(TH)
    
    return Threasholds_1, Threasholds_2, Threasholds_3, Indexes


def evaluate_klnd(NB_CLASSES, model_predictions_test, model_predictions_open, y_test, y_open, 
                  Mean_vectors, Indexes, Threasholds_1, Threasholds_2, Threasholds_3):
    """评估三种K-LND方法，返回结果字典"""
    results = {}
    
    y_test = y_test.astype(int)
    y_open = y_open.astype(int)
    
    # K-LND1
    prediction_classes = []
    for i in range(len(model_predictions_test)):
        d = np.argmax(model_predictions_test[i], axis=0)
        if np.linalg.norm(model_predictions_test[i] - Mean_vectors[d]) > Threasholds_1[d]:
            prediction_classes.append(NB_CLASSES)
        else:
            prediction_classes.append(d)
    prediction_classes = np.array(prediction_classes)
    
    prediction_classes_open = []
    for i in range(len(model_predictions_open)):
        d = np.argmax(model_predictions_open[i], axis=0)
        if np.linalg.norm(model_predictions_open[i] - Mean_vectors[d]) > Threasholds_1[d]:
            prediction_classes_open.append(NB_CLASSES)
        else:
            prediction_classes_open.append(d)
    prediction_classes_open = np.array(prediction_classes_open)
    
    acc_close_1 = accuracy_score(prediction_classes, y_test[:len(prediction_classes)])
    acc_open_1 = accuracy_score(prediction_classes_open, y_open[:len(prediction_classes_open)])
    
    # K-LND2
    prediction_classes = []
    for i in range(len(model_predictions_test)):
        d = np.argmax(model_predictions_test[i], axis=0)
        dist = np.linalg.norm(Mean_vectors[d] - model_predictions_test[i])
        Tot = 0
        for k in range(NB_CLASSES):
            if k != d:
                Tot += np.linalg.norm(Mean_vectors[k] - model_predictions_test[i]) - dist
        if Tot < Threasholds_2[d]:
            prediction_classes.append(NB_CLASSES)
        else:
            prediction_classes.append(d)
    prediction_classes = np.array(prediction_classes)
    
    prediction_classes_open = []
    for i in range(len(model_predictions_open)):
        d = np.argmax(model_predictions_open[i], axis=0)
        dist = np.linalg.norm(Mean_vectors[d] - model_predictions_open[i])
        Tot = 0
        for k in range(NB_CLASSES):
            if k != int(d) and k in Indexes[d]:
                Tot += np.linalg.norm(Mean_vectors[k] - model_predictions_open[i]) - dist
        if Tot < Threasholds_2[d]:
            prediction_classes_open.append(NB_CLASSES)
        else:
            prediction_classes_open.append(d)
    prediction_classes_open = np.array(prediction_classes_open)
    
    acc_close_2 = accuracy_score(prediction_classes, y_test[:len(prediction_classes)])
    acc_open_2 = accuracy_score(prediction_classes_open, y_open[:len(prediction_classes_open)])
    
    # K-LND3
    prediction_classes = []
    for i in range(len(model_predictions_test)):
        d = np.argmax(model_predictions_test[i], axis=0)
        dist = np.linalg.norm(Mean_vectors[d] - model_predictions_test[i])
        Tot = 0
        for k in range(NB_CLASSES):
            if k != d:
                Tot += np.linalg.norm(Mean_vectors[k] - model_predictions_test[i])
        if Tot > 0:
            Tot = dist / Tot
        if Tot > Threasholds_3[d]:
            prediction_classes.append(NB_CLASSES)
        else:
            prediction_classes.append(d)
    prediction_classes = np.array(prediction_classes)
    
    prediction_classes_open = []
    for i in range(len(model_predictions_open)):
        d = np.argmax(model_predictions_open[i], axis=0)
        dist = np.linalg.norm(Mean_vectors[d] - model_predictions_open[i])
        Tot = 0
        for k in range(NB_CLASSES):
            if k != int(d) and k in Indexes[d]:
                Tot += np.linalg.norm(Mean_vectors[k] - model_predictions_open[i])
        if Tot > 0:
            Tot = dist / Tot
        if Tot > Threasholds_3[d]:
            prediction_classes_open.append(NB_CLASSES)
        else:
            prediction_classes_open.append(d)
    prediction_classes_open = np.array(prediction_classes_open)
    
    acc_close_3 = accuracy_score(prediction_classes, y_test[:len(prediction_classes)])
    acc_open_3 = accuracy_score(prediction_classes_open, y_open[:len(prediction_classes_open)])
    
    results = {
        'K-LND1': {'closed_acc': acc_close_1, 'open_acc': acc_open_1},
        'K-LND2': {'closed_acc': acc_close_2, 'open_acc': acc_open_2},
        'K-LND3': {'closed_acc': acc_close_3, 'open_acc': acc_open_3}
    }
    
    return results


def full_evaluation(model, num_labels, K_number, TH_value, qwen_collator, device):
    """完整的评估流程"""
    print("\n" + "="*60)
    print("Starting Full Evaluation")
    print("="*60)
    
    # Load evaluation datasets
    train_dataset_eval = pd.read_csv('./temp_dir/train.csv')
    start_index = int(len(train_dataset_eval) * 0.6)
    train_processed_eval = pre_process(train_dataset_eval)
    train_data_eval = DatasetCreator(train_processed_eval, train=False)
    train_eval_dataloader = DataLoader(train_data_eval, batch_size=32, shuffle=False, collate_fn=qwen_collator)
    
    train_predictions = predict(model, train_eval_dataloader, device, "Train set")
    y_train = get_labels('./temp_dir/train.csv')
    del train_data_eval, train_dataset_eval, train_processed_eval, train_eval_dataloader
    gc.collect()
    
    valid_dataset = pd.read_csv('./temp_dir/valid.csv')
    y_valid = get_labels('./temp_dir/valid.csv')
    
    valid_processed = pre_process(valid_dataset)
    valid_data = DatasetCreator(valid_processed, train=False)
    valid_eval_dataloader = DataLoader(valid_data, batch_size=32, shuffle=False, collate_fn=qwen_collator)
    
    valid_predictions = predict(model, valid_eval_dataloader, device, "Valid set")
    del valid_data, valid_dataset, valid_processed, valid_eval_dataloader
    gc.collect()
    
    test_dataset = pd.read_csv('./temp_dir/test.csv')
    test_processed = pre_process(test_dataset)
    test_data = DatasetCreator(test_processed, train=False)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=qwen_collator)
    
    test_predictions = predict(model, test_dataloader, device, "Test set")
    y_test = get_labels('./temp_dir/test.csv')
    del test_data, test_dataset, test_processed, test_dataloader
    gc.collect()
    
    open_dataset = pd.read_csv('./temp_dir/open.csv')
    open_processed = pre_process(open_dataset)
    open_data = DatasetCreator(open_processed, train=False)
    open_dataloader = DataLoader(open_data, batch_size=32, shuffle=False, collate_fn=qwen_collator)
    
    open_predictions = predict(model, open_dataloader, device, "Open set")
    y_open = np.array([num_labels] * len(open_predictions))
    del open_data, open_dataset, open_processed, open_dataloader
    gc.collect()
    
    print("\nCalculating mean vectors and thresholds...")
    Mean_Vectors = claculate_mean_vectors(num_labels, train_predictions, y_train)
    Threasholds_1, Threasholds_2, Threasholds_3, Indexes = calculate_thresholds(
        num_labels, valid_predictions, y_valid, Mean_Vectors, K_number, TH_value
    )
    
    print("Performing K-LND evaluation...")
    results = evaluate_klnd(
        num_labels, test_predictions, open_predictions, y_test, y_open,
        Mean_Vectors, Indexes, Threasholds_1, Threasholds_2, Threasholds_3
    )
    
    return results


def main(args):
    max_len = args.max_len
    batch_size = args.batch_size
    epochs = args.epochs
    num_labels = args.num_labels
    K_number = args.K_number
    TH_value = args.TH_value
    dataset = args.dataset
    
    # Create output directories
    os.makedirs("./trained_models", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./epoch_results", exist_ok=True)
    
    # Initialize results tracking
    all_results = {
        'dataset': dataset,
        'num_labels': num_labels,
        'batch_size': batch_size,
        'epochs': epochs,
        'K_number': K_number,
        'TH_value': TH_value,
        'start_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'epoch_results': []
    }
    
    train_dataset = pd.read_csv('./temp_dir/train.csv')
    val_dataset = pd.read_csv('./temp_dir/valid.csv')
    
    print('Loading Qwen3 model from:', QWEN_MODEL_PATH)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        QWEN_MODEL_PATH,
        num_labels=num_labels,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_MODEL_PATH,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)
    
    qwen_collator = Qwen_collator(tokenizer=tokenizer, max_seq_len=max_len)
    
    # Prepare training data
    processed_data = pre_process(train_dataset)
    train_data = DatasetCreator(processed_data, train=True)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=qwen_collator)
    
    val_processed = pre_process(val_dataset)
    val_data = DatasetCreator(val_processed, train=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=qwen_collator)
    
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8, weight_decay=0.01)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)
    
    print("\n" + "="*60)
    print(f"Starting Qwen3 Training with Per-Epoch Evaluation")
    print(f"Dataset: {dataset}, Labels: {num_labels}, Epochs: {epochs}")
    print("="*60 + "\n")
    
    # Training loop with per-epoch evaluation
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{epochs}")
        print('='*60)
        
        # Training
        print(f"\n[Training Epoch {epoch+1}]")
        train_labels, true_labels, train_loss = train(model, train_dataloader, optimizer, scheduler, device)
        train_acc = accuracy_score(true_labels, train_labels)
        print(f'✓ Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        
        # Validation
        print(f"\n[Validating Epoch {epoch+1}]")
        val_labels, val_true_labels, val_loss = validate(model, val_dataloader, device)
        val_acc = accuracy_score(val_true_labels, val_labels)
        print(f'✓ Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
        
        # Save checkpoint
        checkpoint_path = f'./trained_models/trained_qwen3_{dataset}_epoch{epoch+1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f'\n✓ Checkpoint saved: {checkpoint_path}')
        
        # Full K-LND evaluation
        print(f"\n[K-LND Evaluation Epoch {epoch+1}]")
        eval_results = full_evaluation(model, num_labels, K_number, TH_value, qwen_collator, device)
        
        # Print results
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1} RESULTS:")
        print('='*60)
        for method, scores in eval_results.items():
            print(f"{method}:")
            print(f"  Closed-set Acc: {scores['closed_acc']:.4f}")
            print(f"  Open-set Acc:   {scores['open_acc']:.4f}")
        
        # Store results
        epoch_result = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'klnd_results': eval_results
        }
        all_results['epoch_results'].append(epoch_result)
        
        # Save intermediate results
        results_file = f'./epoch_results/{dataset}_qwen3_per_epoch.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f'\n✓ Results saved: {results_file}')
    
    all_results['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save final results
    final_results_file = f'./epoch_results/{dataset}_qwen3_final.json'
    with open(final_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary report
    generate_summary_report(all_results, dataset)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Final model: ./trained_models/trained_qwen3_{dataset}_epoch{epochs}.pth")
    print(f"Results JSON: {final_results_file}")
    print(f"Summary report: ./epoch_results/{dataset}_qwen3_summary.txt")


def generate_summary_report(results, dataset):
    """生成可读的总结报告"""
    report_path = f'./epoch_results/{dataset}_qwen3_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"Qwen3 Per-Epoch Evaluation Report - {dataset} Dataset\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Training Configuration:\n")
        f.write(f"  Dataset: {dataset}\n")
        f.write(f"  Num Labels: {results['num_labels']}\n")
        f.write(f"  Batch Size: {results['batch_size']}\n")
        f.write(f"  Epochs: {results['epochs']}\n")
        f.write(f"  K Number: {results['K_number']}\n")
        f.write(f"  Threshold: {results['TH_value']}\n")
        f.write(f"  Start Time: {results['start_time']}\n")
        f.write(f"  End Time: {results['end_time']}\n\n")
        
        f.write("="*80 + "\n")
        f.write("Per-Epoch Results:\n")
        f.write("="*80 + "\n\n")
        
        for epoch_result in results['epoch_results']:
            epoch = epoch_result['epoch']
            f.write(f"Epoch {epoch}:\n")
            f.write(f"  Train Loss: {epoch_result['train_loss']:.4f}, Train Acc: {epoch_result['train_acc']:.4f}\n")
            f.write(f"  Val Loss:   {epoch_result['val_loss']:.4f}, Val Acc:   {epoch_result['val_acc']:.4f}\n")
            f.write(f"\n  K-LND Results:\n")
            for method, scores in epoch_result['klnd_results'].items():
                f.write(f"    {method}: Closed={scores['closed_acc']:.4f}, Open={scores['open_acc']:.4f}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("Best Results:\n")
        f.write("="*80 + "\n\n")
        
        # Find best epoch for each metric
        for method in ['K-LND1', 'K-LND2', 'K-LND3']:
            best_closed_epoch = max(results['epoch_results'], 
                                   key=lambda x: x['klnd_results'][method]['closed_acc'])
            best_open_epoch = max(results['epoch_results'],
                                 key=lambda x: x['klnd_results'][method]['open_acc'])
            
            f.write(f"{method}:\n")
            f.write(f"  Best Closed-set: Epoch {best_closed_epoch['epoch']} "
                   f"({best_closed_epoch['klnd_results'][method]['closed_acc']:.4f})\n")
            f.write(f"  Best Open-set:   Epoch {best_open_epoch['epoch']} "
                   f"({best_open_epoch['klnd_results'][method]['open_acc']:.4f})\n\n")
    
    print(f"\n✓ Summary report generated: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3 Per-Epoch Evaluation")
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num_labels", type=int, default=4)
    parser.add_argument("--K_number", type=int, default=4)
    parser.add_argument("--TH_value", type=float, default=0.85)
    parser.add_argument("--dataset", type=str, default='DC')
    
    args = parser.parse_args()
    main(args)

