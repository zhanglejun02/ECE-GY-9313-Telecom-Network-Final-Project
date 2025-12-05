import argparse
import numpy as np 
import pandas as pd
from tqdm.auto import tqdm
import os
import random
import gc

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


# Function for training
def train(model, dataloader, optimizer, scheduler, device):
    model.train()
    predictions_labels = []
    true_labels = []
    total_loss = 0
    
    for batch in tqdm(dataloader, total=len(dataloader)):
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


# Function for validation 
def validate(model, dataloader, device):
    model.eval()
    predictions_labels = []
    true_labels = []
    total_loss = 0
    
    for batch in tqdm(dataloader, total=len(dataloader)):
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


def predict(model, dataloader, device):
    model.eval()
    predictions_labels = []
    
    for ind, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            _, logits = outputs[:2]
            # Convert bfloat16 to float32 before numpy conversion
            logits_float = logits.float().to('cpu').numpy()
            if ind == 0:
                predictions_labels = logits_float
            else:
                predictions_labels = np.concatenate((predictions_labels, logits_float), axis=0)
    return predictions_labels


def claculate_mean_vectors(NB_CLASSES, model_predictions, y_train):
    for i in range(NB_CLASSES):
        variable_name = f"Mean_{i}"
        locals()[variable_name]=np.array([0] * NB_CLASSES)
    count=[0]*NB_CLASSES
    txt_O = "Mean_{Class1:.0f}"
    Means={}
    for i in range(NB_CLASSES):
        Means[txt_O.format(Class1=i)]=np.array([0]*NB_CLASSES)

    for i in range(len(model_predictions)):
        k=np.argmax(model_predictions[i])
        if (np.argmax(model_predictions[i])==y_train[i]):
            Means[txt_O.format(Class1=y_train[i])]=Means[txt_O.format(Class1=y_train[i])] + model_predictions[i]
            count[y_train[i]]+=1

    Mean_Vectors=[]
    for i in range(NB_CLASSES):
        Means[txt_O.format(Class1=i)]=Means[txt_O.format(Class1=i)]/count[i]
        Mean_Vectors.append(Means[txt_O.format(Class1=i)])

    Mean_Vectors=np.array(Mean_Vectors)
    return Mean_Vectors


def calculate_thresholds(NB_CLASSES, model_predictions, y_valid, Mean_Vectors, K_number, TH_value):

    txt_1 = "Dist_{Class1:.0f}"
    Distances={}
    for i in range(NB_CLASSES):
        Distances[txt_1.format(Class1=i)]=[]

    Indexes=[]
    for i in range(NB_CLASSES):
        Indexes.append([])

    Values={}
    for i in range(NB_CLASSES):
        Values[i]=[0]*NB_CLASSES

    for i in range(len(model_predictions)):
        if (y_valid[i]==np.argmax(model_predictions[i])):
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]]-model_predictions[i])
            for k in range(NB_CLASSES):
                if k!=int(y_valid[i]):
                    Values[y_valid[i]][k]+=np.linalg.norm(Mean_Vectors[k]-model_predictions[i])-dist

    for i in range(NB_CLASSES):
        Tot=0
        for l in range(K_number):
            Min=min(Values[i])
            Tot+=Min
            Indexes[i].append(Values[i].index(Min))
            Values[i][Values[i].index(Min)]=1000000

    Indexes=np.array(Indexes)

    
    txt_1 = "Dist_{Class1:.0f}"
    Distances={}
    for i in range(NB_CLASSES):
        Distances[txt_1.format(Class1=i)]=[]

    for i in range(len(model_predictions)):
        if (y_valid[i]==np.argmax(model_predictions[i])):
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]]-model_predictions[i])
            Distances[txt_1.format(Class1=y_valid[i])].append(dist)

    TH=[0]*NB_CLASSES
    for j in range(NB_CLASSES):
        Distances[txt_1.format(Class1=j)].sort()
        Dist=Distances[txt_1.format(Class1=j)]
        try:
            TH[j]=Dist[int(len(Dist)*TH_value)]
        except:
            if j == 0:
                TH[j] = 10
            else:
                TH[j] = TH[j-1]

    Threasholds_1=np.array(TH)
    print("Thresholds for method 1 calculated")
    
    
    txt_1 = "Dist_{Class1:.0f}"
    Distances={}
    for i in range(NB_CLASSES):
        Distances[txt_1.format(Class1=i)]=[]

    for i in range(len(model_predictions)):
        if (y_valid[i]==np.argmax(model_predictions[i])):
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]]-model_predictions[i])
            Tot=0
            for k in range(NB_CLASSES):
                if k!=int(y_valid[i]) and k in Indexes[y_valid[i]]:
                    Tot+=(np.linalg.norm(Mean_Vectors[k]-model_predictions[i])-dist)
            Distances[txt_1.format(Class1=y_valid[i])].append(Tot)

    TH=[0]*NB_CLASSES
    for j in range(NB_CLASSES):
        Distances[txt_1.format(Class1=j)].sort()
        Dist=Distances[txt_1.format(Class1=j)]
        try:
            TH[j]=Dist[int(len(Dist)*(1-TH_value))]
        except:
            if j == 0:
                TH[j] = 10
            else:
                TH[j] = TH[j-1]

    Threasholds_2=np.array(TH)
    print("Thresholds for method 2 calculated")
    
    
    txt_1 = "Dist_{Class1:.0f}"
    Distances={}
    for i in range(NB_CLASSES):
        Distances[txt_1.format(Class1=i)]=[]

    for i in range(len(model_predictions)):
        if (y_valid[i]==np.argmax(model_predictions[i])):
            dist = np.linalg.norm(Mean_Vectors[y_valid[i]]-model_predictions[i])
            Tot=0
            for k in range(NB_CLASSES):
                if k!=int(y_valid[i]) and k in Indexes[y_valid[i]]:
                    Tot+=np.linalg.norm(Mean_Vectors[k]-model_predictions[i])
            Tot=dist/Tot
            Distances[txt_1.format(Class1=y_valid[i])].append(Tot)

    TH=[0]*NB_CLASSES
    for j in range(NB_CLASSES):
        Distances[txt_1.format(Class1=j)].sort()
        Dist=Distances[txt_1.format(Class1=j)]
        try:
            TH[j]=Dist[int(len(Dist)*TH_value)]
        except:
            if j == 0:
                TH[j] = 10
            else:
                TH[j] = TH[j-1]

    Threasholds_3=np.array(TH)
    print("Thresholds for method 3 calculated")
    
    return Threasholds_1, Threasholds_2, Threasholds_3, Indexes


def print_results(prediction_classes, prediction_classes_open, y_test, y_open, NB_CLASSES, KLND_type, dataset_name):

    y_test = y_test.astype(int)
    y_open = y_open.astype(int)

    acc_Close = accuracy_score(prediction_classes, y_test[:len(prediction_classes)])
    print('Test accuracy Normal model_Closed_set :', acc_Close)

    acc_Open = accuracy_score(prediction_classes_open, y_open[:len(prediction_classes_open)])
    print('Test accuracy Normal model_Open_set :', acc_Open)

    y_test=y_test[:len(prediction_classes)]
    y_open=y_open[:len(prediction_classes_open)]

    Matrix=[]
    for i in range(NB_CLASSES+1):
        Matrix.append(np.zeros(NB_CLASSES+1))

    for i in range(len(y_test)):
        Matrix[y_test[i]][prediction_classes[i]]+=1

    for i in range(len(y_open)):
        Matrix[y_open[i]][prediction_classes_open[i]]+=1

    # Convert Matrix to numpy array for proper indexing
    Matrix = np.array(Matrix)
    
    print("\n", "Micro")
    F1_Score_micro=Micro_F1(Matrix, NB_CLASSES)
    print("Average Micro F1_Score: ", F1_Score_micro)

    print("\n", "Macro")
    F1_Score_macro=Macro_F1(Matrix, NB_CLASSES)
    print("Average Macro F1_Score: ", F1_Score_macro)
    
    text_file = open("./results/results_"+ dataset_name +"_Qwen3.txt", "a")

    text_file.write('########' + KLND_type + '#########\n')
    text_file.write('Test accuracy Normal model_Closed_set :'+ str(acc_Close) + '\n')
    text_file.write('Test accuracy Normal model_Open_set :'+ str(acc_Open) + '\n')
    text_file.write("Average Micro F1_Score: " + str(F1_Score_micro) + '\n')
    text_file.write("Average Macro F1_Score: " + str(F1_Score_macro) + '\n')
    text_file.write('\n')
    text_file.close()


def final_classification(NB_CLASSES, model_predictions_test, model_predictions_open, y_test, y_open, Mean_vectors, Indexes, Threasholds_1, Threasholds_2, Threasholds_3, dataset_name):
     
    
    print("\n", "############## Distance Method 1 #################################")
    prediction_classes=[]
    for i in range(len(model_predictions_test)):

        d=np.argmax(model_predictions_test[i], axis=0)
        if np.linalg.norm(model_predictions_test[i]-Mean_vectors[d])>Threasholds_1[d]:
            prediction_classes.append(NB_CLASSES)

        else:
            prediction_classes.append(d)
    prediction_classes=np.array(prediction_classes)

    prediction_classes_open=[]
    for i in range(len(model_predictions_open)):

        d=np.argmax(model_predictions_open[i], axis=0)
        if np.linalg.norm(model_predictions_open[i]-Mean_vectors[d])>Threasholds_1[d]:
            prediction_classes_open.append(NB_CLASSES)
        else:
            prediction_classes_open.append(d)
    prediction_classes_open=np.array(prediction_classes_open)
    print_results(prediction_classes, prediction_classes_open, y_test, y_open, NB_CLASSES, 'K-LND1', dataset_name)

    print("\n", "############## Distance Method 2 #################################")
    prediction_classes=[]
    for i in range(len(model_predictions_test)):
        d=np.argmax(model_predictions_test[i], axis=0)
        dist=np.linalg.norm(Mean_vectors[d]-model_predictions_test[i])
        Tot=0
        for k in range(NB_CLASSES):
            if k!=d:
                Tot+=np.linalg.norm(Mean_vectors[k]-model_predictions_test[i])-dist

        if Tot<Threasholds_2[d]:
            prediction_classes.append(NB_CLASSES)

        else:
            prediction_classes.append(d)

    prediction_classes_open=[]
    for i in range(len(model_predictions_open)):
        d=np.argmax(model_predictions_open[i], axis=0)
        dist = np.linalg.norm(Mean_vectors[d]-model_predictions_open[i])
        Tot=0
        for k in range(NB_CLASSES):
            if k!=int(d) and k in Indexes[d]:
                Tot+=np.linalg.norm(Mean_vectors[k]-model_predictions_open[i])-dist

        if Tot<Threasholds_2[d]:
            prediction_classes_open.append(NB_CLASSES)
        else:
            prediction_classes_open.append(d)

    prediction_classes_open=np.array(prediction_classes_open)
    print_results(prediction_classes, prediction_classes_open, y_test, y_open, NB_CLASSES, 'K-LND2', dataset_name)

    
    print("\n", "############## Distance Method 3 #################################")

    prediction_classes=[]
    for i in range(len(model_predictions_test)):
        d=np.argmax(model_predictions_test[i], axis=0)
        dist=np.linalg.norm(Mean_vectors[d]-model_predictions_test[i])
        Tot=0
        for k in range(NB_CLASSES):
            if k!=d:
                Tot+=np.linalg.norm(Mean_vectors[k]-model_predictions_test[i])

        Tot=dist/Tot
        if Tot>Threasholds_3[d]:
            prediction_classes.append(NB_CLASSES)

        else:
            prediction_classes.append(d)

    prediction_classes=np.array(prediction_classes)
    
    prediction_classes_open=[]
    for i in range(len(model_predictions_open)):
        d=np.argmax(model_predictions_open[i], axis=0)
        dist=np.linalg.norm(Mean_vectors[d]-model_predictions_open[i])
        Tot=0
        for k in range(NB_CLASSES):
            if k!=int(d) and k in Indexes[d]:
                Tot+=np.linalg.norm(Mean_vectors[k]-model_predictions_open[i])
        Tot=dist/Tot
        if Tot>Threasholds_3[d]:
            prediction_classes_open.append(NB_CLASSES)

        else:
            prediction_classes_open.append(d)

    prediction_classes_open=np.array(prediction_classes_open)
    print_results(prediction_classes, prediction_classes_open, y_test, y_open, NB_CLASSES, 'K-LND3', dataset_name)


def Micro_F1(Matrix, NB_CLASSES):
    epsilon = 1e-8
    TP = 0
    FP = 0
    TN = 0

    for k in range(NB_CLASSES):
        TP += Matrix[k][k]
        FP += (np.sum(Matrix, axis=0)[k] - Matrix[k][k])
        TN += (np.sum(Matrix, axis=1)[k] - Matrix[k][k])

    Micro_Prec = TP / (TP + FP)
    Micro_Rec = TP / (TP + TN)
    print("Micro_Prec:", Micro_Prec)
    print("Micro_Rec:", Micro_Rec)
    Micro_F1 = 2 * Micro_Prec * Micro_Rec / (Micro_Rec + Micro_Prec + epsilon)

    return Micro_F1


def Macro_F1(Matrix, NB_CLASSES):
    epsilon = 1e-8
    F1s = np.zeros(NB_CLASSES)

    for k in range(NB_CLASSES):
        TP = Matrix[k][k]
        FP = np.sum(Matrix[:, k]) - TP
        FN = np.sum(Matrix[k, :]) - TP

        precision = TP / (TP + FP + epsilon)
        recall = TP / (TP + FN + epsilon)
        F1s[k] = 2 * precision * recall / (precision + recall + epsilon)

    macro_F1 = np.mean(F1s)
    print("Per-class F1s:", F1s)
    print("Macro F1:", macro_F1)
    return macro_F1


def main(args):
    max_len = args.max_len
    batch_size = args.batch_size
    epochs = args.epochs
    num_labels = args.num_labels
    K_number = args.K_number
    TH_value = args.TH_value
    dataset = args.dataset

    if not os.path.exists("./trained_models"):  
        os.makedirs("trained_models") 

    train_dataset = pd.read_csv('./temp_dir/train.csv')
    val_dataset = pd.read_csv('./temp_dir/valid.csv')

    print('Loading Qwen3 model from:', QWEN_MODEL_PATH)
    
    # Load Qwen3 model for sequence classification (Full fine-tuning, NO LoRA)
    model = AutoModelForSequenceClassification.from_pretrained(
        QWEN_MODEL_PATH,
        num_labels=num_labels,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16  # Use bf16 for efficiency
    )
    
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_MODEL_PATH,
        trust_remote_code=True
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Use DataParallel for multi-GPU or move to device
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    qwen_collator = Qwen_collator(tokenizer=tokenizer, max_seq_len=max_len)

    # Prepare training data
    processed_data = pre_process(train_dataset)
    train_data = DatasetCreator(processed_data, train=True)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=qwen_collator)

    # Prepare validation data
    val_processed = pre_process(val_dataset)
    val_data = DatasetCreator(val_processed, train=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=qwen_collator)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8, weight_decay=0.01)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)
    
    loss = []
    accuracy = []
    val_loss_list = []
    val_accuracy_list = []

    print("\n" + "="*60)
    print(f"Starting Qwen3 Full Fine-tuning Training")
    print(f"Dataset: {dataset}, Labels: {num_labels}, Epochs: {epochs}")
    print("="*60 + "\n")

    for epoch in tqdm(range(epochs), desc="Epochs"):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        train_labels, true_labels, train_loss = train(model, train_dataloader, optimizer, scheduler, device)    
        train_acc = accuracy_score(true_labels, train_labels) 
        print(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        loss.append(train_loss)
        accuracy.append(train_acc)

        val_labels, val_true_labels, val_loss = validate(model, val_dataloader, device)
        val_acc = accuracy_score(val_true_labels, val_labels)
        print(f'Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_acc)

    # Save model
    print(f"\nSaving model to ./trained_models/trained_qwen3_{dataset}.pth")
    torch.save(model.state_dict(), f'./trained_models/trained_qwen3_{dataset}.pth')

    # ========== Evaluation Phase ==========
    print("\n" + "="*60)
    print("Starting Evaluation Phase")
    print("="*60 + "\n")

    # Prepare evaluation datasets
    train_dataset_eval = pd.read_csv('./temp_dir/train.csv')
    start_index = int(len(train_dataset_eval) * 0.6)
    train_subset = train_dataset_eval[start_index:]
    train_processed_eval = pre_process(train_dataset_eval)
    train_data_eval = DatasetCreator(train_processed_eval, train=False)
    train_eval_dataloader = DataLoader(train_data_eval, batch_size=32, shuffle=False, collate_fn=qwen_collator)

    print("Predicting on train set...")
    train_predictions = predict(model, train_eval_dataloader, device)
    y_train = get_labels('./temp_dir/train.csv')
    del train_data_eval, train_dataset_eval, train_processed_eval, train_eval_dataloader
    gc.collect()

    valid_dataset = pd.read_csv('./temp_dir/valid.csv')
    y_valid = get_labels('./temp_dir/valid.csv')
    if dataset == 'DC':
        valid_dataset = pd.concat([train_subset, valid_dataset], ignore_index=True)
        y_valid = np.concatenate((y_valid, y_train[start_index:]), axis=0)

    valid_processed = pre_process(valid_dataset)
    valid_data = DatasetCreator(valid_processed, train=False)
    valid_eval_dataloader = DataLoader(valid_data, batch_size=32, shuffle=False, collate_fn=qwen_collator)

    print("Predicting on validation set...")
    valid_predictions = predict(model, valid_eval_dataloader, device)
    del valid_data, valid_dataset, valid_processed, valid_eval_dataloader
    gc.collect()

    test_dataset = pd.read_csv('./temp_dir/test.csv')
    test_processed = pre_process(test_dataset)
    test_data = DatasetCreator(test_processed, train=False)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=qwen_collator)

    print("Predicting on test set...")
    test_predictions = predict(model, test_dataloader, device)
    y_test = get_labels('./temp_dir/test.csv')
    del test_data, test_dataset, test_processed, test_dataloader
    gc.collect()

    open_dataset = pd.read_csv('./temp_dir/open.csv')
    open_processed = pre_process(open_dataset)
    open_data = DatasetCreator(open_processed, train=False)
    open_dataloader = DataLoader(open_data, batch_size=32, shuffle=False, collate_fn=qwen_collator)

    print("Predicting on open set...")
    open_predictions = predict(model, open_dataloader, device)
    y_open = get_labels('./temp_dir/open.csv')
    del open_data, open_dataset, open_processed, open_dataloader
    y_open = np.array([num_labels]*len(y_open))

    if not os.path.exists('./results'):
        os.makedirs('./results')
    
    print("\nCalculating mean vectors...")
    Mean_Vectors = claculate_mean_vectors(num_labels, train_predictions, y_train)
    
    print("Calculating thresholds...")
    Threasholds_1, Threasholds_2, Threasholds_3, Indexes = calculate_thresholds(
        num_labels, valid_predictions, y_valid, Mean_Vectors, K_number, TH_value
    )
    
    print("\nPerforming final classification...")
    final_classification(
        num_labels, test_predictions, open_predictions, y_test, y_open, 
        Mean_Vectors, Indexes, Threasholds_1, Threasholds_2, Threasholds_3, dataset
    )

    print("\n" + "="*60)
    print(f"Training and Evaluation Complete!")
    print(f"Model saved: ./trained_models/trained_qwen3_{dataset}.pth")
    print(f"Results saved: ./results/results_{dataset}_Qwen3.txt")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Qwen3 model with full fine-tuning (No LoRA)")
    parser.add_argument("--max_len", type=int, default=1024, help="Max length of the text for input")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--num_labels", type=int, default=4, help="Number of labels for classification")
    parser.add_argument("--K_number", type=int, default=4, help="K nearest neighbors")
    parser.add_argument("--TH_value", type=float, default=0.85, help="Threshold value for distances")
    parser.add_argument("--dataset", type=str, default='DC', help="Dataset name")

    args = parser.parse_args()
    main(args)

