import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

from conv1d_rnncell import Conv1dRNNCell

import copy
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
import os
import sys
from rich.console import Console
from rich.table import Table
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sequence_generation import load_sequences, save_sequences
from pgb_data_processing import overview_csv_files, process_pgb_data
from data_scaling import load_and_scale_data
from util import concatenate_and_delete_ltn_csv_files
import commons as commons
from tqdm import tqdm
from numpy import mean
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import pandas as pd


# Append config directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))  # Absolute dir the script is in
sys.path.append(os.path.join(script_dir, '..', 'config'))

import config as config


# Configurations
results_path_ltn = config.results_path_ltn
model_path = config.model_path
dataset_path = config.dataset_path
PGB_path = config.PGB_path
RGB_path = config.RGB_path
csv_file = config.csv_file
preprocessor_file = config.preprocessor_file
train_path = config.train_path
val_path = config.val_path
chunk_size = config.chunk_size
csv_directory = config.csv_directory
data_root_folder = config.data_root_folder
sequence_length = config.sequence_length
sequences_directory = config.sequences_directory
num_features = config.num_features
processed_bases = config.processed_bases
batch_size = config.batch_size
epochs = config.epochs
patience = config.patience
learning_rate = config.learning_rate
n_splits = config.n_splits
model_save_directory = config.model_save_directory
reg_value = config.reg_value
num_train_samples = config.num_train_samples
num_test_samples = config.num_test_samples
reg_type = config.reg_type
n_samples = config.n_samples
num_classes = config.num_classes
buffer_size = config.buffer_size
ltn_batch = config.ltn_batch
results_path = config.results_path
S = config.S
lr_ltn = config.lr_ltn
processed_file_tracker = config.processed_file_tracker


class Swish_act(nn.Module):
    def __init__(self):
        super(Swish_act, self).__init__()

    def forward(self, x):
        x = x * F.sigmoid(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(output_channel),
            Swish_act(),

            nn.Conv1d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(output_channel)
        )

        self.skip_connection = nn.Sequential()
        if output_channel != input_channel:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_channel)
            )

        self.Lrelu = Swish_act()

    def forward(self, x):
        out = self.conv(x)
        out = self.skip_connection(x) + out
        out = self.Lrelu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_channel=num_features, num_class=num_classes):    # batch_size, 33, 1000
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channel, 64, kernel_size=7, stride=1, padding=3),  # 1, 64, 1000
            nn.BatchNorm1d(64),
            Swish_act(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # batch, 64, 500
        )

        self.layer1 = ResBlock(64, 96, stride=1)  # batch, 96, 500
        self.layer2 = ResBlock(96, 128, stride=2)  # batch, 128, 250
        self.layer3 = ResBlock(128, 192, stride=2)  # batch, 192, 125
        self.layer4 = ResBlock(192, 256, stride=2)  # batch, 256, 63
        self.layer5 = ResBlock(256, 384, stride=2)  # batch, 384, 32

        self.linear = nn.Sequential(
            nn.Linear(384 * 32, 100),
            nn.Dropout(),
            nn.Linear(100, num_class)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        pred = self.linear(out.view(out.size(0), -1))
        return pred


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
    
    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = correct_preds / total_preds
    return epoch_loss, accuracy



def create_dataloaders(X_train, y_train, X_val, y_val, batch_size):
    # Convert the input data from [batch, steps, channels] to [batch, channels, steps]
    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)

    # Convert labels into tensors
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == '__main__':    
    kf = KFold(n_splits=n_splits, shuffle=False)
    counter = 0
    console = Console()
    processed_bases = set()
    
    if os.path.exists(processed_file_tracker):
        with open(processed_file_tracker, "r") as file:
            processed_bases = set(file.read().splitlines())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    metrics_summary = []

    for file in sorted(os.listdir(sequences_directory)):

        if "_train_scaled_sequences.npy" in file:
            # if counter >= S:
            #     break
            
            base_name = file.replace("_train_scaled_sequences.npy", "")
            if base_name in processed_bases:
                continue
            processed_bases.add(base_name)
            counter += 1
     
                    
            # Load sequences and labels
            train_sequence_file_path = os.path.join(sequences_directory, f"{base_name}_train_scaled_sequences.npy")
            train_label_file_path = os.path.join(sequences_directory, f"{base_name}_train_scaled_labels.npy")
            X_train, y_train = load_sequences(train_sequence_file_path, train_label_file_path)
            
            test_sequence_file_path = os.path.join(sequences_directory, f"{base_name}_test_scaled_sequences.npy")
            test_label_file_path = os.path.join(sequences_directory, f"{base_name}_test_scaled_labels.npy")
            X_test, y_test = load_sequences(test_sequence_file_path, test_label_file_path)

            # Shuffle the sequences and corresponding labels. Before this they were kept ordered.
            train_indices = np.arange(len(X_train))
            np.random.shuffle(train_indices)
            X_train = X_train[train_indices]
            y_train = y_train[train_indices]

            test_indices = np.arange(len(X_test))
            np.random.shuffle(test_indices)
            X_test = X_test[test_indices]
            y_test = y_test[test_indices]

            # Merge for cross-validation
            X = np.concatenate((X_train, X_test), axis=0)
            y = np.concatenate((y_train, y_test), axis=0)

            input_shape = (sequence_length, num_features)
            fold_metrics = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                print(f"Class distribution in train fold {fold+1}:", Counter(y[train_idx]))
                print(f"Class distribution in validation fold {fold+1}:", Counter(y[val_idx]))
                console.print(f"[bold green]Training fold {fold + 1}/{n_splits} for {base_name}[/]")
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                print(X_train_fold.shape)
                print(y_train_fold)
                
                train_loader, val_loader = create_dataloaders(X_train_fold, y_train_fold, X_val_fold, y_val_fold, batch_size)
                train_batch = next(iter(train_loader))
                val_batch = next(iter(val_loader))
                console.print(f"Train Loader Batch Shape: {train_batch[0].shape}, Labels Shape: {train_batch[1].shape}")
                console.print(f"Val Loader Batch Shape: {val_batch[0].shape}, Labels Shape: {val_batch[1].shape}")

                
                model = ResNet(num_class=num_classes)
                model.to(device)  # Move the model to the appropriate device
                
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                
                num_epochs = 40
                for epoch in range(num_epochs):
                    train_loss = train_model(model, train_loader, criterion, optimizer, device)
                    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
                    console.print(f"Fold {fold + 1} Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

                fold_metrics.append((val_loss, val_accuracy))
            
            avg_loss = np.mean([metric[0] for metric in fold_metrics])
            avg_accuracy = np.mean([metric[1] for metric in fold_metrics])
            metrics_summary.append((base_name, avg_loss, avg_accuracy))
            console.print(f"[bold blue]Finished training {base_name}: Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}[/]")

            # Save processed file name to track
            with open(processed_file_tracker, "a") as file:
                file.write(f"{base_name}\n")
                
    # Save metrics summary to CSV
    df_metrics_summary = pd.DataFrame(metrics_summary, columns=['Base Name', 'Avg Loss', 'Avg Accuracy'])
    csv_path = os.path.join(results_path_ltn, 'ResNet.csv')
    df_metrics_summary.to_csv(csv_path, index=False)
    
    console.print("[bold yellow]Summary of all base names and their metrics[/]")
    for base_name, avg_loss, avg_accuracy in metrics_summary:
        console.print(f"{base_name}: Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")

