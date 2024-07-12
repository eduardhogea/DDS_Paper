import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# from parameter import parse_option
# args = parse_option()

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

#!!
model_save_directory = "/home/ubuntu/dds_paper/DDS_Paper/model_weights"

class Feed_Forward(nn.Module):
    """
    Feedforward network in encoder
    """
    def __init__(self, hidden_size, mlp_ratio, attn_drop_rate):
        super(Feed_Forward, self).__init__()
        mlp_dim = int(mlp_ratio * hidden_size)
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.activation = F.gelu
        self.dropout = nn.Dropout(attn_drop_rate)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.dropout(out)
        # out = out + x  # 残差连接
        # out = self.layer_norm(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    """
    Scaled Dot-Product Attention
    """
    def __init__(self, attn_dropout=0.1):
        super(Scaled_Dot_Product_Attention, self).__init__()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, Q, K, V):
        attention = torch.matmul(Q, K.transpose(-1, -2))
        scale = K.size(-1) ** -0.5
        attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        weights = attention
        output = torch.matmul(attention, V)
        return output, weights


class DenseAttention(nn.Module):

    def __init__(self, max_seq_len, d_k, d_hid=32, attn_dropout=0.1):
        super(DenseAttention, self).__init__()
        self.w_1 = nn.Linear(d_k, d_hid)
        self.w_2 = nn.Linear(d_hid, max_seq_len)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, Q, K, V, len_q):
        dense_attn = self.w_2(self.relu(self.w_1(Q)))[:, :, :, :len_q]
        dense_attn = F.softmax(dense_attn, dim=-1)
        weights = dense_attn
        output = torch.matmul(dense_attn, V)
        return output, weights


class RandomAttention(nn.Module):

    def __init__(self, batch_size, n_head, max_seq_len, attn_dropout=0.1):
        super(RandomAttention, self).__init__()
        self.random_attn = torch.randn(batch_size, n_head, max_seq_len, max_seq_len, requires_grad=True)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, Q, K, V, len_q):
        random_attn = self.random_attn[:Q.size()[0], :, :len_q, :len_q]
       #  random_attn = random_attn.to(torch.device("cuda"))
        random_attn = F.softmax(random_attn, dim=-1)
        weights = random_attn
        output = torch.matmul(random_attn, V)
        return output, weights


class Multi_Head_Attention(nn.Module):

    def __init__(self, hidden_size, num_heads, drop_rate, attention_choice):
        super(Multi_Head_Attention, self).__init__()
        self.num_attention_heads = num_heads
        self.head_dim = int(hidden_size / num_heads)
        self.all_head_size = self.head_dim * self.num_attention_heads
        self.fc_query = nn.Linear(hidden_size, self.all_head_size)
        self.fc_key = nn.Linear(hidden_size, self.all_head_size)
        self.fc_value = nn.Linear(hidden_size, self.all_head_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(drop_rate)
        self.attention_choice = attention_choice
        if self.attention_choice == "dot":
            self.attention = Scaled_Dot_Product_Attention()  # 点积Attention
        elif self.attention_choice == "dense":
            self.attention = DenseAttention(max_seq_len=40, d_k=self.head_dim, d_hid=32)
        elif self.attention_choice == "random":
            self.attention = RandomAttention(batch_size=1, n_head=num_heads, max_seq_len=40)
        else:
            self.attention = None
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.fc_query(hidden_states)
        mixed_key_layer = self.fc_key(hidden_states)
        mixed_value_layer = self.fc_value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        if self.attention_choice == "dot":
            attention_scores, weights = self.attention(query_layer, key_layer, value_layer)
        elif self.attention_choice == "dense":
            attention_scores, weights = self.attention(query_layer, key_layer, value_layer, len_q=query_layer.size()[-2])
        elif self.attention_choice == "random":
            attention_scores, weights = self.attention(query_layer, key_layer, value_layer, len_q=query_layer.size()[-2])
        else:
            attention_scores, weights = None, None
        context_layer = attention_scores.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        out = self.fc(context_layer)
        out = self.dropout(out)
        out = out + hidden_states  # residual connection like in ResNet
        out = self.layer_norm(out)
        return out, weights


class Position_Embedding(nn.Module):
    """
    Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, input_channel, signal_size, patch_size, hidden_size, drop_rate):
        super(Position_Embedding, self).__init__()
        seq_size = (1, signal_size)
        n_patches = seq_size[1] // patch_size
        self.patch_embeddings = nn.Conv1d(in_channels=input_channel, out_channels=hidden_size,
                                          kernel_size=patch_size, stride=patch_size // 2)
        self.cls_token = nn.Parameter(torch.rand(1, hidden_size))  # Classification token
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x0 = x.shape[0]
        cls_tokens = self.cls_token.expand(x0, -1, -1)
        x = self.patch_embeddings(x)
        x = x.transpose(-2, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = self.dropout(x)
        return embeddings


class Encoder_Block(nn.Module):
    """
    Block in encoder, including multihead attention and feedforward neural network
    Encoder层里的单元，包括多头注意力层和前馈神经网络
    """
    def __init__(self, hidden_size, mlp_ratio, attn_drop_rate, num_heads, drop_rate, drop_path_rate, attention_choice):
        super(Encoder_Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)  # Normalization layer before multihead attentiuon
        self.attention = Multi_Head_Attention( hidden_size, num_heads, drop_rate, attention_choice)  # multi head attention
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)  # nomalization layer before feed network
        self.ffn = Feed_Forward(hidden_size, mlp_ratio, attn_drop_rate)  # feed network

    def forward(self, x):
        x = self.attention_norm(x)
        x, weights = self.attention(x)
        x = self.ffn_norm(x)
        x = self.ffn(x)
        return x, weights


class Encoder(nn.Module):
    """
    Transformer Encoder 层
    """
    def __init__(self, hidden_size, depth, mlp_ratio, attn_drop_rate, num_heads, drop_rate,
                 drop_path_rate, attention_choice):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        for _ in range(depth):
            layer = Encoder_Block(hidden_size, mlp_ratio, attn_drop_rate, num_heads, drop_rate, drop_path_rate,
                                  attention_choice)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            attn_weights.append(weights)
        # encoded = self.encoder_norm(hidden_states)
        return hidden_states, attn_weights


class Transformer(nn.Module):
    """
    Transformer layer
    """
    def __init__(self, input_channel, signal_size, patch_size, hidden_size, drop_rate, depth, mlp_ratio, attn_drop_rate, num_heads,
                 drop_path_rate, attention_choice):
        super(Transformer, self).__init__()
        self.embeddings = Position_Embedding(input_channel, signal_size, patch_size, hidden_size, drop_rate)  # 将序列编码
        self.encoder = Encoder(hidden_size, depth, mlp_ratio, attn_drop_rate, num_heads, drop_rate, drop_path_rate,
                               attention_choice)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Module):
    """
    Transformer for classification
    """
    def __init__(self, input_channel=20, signal_size=1000, patch_size=10, num_class=12, hidden_size=256, depth=3,
                 attention_choice="dot", num_heads=8, mlp_ratio=4., drop_rate=0.2, attn_drop_rate=0.2, drop_path_rate=0.2, classifier="gap"):
        super(VisionTransformer, self).__init__()
        self.num_class = num_class
        self.transformer = Transformer(input_channel, signal_size, patch_size, hidden_size, drop_rate,
                                       depth, mlp_ratio, attn_drop_rate, num_heads, drop_path_rate, attention_choice)
        self.linear_dim = int(hidden_size / 2)
        self.classifier = classifier
        self.avg_pool = nn.AdaptiveAvgPool1d(4)
        self.max_pool = nn.AdaptiveMaxPool1d(4)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 8, 256),  # Change from 256 * 8 to hidden_size * 8
            nn.ReLU(),
            nn.Linear(256, num_class)
        )

    def forward(self, x):
        hidden_state, attn_weights = self.transformer(x)
        if self.classifier == "token":
            cls_head = hidden_state[:, 0]
            x_logits = self.head(cls_head)
        elif self.classifier == "gap":
            features = hidden_state[:, 1:]
            features = features.transpose(-2, -1)
            avg_features = self.avg_pool(features)
            max_features = self.max_pool(features)
            features = torch.cat([avg_features, max_features], dim=-1)
            features = features.view(features.size()[0], -1)
            x_logits = self.fc(features)
        else:
            raise ValueError("config.classifier is non-existent")
        return x_logits, attn_weights

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, _ = model(inputs)
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

            outputs, _ = model(inputs)
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

                
                model = VisionTransformer(input_channel=num_features, signal_size=sequence_length, num_class=num_classes, attention_choice="dot").to(device)
               
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
    csv_path = os.path.join(results_path_ltn, 'transformer.csv')
    df_metrics_summary.to_csv(csv_path, index=False)
    
    console.print("[bold yellow]Summary of all base names and their metrics[/]")
    for base_name, avg_loss, avg_accuracy in metrics_summary:
        console.print(f"{base_name}: Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")
