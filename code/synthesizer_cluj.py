import copy
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rich.console import Console
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

# LTN imports
import ltn

# Local utilities -----------------------------------------------------------
from sequence_generation import load_sequences  # expects two .npy paths and returns (X, y)

# Add config folder to python path -----------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "..", "config"))

import config_uoc as config  # custom config that already points to the UOC data folder

# --------------------------------------------------------------------------
# Configuration values (taken from config_uoc)
# --------------------------------------------------------------------------
results_path_ltn = config.results_path_ltn
sequence_length = config.sequence_length
num_features = config.num_features
batch_size = config.batch_size
epochs = config.epochs
learning_rate = config.learning_rate
n_splits = config.n_splits
num_classes = config.num_classes
sequences_directory = "/home/ubuntu/dds_paper/DDS_Paper/data_uoc/output_sequences"
model_save_directory = "/home/ubuntu/dds_paper/DDS_Paper/model_weights"

# --------------------------------------------------------------------------
# Model definitions (unchanged)
# --------------------------------------------------------------------------
class Feed_Forward(nn.Module):
    """Feed‑forward network used inside the Transformer encoder."""

    def __init__(self, hidden_size: int, mlp_ratio: float, attn_drop_rate: float):
        super().__init__()
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
        x = self.activation(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return x


class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self, attn_dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, Q, K, V):
        scale = K.size(-1) ** -0.5
        attention = torch.matmul(Q, K.transpose(-1, -2)) * scale
        attention = F.softmax(attention, dim=-1)
        return torch.matmul(self.dropout(attention), V), attention


class Multi_Head_Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, drop_rate: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.fc_query = nn.Linear(hidden_size, hidden_size)
        self.fc_key = nn.Linear(hidden_size, hidden_size)
        self.fc_value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(drop_rate)
        self.attention = Scaled_Dot_Product_Attention()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def _transpose(self, x):
        # (B, T, H) -> (B, heads, T, head_dim)
        B, T, H = x.size()
        x = x.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        return x

    def forward(self, hidden_states):
        Q = self._transpose(self.fc_query(hidden_states))
        K = self._transpose(self.fc_key(hidden_states))
        V = self._transpose(self.fc_value(hidden_states))
        context, weights = self.attention(Q, K, V)
        context = context.permute(0, 2, 1, 3).contiguous().view(hidden_states.size())
        out = self.dropout(self.fc_out(context)) + hidden_states  # residual
        return self.layer_norm(out), weights


class Encoder_Block(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float, attn_drop_rate: float, num_heads: int, drop_rate: float):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attention = Multi_Head_Attention(hidden_size, num_heads, drop_rate)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Feed_Forward(hidden_size, mlp_ratio, attn_drop_rate)

    def forward(self, x):
        x, weights = self.attention(self.attention_norm(x))
        x = self.ffn_norm(x)
        x = self.ffn(x)
        return x, weights


class Encoder(nn.Module):
    def __init__(self, depth: int, hidden_size: int, mlp_ratio: float, attn_drop_rate: float, num_heads: int, drop_rate: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Encoder_Block(hidden_size, mlp_ratio, attn_drop_rate, num_heads, drop_rate) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x):
        attn_maps = []
        for block in self.blocks:
            x, w = block(x)
            attn_maps.append(w)
        return self.final_norm(x), attn_maps


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_channel: int = num_features,
        signal_size: int = sequence_length,
        patch_size: int = 10,
        num_class: int = num_classes,
        hidden_size: int = 64,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.1,
        attn_drop_rate: float = 0.1,
        classifier: str = "gap",
    ):
        super().__init__()
        self.classifier = classifier
        self.patch_embed = nn.Conv1d(input_channel, hidden_size, kernel_size=patch_size, stride=patch_size // 2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.dropout = nn.Dropout(drop_rate)
        self.encoder = Encoder(depth, hidden_size, mlp_ratio, attn_drop_rate, num_heads, drop_rate)
        self.avg_pool = nn.AdaptiveAvgPool1d(4)
        self.max_pool = nn.AdaptiveMaxPool1d(4)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 8, 256), nn.ReLU(), nn.Linear(256, num_class)
        )

    def forward(self, x):  # x: (B, C, T)
        B = x.size(0)
        x = self.patch_embed(x).transpose(1, 2)  # (B, T', H)
        cls_tok = self.cls_token.expand(B, -1, -1)
        x = self.dropout(torch.cat([cls_tok, x], dim=1))
        x, attn = self.encoder(x)
        if self.classifier == "token":
            logits = self.head(x[:, 0])
        else:  # GAP‑style pooling
            feats = x[:, 1:].transpose(1, 2)  # (B, H, T')
            feats = torch.cat([self.avg_pool(feats), self.max_pool(feats)], dim=-1)
            logits = self.head(feats.flatten(start_dim=1))
        return logits, attn

# --------------------------------------------------------------------------
# LTN Predicate Wrapper
# --------------------------------------------------------------------------
class ClassPredicate(nn.Module):
    """Wraps the VisionTransformer into a 0-1 predicate for a single class."""
    def __init__(self, base_model: VisionTransformer, class_idx: int):
        super().__init__()
        self.base = base_model
        self.class_idx = class_idx

    def forward(self, x):
        logits, _ = self.base(x)
        probs = F.softmax(logits, dim=1)
        return probs[:, self.class_idx].unsqueeze(-1)

# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def create_dataloaders(X_train, y_train, X_val, y_val):
    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )

# LTN satisfaction
def compute_sat_level(loader, predicates, Not, Forall, SatAgg, device):
    total_sat = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        rules = []
        for c, P_c in enumerate(predicates):
            mask_pos = (y == c)
            mask_neg = (y != c)
            if mask_pos.any():
                x_pos = ltn.Variable(f"x_pos_{c}", x[mask_pos])
                rules.append(Forall(x_pos, P_c(x_pos)))
            if mask_neg.any():
                x_neg = ltn.Variable(f"x_neg_{c}", x[mask_neg])
                rules.append(Forall(x_neg, Not(P_c(x_neg))))
        sat = SatAgg(*rules)
        total_sat += sat.item()
    return total_sat / len(loader)

# Standard accuracy
def compute_accuracy(loader, model, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out, _ = model(x)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# --------------------------------------------------------------------------
# Main script
# --------------------------------------------------------------------------
def main():
    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load pre-processed data
    train_seq_path = os.path.join(sequences_directory, "train_sequences.npy")
    train_lbl_path = os.path.join(sequences_directory, "train_labels.npy")
    test_seq_path = os.path.join(sequences_directory, "test_sequences.npy")
    test_lbl_path = os.path.join(sequences_directory, "test_labels.npy")

    X_train, y_train = load_sequences(train_seq_path, train_lbl_path)
    X_test, y_test = load_sequences(test_seq_path, test_lbl_path)

    console.print(f"Loaded: X_train {X_train.shape}, X_test {X_test.shape}")

    # Shuffle both sets the same way
    rng = np.random.default_rng(42)
    train_idx = rng.permutation(len(X_train))
    test_idx = rng.permutation(len(X_test))
    X_train, y_train = X_train[train_idx], y_train[train_idx]
    X_test, y_test = X_test[test_idx], y_test[test_idx]

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    console.print(f"Merged dataset shape: {X.shape}")

    # Cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    # LTN operators
    Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    SatAgg = ltn.fuzzy_ops.SatAgg()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
        console.rule(f"Fold {fold}/{n_splits}")
        console.print(f"Class dist train: {Counter(y[train_idx])}")
        console.print(f"Class dist val  : {Counter(y[val_idx])}")

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        train_loader, val_loader = create_dataloaders(X_tr, y_tr, X_val, y_val)

        # Model + optimizer
        model = VisionTransformer().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Wrap LTN predicates (one per class)
        predicates = [ltn.Predicate(ClassPredicate(model, c)) for c in range(num_classes)]

        # Training loop with LTN loss
        for epoch in range(1, epochs + 1):
            model.train()
            running = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()

                # Ground rules for this batch
                rules = []
                for c, P_c in enumerate(predicates):
                    mask_pos = (y_batch == c)
                    mask_neg = (y_batch != c)
                    if mask_pos.any():
                        x_pos = ltn.Variable(f"x_pos_{c}", x_batch[mask_pos])
                        rules.append(Forall(x_pos, P_c(x_pos)))
                    if mask_neg.any():
                        x_neg = ltn.Variable(f"x_neg_{c}", x_batch[mask_neg])
                        rules.append(Forall(x_neg, Not(P_c(x_neg))))

                sat = SatAgg(*rules)
                loss = 1.0 - sat
                loss.backward()
                optimizer.step()
                running += loss.item() * x_batch.size(0)

            train_sat = running / len(train_loader.dataset)

            # Evaluation on validation fold
            val_sat = compute_sat_level(val_loader, predicates, Not, Forall, SatAgg, device)
            val_acc = compute_accuracy(val_loader, model, device)
            console.print(f"Epoch {epoch:02d}: train_sat {train_sat:.4f} | val_sat {val_sat:.4f} | val_acc {val_acc:.4f}")

        fold_metrics.append((val_sat, val_acc))

    # Summary
    avg_sat = np.mean([m[0] for m in fold_metrics])
    avg_acc = np.mean([m[1] for m in fold_metrics])
    console.rule("Summary")
    console.print(f"Avg Sat: {avg_sat:.4f} | Avg accuracy: {avg_acc:.4f}")

    # Save results to CSV
    df = pd.DataFrame({
        "Fold": list(range(1, n_splits + 1)),
        "Val Sat": [m[0] for m in fold_metrics],
        "Val Acc": [m[1] for m in fold_metrics],
    })
    os.makedirs(results_path_ltn, exist_ok=True)
    results_csv = os.path.join(results_path_ltn, "transformer_uoc_ltn.csv")
    df.to_csv(results_csv, index=False)
    console.print(f"Metrics written to {results_csv}")

if __name__ == "__main__":
    main()
