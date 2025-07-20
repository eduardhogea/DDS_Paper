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
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange
import ltn

from sequence_generation import load_sequences

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "..", "config"))

import config_revision_data as config

results_path_ltn = config.results_path_ltn
sequence_length = config.sequence_length
num_features = config.num_features
batch_size = config.batch_size
epochs = config.epochs
learning_rate = config.learning_rate
num_classes = config.num_classes
sequences_directory = "/home/ubuntu/dds_paper/DDS_Paper/data_revision/csvs/sequences"
model_save_directory = "/home/ubuntu/dds_paper/DDS_Paper/model_weights_cluj_spectra"
os.makedirs(model_save_directory, exist_ok=True)


class Feed_Forward(nn.Module):
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
        B, T, H = x.size()
        x = x.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        return x
    def forward(self, hidden_states):
        Q = self._transpose(self.fc_query(hidden_states))
        K = self._transpose(self.fc_key(hidden_states))
        V = self._transpose(self.fc_value(hidden_states))
        context, weights = self.attention(Q, K, V)
        context = context.permute(0, 2, 1, 3).contiguous().view(hidden_states.size())
        out = self.dropout(self.fc_out(context)) + hidden_states
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
    def embed(self, x):
        B = x.size(0)
        x = self.patch_embed(x).transpose(1, 2)
        cls_tok = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tok, x], dim=1)
        x, _ = self.encoder(x)
        return x[:, 0]
    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x).transpose(1, 2)
        cls_tok = self.cls_token.expand(B, -1, -1)
        x = self.dropout(torch.cat([cls_tok, x], dim=1))
        x, attn = self.encoder(x)
        if self.classifier == "token":
            logits = self.head(x[:, 0])
        else:
            feats = x[:, 1:].transpose(1, 2)
            feats = torch.cat([self.avg_pool(feats), self.max_pool(feats)], dim=-1)
            logits = self.head(feats.flatten(start_dim=1))
        return logits, attn


class ClassPredicate(nn.Module):
    def __init__(self, base_model: VisionTransformer, class_idx: int):
        super().__init__()
        self.base = base_model
        self.class_idx = class_idx
    def forward(self, x):
        logits, _ = self.base(x)
        probs = F.softmax(logits, dim=1)
        return probs[:, self.class_idx].unsqueeze(-1)


class SimilarityPredicate(nn.Module):
    def __init__(self, base_model: VisionTransformer, centroid: np.ndarray):
        super().__init__()
        self.base = base_model
        self.register_buffer("centroid", torch.tensor(centroid, dtype=torch.float32))
    def forward(self, x):
        embed = self.base.embed(x)
        dist = torch.norm(embed - self.centroid.to(x.device), dim=1, keepdim=True)
        return torch.exp(-0.5 * dist)


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


def recompute_centroids(model, loader, device):
    embeddings = [[] for _ in range(num_classes)]
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            emb = model.embed(x).cpu().numpy()
            for cls in range(num_classes):
                mask = (y.numpy() == cls)
                if mask.any():
                    embeddings[cls].append(emb[mask])
    centroids = {}
    for cls in range(num_classes):
        if embeddings[cls]:
            data = np.concatenate(embeddings[cls], axis=0)
            n_clusters = min(2, data.shape[0])
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(data)
            centroids[cls] = kmeans.cluster_centers_
    return centroids


def build_similarity_predicates(model, centroids):
    preds = [[] for _ in range(num_classes)]
    for cls, c_arr in centroids.items():
        for c in c_arr:
            preds[cls].append(ltn.Predicate(SimilarityPredicate(model, c)))
    return preds


def build_rules(x, y, predicates, sim_preds, Not, Forall, Implies):
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
    if any(any(lst) for lst in sim_preds):
        x_all = ltn.Variable("x_all", x)
        for c, plist in enumerate(sim_preds):
            for S_c in plist:
                rules.append(Forall(x_all, Implies(S_c(x_all), predicates[c](x_all))))
    return rules


def compute_sat_level(loader, predicates, sim_preds, Not, Forall, Implies, SatAgg, device):
    total_sat = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        rules = build_rules(x, y, predicates, sim_preds, Not, Forall, Implies)
        sat = SatAgg(*rules)
        total_sat += sat.item()
    return total_sat / len(loader)


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


def main():
    import matplotlib.pyplot as plt

    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------
    # Load data by iterating over all sequence files in sequences_directory
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    # Expect files named like "<base>_train_scaled_sequences.npy" and "<base>_train_scaled_labels.npy"
    # and corresponding "<base>_test_scaled_sequences.npy" and "<base>_test_scaled_labels.npy".
    for file in sorted(os.listdir(sequences_directory)):
        if file.endswith("_train_scaled_sequences.npy"):
            base_name = file.replace("_train_scaled_sequences.npy", "")
            train_seq_path = os.path.join(sequences_directory, f"{base_name}_train_scaled_sequences.npy")
            train_label_path = os.path.join(sequences_directory, f"{base_name}_train_scaled_labels.npy")
            test_seq_path = os.path.join(sequences_directory, f"{base_name}_test_scaled_sequences.npy")
            test_label_path = os.path.join(sequences_directory, f"{base_name}_test_scaled_labels.npy")

            # Ensure both train and test files exist for this base
            if os.path.exists(train_seq_path) and os.path.exists(train_label_path) and \
               os.path.exists(test_seq_path) and os.path.exists(test_label_path):

                # Load train split
                X_tr, y_tr = load_sequences(train_seq_path, train_label_path)
                # Shuffle train portion
                train_idx = np.arange(len(X_tr))
                np.random.shuffle(train_idx)
                X_tr, y_tr = X_tr[train_idx], y_tr[train_idx]
                X_train_list.append(X_tr)
                y_train_list.append(y_tr)

                # Load test split
                X_te, y_te = load_sequences(test_seq_path, test_label_path)
                # Shuffle test portion
                test_idx = np.arange(len(X_te))
                np.random.shuffle(test_idx)
                X_te, y_te = X_te[test_idx], y_te[test_idx]
                X_test_list.append(X_te)
                y_test_list.append(y_te)

    # Concatenate all collected train and test arrays
    if X_train_list and X_test_list:
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
    else:
        print(f"[ERROR] sequences_directory path: {sequences_directory}")

        raise RuntimeError("No sequence files found in sequences_directory with the expected naming convention.")

    console.print(f"Loaded: X_train {X_train.shape}, X_test {X_test.shape}")
    # -------------------------------------------------------------------

    # prepare output directories
    os.makedirs(results_path_ltn, exist_ok=True)
    plots_dir   = os.path.join(results_path_ltn, "plots_cluj")
    metrics_dir = os.path.join(results_path_ltn, "metrics_cluj")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # set up LTN
    Not     = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    Forall  = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesGoguen())
    SatAgg  = ltn.fuzzy_ops.SatAgg()

    # create dataloaders
    train_loader, val_loader = create_dataloaders(X_train, y_train, X_test, y_test)
    emb_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32).permute(0,2,1),
            torch.tensor(y_train, dtype=torch.long)
        ),
        batch_size=batch_size, shuffle=False
    )

    # instantiate model, optimizer, predicates
    model     = VisionTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    predicates = [ltn.Predicate(ClassPredicate(model, c)) for c in range(num_classes)]
    centroids  = recompute_centroids(model, emb_loader, device)
    sim_preds  = build_similarity_predicates(model, centroids)

    best_acc  = 0.0
    best_path = os.path.join(model_save_directory, "best_model.pt")

    epochs_list    = []
    train_sat_list = []
    val_sat_list   = []
    val_acc_list   = []

    # training loop
    for epoch in trange(1, epochs+1, desc="Training"):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            rules = build_rules(xb, yb, predicates, sim_preds, Not, Forall, Implies)
            sat   = SatAgg(*rules)
            loss  = 1.0 - sat
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)

        train_sat = running / len(train_loader.dataset)
        val_sat   = compute_sat_level(val_loader, predicates, sim_preds, Not, Forall, Implies, SatAgg, device)
        val_acc   = compute_accuracy(val_loader, model, device)

        epochs_list.append(epoch)
        train_sat_list.append(train_sat)
        val_sat_list.append(val_sat)
        val_acc_list.append(val_acc)

        console.print(f"Epoch {epoch:02d} • "
                      f"train_sat={train_sat:.4f} • "
                      f"val_sat={val_sat:.4f} • "
                      f"val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            console.log(f"Saved best (acc={best_acc:.4f})")

        centroids = recompute_centroids(model, emb_loader, device)
        sim_preds  = build_similarity_predicates(model, centroids)

    # save per‐epoch metrics to CSV
    metrics_df = pd.DataFrame({
        'epoch':       epochs_list,
        'train_sat':   train_sat_list,
        'val_sat':     val_sat_list,
        'val_acc':     val_acc_list,
        'train_loss':  [1 - s for s in train_sat_list],
        'val_loss':    [1 - s for s in val_sat_list],
    })
    metrics_csv = os.path.join(metrics_dir, "metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    # plot & save loss
    plt.figure()
    plt.plot(epochs_list, metrics_df['train_loss'], label="train_loss")
    plt.plot(epochs_list, metrics_df['val_loss'],   label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "loss.png"), dpi=400)
    plt.close()

    # plot & save accuracy
    plt.figure()
    plt.plot(epochs_list, val_acc_list, label="val_accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "accuracy.png"), dpi=400)
    plt.close()

    # final evaluation on best model
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits, _ = model(xb.to(device))
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(yb.numpy())

    console.print(classification_report(labels, preds))
    report_df = pd.DataFrame(classification_report(labels, preds, output_dict=True)).T
    report_df.to_csv(os.path.join(results_path_ltn, "classification_report.csv"))

    console.print(f"Reports, metrics, and plots saved under:\n"
                  f"Metrics CSVs: {metrics_dir}\n"
                  f"Plots: {plots_dir}\n"
                  f"Summary CSV: {results_path_ltn}/classification_report.csv")
    console.print("Done!")


if __name__ == "__main__":
    main()