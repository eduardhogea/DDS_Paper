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
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange
import ltn
import random

from sequence_generation import load_sequences

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "..", "config"))

SEED = 1
random.seed(SEED)

# 2) Hash seed (so e.g. dict ordering is fixed)
os.environ['PYTHONHASHSEED'] = str(SEED)

# 3) NumPy RNG
np.random.seed(SEED)

# 4) PyTorch RNG
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 5) Force deterministic CuDNN (may slow you down)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

def seed_worker(worker_id):
    # seed numpy in each worker from the global torch seed
    np.random.seed(torch.initial_seed() % (2**32))


import config_uoc as config

results_path_ltn = config.results_path_ltn
sequence_length = config.sequence_length
num_features = config.num_features
batch_size = config.batch_size
epochs = config.epochs
learning_rate = config.learning_rate
num_classes = config.num_classes
merge_cosine_threshold = config.rule_merge_tau
support_min = config.rule_support_min
n_cluster = config.n_cluster
patience = config.patience

sequences_directory = "data_uoc/output_sequences"
model_save_directory = "model_weights_cluj"
os.makedirs(model_save_directory, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap  = ".".join(map(str, torch.cuda.get_device_capability(0)))
        print(f"[CUDA] Using {name} (compute capability {cap})")
        return torch.device("cuda")
    else:
        print("[CUDA] Not available. Falling back to CPU.")
        return torch.device("cpu")


#get_device()

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

###

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
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available(), num_workers=0 if torch.cuda.is_available() else 0),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available(), num_workers=0 if torch.cuda.is_available() else 0),
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
            n_clusters = min(n_cluster, data.shape[0])
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(data)
            #centroids[cls] = kmeans.cluster_centers_
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            # prune by support_min
            surviving = []
            for idx, center in enumerate(centers):
                if (labels == idx).sum() >= support_min:
                    surviving.append(center)
            # merge by cosine similarity
            final_centroids = []
            for c in surviving:
                if final_centroids:
                    sims = cosine_similarity([c], final_centroids)[0]
                    if sims.max() <= merge_cosine_threshold:
                        #print("------------!!!Check this:")
                        final_centroids.append(c)
                else:
                    final_centroids.append(c)
            centroids[cls] = np.vstack(final_centroids) if final_centroids else np.empty((0, centers.shape[1]))
    return centroids


####

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
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        rules = build_rules(x, y, predicates, sim_preds, Not, Forall, Implies)
        sat = SatAgg(*rules)
        total_sat += sat.item()
    return total_sat / len(loader)


def compute_accuracy(loader, model, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out, _ = model(x)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    import matplotlib.pyplot as plt
    get_device()

    console = Console()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    X_train, y_train = load_sequences(
        os.path.join(sequences_directory, "train_sequences.npy"),
        os.path.join(sequences_directory, "train_labels.npy"),
    )
    X_test, y_test = load_sequences(
        os.path.join(sequences_directory, "test_sequences.npy"),
        os.path.join(sequences_directory, "test_labels.npy"),
    )
    console.print(f"Loaded: X_train {X_train.shape}, X_test {X_test.shape}")

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
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #scaler    = torch.cuda.amp.GradScaler()
    scaler    = torch.cuda.amp.GradScaler() if device.type=="cuda" else None
    predicates = [ltn.Predicate(ClassPredicate(model, c)) for c in range(num_classes)]
    centroids  = recompute_centroids(model, emb_loader, device)
    sim_preds  = build_similarity_predicates(model, centroids)

    best_acc  = 0.0
    best_path = os.path.join(model_save_directory, "best_model.pt")

    no_improve_epochs  = 0
    best_val_loss      = float("inf")

    epochs_list    = []
    train_sat_list = []
    val_sat_list   = []
    val_acc_list   = []

    # training loop
    for epoch in trange(1, epochs+1, desc="Training"):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    rules = build_rules(xb, yb, predicates, sim_preds, Not, Forall, Implies)
                    sat   = SatAgg(*rules)
                    loss  = 1.0 - sat

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                rules = build_rules(xb, yb, predicates, sim_preds, Not, Forall, Implies)
                sat   = SatAgg(*rules)
                loss  = 1.0 - sat
                loss.backward()
                optimizer.step()

            running += loss.item() * xb.size(0)

        train_sat = running / len(train_loader.dataset)
        val_sat   = compute_sat_level(val_loader, predicates, sim_preds, Not, Forall, Implies, SatAgg, device)
        val_acc   = compute_accuracy(val_loader, model, device)

        # compute "validation loss" from sat (or use 1 - val_sat)
        val_loss = 1.0 - val_sat
        
        epochs_list.append(epoch)
        train_sat_list.append(train_sat)
        val_sat_list.append(val_sat)
        val_acc_list.append(val_acc)

        # Early‐stopping logic
        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            console.log(f"Early stopping after {patience} epochs without improvement")
            break



        console.print(f"Epoch {epoch:02d} • "
                      f"train_sat={train_sat:.4f} • "
                      f"val_sat={val_sat:.4f} • "
                      f"val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            console.log(f"Saved best (acc={best_acc:.4f})")

        # centroids = recompute_centroids(model, emb_loader, device)
        # sim_preds  = build_similarity_predicates(model, centroids)
            
        #if epoch % 5 == 0: #TODO check this
        centroids = recompute_centroids(model, emb_loader, device)
        sim_preds = build_similarity_predicates(model, centroids)

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
            logits, _ = model(xb.to(device, non_blocking=True))
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(yb.cpu().numpy())

    console.print(classification_report(labels, preds, digits=4))
    report_df = pd.DataFrame(classification_report(labels, preds, output_dict=True, digits=4)).T

    report_df = report_df.round(4)
    report_df.to_csv(os.path.join(results_path_ltn, "classification_report.csv"))

    console.print(f"Reports, metrics, and plots saved under:\n"
                  f"Metrics CSVs: {metrics_dir}\n"
                  f"Plots: {plots_dir}\n"
                  f"Summary CSV: {results_path_ltn}/classification_report.csv")
    console.print("Done!")


if __name__ == "__main__":
    main()

