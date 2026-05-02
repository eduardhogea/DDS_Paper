import os
import sys
import random
from collections import Counter
from typing import Dict, List, Tuple

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
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
import ltn
import multiprocessing as mp

# ---------------------------------------------------------------------
# Config & seeding
# ---------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "..", "config"))
import config as config  # noqa: E402

SEED = 1
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

console = Console()

# Project config
results_root           = config.results_path_ltn
sequence_length        = config.sequence_length
batch_size             = config.batch_size
epochs                 = config.epochs
learning_rate          = config.learning_rate
num_classes            = config.num_classes
merge_cosine_threshold = config.rule_merge_tau
support_min            = config.rule_support_min
n_cluster              = config.n_cluster
patience               = config.patience
n_splits               = getattr(config, "n_splits", 2)
bases_limit            = getattr(config, "bases_limit", None)

sequences_directory    = config.sequences_directory

model_save_directory   = "model_weights_cluj"
os.makedirs(model_save_directory, exist_ok=True)

# ================= DRIFT EXPERIMENT ADDITIONS =================
# Number of epochs per stage in the sequential drift experiment.
# Defaults to the same value as 'epochs' if not set in config.
drift_epochs_per_stage = getattr(config, "drift_epochs_per_stage", epochs)
RUN_DRIFT_EXPERIMENT   = True
# ===============================================================

from sequence_generation import load_sequences  # noqa: E402


def seed_worker(_):
    np.random.seed(torch.initial_seed() % (2**32))


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
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
        input_channel: int,
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
        assert signal_size >= patch_size, f"sequence_length ({signal_size}) must be >= patch_size ({patch_size})"
        self.classifier = classifier
        self.patch_embed = nn.Conv1d(input_channel, hidden_size, kernel_size=patch_size, stride=patch_size // 2)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.dropout = nn.Dropout(drop_rate)
        self.encoder = Encoder(depth, hidden_size, mlp_ratio, attn_drop_rate, num_heads, drop_rate)
        self.avg_pool = nn.AdaptiveAvgPool1d(4)
        self.max_pool = nn.AdaptiveMaxPool1d(4)
        self.head = nn.Sequential(nn.Linear(hidden_size * 8, 256), nn.ReLU(), nn.Linear(256, num_class))

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


# ---------------------------------------------------------------------
# LTN predicates & rules
# ---------------------------------------------------------------------
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


def _suggest_num_workers(default: int = 0) -> int:
    """Heuristic: use a few CPU workers to speed data loading without changing training semantics."""
    try:
        cores = mp.cpu_count()
    except NotImplementedError:
        return default
    if cores is None or cores <= 2:
        return default
    return min(8, cores)  # cap to avoid oversubscribing


def create_dataloaders(X_train, y_train, X_val, y_val, seed=SEED):
    g = torch.Generator()
    g.manual_seed(seed)

    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    X_val   = torch.tensor(X_val,   dtype=torch.float32).permute(0, 2, 1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val   = torch.tensor(y_val,   dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val,   y_val)

    pin = torch.cuda.is_available()
    workers = _suggest_num_workers()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        generator=g, worker_init_fn=seed_worker, pin_memory=pin,
        num_workers=workers, persistent_workers=workers > 0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin,
        num_workers=workers, persistent_workers=workers > 0
    )
    return train_loader, val_loader


def recompute_centroids(model, loader, device):
    embeddings = [[] for _ in range(num_classes)]
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            emb = model.embed(x).cpu().numpy()
            y_np = y.numpy()
            for cls in range(num_classes):
                mask = (y_np == cls)
                if mask.any():
                    embeddings[cls].append(emb[mask])
    centroids: Dict[int, np.ndarray] = {}
    for cls in range(num_classes):
        if embeddings[cls]:
            data = np.concatenate(embeddings[cls], axis=0)
            n_clusters = min(n_cluster, data.shape[0])
            kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10).fit(data)
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            # prune by support
            surviving = [centers[i] for i in range(len(centers)) if (labels == i).sum() >= support_min]
            # merge by cosine sim
            final_centroids: List[np.ndarray] = []
            for c in surviving:
                if final_centroids:
                    sims = cosine_similarity([c], final_centroids)[0]
                    if sims.max() <= merge_cosine_threshold:
                        final_centroids.append(c)
                else:
                    final_centroids.append(c)
            centroids[cls] = np.vstack(final_centroids) if final_centroids else np.empty((0, centers.shape[1]))
        else:
            centroids[cls] = np.empty((0, 1))
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


# ---------------------------------------------------------------------
# Dataset discovery: per-base (speed/load) evaluation
# ---------------------------------------------------------------------
def list_bases(root: str, prefer_scaled: bool = True) -> List[Tuple[str, bool]]:
    """
    Returns list of (base_name, use_scaled) tuples.
    A base is detected if it has train/test sequences and labels.
    If prefer_scaled=True and scaled files exist, they are used; else fall back to unscaled.
    """
    files = os.listdir(root) if os.path.isdir(root) else []
    scaled_suffixes = [
        "_train_scaled_sequences.npy", "_train_scaled_labels.npy",
        "_test_scaled_sequences.npy",  "_test_scaled_labels.npy",
    ]
    raw_suffixes = [
        "_train_sequences.npy", "_train_labels.npy",
        "_test_sequences.npy",  "_test_labels.npy",
    ]

    # collect candidate bases
    bases = set()
    for f in files:
        for suf in (scaled_suffixes + raw_suffixes):
            if f.endswith(suf):
                base = f[:-len(suf)]
                bases.add(base)
                break

    out: List[Tuple[str, bool]] = []
    for base in sorted(bases):
        scaled_exist = all(os.path.exists(os.path.join(root, base + suf)) for suf in scaled_suffixes)
        raw_exist    = all(os.path.exists(os.path.join(root, base + suf)) for suf in raw_suffixes)
        if prefer_scaled and scaled_exist:
            out.append((base, True))
        elif raw_exist:
            out.append((base, False))
        elif scaled_exist:
            out.append((base, True))
        # else: ignore incomplete bases
    return out


def load_base(root: str, base: str, use_scaled: bool):
    if use_scaled:
        X_tr, y_tr = load_sequences(
            os.path.join(root, f"{base}_train_scaled_sequences.npy"),
            os.path.join(root, f"{base}_train_scaled_labels.npy"),
        )
        X_te, y_te = load_sequences(
            os.path.join(root, f"{base}_test_scaled_sequences.npy"),
            os.path.join(root, f"{base}_test_scaled_labels.npy"),
        )
    else:
        X_tr, y_tr = load_sequences(
            os.path.join(root, f"{base}_train_sequences.npy"),
            os.path.join(root, f"{base}_train_labels.npy"),
        )
        X_te, y_te = load_sequences(
            os.path.join(root, f"{base}_test_sequences.npy"),
            os.path.join(root, f"{base}_test_labels.npy"),
        )
    return X_tr, y_tr, X_te, y_te


# ================= DRIFT EXPERIMENT ADDITIONS =================
def run_drift_experiment(bases, device, Not, Forall, Implies, SatAgg):
    """
    Sequential drift experiment:
    - One single model.
    - Stages: sequentially add bases in sorted order.
    - At each stage, train on the new base only, but recompute centroids on the
      union of all seen bases so far.
    - Logs per-epoch rule counts and per-stage test accuracy on all seen bases.
    """
    if len(bases) < 2:
        console.rule("[magenta]Sequential drift experiment[/]")
        console.print("[yellow]Need at least 2 bases for drift; skipping.[/]")
        return

    console.rule("[magenta]Sequential drift experiment across bases[/]")

    drift_out_dir = os.path.join(results_root, "drift_experiment")
    os.makedirs(drift_out_dir, exist_ok=True)

    # fixed order of bases defines the drift path
    bases_sorted = sorted(bases, key=lambda x: x[0])

    # load data for all bases
    base_data = {}
    for base, use_scaled in bases_sorted:
        X_train, y_train, X_test, y_test = load_base(sequences_directory, base, use_scaled)
        base_data[base] = {
            "use_scaled": use_scaled,
            "X_train_full": X_train,
            "y_train_full": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }
        console.print(
            f"Drift data loaded for base [bold]{base}[/] "
            f"(scaled={use_scaled}) → X_train {X_train.shape}, X_test {X_test.shape}"
        )

    # assume same temporal length & feature dimension across bases
    first_base = bases_sorted[0][0]
    X0 = base_data[first_base]["X_train_full"]
    assert X0.ndim == 3, f"Expected (N, T, F), got {X0.shape}"
    T0, Fdim = X0.shape[1], X0.shape[2]
    assert T0 >= 10, "sequence_length must be >= 10 (patch_size) for Conv1d"

    # single model + optimizer + scaler
    model = VisionTransformer(input_channel=Fdim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # LTN predicates (classification predicates are fixed; similarity predicates will be rebuilt)
    predicates = [ltn.Predicate(ClassPredicate(model, c)) for c in range(num_classes)]
    sim_preds = [[] for _ in range(num_classes)]

    # helper: train/val split for each base
    def make_train_val(X, y, val_ratio=0.2):
        n = X.shape[0]
        n_val = max(1, int(val_ratio * n))
        rng = np.random.RandomState(SEED)
        idx = np.arange(n)
        rng.shuffle(idx)
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]

    # logs
    drift_metrics_rows = []        # per-epoch metrics
    stage_perf_rows = []           # per-stage test accuracy on each seen base
    stage_rule_stats_rows = []     # per-stage rule stats
    global_epoch = 0
    seen_bases = []

    # generator for emb_loader (deterministic)
    g = torch.Generator()
    g.manual_seed(SEED)
    loader_workers = _suggest_num_workers()

    import matplotlib.pyplot as plt  # local import is fine

    for stage_idx, (base, use_scaled) in enumerate(bases_sorted, start=1):
        console.rule(f"[magenta]Drift stage {stage_idx}: add base {base} (scaled={use_scaled})[/]")

        # train/val split for this base (once)
        if "X_train" not in base_data[base]:
            X_train_full = base_data[base]["X_train_full"]
            y_train_full = base_data[base]["y_train_full"]
            X_tr, y_tr, X_val, y_val = make_train_val(X_train_full, y_train_full, val_ratio=0.2)
            base_data[base]["X_train"] = X_tr
            base_data[base]["y_train"] = y_tr
            base_data[base]["X_val"] = X_val
            base_data[base]["y_val"] = y_val
        else:
            X_tr = base_data[base]["X_train"]
            y_tr = base_data[base]["y_train"]
            X_val = base_data[base]["X_val"]
            y_val = base_data[base]["y_val"]

        seen_bases.append(base)

        console.print(
            f"  Stage {stage_idx}: train on base {base} → "
            f"X_train {X_tr.shape}, X_val {X_val.shape}; "
            f"seen_bases={seen_bases}"
        )

        # DataLoaders for current base
        train_loader, val_loader = create_dataloaders(X_tr, y_tr, X_val, y_val)

        # Embedding loader built on union of train splits for all seen bases
        X_seen_list, y_seen_list = [], []
        for b_seen in seen_bases:
            X_seen_list.append(base_data[b_seen]["X_train"])
            y_seen_list.append(base_data[b_seen]["y_train"])
        X_seen = np.concatenate(X_seen_list, axis=0)
        y_seen = np.concatenate(y_seen_list, axis=0)

        emb_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_seen, dtype=torch.float32).permute(0, 2, 1),
                torch.tensor(y_seen, dtype=torch.long),
            ),
            batch_size=batch_size,
            shuffle=False,
            generator=g,
            worker_init_fn=seed_worker,
            num_workers=loader_workers,
            persistent_workers=loader_workers > 0,
        )

        # per-stage rule counts
        stage_rule_counts = []

        # train for a fixed number of epochs in this stage
        for local_epoch in range(1, drift_epochs_per_stage + 1):
            model.train()
            running_loss = 0.0

            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        rules_batch = build_rules(xb, yb, predicates, sim_preds, Not, Forall, Implies)
                        sat = SatAgg(*rules_batch)
                        loss = 1.0 - sat
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    rules_batch = build_rules(xb, yb, predicates, sim_preds, Not, Forall, Implies)
                    sat = SatAgg(*rules_batch)
                    loss = 1.0 - sat
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * xb.size(0)

            train_loss = running_loss / len(train_loader.dataset)
            train_sat = 1.0 - train_loss

            # validation on current base
            model.eval()
            val_sat = compute_sat_level(val_loader, predicates, sim_preds, Not, Forall, Implies, SatAgg, device)
            val_acc = compute_accuracy(val_loader, model, device)
            val_loss = 1.0 - val_sat

            # recompute centroids & similarity predicates using all seen bases
            centroids = recompute_centroids(model, emb_loader, device)
            sim_preds = build_similarity_predicates(model, centroids)
            num_sim_rules = sum(len(lst) for lst in sim_preds)
            stage_rule_counts.append(num_sim_rules)

            global_epoch += 1

            console.print(
                f"[magenta]Stage {stage_idx} / Base {base}[/] "
                f"Epoch {local_epoch:02d} (global {global_epoch:03d}) • "
                f"train_sat={train_sat:.4f} • val_sat={val_sat:.4f} • "
                f"val_acc={val_acc:.4f} • num_rules={num_sim_rules}"
            )

            drift_metrics_rows.append(
                {
                    "stage": stage_idx,
                    "base": base,
                    "global_epoch": global_epoch,
                    "local_epoch": local_epoch,
                    "train_sat": train_sat,
                    "val_sat": val_sat,
                    "val_acc": val_acc,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "num_similarity_rules": num_sim_rules,
                }
            )

        # per-stage rule statistics
        num_epochs_stage = len(stage_rule_counts)
        max_rules = int(max(stage_rule_counts))
        min_rules = int(min(stage_rule_counts))
        final_rules = int(stage_rule_counts[-1])
        mean_rules = float(np.mean(stage_rule_counts))
        std_rules = float(np.std(stage_rule_counts))
        k = max(1, int(0.2 * num_epochs_stage))
        stable_segment = stage_rule_counts[-k:]
        stable_mean_rules = float(np.mean(stable_segment))
        stable_std_rules = float(np.std(stable_segment))

        stage_rule_stats_rows.append(
            {
                "stage": stage_idx,
                "added_base": base,
                "epochs_trained_this_stage": num_epochs_stage,
                "max_num_rules": max_rules,
                "min_num_rules": min_rules,
                "final_num_rules": final_rules,
                "mean_num_rules": mean_rules,
                "std_num_rules": std_rules,
                "mean_num_rules_last_20pct": stable_mean_rules,
                "std_num_rules_last_20pct": stable_std_rules,
            }
        )

        console.print(
            f"[bold]Drift stage {stage_idx} summary[/] "
            f"final_rules={final_rules}  stable_mean_rules={stable_mean_rules:.1f}"
        )

        # test accuracy on all seen bases so far
        for test_base in seen_bases:
            X_te = base_data[test_base]["X_test"]
            y_te = base_data[test_base]["y_test"]
            test_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_te, dtype=torch.float32).permute(0, 2, 1),
                    torch.tensor(y_te, dtype=torch.long),
                ),
                batch_size=batch_size,
                shuffle=False,
                num_workers=loader_workers,
                persistent_workers=loader_workers > 0,
            )
            test_acc = compute_accuracy(test_loader, model, device)
            stage_perf_rows.append(
                {
                    "stage": stage_idx,
                    "train_bases_seen": "+".join(seen_bases),
                    "test_base": test_base,
                    "test_accuracy": test_acc,
                }
            )
            console.print(
                f"  Test accuracy after stage {stage_idx} on base {test_base}: {test_acc:.4f}"
            )

    # save logs
    drift_metrics_df = pd.DataFrame(drift_metrics_rows)
    drift_metrics_df.to_csv(os.path.join(drift_out_dir, "drift_metrics.csv"), index=False)

    stage_perf_df = pd.DataFrame(stage_perf_rows)
    stage_perf_df.to_csv(os.path.join(drift_out_dir, "drift_stage_performance.csv"), index=False)

    stage_rule_stats_df = pd.DataFrame(stage_rule_stats_rows)
    stage_rule_stats_df.to_csv(os.path.join(drift_out_dir, "drift_stage_rule_stats.csv"), index=False)

    # plot: number of rules vs global epoch, colored by stage
    plt.figure()
    for stage_idx in sorted(drift_metrics_df["stage"].unique()):
        df_stage = drift_metrics_df[drift_metrics_df["stage"] == stage_idx]
        label = f"stage {stage_idx} ({df_stage['base'].iloc[0]})"
        plt.plot(df_stage["global_epoch"], df_stage["num_similarity_rules"], label=label)
    plt.xlabel("Global epoch")
    plt.ylabel("Number of similarity rules")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(drift_out_dir, "drift_num_rules_vs_epoch.png"), dpi=400)
    plt.close()

    # stage vs test accuracy per base (line plot)
    pivot = stage_perf_df.pivot(index="stage", columns="test_base", values="test_accuracy")
    pivot.to_csv(os.path.join(drift_out_dir, "drift_stage_vs_base_accuracy.csv"))

    plt.figure()
    for test_base in pivot.columns:
        plt.plot(pivot.index.values, pivot[test_base].values, marker="o", label=f"test base {test_base}")
    plt.xlabel("Stage")
    plt.ylabel("Test accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(drift_out_dir, "drift_accuracy_per_base_vs_stage.png"), dpi=400)
    plt.close()

    console.print(
        f"[green]Sequential drift experiment outputs saved in[/] {drift_out_dir}"
    )
# ===============================================================


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = ".".join(map(str, torch.cuda.get_device_capability(0)))
        console.print(f"[bold green]CUDA[/]: {name} (cc {cap})")
    else:
        console.print("[bold yellow]CUDA not available; using CPU[/]")

    abs_data_dir = os.path.abspath(sequences_directory)
    console.rule("Dataset")
    console.print(f"[bold]Input data folder:[/] {abs_data_dir}")

    os.makedirs(results_root, exist_ok=True)

    # Discover bases (speeds/loads) and which variant (scaled/raw) to use
    bases = list_bases(sequences_directory, prefer_scaled=True)
    if not bases:
        console.print("[red]No complete bases found in data folder.[/]")
        return
    if bases_limit is not None:
        bases = bases[:bases_limit]
        console.print(f"[yellow]Using first {len(bases)} bases due to config.bases_limit={bases_limit}[/]")

    console.print(f"Found [bold]{len(bases)}[/] bases:")
    for base, use_scaled in bases:
        console.print(f" • {base}  (scaled={use_scaled})")

    overall_rows = []  # (base, acc, prec, rec, f1)

    # LTN ops (same across runs)
    Not     = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    Forall  = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesGoguen())
    SatAgg  = ltn.fuzzy_ops.SatAgg()

    # generator for deterministic emb_loader
    g = torch.Generator()
    g.manual_seed(SEED)
    loader_workers = _suggest_num_workers()

    for base, use_scaled in bases:
        console.rule(f"[cyan]Base[/] {base}  (scaled={use_scaled})")

        # Output structure per base
        base_out_dir = os.path.join(results_root, base)
        plots_dir    = os.path.join(base_out_dir, "plots")
        metrics_dir  = os.path.join(base_out_dir, "metrics")
        reports_dir  = os.path.join(base_out_dir, "reports")
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        # Load data for this base
        X_train, y_train, X_test, y_test = load_base(sequences_directory, base, use_scaled)
        console.print(f"Loaded: X_train {X_train.shape}, X_test {X_test.shape}")

        # Merge (like LogicLSTM) then fold
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)

        # Sanity
        assert X.ndim == 3, f"Expected (N, T, F), got {X.shape}"
        T, Fdim = X.shape[1], X.shape[2]
        assert T >= 10, "sequence_length must be >= 10 (patch_size) for Conv1d"

        # Report class dist overall
        class_dist = Counter(y.tolist())
        console.print(f"Class distribution (merged): {dict(class_dist)}")

        fold_rows = []

        kf = KFold(n_splits=n_splits, shuffle=False)
        for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X, y), start=1):
            console.rule(f"Fold {fold_idx}")

            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]

            console.print(f"Train fold shape: {X_tr.shape} | Val fold shape: {X_va.shape}")
            console.print(f"Train fold class dist: {dict(Counter(y_tr.tolist()))}")
            console.print(f"Val   fold class dist: {dict(Counter(y_va.tolist()))}")

            # DataLoaders
            train_loader, val_loader = create_dataloaders(X_tr, y_tr, X_va, y_va)

            emb_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_tr, dtype=torch.float32).permute(0, 2, 1),
                    torch.tensor(y_tr, dtype=torch.long),
                ),
                batch_size=batch_size,
                shuffle=False,
                generator=g,
                worker_init_fn=seed_worker,
                num_workers=loader_workers,
                persistent_workers=loader_workers > 0,
            )

            # Model
            model = VisionTransformer(input_channel=Fdim).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

            # Predicates
            predicates = [ltn.Predicate(ClassPredicate(model, c)) for c in range(num_classes)]
            centroids = recompute_centroids(model, emb_loader, device)
            sim_preds = build_similarity_predicates(model, centroids)

            # Early stopping on val_loss
            best_path = os.path.join(model_save_directory, f"{base}_best_fold{fold_idx}.pt")
            best_val_loss = float("inf")
            best_acc = 0.0
            no_improve = 0

            # Logs
            epochs_list, train_sat_list, val_sat_list, val_acc_list = [], [], [], []
            rule_count_list = []  # track number of similarity rules per epoch

            for epoch in trange(1, epochs + 1, desc=f"Training ({base}, fold {fold_idx})"):
                model.train()
                running = 0.0

                for xb, yb in train_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)

                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            rules = build_rules(xb, yb, predicates, sim_preds, Not, Forall, Implies)
                            sat = SatAgg(*rules)
                            loss = 1.0 - sat
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        rules = build_rules(xb, yb, predicates, sim_preds, Not, Forall, Implies)
                        sat = SatAgg(*rules)
                        loss = 1.0 - sat
                        loss.backward()
                        optimizer.step()

                    running += loss.item() * xb.size(0)

                train_loss = running / len(train_loader.dataset)
                train_sat = 1.0 - train_loss
                val_sat = compute_sat_level(val_loader, predicates, sim_preds, Not, Forall, Implies, SatAgg, device)
                val_acc = compute_accuracy(val_loader, model, device)
                val_loss = 1.0 - val_sat

                epochs_list.append(epoch)
                train_sat_list.append(train_sat)
                val_sat_list.append(val_sat)
                val_acc_list.append(val_acc)

                # refresh centroids/rules (after this epoch update)
                centroids = recompute_centroids(model, emb_loader, device)
                sim_preds = build_similarity_predicates(model, centroids)

                # count active similarity rules after update
                num_sim_rules = sum(len(lst) for lst in sim_preds)
                rule_count_list.append(num_sim_rules)

                console.print(
                    f"Epoch {epoch:02d} • train_sat={train_sat:.4f} • val_sat={val_sat:.4f} "
                    f"• val_acc={val_acc:.4f} • num_rules={num_sim_rules}"
                )

                # Early stopping bookkeeping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                else:
                    no_improve += 1

                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), best_path)
                    console.log(f"Saved best ({base}, fold {fold_idx}) acc={best_acc:.4f}")

                if no_improve >= patience:
                    console.log(f"[yellow]Early stopping[/] after {patience} epochs without val_loss improvement")
                    break

            # Save per-epoch metrics for this fold (temporal evolution)
            metrics_df = pd.DataFrame(
                {
                    "epoch": epochs_list,
                    "train_sat": train_sat_list,
                    "val_sat": val_sat_list,
                    "val_acc": val_acc_list,
                    "train_loss": [1.0 - s for s in train_sat_list],
                    "val_loss": [1.0 - s for s in val_sat_list],
                    "num_similarity_rules": rule_count_list,
                }
            )
            metrics_df.to_csv(os.path.join(metrics_dir, f"metrics_fold{fold_idx}.csv"), index=False)

            # Plots
            plt.figure()
            plt.plot(epochs_list, metrics_df["train_loss"], label="train_loss")
            plt.plot(epochs_list, metrics_df["val_loss"], label="val_loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(os.path.join(plots_dir, f"loss_fold{fold_idx}.png"), dpi=400)
            plt.close()

            plt.figure()
            plt.plot(epochs_list, val_acc_list, label="val_accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Validation Accuracy")
            plt.legend()
            plt.savefig(os.path.join(plots_dir, f"accuracy_fold{fold_idx}.png"), dpi=400)
            plt.close()

            # Plot number of similarity rules vs epoch
            plt.figure()
            plt.plot(epochs_list, rule_count_list, label="num_similarity_rules")
            plt.xlabel("Epoch")
            plt.ylabel("Number of similarity rules")
            plt.legend()
            plt.savefig(os.path.join(plots_dir, f"rules_fold{fold_idx}.png"), dpi=400)
            plt.close()

            # --- NEW: summary statistics of rule dynamics for this fold ---
            if rule_count_list:
                num_epochs_this_fold = len(rule_count_list)
                max_rules = int(max(rule_count_list))
                min_rules = int(min(rule_count_list))
                final_rules = int(rule_count_list[-1])
                mean_rules = float(np.mean(rule_count_list))
                std_rules = float(np.std(rule_count_list))
                # stability on last 20% of epochs
                k = max(1, int(0.2 * num_epochs_this_fold))
                stable_segment = rule_count_list[-k:]
                stable_mean_rules = float(np.mean(stable_segment))
                stable_std_rules = float(np.std(stable_segment))
            else:
                num_epochs_this_fold = 0
                max_rules = min_rules = final_rules = 0
                mean_rules = std_rules = 0.0
                stable_mean_rules = stable_std_rules = 0.0
            # --------------------------------------------------------------

            # Evaluate best checkpoint for this fold
            model.load_state_dict(torch.load(best_path, map_location=device))
            model.eval()

            preds, labels = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    logits, _ = model(xb.to(device, non_blocking=True))
                    preds.extend(logits.argmax(1).cpu().numpy())
                    labels.extend(yb.cpu().numpy())

            prec, rec, f1, _ = precision_recall_fscore_support(
                labels, preds, average="macro", zero_division=0
            )
            acc = (np.array(labels) == np.array(preds)).mean()

            # store per-fold summary including rule stats
            fold_rows.append(
                (
                    fold_idx,
                    acc,
                    prec,
                    rec,
                    f1,
                    num_epochs_this_fold,
                    max_rules,
                    min_rules,
                    final_rules,
                    mean_rules,
                    std_rules,
                    stable_mean_rules,
                    stable_std_rules,
                )
            )
            console.print(
                f"[bold]Fold {fold_idx}[/] Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}  "
                f"final_rules={final_rules}  stable_mean_rules={stable_mean_rules:.1f}"
            )

            # Per-fold classification report
            report = classification_report(labels, preds, output_dict=True, digits=4)
            report_df = pd.DataFrame(report).T.round(4)
            report_df.to_csv(os.path.join(reports_dir, f"classification_report_fold{fold_idx}.csv"))

        # Per-base summary (now includes rule statistics by fold)
        if fold_rows:
            df = pd.DataFrame(
                fold_rows,
                columns=[
                    "fold",
                    "accuracy",
                    "precision_macro",
                    "recall_macro",
                    "f1_macro",
                    "epochs_trained",
                    "max_num_rules",
                    "min_num_rules",
                    "final_num_rules",
                    "mean_num_rules",
                    "std_num_rules",
                    "mean_num_rules_last_20pct",
                    "std_num_rules_last_20pct",
                ],
            )
            df.to_csv(os.path.join(base_out_dir, "cv_summary.csv"), index=False)

            # keep global averages for standard metrics as before
            avg = df[["accuracy", "precision_macro", "recall_macro", "f1_macro"]].mean().to_dict()
            console.rule(f"[bold blue]Summary {base}[/]")
            console.print(
                f"{base} → Acc={avg['accuracy']:.4f}  P={avg['precision_macro']:.4f}  "
                f"R={avg['recall_macro']:.4f}  F1={avg['f1_macro']:.4f}"
            )
            overall_rows.append((base, avg["accuracy"], avg["precision_macro"], avg["recall_macro"], avg["f1_macro"]))

        console.print(
            f"Outputs for [bold]{base}[/]:\n"
            f"  {base_out_dir}\n"
            f"   ├─ metrics/metrics_fold*.csv\n"
            f"   ├─ plots/loss_fold*.png, accuracy_fold*.png, rules_fold*.png\n"
            f"   └─ reports/classification_report_fold*.csv\n"
            f"   └─ cv_summary.csv (per-fold rule dynamics + performance)"
        )

    # Overall summary across bases (standard metrics)
    if overall_rows:
        overall_df = pd.DataFrame(
            overall_rows, columns=["base", "accuracy", "precision_macro", "recall_macro", "f1_macro"]
        )
        overall_df.to_csv(os.path.join(results_root, "overall_summary_across_bases.csv"), index=False)
        gavg = overall_df[["accuracy", "precision_macro", "recall_macro", "f1_macro"]].mean().to_dict()
        console.rule("[bold magenta]Overall summary across bases[/]")
        console.print(
            f"Global avg → Acc={gavg['accuracy']:.4f}  P={gavg['precision_macro']:.4f}  "
            f"R={gavg['recall_macro']:.4f}  F1={gavg['f1_macro']:.4f}"
        )

    # ================= DRIFT EXPERIMENT CALL =================
    if RUN_DRIFT_EXPERIMENT:
        """
        TL;DR: Sequential operating-condition drift experiment.

        What it does
        ------------
        - Uses ONE shared model.
        - Stages = sorted bases (operating regimes), e.g. A → B → C → ...
        - Stage 1: train on base A.
        - Stage 2: continue from same weights, fine-tune on base B.
        - Stage 3: continue, fine-tune on base C, etc.
        - At each stage:
            * Gradients see ONLY the current base (new operating condition).
            * Similarity rules are recomputed on the UNION of all train data
            from bases seen so far (A, then A+B, then A+B+C, ...).
            * We log:
                - train/val sat, losses, val accuracy,
                - number of similarity rules per epoch,
                - test accuracy on ALL seen bases so far.

        Why (for the paper / rebuttal)
        -------------------------------
        - Simulates "operating-condition drift" in time: model is exposed
        sequentially to different speeds/loads instead of training on a
        static mixture.
        - Lets us show:
            * Temporal evolution of the rule base under drift.
            * That the rule count grows then stabilizes (no explosion).
            * That previously learned rules remain useful as new regimes
            are added (stability and boundedness of merge/prune).
            * Cross-domain generalization: performance on older regimes
            remains high while accuracy on new regimes increases.

        Where results are stored
        ------------------------
        All outputs are written under:
            {results_root}/drift_experiment/

        Main files:
        - drift_metrics.csv
            Per-epoch metrics during drift:
                stage, base, global_epoch, local_epoch,
                train_sat, val_sat, train_loss, val_loss,
                val_acc, num_similarity_rules.
            → Use for the main "num rules vs epoch" plot and temporal curves.

        - drift_stage_rule_stats.csv
            Per-stage rule statistics:
                stage, added_base, epochs_trained_this_stage,
                max_num_rules, min_num_rules, final_num_rules,
                mean_num_rules, std_num_rules,
                mean_num_rules_last_20pct, std_num_rules_last_20pct.
            → Use for convergence/stability discussion (rule-base stability).

        - drift_stage_performance.csv
            Per (stage, test_base) test accuracy:
                stage, train_bases_seen, test_base, test_accuracy.
            → Use to build a table: stage × test_base accuracy.

        - drift_stage_vs_base_accuracy.csv
            Pivoted version: stage as rows, test_base as columns.
            → Direct input for a "accuracy per base vs stage" figure.

        - drift_num_rules_vs_epoch.png
            Line plot: global_epoch vs num_similarity_rules,
            colored by stage.
            → Visual evidence that the rule base grows then stabilizes
                across drift stages.

        - drift_accuracy_per_base_vs_stage.png
            Line plot: stage vs test_accuracy, one curve per test_base.
            → Shows cross-domain generalization under drift:
                old regimes stay good, new regimes improve with training.
        """
        run_drift_experiment(bases, device, Not, Forall, Implies, SatAgg)
    # =========================================================

    console.print("\nDone.")


if __name__ == "__main__":
    main()
