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
results_root          = config.results_path_ltn
sequence_length       = config.sequence_length
batch_size            = config.batch_size
epochs                = config.epochs
learning_rate         = config.learning_rate
num_classes           = config.num_classes
merge_cosine_threshold= config.rule_merge_tau
support_min           = config.rule_support_min
n_cluster             = config.n_cluster
patience              = config.patience
n_splits              = getattr(config, "n_splits", 2)

#sequences_directory   = "data/DDS_Data_SEU/PGB/PGB"
sequences_directory = config.sequences_directory

model_save_directory  = "model_weights_cluj"
os.makedirs(model_save_directory, exist_ok=True)

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
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        generator=g, worker_init_fn=seed_worker, pin_memory=pin, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin, num_workers=0)
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

                console.print(
                    f"Epoch {epoch:02d} • train_sat={train_sat:.4f} • val_sat={val_sat:.4f} • val_acc={val_acc:.4f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                else:
                    no_improve += 1

                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), best_path)
                    console.log(f"Saved best ({base}, fold {fold_idx}) acc={best_acc:.4f}")

                # refresh centroids/rules
                centroids = recompute_centroids(model, emb_loader, device)
                sim_preds = build_similarity_predicates(model, centroids)

                if no_improve >= patience:
                    console.log(f"[yellow]Early stopping[/] after {patience} epochs without val_loss improvement")
                    break

            # Save per-epoch metrics for this fold
            metrics_df = pd.DataFrame(
                {
                    "epoch": epochs_list,
                    "train_sat": train_sat_list,
                    "val_sat": val_sat_list,
                    "val_acc": val_acc_list,
                    "train_loss": [1.0 - s for s in train_sat_list],
                    "val_loss": [1.0 - s for s in val_sat_list],
                }
            )
            metrics_df.to_csv(os.path.join(metrics_dir, f"metrics_fold{fold_idx}.csv"), index=False)

            # Plots
            import matplotlib.pyplot as plt

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

            fold_rows.append((fold_idx, acc, prec, rec, f1))
            console.print(f"[bold]Fold {fold_idx}[/] Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")

            # Per-fold classification report
            report = classification_report(labels, preds, output_dict=True, digits=4)
            report_df = pd.DataFrame(report).T.round(4)
            report_df.to_csv(os.path.join(reports_dir, f"classification_report_fold{fold_idx}.csv"))

        # Per-base summary
        if fold_rows:
            df = pd.DataFrame(
                fold_rows, columns=["fold", "accuracy", "precision_macro", "recall_macro", "f1_macro"]
            )
            df.to_csv(os.path.join(base_out_dir, "cv_summary.csv"), index=False)
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
            f"   ├─ plots/loss_fold*.png, accuracy_fold*.png\n"
            f"   └─ reports/classification_report_fold*.csv"
        )

    # Overall summary across bases
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

    console.print("\nDone.")


if __name__ == "__main__":
    main()
