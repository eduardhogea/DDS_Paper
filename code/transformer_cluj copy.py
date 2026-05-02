#!/usr/bin/env python3
# ---------------------------------------------------------------
# Transformer-LTN with weighted similarity rules
# ---------------------------------------------------------------
import os, sys
from collections import Counter

import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from rich.console import Console
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import ltn
from sequence_generation import load_sequences

# ----------------------------------------------------------------
# Config ----------------------------------------------------------
# ----------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "..", "config"))
import config_uoc as config

results_path_ltn   = config.results_path_ltn
sequence_length    = config.sequence_length          # 20
num_features       = config.num_features            # 1
batch_size         = config.batch_size              # 150
epochs             = config.epochs                  # 500
learning_rate      = config.learning_rate           # 1e-4
n_splits           = config.n_splits                # 2
num_classes        = config.num_classes             # 9
sequences_directory = "/home/ubuntu/dds_paper/DDS_Paper/data_uoc/output_sequences"

# similarity-rule hyper-parameters (can also be added to config_uoc.py)
sim_weight        = getattr(config, "sim_weight", 0.30)   # weight of similarity satisfactions
k_centroids       = getattr(config, "k_means_per_class", 2)
similarity_scale  = getattr(config, "similarity_scale", 0.5)

# ----------------------------------------------------------------
# Positional encoding --------------------------------------------
# ----------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ----------------------------------------------------------------
# Vanilla Transformer --------------------------------------------
# ----------------------------------------------------------------
class VanillaTransformer(nn.Module):
    def __init__(self,
                 input_channel=num_features,
                 patch_size=10,
                 num_class=num_classes,
                 hidden_size=64,
                 depth=6,
                 num_heads=8,
                 mlp_ratio=4.0,
                 drop_rate=0.1,
                 classifier="gap"):
        super().__init__()
        self.classifier = classifier.lower()
        stride = patch_size // 2
        self.patch_embed = nn.Conv1d(input_channel, hidden_size, patch_size, stride)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.posenc = PositionalEncoding(hidden_size)
        self.dropout = nn.Dropout(drop_rate)
        enc_layer = nn.TransformerEncoderLayer(
            hidden_size, num_heads,
            int(mlp_ratio * hidden_size),
            drop_rate, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, depth)
        self.avg_pool = nn.AdaptiveAvgPool1d(4)
        self.max_pool = nn.AdaptiveMaxPool1d(4)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 8, 256), nn.ReLU(),
            nn.Linear(256, num_class)
        )
        nn.init.normal_(self.cls_token, std=1e-6)
    def forward(self, x):                        # x:(B,1,T)
        b = x.size(0)
        x = self.patch_embed(x).transpose(1, 2)  # (B,T',H)
        x = torch.cat([self.cls_token.expand(b, -1, -1), x], 1)
        x = self.dropout(self.posenc(x))
        x = self.encoder(x)
        emb = x[:, 0]                            # (B,H)
        if self.classifier == "token":
            logits = self.head(emb)
        else:
            f = x[:, 1:].transpose(1, 2)
            f = torch.cat([self.avg_pool(f), self.max_pool(f)], -1)
            logits = self.head(f.flatten(1))
        return logits, emb

VisionTransformer = VanillaTransformer

# ----------------------------------------------------------------
# LTN predicates -------------------------------------------------
# ----------------------------------------------------------------
class ClassPredicate(nn.Module):
    def __init__(self, model, idx):
        super().__init__(); self.base, self.idx = model, idx
    def forward(self, x):                        # x tensor (B,1,T)
        logits, _ = self.base(x)
        return F.softmax(logits, 1)[:, self.idx].unsqueeze(-1)

class SimilarityPredicate(nn.Module):
    """Gaussian similarity computed on an *embedding tensor*."""
    def __init__(self, centroid, scale=0.5):
        super().__init__(); self.register_buffer("centroid", centroid.unsqueeze(0)); self.scale = scale
    def forward(self, emb):                      # emb (B,H)
        dist = torch.norm(emb - self.centroid, dim=-1, keepdim=True)
        return torch.exp(-self.scale * dist)

# ----------------------------------------------------------------
# Helpers --------------------------------------------------------
# ----------------------------------------------------------------
def create_dataloaders(Xt, yt, Xv, yv):
    Xt, Xv = map(lambda a: torch.tensor(a, dtype=torch.float32).permute(0,2,1), (Xt, Xv))
    yt, yv = map(torch.tensor, (yt, yv))
    return (DataLoader(TensorDataset(Xt, yt), batch_size, True),
            DataLoader(TensorDataset(Xv, yv), batch_size, False))

def gen_cluster_predicates(model, loader, device):
    """Return dict[class] -> list[LTN predicate]  (centroids already on device)."""
    model.eval()
    embs, lbls = [], []
    with torch.no_grad():
        for x, y in loader:
            emb = model(x.to(device))[1]         # cache once
            embs.append(emb.cpu()); lbls.append(y)
    embs = torch.cat(embs).numpy()
    lbls = torch.cat(lbls).numpy()
    preds = {}
    for c in range(num_classes):
        pts = embs[lbls == c]
        if pts.size == 0: continue
        kc = min(k_centroids, len(pts))
        ctr = pts.mean(0, keepdims=True) if kc == 1 else KMeans(kc, 0).fit(pts).cluster_centers_
        preds[c] = [ltn.Predicate(SimilarityPredicate(torch.tensor(mu, dtype=torch.float32).to(device),
                                                      similarity_scale))
                     for mu in ctr]
    model.train()
    return preds

def compute_accuracy(loader, model, device):
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device))[0].argmax(1)
            correct += (preds.cpu() == y).sum().item(); total += y.size(0)
    return correct / total

# ----------------------------------------------------------------
# Main -----------------------------------------------------------
# ----------------------------------------------------------------
def main():
    console = Console()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data -------------------------------------------------------
    Xtr = np.load(os.path.join(sequences_directory, "train_sequences.npy"))
    ytr = np.load(os.path.join(sequences_directory, "train_labels.npy"))
    Xte = np.load(os.path.join(sequences_directory, "test_sequences.npy"))
    yte = np.load(os.path.join(sequences_directory, "test_labels.npy"))
    console.print(f"Loaded  train {Xtr.shape}  test {Xte.shape}")

    rng = np.random.default_rng(42)
    Xtr, ytr = Xtr[rng.permutation(len(Xtr))], ytr[rng.permutation(len(ytr))]
    Xte, yte = Xte[rng.permutation(len(Xte))], yte[rng.permutation(len(yte))]
    X, y = np.concatenate([Xtr, Xte]), np.concatenate([ytr, yte])
    console.print(f"Merged dataset {X.shape}")

    # LTN operators ---------------------------------------------
    Not     = ltn.Connective(ltn.fuzzy_ops.NotStandard())
    Forall  = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), "f")
    SatAgg  = ltn.fuzzy_ops.SatAgg()
    Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (tr, va) in enumerate(kf.split(X, y), 1):
        console.rule(f"Fold {fold}/{n_splits}")
        tl, vl = create_dataloaders(X[tr], y[tr], X[va], y[va])

        model     = VisionTransformer().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        class_preds = [ltn.Predicate(ClassPredicate(model, c)) for c in range(num_classes)]

        for epoch in range(1, epochs + 1):
            cluster_pred = gen_cluster_predicates(model, tl, device)  # refresh centroids
            batch_iter   = tqdm(tl, desc=f"Fold {fold} Epoch {epoch}", leave=False)
            run_loss = 0.0

            for xb, yb in batch_iter:
                xb, yb = xb.to(device), yb.to(device)
                logits, emb = model(xb)                # single forward pass
                optimizer.zero_grad()

                label_rules, sim_rules = [], []
                for c, Pc in enumerate(class_preds):
                    pos, neg = (yb == c), (yb != c)
                    if pos.any():
                        label_rules.append(Forall(ltn.Variable("xp", xb[pos]), Pc(xb[pos])))
                    if neg.any():
                        label_rules.append(Forall(ltn.Variable("xn", xb[neg]), Not(Pc(xb[neg]))))
                    for Sim in cluster_pred.get(c, []):
                        sim_rules.append(Forall(ltn.Variable("e", emb), Implies(Sim(emb), Pc(xb))))

                sat_label = SatAgg(*label_rules)
                sat_sim   = SatAgg(*sim_rules) if sim_rules else 0.0
                sat       = (sat_label + sim_weight * sat_sim) / (1 + sim_weight)

                loss = 1.0 - sat
                loss.backward(); optimizer.step()
                run_loss += loss.item() * xb.size(0)
                batch_iter.set_postfix(loss=f"{loss.item():.3f}")

            train_sat = 1.0 - run_loss / len(tl.dataset)

            # validation: label rules only
            model.eval(); vsat = 0.0
            with torch.no_grad():
                for xv, yv in vl:
                    xv, yv = xv.to(device), yv.to(device)
                    vrules = []
                    for c, Pc in enumerate(class_preds):
                        pos, neg = (yv == c), (yv != c)
                        if pos.any():
                            vrules.append(Forall(ltn.Variable("xp", xv[pos]), Pc(xv[pos])))
                        if neg.any():
                            vrules.append(Forall(ltn.Variable("xn", xv[neg]), Not(Pc(xv[neg]))))
                    vsat += SatAgg(*vrules).item()
            val_sat = vsat / len(vl)
            val_acc = compute_accuracy(vl, model, device)
            model.train()

            console.print(f"Epoch {epoch:03d}  train_sat {train_sat:.4f} "
                          f"val_sat {val_sat:.4f}  val_acc {val_acc:.4f}")

        fold_metrics.append((val_sat, val_acc))

    # summary ---------------------------------------------------
    avg_sat = np.mean([m[0] for m in fold_metrics])
    avg_acc = np.mean([m[1] for m in fold_metrics])
    console.rule("Summary")
    console.print(f"Avg Sat {avg_sat:.4f} | Avg Acc {avg_acc:.4f}")

    # save csv --------------------------------------------------
    df = pd.DataFrame({"Fold": range(1, n_splits+1),
                       "Val Sat": [m[0] for m in fold_metrics],
                       "Val Acc": [m[1] for m in fold_metrics]})
    os.makedirs(results_path_ltn, exist_ok=True)
    csv_path = os.path.join(results_path_ltn, "transformer_uoc_ltn.csv")
    df.to_csv(csv_path, index=False); console.print(f"Wrote {csv_path}")

# ----------------------------------------------------------------
if __name__ == "__main__":
    if sys.version_info < (3, 8):
        raise RuntimeError("Python ≥3.8 required")
    main()
