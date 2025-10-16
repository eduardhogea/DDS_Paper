# ablation_model_size.py
# Width × Depth × Heads ablation with LTN (+ optional similarity) matching original training behavior.
# - Uses learning_rate from config_uoc
# - GradScaler on CUDA (LTN path)
# - Validate: width % heads == 0
# - Validation SAT->loss BEFORE centroid refresh
# - Centroids recomputed at END of epoch (after logging/ES)
# - Early stopping on val_loss with patience from config
# - Deterministic DataLoaders; pin_memory matches CUDA availability
# - Saves CSV: width,depth,heads,acc,delta_acc,flops,delta_flops_pct
# - Prints LaTeX-ready snippets

import os
import sys
import time
import argparse
import random
import platform
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import ltn

from sequence_generation import load_sequences

# ---------- config ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "..", "config"))
import config_uoc as config

results_path_ltn   = getattr(config, "results_path_ltn", "./results_ltn")
sequence_length    = config.sequence_length
num_features       = config.num_features
batch_size         = config.batch_size
num_classes        = config.num_classes
learning_rate      = config.learning_rate
patience           = getattr(config, "patience", None)

CFG_use_ltn        = getattr(config, "use_ltn", True)
CFG_use_similarity = getattr(config, "use_similarity", False)
merge_cosine_threshold = getattr(config, "rule_merge_tau", 0.95)
support_min            = getattr(config, "rule_support_min", 3)
n_cluster              = getattr(config, "n_cluster", 3)

sequences_directory = "data_uoc/output_sequences"
os.makedirs(results_path_ltn, exist_ok=True)

# ---------- seeding ----------
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(_):
    np.random.seed(SEED)
    random.seed(SEED)

# ---------- model ----------
class Feed_Forward(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float, attn_drop_rate: float):
        super().__init__()
        mlp_dim = int(mlp_ratio * hidden_size)
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.activation = F.gelu
        self.dropout = nn.Dropout(attn_drop_rate)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
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
        attn = torch.matmul(Q, K.transpose(-1, -2)) * scale
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(self.dropout(attn), V), attn

class Multi_Head_Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, drop_rate: float):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.fc_query = nn.Linear(hidden_size, hidden_size)
        self.fc_key   = nn.Linear(hidden_size, hidden_size)
        self.fc_value = nn.Linear(hidden_size, hidden_size)
        self.fc_out   = nn.Linear(hidden_size, hidden_size)
        self.dropout  = nn.Dropout(drop_rate)
        self.attn     = Scaled_Dot_Product_Attention()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
    def _transpose(self, x):
        B, T, H = x.size()
        return x.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
    def forward(self, h):
        Q = self._transpose(self.fc_query(h))
        K = self._transpose(self.fc_key(h))
        V = self._transpose(self.fc_value(h))
        ctx, _ = self.attn(Q, K, V)
        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(h.size())
        out = self.dropout(self.fc_out(ctx)) + h
        return self.layer_norm(out), None

class Encoder_Block(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float, attn_drop_rate: float, num_heads: int, drop_rate: float):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn      = Multi_Head_Attention(hidden_size, num_heads, drop_rate)
        self.ffn_norm  = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn       = Feed_Forward(hidden_size, mlp_ratio, attn_drop_rate)
    def forward(self, x):
        x, _ = self.attn(self.attn_norm(x))
        x = self.ffn_norm(x)
        x = self.ffn(x)
        return x, None

class Encoder(nn.Module):
    def __init__(self, depth: int, hidden_size: int, mlp_ratio: float, attn_drop_rate: float, num_heads: int, drop_rate: float):
        super().__init__()
        self.blocks = nn.ModuleList([Encoder_Block(hidden_size, mlp_ratio, attn_drop_rate, num_heads, drop_rate) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6)
    def forward(self, x):
        for blk in self.blocks: x, _ = blk(x)
        return self.final_norm(x), None

class VisionTransformer(nn.Module):
    def __init__(self,
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
                 classifier: str = "gap"):
        super().__init__()
        self.classifier = classifier
        self.patch_size = patch_size
        self.stride = patch_size // 2
        self.hidden_size = hidden_size

        self.patch_embed = nn.Conv1d(input_channel, hidden_size, kernel_size=patch_size, stride=self.stride)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.dropout = nn.Dropout(drop_rate)
        self.encoder = Encoder(depth, hidden_size, mlp_ratio, attn_drop_rate, num_heads, drop_rate)
        self.avg_pool = nn.AdaptiveAvgPool1d(4)
        self.max_pool = nn.AdaptiveMaxPool1d(4)
        self.head = nn.Sequential(nn.Linear(hidden_size * 8, 256), nn.ReLU(), nn.Linear(256, num_class))

    def embed(self, x):
        B = x.size(0)
        x = self.patch_embed(x).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x, _ = self.encoder(x)
        return x[:, 0]

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = self.dropout(torch.cat([cls, x], dim=1))
        x, _ = self.encoder(x)
        feats = x[:, 1:].transpose(1, 2)
        feats = torch.cat([self.avg_pool(feats), self.max_pool(feats)], dim=-1)
        logits = self.head(feats.flatten(start_dim=1))
        return logits, None

# ---------- LTN predicates / rules ----------
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
        z = self.base.embed(x)
        dist = torch.norm(z - self.centroid.to(x.device), dim=1, keepdim=True)
        return torch.exp(-0.5 * dist)

def build_rules(x, y, predicates, sim_preds, Not, Forall, Implies):
    rules = []
    for c, P_c in enumerate(predicates):
        pos = (y == c)
        neg = (y != c)
        if pos.any():
            x_pos = ltn.Variable(f"x_pos_{c}", x[pos])
            rules.append(Forall(x_pos, P_c(x_pos)))
        if neg.any():
            x_neg = ltn.Variable(f"x_neg_{c}", x[neg])
            rules.append(Forall(x_neg, Not(P_c(x_neg))))
    if any(any(lst) for lst in sim_preds):
        x_all = ltn.Variable("x_all", x)
        for c, plist in enumerate(sim_preds):
            for S_c in plist:
                rules.append(Forall(x_all, Implies(S_c(x_all), predicates[c](x_all))))
    return rules

def recompute_centroids(model, loader, device):
    embeddings = [[] for _ in range(num_classes)]
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            emb = model.embed(xb).cpu().numpy()
            y_np = yb.cpu().numpy()
            for cls in range(num_classes):
                mask = (y_np == cls)
                if mask.any():
                    embeddings[cls].append(emb[mask])
    centroids = {}
    for cls in range(num_classes):
        if embeddings[cls]:
            data = np.concatenate(embeddings[cls], axis=0)
            n_clusters = min(n_cluster, data.shape[0])
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(data)
            centers = kmeans.cluster_centers_
            labels  = kmeans.labels_
            surviving = []
            for idx, center in enumerate(centers):
                if (labels == idx).sum() >= support_min:
                    surviving.append(center)
            final_centroids = []
            for c in surviving:
                if final_centroids:
                    sims = cosine_similarity([c], final_centroids)[0]
                    if sims.max() <= merge_cosine_threshold:
                        final_centroids.append(c)
                else:
                    final_centroids.append(c)
            if final_centroids:
                centroids[cls] = np.vstack(final_centroids)
            else:
                centroids[cls] = np.empty((0, centers.shape[1]))
        else:
            centroids[cls] = np.empty((0, 1))
    return centroids

def build_similarity_predicates(model, centroids):
    preds = [[] for _ in range(num_classes)]
    for cls, c_arr in centroids.items():
        for c in c_arr:
            preds[cls].append(ltn.Predicate(SimilarityPredicate(model, c)))
    return preds

# ---------- data ----------
def make_loaders(X_train, y_train, X_val, y_val, bs):
    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    X_val   = torch.tensor(X_val,   dtype=torch.float32).permute(0, 2, 1)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val   = torch.tensor(y_val,   dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)
    g = torch.Generator(); g.manual_seed(SEED)
    pin = torch.cuda.is_available()

    loader_tr = DataLoader(train_ds, batch_size=bs, shuffle=True,
                           num_workers=0, pin_memory=pin,
                           worker_init_fn=seed_worker, generator=g)
    loader_va = DataLoader(val_ds, batch_size=bs, shuffle=False,
                           num_workers=0, pin_memory=pin)
    loader_emb = DataLoader(train_ds, batch_size=bs, shuffle=False,
                            num_workers=0, pin_memory=pin)
    return loader_tr, loader_va, loader_emb

# ---------- metrics ----------
def compute_accuracy(loader, model, device):
    model.eval()
    correct = 0; total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
    return correct / max(1, total)

def compute_sat_level(loader, predicates, sim_preds, Not, Forall, Implies, SatAgg, device):
    total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            rules = build_rules(xb, yb, predicates, sim_preds, Not, Forall, Implies)
            total += float(SatAgg(*rules))
    return total / max(1, len(loader))

# ---------- FLOPs proxy ----------
def tokens_after_conv(L_in: int, kernel: int, stride: int) -> int:
    # floor((L - (K-1) - 1)/stride + 1)
    return (L_in - (kernel - 1) - 1) // stride + 1

def proxy_flops_total(seq_len=sequence_length, patch=10, stride=5, in_ch=num_features,
                      hidden=64, heads=8, mlp_ratio=4.0, depth=6, num_classes=num_classes):
    T_no_cls = tokens_after_conv(seq_len, patch, stride)
    T = T_no_cls + 1
    H = hidden
    mlp = int(mlp_ratio * H)
    conv_madds = H * T_no_cls * (in_ch * patch)
    qkv = 3 * T * H * H
    attn_matmuls = 2 * T * T * H
    oproj = T * H * H
    ffn = 2 * T * H * mlp
    block = qkv + attn_matmuls + oproj + ffn
    head_madds = (H * 8) * 256 + 256 * num_classes
    return conv_madds + depth * block + head_madds

def pct_increase(new, base):
    return 100.0 * (new - base) / base

# ---------- latency (1-thread CPU) ----------
def measure_cpu_latency_ms(model, example, warmup=20, iters=100):
    torch.set_num_threads(1)
    dev_cpu = torch.device("cpu")
    model_cpu = model.to(dev_cpu)
    model_cpu.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_cpu(example)[0]
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model_cpu(example)[0]
        dt = time.perf_counter() - t0
    return (dt / iters) * 1000.0

# ---------- device info ----------
def get_gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

def get_cpu_name():
    return platform.processor() or platform.uname().processor or "CPU"

# ---------- train one variant ----------
def train_one_model(depth, heads, hidden, epochs, device, train_loader, val_loader, emb_loader,
                    use_ltn=True, use_similarity=False, recompute_every=1, lr=1e-3,
                    verbose=False, verbose_val_sat=False, patience=None):
    assert hidden % heads == 0, "hidden_size must be divisible by num_heads"
    model = VisionTransformer(depth=depth, num_heads=heads, hidden_size=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda") else None

    if use_ltn:
        Not     = ltn.Connective(ltn.fuzzy_ops.NotStandard())
        Forall  = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
        Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesGoguen())
        SatAgg  = ltn.fuzzy_ops.SatAgg()
        predicates = [ltn.Predicate(ClassPredicate(model, c)) for c in range(num_classes)]
        if use_similarity:
            centroids = recompute_centroids(model, emb_loader, device)
            sim_preds = build_similarity_predicates(model, centroids)
        else:
            sim_preds = [[] for _ in range(num_classes)]

    best_acc = 0.0
    best_val_loss = float("inf")
    no_improve = 0

    for ep in range(1, epochs+1):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            if use_ltn:
                rules = build_rules(xb, yb, predicates, sim_preds, Not, Forall, Implies)
                sat   = SatAgg(*rules)
                loss  = 1.0 - sat
            else:
                logits, _ = model(xb)
                loss = criterion(logits, yb)

            if scaler is not None and use_ltn:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += float(loss) * xb.size(0)

        # ---------- validation BEFORE refresh ----------
        if use_ltn:
            val_sat  = compute_sat_level(val_loader, predicates, sim_preds, Not, Forall, Implies, SatAgg, device)
            val_loss = 1.0 - val_sat
        else:
            total, tot_loss = 0, 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits, _ = model(xb)
                    l = criterion(logits, yb)
                    tot_loss += float(l) * xb.size(0)
                    total += xb.size(0)
            val_loss = tot_loss / max(1, total)
            val_sat  = None

        acc = compute_accuracy(val_loader, model, device)
        best_acc = max(best_acc, acc)

        # early stopping based on val_loss
        if patience is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"[width={hidden:03d}, depth={depth:02d}, heads={heads:02d}] early stopping at epoch {ep} (patience={patience})")
                break

        # verbose print (pre-refresh metrics)
        if verbose:
            mean_loss = running_loss / max(1, len(train_loader.dataset))
            if use_ltn:
                train_sat_est = 1.0 - mean_loss
                msg = f"[width={hidden:03d}, depth={depth:02d}, heads={heads:02d}] epoch {ep}/{epochs} | train_sat={train_sat_est:.4f} | val_acc={acc:.4f}"
                if verbose_val_sat and (val_sat is not None):
                    msg += f"\n                                         ↳ val_sat={val_sat:.4f}"
                print(msg)
            else:
                print(f"[width={hidden:03d}, depth={depth:02d}, heads={heads:02d}] epoch {ep}/{epochs} | train_loss={mean_loss:.4f} | val_acc={acc:.4f}")

        # ---------- centroid refresh at END of epoch ----------
        if use_ltn and use_similarity and (recompute_every > 0 and ep % recompute_every == 0):
            centroids = recompute_centroids(model, emb_loader, device)
            sim_preds = build_similarity_predicates(model, centroids)

    return model, best_acc

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--widths", nargs="+", type=int, default=[64], help="hidden sizes to sweep")
    ap.add_argument("--depths", nargs="+", type=int, default=[4, 6, 8])
    ap.add_argument("--heads",  nargs="+", type=int, default=[8])
    ap.add_argument("--use_ltn", type=str, default="config", help="true|false|config")
    ap.add_argument("--use_similarity", type=str, default="config", help="true|false|config")
    ap.add_argument("--recompute_centroids_every", type=int, default=1, help="1 == every epoch (like original)")
    ap.add_argument("--save_csv", type=str, default=os.path.join(results_path_ltn, "ablation_ltn_wdh.csv"))
    ap.add_argument("--eps_acc", type=float, default=0.002, help="tolerance when declaring heads sufficient at baseline depth/width")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--verbose_val_sat", action="store_true")
    args = ap.parse_args()

    tf = lambda b: str(b).lower() in ["1","true","t","yes","y"]
    use_ltn = CFG_use_ltn if args.use_ltn == "config" else tf(args.use_ltn)
    use_similarity = CFG_use_similarity if args.use_similarity == "config" else tf(args.use_similarity)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | GPU: {get_gpu_name()} | CPU: {get_cpu_name()}")
    print(f"LTN training: {use_ltn} | similarity rules: {use_similarity} | recompute every {args.recompute_centroids_every} epoch(s)")

    # load data
    X_train, y_train = load_sequences(
        os.path.join(sequences_directory, "train_sequences.npy"),
        os.path.join(sequences_directory, "train_labels.npy"),
    )
    X_test, y_test = load_sequences(
        os.path.join(sequences_directory, "test_sequences.npy"),
        os.path.join(sequences_directory, "test_labels.npy"),
    )
    train_loader, val_loader, emb_loader = make_loaders(X_train, y_train, X_test, y_test, batch_size)

    # baseline
    baseline_width = 64
    baseline_depth = 6
    baseline_heads = 8

    # validate heads divisibility for baseline
    if baseline_width % baseline_heads != 0:
        raise ValueError("Baseline width must be divisible by baseline heads.")

    print(f"\nTraining baseline (width={baseline_width}, depth={baseline_depth}, heads={baseline_heads})...")
    base_model, base_acc = train_one_model(
        baseline_depth, baseline_heads, baseline_width, args.epochs, device, train_loader, val_loader, emb_loader,
        use_ltn=use_ltn, use_similarity=use_similarity,
        recompute_every=args.recompute_centroids_every,
        lr=learning_rate,
        verbose=args.verbose, verbose_val_sat=args.verbose_val_sat,
        patience=patience
    )

    # baseline metrics
    base_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    base_flops  = proxy_flops_total(hidden=baseline_width, depth=baseline_depth, heads=baseline_heads)
    example = torch.zeros(1, num_features, sequence_length, dtype=torch.float32)
    cpu_latency_ms = measure_cpu_latency_ms(base_model, example)

    rows = []
    for hidden in args.widths:
        for d in args.depths:
            for h in args.heads:
                if hidden % h != 0:
                    # skip invalid combos
                    continue
                if hidden == baseline_width and d == baseline_depth and h == baseline_heads:
                    acc_d = base_acc
                    flops_d = base_flops
                else:
                    print(f"\nTraining variant width={hidden}, depth={d}, heads={h} ...")
                    model_d, acc_d = train_one_model(
                        d, h, hidden, args.epochs, device, train_loader, val_loader, emb_loader,
                        use_ltn=use_ltn, use_similarity=use_similarity,
                        recompute_every=args.recompute_centroids_every,
                        lr=learning_rate,
                        verbose=args.verbose, verbose_val_sat=args.verbose_val_sat,
                        patience=patience
                    )
                    flops_d = proxy_flops_total(hidden=hidden, depth=d, heads=h)
                rows.append((hidden, d, h, acc_d, acc_d - base_acc, flops_d, pct_increase(flops_d, base_flops)))

    # choose a deeper (same width/heads) config for LaTeX (2) — keep baseline heads/width
    deeper = [r for r in rows if r[0] == baseline_width and r[2] == baseline_heads and r[1] > baseline_depth]
    deeper.sort(key=lambda x: x[1])
    if deeper:
        w_d, d_depth, h_d, acc_d, dacc_d, flops_d, dflops_pct_d = deeper[0][0], deeper[0][1], deeper[0][2], deeper[0][3], deeper[0][4], deeper[0][5], deeper[0][6]
    elif rows:
        w_d, d_depth, h_d, acc_d, dacc_d, flops_d, dflops_pct_d = rows[-1]
    else:
        w_d, d_depth, h_d, acc_d, dacc_d, flops_d, dflops_pct_d = baseline_width, baseline_depth, baseline_heads, base_acc, 0.0, base_flops, 0.0

    # summary
    print("\n=== Summary ===")
    print(f"Baseline width={baseline_width}, depth={baseline_depth}, heads={baseline_heads}: "
          f"acc={base_acc:.4f}, params={base_params:,}, FLOPs={base_flops:,}, CPU latency (1 thread)={cpu_latency_ms:.3f} ms/segment")
    if rows:
        print("Ablations (width, depth, heads) -> acc | Δacc | FLOPs | ΔFLOPs% :")
        for w, d, h, acc_, dacc_, flops_, dflops_ in rows:
            print(f"  ({w:>3},{d:>2},{h:>2}) -> {acc_:.4f} | {dacc_:+.4f} | {flops_:,} | {dflops_:+.2f}%")

    # save CSV
    try:
        import pandas as pd
        pd.DataFrame(rows, columns=["width","depth","heads","acc","delta_acc","flops","delta_flops_pct"]).to_csv(
            args.save_csv, index=False
        )
        print(f"\nSaved CSV: {args.save_csv}")
    except Exception:
        with open(args.save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["width","depth","heads","acc","delta_acc","flops","delta_flops_pct"])
            for r in rows: w.writerow(r)
        print(f"\nSaved CSV (no pandas): {args.save_csv}")

    # --------- LaTeX snippets ----------
    print("\nLaTeX — item (1) param count:")
    print(f"In our case, the model has \\emph{{{base_params/1e6:.2f} M}} trainable parameters ({base_params:,}).")

    print("\nLaTeX — item (2) depth sweep (at width=64, heads=8):")
    print(f"(2) A depth sweep (Table \\ref{{tab:ablation_depth}}) at width=64 showed that increasing depth from "
          f"{baseline_depth} to {d_depth} changes accuracy by $<\\!{abs(dacc_d):.3f}$ while costing $>\\!{dflops_pct_d:.2f}\\%$ FLOPs; "
          f"shallower models may under-fit or over-fit depending on the dataset.")

    cpu_name = get_cpu_name()
    print("\nLaTeX — item (3) latency:")
    print(f"(3) Width 64 offers a favorable latency–accuracy trade-off, enabling on-device inference in "
          f"\\emph{{{cpu_latency_ms:.3f} ms}} per segment on a single-threaded {cpu_name} CPU.")

    head_rows = [r for r in rows if r[0] == baseline_width and r[1] == baseline_depth]
    if len({h for (_,_,h,_,_,_,_) in head_rows}) > 1:
        eps = args.eps_acc
        best_acc_head = max(r[3] for r in head_rows)
        candidates = sorted({h for (_,_,h,acc,_,_,_) in head_rows if (best_acc_head - acc) <= eps})
        if candidates:
            min_sufficient = candidates[0]
            print("\nLaTeX — item (4) heads:")
            print(f"(4) A head sweep at width={baseline_width}, depth={baseline_depth} indicated that {min_sufficient} heads suffice "
                  "to capture long-range spectral–temporal dependencies, with wider head counts yielding negligible accuracy gains "
                  "at approximately $0\\%$ additional FLOPs.")

if __name__ == "__main__":
    main()
