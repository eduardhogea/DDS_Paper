import os
import sys
import math
import random
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rich.console import Console
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

# ---------------------------------------------------------------------
# Config & seeding
# ---------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "..", "config"))
import ewsnet_config as config  # noqa: E402

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
results_root          = config.results_root
sequence_length       = config.sequence_length
batch_size            = config.batch_size
epochs                = config.epochs
learning_rate         = config.learning_rate
num_classes           = config.num_classes
patience              = config.patience
n_splits              = getattr(config, "n_splits", 2)
input_pad_length      = getattr(config, "input_pad_length", 512)
prefer_scaled         = getattr(config, "prefer_scaled", True)

sequences_directory = config.sequences_directory
# Ensure output directories exist
model_save_directory  = config.model_save_directory
os.makedirs(model_save_directory, exist_ok=True)

from sequence_generation import load_sequences  # noqa: E402


def seed_worker(_):
    np.random.seed(torch.initial_seed() % (2**32))


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
def _laplace_wavelet_filter(out_channels: int, kernel_size: int, eps: float = -0.3) -> torch.Tensor:
    time_disc = torch.linspace(0, kernel_size - 1, steps=kernel_size, dtype=torch.float32).view(1, -1)
    a = torch.linspace(0, out_channels, out_channels, dtype=torch.float32).view(-1, 1)
    b = torch.linspace(0, out_channels, out_channels, dtype=torch.float32).view(-1, 1)
    p = (time_disc - b) / (a - eps)
    w = 2 * math.pi * 80
    q = math.sqrt(1 - 0.03 ** 2)
    filter_vals = 0.08 * torch.exp((-0.03 / q) * (w * (p - 0.1))) * torch.sin(w * (p - 0.1))
    return filter_vals.view(out_channels, 1, kernel_size)


class Shrinkagev3ppp2(nn.Module):
    def __init__(self, channel: int, gap_size: int):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([0.48], dtype=torch.float32))
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_raw = x
        x_abs = x.abs()
        pooled = self.gap(x)
        pooled_flat = torch.flatten(pooled, 1)
        avg = pooled_flat
        scale = self.fc(pooled_flat)
        thresh = torch.mul(avg, scale).unsqueeze(2)
        sub = torch.max(x_abs - thresh, torch.zeros_like(x_abs))
        mask = (sub > 0).to(x.dtype)
        shrink = sub + (1 - self.a) * thresh
        return torch.sign(x_raw) * (shrink * mask)


class EWSNet(nn.Module):
    def __init__(self, input_channel: int, num_class: int = num_classes, min_length: int = input_pad_length):
        super().__init__()
        first_in = 1 if input_channel != 1 else 1
        self.preconv = nn.Conv1d(input_channel, 1, kernel_size=1, bias=False) if input_channel != 1 else None
        self.required_input_length = max(min_length, 300)  # ensure convolutional stack remains valid

        self.p1_0 = nn.Sequential(
            nn.Conv1d(first_in, 64, kernel_size=250, stride=1, bias=True),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.p1_1 = nn.Sequential(
            nn.Conv1d(64, 16, kernel_size=18, stride=2, bias=True),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.p1_2 = nn.Sequential(
            nn.Conv1d(16, 10, kernel_size=10, stride=2, bias=True),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
        )
        self.p1_3 = nn.MaxPool1d(kernel_size=2)

        self.p2_1 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=6, stride=1, bias=True),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.p2_2 = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=6, stride=1, bias=True),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.p2_3 = nn.MaxPool1d(kernel_size=2)
        self.p2_4 = nn.Sequential(
            nn.Conv1d(16, 10, kernel_size=6, stride=1, bias=True),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
        )
        self.p2_5 = nn.Sequential(
            nn.Conv1d(10, 10, kernel_size=8, stride=2, bias=True),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
        )
        self.p2_6 = nn.MaxPool1d(kernel_size=2)

        self.p3_0 = Shrinkagev3ppp2(channel=64, gap_size=1)
        self.p3_1 = nn.Sequential(
            nn.Conv1d(64, 10, kernel_size=43, stride=4, bias=True),
            nn.BatchNorm1d(10),
            nn.ReLU(inplace=True),
        )
        self.p3_2 = nn.MaxPool1d(kernel_size=2)
        self.p3_3 = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Linear(10, num_class)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                ksize = module.kernel_size[0]
                if ksize == 250:
                    base_filter = _laplace_wavelet_filter(module.out_channels, ksize)
                    if module.in_channels == 1:
                        weight = base_filter
                    else:
                        weight = base_filter.repeat(1, module.in_channels, 1) / math.sqrt(module.in_channels)
                    module.weight.data.copy_(weight.to(module.weight.device, dtype=module.weight.dtype))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
                else:
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.preconv is not None:
            x = self.preconv(x)
        if x.size(-1) < self.required_input_length:
            pad_len = self.required_input_length - x.size(-1)
            x = F.pad(x, (0, pad_len))
        x = self.p1_0(x)
        p1 = self.p1_3(self.p1_2(self.p1_1(x)))
        p2 = self.p2_6(self.p2_5(self.p2_4(self.p2_3(self.p2_2(self.p2_1(x))))))
        x = self.p3_2(self.p3_1(x + self.p3_0(x)))
        x = x + p1 + p2
        x = self.p3_3(x)
        return x.flatten(start_dim=1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)

    def forward(self, x: torch.Tensor):
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits, None


# ---------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------
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


def evaluate(loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    if total == 0:
        return 0.0, 0.0
    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


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
    bases = list_bases(sequences_directory, prefer_scaled=prefer_scaled)
    if not bases:
        console.print("[red]No complete bases found in data folder.[/]")
        return

    console.print(f"Found [bold]{len(bases)}[/] bases:")
    for base, use_scaled in bases:
        console.print(f" • {base}  (scaled={use_scaled})")

    overall_rows = []  # (base, acc, prec, rec, f1)

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
        if T < input_pad_length:
            console.print(
                f"[yellow]Sequence length {T} < required {input_pad_length}; zero-padding will be applied inside the model[/]"
            )

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

            # Model
            model = EWSNet(input_channel=Fdim, num_class=num_classes, min_length=input_pad_length).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
            criterion = nn.CrossEntropyLoss()

            # Early stopping on val_loss
            best_path = os.path.join(model_save_directory, f"{base}_best_fold{fold_idx}.pt")
            best_val_loss = float("inf")
            best_acc = 0.0
            no_improve = 0

            # Logs
            epochs_list, train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], [], []

            for epoch in trange(1, epochs + 1, desc=f"Training ({base}, fold {fold_idx})"):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                for xb, yb in train_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)

                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            logits, _ = model(xb)
                            loss = criterion(logits, yb)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        logits, _ = model(xb)
                        loss = criterion(logits, yb)
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * xb.size(0)
                    preds = logits.argmax(1)
                    correct += (preds == yb).sum().item()
                    total += yb.size(0)

                if total == 0:
                    train_loss = 0.0
                    train_acc = 0.0
                else:
                    train_loss = running_loss / total
                    train_acc = correct / total

                val_loss, val_acc = evaluate(val_loader, model, criterion, device)

                epochs_list.append(epoch)
                train_loss_list.append(train_loss)
                train_acc_list.append(train_acc)
                val_loss_list.append(val_loss)
                val_acc_list.append(val_acc)

                console.print(
                    f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
                    f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
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

                if no_improve >= patience:
                    console.log(f"[yellow]Early stopping[/] after {patience} epochs without val_loss improvement")
                    break

            # Save per-epoch metrics for this fold
            metrics_df = pd.DataFrame(
                {
                    "epoch": epochs_list,
                    "train_loss": train_loss_list,
                    "train_acc": train_acc_list,
                    "val_loss": val_loss_list,
                    "val_acc": val_acc_list,
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
