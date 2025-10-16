import os
import random
import importlib.util
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def set_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

HERE = Path(__file__).resolve().parent
BASE_FILE = HERE / "transformer_cluj_evaluate_justotomakesure.py"

def _import_module_from(path: Path, name: str = "tf_eval"):
    if not path.exists():
        raise FileNotFoundError(f"Missing module next to explain.py: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

tfmod = _import_module_from(BASE_FILE, "tf_eval")

NUM_CLASSES = getattr(tfmod, "num_classes")
BATCH_SIZE = getattr(tfmod, "batch_size")
DEFAULT_RESULTS_DIR = getattr(tfmod, "results_path_ltn")
DEFAULT_SEQ_DIR = getattr(tfmod, "sequences_directory")
DEFAULT_MODEL_DIR = getattr(tfmod, "model_save_directory")

def _resolve_dir(default_value: str, env_key: str) -> Path:
    v = os.environ.get(env_key, default_value)
    return Path(v).expanduser().resolve()

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

RESULTS_DIR = _resolve_dir(DEFAULT_RESULTS_DIR, "EXPLAIN_RESULTS_DIR")
SEQ_DIR = _resolve_dir(DEFAULT_SEQ_DIR, "EXPLAIN_SEQUENCES_DIR")
MODEL_DIR = _resolve_dir(DEFAULT_MODEL_DIR, "EXPLAIN_MODEL_DIR")

PLOTS_DIR = _ensure_dir(RESULTS_DIR / "plots_explain")
METRICS_DIR = _ensure_dir(RESULTS_DIR / "metrics_cluj")
#CLASS_ABBR_DEFAULT = ["HEA", "CTF", "MTF", "RCF", "SWF", "BWF", "CWF", "IRF", "ORF"] #DDS
CLASS_ABBR_DEFAULT = ["HEA","CTF-1","CTF-2","CTF-3","CTF-4","CTF-5","MTF","RCF","SPL"] #UOC

def _class_abbr(idx: int) -> str:
    try:
        if isinstance(CLASS_ABBR_DEFAULT, (list, tuple)) and len(CLASS_ABBR_DEFAULT) == NUM_CLASSES:
            return CLASS_ABBR_DEFAULT[idx]
    except Exception:
        pass
    # Fallback to numeric if something is off
    return str(idx)


def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "font.family": "DejaVu Sans",
        "figure.constrained_layout.use": True,
    })

def beautify_axes(ax, equal=False):
    ax.grid(False)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    if equal:
        ax.set_aspect("equal", adjustable="box")

def savefig(fig, out_base: Path, dpi: int = 300):
    # Strip any titles as a last step
    for ax in fig.get_axes():
        try:
            ax.set_title("")
        except Exception:
            pass
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base.with_suffix(".png")), dpi=dpi, bbox_inches="tight")
    fig.savefig(str(out_base.with_suffix(".pdf")), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _check_file(p: Path, hint: str):
    if not p.exists():
        raise FileNotFoundError(f"Missing {hint}: {p}")

def load_data(seq_dir: Path):
    train_x = seq_dir / "train_sequences.npy"
    train_y = seq_dir / "train_labels.npy"
    test_x = seq_dir / "test_sequences.npy"
    test_y = seq_dir / "test_labels.npy"
    for f in [train_x, train_y, test_x, test_y]:
        _check_file(f, "NPY")
    Xtr, ytr = tfmod.load_sequences(str(train_x), str(train_y))
    Xte, yte = tfmod.load_sequences(str(test_x), str(test_y))
    return Xtr, ytr, Xte, yte

def make_loader(X, y, bs: int, shuffle=False):
    X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
    y = torch.tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=shuffle, drop_last=False)

@torch.no_grad()
def collect_embeddings_and_probs(model, loader, device):
    model.eval()
    E, P, Y = [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits, _ = model(xb)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        emb = model.embed(xb).cpu().numpy()
        E.append(emb); P.append(probs); Y.append(yb.numpy())
    return np.concatenate(E), np.concatenate(P), np.concatenate(Y)

def batch_S_true_for_M(E_batch: np.ndarray, y_batch: np.ndarray, centroids) -> np.ndarray:
    N = E_batch.shape[0]
    out = np.zeros((N,), dtype=np.float32)
    for cls in range(NUM_CLASSES):
        idx = np.where(y_batch == cls)[0]
        if idx.size == 0:
            continue
        mu = centroids.get(cls, np.empty((0, E_batch.shape[1]), dtype=np.float32))
        if mu.size == 0:
            out[idx] = 0.0
            continue
        Ei = E_batch[idx]
        d2 = ((Ei[:, None, :] - mu[None, :, :])**2).sum(axis=2)
        s_vals = np.exp(-0.5 * d2).max(axis=1)
        out[idx] = s_vals.astype(np.float32)
    return out

def all_class_S(E: np.ndarray, centroids) -> np.ndarray:
    N, D = E.shape
    S = np.zeros((N, NUM_CLASSES), dtype=np.float32)
    for cls in range(NUM_CLASSES):
        mu = centroids.get(cls, np.empty((0, D), dtype=np.float32))
        if mu.size == 0:
            continue
        d2 = ((E[:, None, :] - mu[None, :, :])**2).sum(axis=2)
        S[:, cls] = np.exp(-0.5 * d2).max(axis=1)
    return S

def compute_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15):
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == y_true).astype(np.float32)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_confs, bin_accs = np.full(n_bins, np.nan), np.full(n_bins, np.nan)
    for b in range(n_bins):
        if b < n_bins - 1:
            mask = (conf >= bins[b]) & (conf < bins[b + 1])
        else:
            mask = (conf >= bins[b]) & (conf <= bins[b + 1])
        n_b = mask.sum()
        if n_b == 0:
            continue
        avg_conf = float(conf[mask].mean())
        avg_acc = float(correct[mask].mean())
        w = n_b / len(conf)
        ece += w * abs(avg_acc - avg_conf)
        bin_confs[b] = avg_conf
        bin_accs[b] = avg_acc
    return float(ece), bin_confs, bin_accs

def plot_reliability(bin_confs: np.ndarray, bin_accs: np.ndarray, out_path: Path):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
    m = ~np.isnan(bin_confs) & ~np.isnan(bin_accs)
    ax.plot([0, 1], [0, 1], ls="--", lw=1.0, c="gray", label="Perfect calibration")
    ax.plot(bin_confs[m], bin_accs[m], marker="o", lw=1.5, label="Model")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed accuracy")
    beautify_axes(ax, equal=False)
    ax.legend(loc="lower right")
    savefig(fig, out_path)

def plot_prob_matrix(M: np.ndarray, out_img: Path, title: str = "Probability mass when the class-i rule fires"):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)

    # High-contrast heatmap
    norm = mcolors.PowerNorm(gamma=0.6, vmin=0.0, vmax=1.0)
    im = ax.imshow(M, interpolation="nearest", cmap="magma", norm=norm)

    cbar = fig.colorbar(im, ax=ax, label="Average P(predicted=j | true=i, weighted by rule activation)")
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel("Predicted class j")
    ax.set_ylabel("True class i")

    ax.set_xticks(range(M.shape[1]))
    ax.set_yticks(range(M.shape[0]))

    # Subtle gridlines for readability
    ax.set_xticks(np.arange(-.5, M.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, M.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    beautify_axes(ax, equal=False)
    savefig(fig, out_img)

def nearest_centroid_and_training_exemplar(e_test: np.ndarray, cls_hat: int, centroids, E_train: np.ndarray, y_train: np.ndarray):
    mu = centroids.get(cls_hat, np.empty((0, e_test.shape[0]), dtype=np.float32))
    if mu.size == 0:
        return -1, float("inf"), -1, float("inf")
    d2 = ((mu - e_test[None, :])**2).sum(axis=1)
    i_star = int(np.argmin(d2))
    dist_mu = float(np.sqrt(d2[i_star]))
    mu_star = mu[i_star]
    idx_cls = np.where(y_train == cls_hat)[0]
    if idx_cls.size == 0:
        return i_star, dist_mu, -1, float("inf")
    E_cls = E_train[idx_cls]
    d2_ex = ((E_cls - mu_star[None, :])**2).sum(axis=1)
    j_local = int(np.argmin(d2_ex))
    train_idx = int(idx_cls[j_local])
    dist_ex = float(np.sqrt(d2_ex[j_local]))
    return i_star, dist_mu, train_idx, dist_ex

def plot_local_rationale_panel(x_test: np.ndarray, x_train_ex: np.ndarray, info_title: str, info_sub: str, out_path: Path, chan: int = 0):
    set_plot_style()
    T = x_test.shape[0]
    fig, ax = plt.subplots(2, 1, figsize=(8.6, 5.8), sharex=True, constrained_layout=True)
    ax[0].plot(np.arange(T), x_test[:, chan], lw=1.2)
    ax[0].set_ylabel("Amplitude")
    ax[0].text(0.01, 0.98, info_sub, transform=ax[0].transAxes, va="top", ha="left", fontsize=9)
    beautify_axes(ax[0], equal=False)
    ax[1].plot(np.arange(T), x_train_ex[:, chan], lw=1.2, color="tab:green")
    ax[1].set_ylabel("Amplitude")
    ax[1].set_xlabel("Time (sample)")
    beautify_axes(ax[1], equal=False)
    savefig(fig, out_path)

def _get_patch_spans(T: int, kernel: int, stride: int):
    starts = list(range(0, max(0, T - kernel) + 1, stride))
    return [(s, min(s + kernel, T)) for s in starts]

@torch.no_grad()
def _attention_rollout_from_attn_list(attn_list):
    A_blocks = [a.mean(dim=1) for a in attn_list]
    B, T, _ = A_blocks[0].shape
    I = torch.eye(T, device=A_blocks[0].device).unsqueeze(0).expand(B, -1, -1)
    R = I.clone()
    for A in A_blocks:
        A_hat = A + I
        A_hat = A_hat / (A_hat.sum(dim=-1, keepdim=True).clamp_min(1e-9))
        R = torch.bmm(A_hat, R)
    rollout = R[:, 0, 1:]
    rollout = rollout / rollout.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return rollout

def _mask_one_patch(x_single: torch.Tensor, span: tuple):
    xs = x_single.clone()
    s, e = span
    xs[:, s:e] = 0.0
    return xs

@torch.no_grad()
def _rule_truth_for_class(model, x_single: torch.Tensor, cls_idx: int, centroids_for_cls: np.ndarray, device):
    emb = model.embed(x_single.unsqueeze(0).to(device)).cpu().numpy()[0]
    if centroids_for_cls.size == 0:
        return 0.0
    dif = centroids_for_cls - emb[None, :]
    d2 = (dif * dif).sum(axis=1)
    s_vals = np.exp(-0.5 * d2)
    return float(s_vals.max())

def _fmt_array(a: np.ndarray, decimals: int = 6):
    if a.size == 0:
        return "[]"
    return "[" + ", ".join(f"{float(x):.{decimals}f}" for x in a) + "]"

def _plot_attention_panel(
    x_time: np.ndarray,
    rollout_vec: np.ndarray,
    deltaS_vec: np.ndarray,
    spans: list,
    out_path: Path,
    scale_rule: bool = True,
    print_values: bool = True,
    panel_id: str = ""
):
    set_plot_style()
    T = len(x_time)

    # Patch positions & labels (ASCII hyphen for portability)
    patch_x = np.arange(1, len(spans) + 1)
    patch_labels = [f"{s+1}-{e}" for (s, e) in spans]

    fig, ax = plt.subplots(2, 1, figsize=(10.0, 6.6), sharex=False, constrained_layout=True)

    # Top: raw signal
    ax[0].plot(np.arange(T), x_time, lw=1.2)
    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlabel("Time (sample)")
    ax[0].set_xticks([0, 5, 10, 15, 20] if T >= 20 else np.linspace(0, T-1, 5, dtype=int))
    beautify_axes(ax[0], equal=False)

    # Bottom: Attention + Rule sensitivity (ΔS)
    width = 0.40

    # Compute scaling for plotting only (ΔS can be orders smaller than attention)
    k = 1.0
    if scale_rule:
        attn_max = float(np.max(rollout_vec)) if rollout_vec.size else 0.0
        dmax = float(np.max(deltaS_vec)) if deltaS_vec.size else 0.0
        if attn_max > 0 and dmax > 0:
            k = attn_max / dmax
        else:
            k = 1.0
    deltaS_scaled = deltaS_vec * k

    # Bars
    ax[1].bar(patch_x - width/2, rollout_vec, width=width, label="Attention",
              color="tab:orange", alpha=0.9)
    ax[1].bar(patch_x + width/2, deltaS_scaled, width=width, label="Rule sensitivity",
              color="tab:blue", alpha=0.9)

    # Cosmetics
    ymax = max(1e-9, float(np.max(rollout_vec)) if rollout_vec.size else 0.0,
               float(np.max(deltaS_scaled)) if deltaS_scaled.size else 0.0)
    ax[1].set_ylim(0, ymax * 1.15)
    ax[1].set_ylabel("Value")
    ax[1].set_xlabel("Patch span (samples)")
    ax[1].set_xticks(patch_x)
    ax[1].set_xticklabels(patch_labels)
    ax[1].legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=2, borderaxespad=0.0)
    beautify_axes(ax[1], equal=False)

    # Print values for the plotted panel
    if print_values:
        print("\n[attention panel]" + (f" id={panel_id}" if panel_id else ""))
        print(f"  scale_factor_for_rule_sensitivity (plot-only) = {k:.6f}")
        print(f"  patch_spans = {patch_labels}")
        print(f"  Attention = { _fmt_array(rollout_vec, 6) }")
        print(f"  Rule_sensitivity_raw = { _fmt_array(deltaS_vec, 6) }")
        print(f"  Rule_sensitivity_scaled = { _fmt_array(deltaS_scaled, 6) }")

    savefig(fig, out_path)

@torch.no_grad()
def run_attention_rollout_checks(
    model,
    te_loader,
    centroids_dict: Dict[int, np.ndarray],
    device,
    plots_dir: Path,
    metrics_dir: Path,
    max_examples: int = 200,
    example_panels: int = 1,   # number of random panels to plot
    channel_to_plot: int = 0
):
    out_dir = plots_dir / "attention_rollout"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    kernel = model.patch_embed.kernel_size[0]
    stride = model.patch_embed.stride[0]

    n_done = 0
    panels_done = 0

    for xb, yb in te_loader:
        xb = xb.to(device, non_blocking=True)  # [B, C, T]
        B, C, T = xb.shape

        logits, attn_list = model(xb)
        rollout = _attention_rollout_from_attn_list(attn_list).cpu().numpy()  # [B, n_patches]
        probs = F.softmax(logits, dim=1)
        yhat = probs.argmax(dim=1).cpu().numpy()
        ytrue = yb.cpu().numpy()

        spans = _get_patch_spans(T, kernel, stride)

        # RANDOMIZE traversal order for this batch
        idx_order = list(range(B))
        random.shuffle(idx_order)

        for j, i in enumerate(idx_order):
            if n_done >= max_examples:
                break

            x_i = xb[i].detach().cpu()          # [C,T]
            y_i = int(ytrue[i])
            yh_i = int(yhat[i])
            rollout_i = rollout[i]              # [n_patches]

            C_for_hat = centroids_dict.get(
                yh_i, np.empty((0, model.embed(xb[:1]).shape[1]), dtype=np.float32)
            )
            S_full = _rule_truth_for_class(model, x_i, yh_i, C_for_hat, device)

            # Absolute ΔS via leave-one-patch-out
            deltaS = []
            for (s, e) in spans:
                xm = _mask_one_patch(x_i, (s, e))
                S_m = _rule_truth_for_class(model, xm, yh_i, C_for_hat, device)
                deltaS.append(S_full - S_m)
            deltaS = np.asarray(deltaS, dtype=np.float32)

            L = min(len(rollout_i), len(deltaS), len(spans))
            rollout_i = rollout_i[:L]
            deltaS = deltaS[:L]
            spans_L = spans[:L]

            # Correlation for logging
            if np.std(rollout_i) > 1e-9 and np.std(deltaS) > 1e-9:
                RA = float(np.corrcoef(rollout_i, deltaS)[0, 1])
            else:
                RA = float("nan")

            row = {
                "sample_index": n_done,
                "true": y_i,
                "pred": yh_i,
                "S_full_predclass": S_full,
                "RA_corr": RA
            }
            for p in range(L):
                s, e = spans_L[p]
                row[f"rollout_p{p}"] = float(rollout_i[p])
                row[f"deltaS_p{p}"] = float(deltaS[p])
                row[f"span_p{p}"] = f"{s}:{e}"
            rows.append(row)

            # Save the FIRST processed item in this batch if we still need panels.
            # Because idx_order is shuffled, this is random and guaranteed.
            if (panels_done < example_panels) and (j == 0):
                x_time = x_i[channel_to_plot].numpy()
                panel_path = out_dir / f"panel_{n_done:04d}"
                _plot_attention_panel(
                    x_time=x_time,
                    rollout_vec=rollout_i,
                    deltaS_vec=deltaS,
                    spans=spans_L,
                    out_path=panel_path,
                    scale_rule=True,
                    print_values=True,
                    panel_id=f"{n_done:04d}"
                )
                panels_done += 1

            n_done += 1

        if n_done >= max_examples:
            break

    # Save per-sample CSV
    df = pd.DataFrame(rows)
    df.to_csv(str(metrics_dir / "attention_rollout_per_sample.csv"), index=False)

    # Summary
    RA_vals = df["RA_corr"].replace([np.inf, -np.inf], np.nan).dropna().values
    if RA_vals.size > 0:
        RA_mean = float(np.mean(RA_vals))
        RA_std  = float(np.std(RA_vals))
        N = int(RA_vals.size)
    else:
        RA_mean, RA_std, N = float("nan"), float("nan"), 0

    with open(metrics_dir / "attention_rollout_summary.txt", "w") as f:
        f.write(f"RA_mean={RA_mean:.4f}, RA_std={RA_std:.4f}, N={N}\n")

    print(f"[attention] rows: {len(df)}  RA_mean={RA_mean:.4f} ± {RA_std:.4f} (N={N})")
    print(f"[attention] panels saved: {panels_done} → {out_dir}")

def plot_reliability(bin_confs: np.ndarray, bin_accs: np.ndarray, out_path: Path):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
    m = ~np.isnan(bin_confs) & ~np.isnan(bin_accs)
    ax.plot([0, 1], [0, 1], ls="--", lw=1.0, c="gray", label="Perfect calibration")
    ax.plot(bin_confs[m], bin_accs[m], marker="o", lw=1.5, label="Model")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Observed accuracy")
    beautify_axes(ax, equal=False)
    ax.legend(loc="lower right")
    savefig(fig, out_path)

def plot_prob_matrix(M: np.ndarray, out_img: Path, title: str = "Probability mass when the class-i rule fires"):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.4, 6.2), constrained_layout=True)
    norm = mcolors.PowerNorm(gamma=0.6, vmin=0.0, vmax=1.0)
    im = ax.imshow(M, interpolation="nearest", cmap="magma", norm=norm)
    cbar = fig.colorbar(im, ax=ax, label="Average P(predicted=j | true=i, weighted by rule activation)")
    cbar.ax.tick_params(labelsize=9)
    ax.set_xlabel("Predicted class j")
    ax.set_ylabel("True class i")
    ax.set_xticks(range(M.shape[1]))
    ax.set_yticks(range(M.shape[0]))
    ax.set_xticks(np.arange(-.5, M.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, M.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    beautify_axes(ax, equal=False)
    savefig(fig, out_img)

def plot_prob_bars_by_true_class(M: np.ndarray, out_img: Path, topk: int = None, font_scale: float = 1.75):
    """
    One small bar chart per true class i, showing P(pred=j | true=i).
    - First bar (largest prob) is hatched to de-emphasize it and ignored in y-scale.
    - y-axis fixed to [0.0, 0.5].
    - Always show value labels (even <1%).
    - First bar's value label is shifted further right to avoid overlap.
    - Uses class abbreviations (HEA, CTF, ..., ORF) for true/pred classes.
    - 'font_scale' enlarges all text in the plot (ticks, labels, value labels).
    """
    set_plot_style()
    K, C = M.shape

    # ---- sizes scaled by font_scale ----
    xtick_fs   = max(8, int(round(9 * font_scale)))
    ytick_fs   = max(8, int(round(9 * font_scale)))
    ylabel_fs  = max(9, int(round(11 * font_scale)))
    vlabel_fs  = max(8, int(round(9 * font_scale)))
    # enlarge figure accordingly
    ncols = min(3, K)
    nrows = int(np.ceil(K / ncols))
    fig_w = ncols * 3.4 * font_scale
    fig_h = nrows * 2.8 * font_scale

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        constrained_layout=True
    )

    for i in range(K):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        row = M[i].astype(float)

        # pick classes to show
        if topk is not None and topk < C:
            idx = np.argsort(row)[::-1][:topk]
            vals = row[idx]
            labels = [_class_abbr(j) for j in idx]
        else:
            idx = np.arange(C)
            vals = row
            labels = [_class_abbr(j) for j in idx]

        # sort descending inside the selection (keep labels in sync)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        labels = [labels[k] for k in order]

        x = np.arange(len(vals))
        bar_w = 0.8
        FIRST_LABEL_SHIFT = (bar_w / 2) + 0.05  # ~0.45 with bar_w=0.8

        # draw all "non-first" bars
        if len(vals) > 1:
            ax.bar(x[1:], vals[1:], width=bar_w, color="tab:blue", alpha=0.85)

        # first bar: hatched
        if len(vals) >= 1:
            ax.bar(
                x[0], vals[0], width=bar_w,
                facecolor="none", edgecolor="black",
                hatch="//", linewidth=1.0, alpha=1.0
            )

        # y-axis: fixed 0..0.5
        ax.set_ylim(0.0, 0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=xtick_fs)
        ax.set_yticks([0.0, 0.25, 0.5])
        ax.tick_params(axis="y", labelsize=ytick_fs)

        # TRUE class label with abbreviation
        ax.set_ylabel(f"true class {_class_abbr(i)}", fontsize=ylabel_fs)

        # value labels (always show; keep within axis bounds)
        for xi, v in zip(x, vals):
            y_text = min(max(v + 0.015, 0.015), 0.49)
            text = f"{100*v:.1f}%"
            if xi == 0:
                ax.text(
                    xi + FIRST_LABEL_SHIFT, y_text,
                    text, ha="left", va="bottom", fontsize=vlabel_fs
                )
            else:
                ax.text(
                    xi, y_text,
                    text, ha="center", va="bottom", fontsize=vlabel_fs
                )

        # minimal styling
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # remove unused axes
    for j in range(K, nrows * ncols):
        fig.delaxes(axes.flat[j])

    savefig(fig, out_img)



def main():
    set_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Resolved paths:")
    print(f"  RESULTS_DIR = {RESULTS_DIR}")
    print(f"  SEQ_DIR     = {SEQ_DIR}")
    print(f"  MODEL_DIR   = {MODEL_DIR}")
    Xtr, ytr, Xte, yte = load_data(SEQ_DIR)
    tr_loader = make_loader(Xtr, ytr, BATCH_SIZE, shuffle=False)
    te_loader = make_loader(Xte, yte, BATCH_SIZE, shuffle=False)
    best_path = MODEL_DIR / "best_model.pt"
    _check_file(best_path, "best model checkpoint")
    model = tfmod.VisionTransformer().to(device)
    state = torch.load(str(best_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Embeddings/probs
    E_tr, P_tr, Y_tr = collect_embeddings_and_probs(model, tr_loader, device)
    E_te, P_te, Y_te = collect_embeddings_and_probs(model, te_loader, device)

    # Accuracy
    acc = float((P_te.argmax(1) == Y_te).mean())
    print(f"Accuracy: {acc:.4f}")
    pd.DataFrame([{"accuracy": acc}]).to_csv(str(METRICS_DIR / "accuracies_explain.csv"), index=False)

    # Centroids
    emb_loader = DataLoader(
        TensorDataset(
            torch.tensor(Xtr, dtype=torch.float32).permute(0, 2, 1),
            torch.tensor(ytr, dtype=torch.long)
        ),
        batch_size=BATCH_SIZE, shuffle=False
    )
    centroids = tfmod.recompute_centroids(model, emb_loader, device)

    # Weighted prob matrix by rule activation
    S_true = batch_S_true_for_M(E_te, Y_te, centroids)
    M = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
    for c in range(NUM_CLASSES):
        mask = (Y_te == c)
        if not mask.any():
            continue
        w = S_true[mask].astype(np.float64)
        if np.allclose(w.sum(), 0.0):
            continue
        P_c = P_te[mask].astype(np.float64)
        num = (w[:, None] * P_c).sum(axis=0)
        den = w.sum()
        M[c, :] = num / max(den, 1e-12)

    pd.DataFrame(
        M,
        columns=[f"pred_{j}" for j in range(NUM_CLASSES)],
        index=[f"true_{i}" for i in range(NUM_CLASSES)]
    ).to_csv(str(METRICS_DIR / "prob_confusion_matrix_ltn.csv"))

    plot_prob_matrix(M, PLOTS_DIR / "prob_confusion_matrix_ltn")
    plot_prob_bars_by_true_class(M, PLOTS_DIR / "prob_confusion_bars_top5", topk=5)

    # Rule–classifier consistency
    S_all = all_class_S(E_te, centroids)
    eps = 1e-9
    A = S_all
    B = P_te
    imply = np.where(A <= B + 1e-12, 1.0, (B + eps) / (A + eps))
    Cons_c = imply.mean(axis=0)
    Cons_overall = float(Cons_c.mean())
    df_cons = pd.DataFrame({"class": list(range(NUM_CLASSES)), "consistency": Cons_c})
    df_cons.loc[len(df_cons)] = {"class": "overall", "consistency": Cons_overall}
    df_cons.to_csv(str(METRICS_DIR / "rule_classifier_consistency.csv"), index=False)
    print("Rule–classifier consistency (overall): {:.4f}".format(Cons_overall))

    # ECE + reliability
    ece, bin_confs, bin_accs = compute_ece(P_te, Y_te, n_bins=15)
    with open(METRICS_DIR / "ece_summary.txt", "w") as f:
        f.write(f"ECE={ece:.4f}\n")
    print(f"ECE: {ece:.4f}")
    plot_reliability(bin_confs, bin_accs, PLOTS_DIR / "reliability_diagram")

    # Local prototype rationale (unchanged selection)
    lr_dir = _ensure_dir(PLOTS_DIR / "local_rationale")
    rows = []
    probs_hat = P_te
    preds = probs_hat.argmax(axis=1)
    margins = probs_hat.max(axis=1) - np.partition(probs_hat, -2, axis=1)[:, -2]
    order = np.argsort(margins)
    n_panels = 1
    for k in range(n_panels):
        i = int(order[k])
        y_i = int(Y_te[i])
        yh_i = int(preds[i])
        e_i = E_te[i]
        c_idx, dist_mu, tr_idx, dist_ex = nearest_centroid_and_training_exemplar(
            e_test=e_i, cls_hat=yh_i, centroids=centroids, E_train=E_tr, y_train=Y_tr
        )
        mu_pred = centroids.get(yh_i, np.empty((0, E_tr.shape[1]), dtype=np.float32))
        if mu_pred.size == 0:
            S_pred = 0.0
        else:
            d2 = ((mu_pred - e_i[None, :])**2).sum(axis=1)
            S_pred = float(np.exp(-0.5 * d2).max())
        subtitle = f"Nearest centroid {c_idx}, distance {dist_mu:.3f}, rule score {S_pred:.3f}"
        x_test_tc = Xte[i]
        x_train_tc = Xtr[tr_idx] if tr_idx >= 0 else np.zeros_like(x_test_tc)
        plot_local_rationale_panel(
            x_test=x_test_tc,
            x_train_ex=x_train_tc,
            info_title="",
            info_sub=subtitle,
            out_path=lr_dir / f"panel_{i:05d}"
        )
        rows.append({
            "test_index": i, "true": y_i, "pred": yh_i,
            "centroid_idx": c_idx, "dist_to_centroid": dist_mu,
            "S_predclass": S_pred, "nearest_train_index": tr_idx,
            "dist_exemplar_to_centroid": dist_ex, "margin": float(margins[i])
        })
    pd.DataFrame(rows).to_csv(str(METRICS_DIR / "local_rationale.csv"), index=False)

    # Attention rollout (random example, ΔS scaled for plotting; prints values)
    run_attention_rollout_checks(
        model=model,
        te_loader=te_loader,
        centroids_dict=centroids,
        device=device,
        plots_dir=PLOTS_DIR,
        metrics_dir=METRICS_DIR,
        max_examples=200,
        example_panels=1,
        channel_to_plot=0
    )

    print("\nSaved to:")
    print(f"  Accuracies CSV:      {METRICS_DIR / 'accuracies_explain.csv'}")
    print(f"  M_LTN CSV:           {METRICS_DIR / 'prob_confusion_matrix_ltn.csv'}")
    print(f"  Consistency CSV:     {METRICS_DIR / 'rule_classifier_consistency.csv'}")
    print(f"  ECE summary:         {METRICS_DIR / 'ece_summary.txt'}")
    print(f"  Local rationale CSV: {METRICS_DIR / 'local_rationale.csv'}")
    print(f"  Attention CSV:       {METRICS_DIR / 'attention_rollout_per_sample.csv'}")
    print(f"  Attention summary:   {METRICS_DIR / 'attention_rollout_summary.txt'}")
    print(f"  Prob heatmap:        {PLOTS_DIR / 'prob_confusion_matrix_ltn.png'}")
    print(f"  Reliability:         {PLOTS_DIR / 'reliability_diagram.png'}")
    print(f"  Local panel:         {PLOTS_DIR / 'local_rationale'}")
    print(f"  Attention panel:     {PLOTS_DIR / 'attention_rollout'}")

if __name__ == "__main__":
    main()
