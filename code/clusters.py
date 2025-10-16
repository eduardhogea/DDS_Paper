#!/usr/bin/env python3
# Embedding sweep + per-class KMeans with:
# - Panel A: metrics over ks_metrics (e.g., 1,2,3,4,5,9,10,20) with dual y-axes
# - Panels B–E: two ks (default 2,3) for class-focused views
# - Optional AUTO-PICK of representative classes and downsample_per_class to minimize 2σ ellipse overlap in t-SNE
# - Single, publication-grade composite figure with a RIGHT legend

import os, argparse, numpy as np, torch, matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7.3,
    "axes.titlesize": 9.2,
    "axes.labelsize": 8.4,
    "legend.fontsize": 7.0,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
})

from transformer_cluj_evaluate import VisionTransformer  # must be importable

# ---------------- I/O ----------------
def load_data(seq_path, lab_path):
    X = np.load(seq_path).astype(np.float32)   # (N, T, C)
    y = np.load(lab_path)
    assert X.ndim == 3 and len(y) == len(X), f"Data shapes mismatch: {X.shape}, {y.shape}"
    return X, y

def extract_embeddings(model_ckpt, X_cf, batch=512):
    model = VisionTransformer()
    sd = torch.load(model_ckpt, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X_cf), batch):
            xb = torch.from_numpy(X_cf[i:i+batch])
            outs.append(model.embed(xb).numpy())
    return np.concatenate(outs, 0)

# ---------------- Clustering + metrics ----------------
def per_class_kmeans(E, y, k):
    sublab = -np.ones(len(E), dtype=int)
    centroids = {}
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        Em = E[idx]
        kk = min(k, len(Em)) if len(Em) >= 2 else 1
        if len(Em) < 2:
            centroids[int(c)] = Em.mean(axis=0, keepdims=True)
            sublab[idx] = 0
        else:
            km = KMeans(n_clusters=kk, n_init=10, random_state=0).fit(Em)
            sublab[idx] = km.labels_
            centroids[int(c)] = km.cluster_centers_
    return sublab, centroids

def silhouette_within_class(E, y, sublab):
    vals = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        labs = sublab[idx]
        if len(idx) < 3 or len(np.unique(labs)) < 2:
            continue
        try:
            vals.append(silhouette_score(E[idx], labs))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else float("nan")

def support_stats(y, sublab, k):
    medians, tiny_pct = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        n = len(idx)
        if n == 0:
            continue
        counts = np.array([np.sum(sublab[idx] == sc) for sc in range(k)], dtype=int)
        medians.append(float(np.median(counts)))
        tiny = np.sum(counts < max(1, int(0.01 * n)))  # <1% of class (>=1)
        tiny_pct.append(100.0 * tiny / max(1, len(counts)))
    return (float(np.median(medians)) if medians else float("nan"),
            float(np.mean(tiny_pct)) if tiny_pct else float("nan"))

def project_centroids_linear(E, Z, centroids):
    """Fit a linear map from high-D E to 2D Z and project given centroids."""
    E_aug = np.hstack([E, np.ones((len(E), 1))])
    A, *_ = np.linalg.lstsq(E_aug, Z, rcond=None)
    return {c: np.hstack([Cc, np.ones((len(Cc), 1))]) @ A for c, Cc in centroids.items()}

# ---------------- Visual helpers ----------------
def _plot_cov_ellipse(ax, pts, n_std=2.0, face='C0', edge='C0', lw=1.4, alpha=0.14, z=1.2):
    if pts.shape[0] < 3:
        return
    mu = pts.mean(axis=0)
    C = np.cov(pts.T)
    if not np.all(np.isfinite(C)):
        return
    vals, vecs = np.linalg.eigh(C)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-12))
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    e = mpatches.Ellipse((float(mu[0]), float(mu[1])),
                         float(width), float(height),
                         angle=float(theta),
                         facecolor=face, edgecolor=edge,
                         alpha=alpha, linewidth=lw, zorder=z)
    ax.add_patch(e)

def _mean_iqr_finite(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return float(arr.mean()), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))

def _mean_sd_ci(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan, np.nan, 0
    m = float(arr.mean()); sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    se = sd / max(1.0, np.sqrt(arr.size))
    ci = 1.96 * se
    return m, sd, ci, arr.size

# Global per-k panel (saved individually; not used in composite layout)
def plot_panel(Z, y, sublab, Cproj, k, out_png, title, point_size=7.5, point_alpha=0.72):
    markers = ['o', '^', 's', 'D', 'P', 'v', 'X']
    plt.figure(figsize=(7,5.7))
    cmap = plt.get_cmap("tab10")
    for ci, c in enumerate(np.unique(y)):
        idx = np.where(y == c)[0]
        for sc in sorted(np.unique(sublab[idx])):
            m = idx[sublab[idx] == sc]
            col = cmap(sc % 10)
            plt.scatter(Z[m,0], Z[m,1], s=point_size, alpha=point_alpha,
                        marker=markers[int(sc) % len(markers)],
                        edgecolors='white', linewidths=0.25, c=[col])
    for c, CZ in Cproj.items():
        for j in range(CZ.shape[0]):
            col = cmap(j % 10)
            plt.scatter(CZ[j,0], CZ[j,1], s=65, marker='x',
                        linewidths=2.1, c=[col], zorder=4)
    plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()

# ---------------- Extra statistics ----------------
def _safe_inv_cov(X):
    C = np.cov(X.T)
    if not np.all(np.isfinite(C)):
        C = np.eye(X.shape[1])
    lam = 1e-6 * np.trace(C) / C.shape[0]
    try:
        inv = np.linalg.inv(C + lam * np.eye(C.shape[0]))
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(C + lam * np.eye(C.shape[0]))
    return inv

def _cohens_d_along_axis(Em, labels, mu0, mu1):
    if mu0 is None or mu1 is None: return np.nan
    d = mu1 - mu0
    nrm = np.linalg.norm(d)
    if nrm < 1e-9: return 0.0
    d = d / nrm
    proj = Em @ d
    a = proj[labels == 0]; b = proj[labels == 1]
    if len(a) < 2 or len(b) < 2: return np.nan
    m1, m2 = a.mean(), b.mean()
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    sp = np.sqrt(((len(a)-1)*s1*s1 + (len(b)-1)*s2*s2) / max(1, (len(a)+len(b)-2)))
    return float(abs(m1 - m2) / (sp + 1e-9))

def per_class_stats(E, y, sublab, cents, k):
    rows = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        Em = E[idx]
        labs = sublab[idx]
        row = {"class": int(c), "k": int(k), "n": int(len(idx))}
        counts = np.array([np.sum(labs == j) for j in range(k)], dtype=int)
        row["median_support"] = float(np.median(counts)) if counts.size else np.nan
        row["balance_ratio"]  = float(counts.min() / max(1, counts.max())) if counts.size else np.nan
        if len(np.unique(labs)) >= 2 and len(idx) >= 3:
            try: row["silhouette"] = float(silhouette_score(Em, labs))
            except Exception: row["silhouette"] = np.nan
        else:
            row["silhouette"] = np.nan
        row["cohens_d"] = np.nan
        row["mahal_sq"] = np.nan
        if k >= 2 and c in cents and cents[c].shape[0] >= 2:
            mu0, mu1 = cents[c][0], cents[c][1]
            row["cohens_d"] = _cohens_d_along_axis(Em, labs, mu0, mu1)
            invS = _safe_inv_cov(Em)
            diff = (mu1 - mu0)
            row["mahal_sq"] = float(diff @ invS @ diff)
        rows.append(row)
    return pd.DataFrame(rows)

def zscore_series(s: pd.Series) -> pd.Series:
    v = s.values.astype(float)
    sd = np.nanstd(v)
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(np.zeros_like(v), index=s.index)
    return (s - np.nanmean(v)) / sd

def pick_top_classes(stats_k2, topn=2, min_support=20):
    df = stats_k2.copy()
    df = df[df["k"] == 2].copy()
    df = df[df["median_support"] >= min_support]
    s = zscore_series(df["silhouette"]) + zscore_series(df["cohens_d"]) + \
        zscore_series(np.log1p(df["mahal_sq"])) + zscore_series(df["balance_ratio"])
    df["rank_score"] = s
    df = df.sort_values("rank_score", ascending=False)
    return df["class"].astype(int).head(topn).tolist(), df

# ---------------- Embedding sweep helpers ----------------
def postprocess(E, dim):
    if dim is None or dim < 0 or dim >= E.shape[1]:
        return E
    p = PCA(n_components=dim, whiten=True, random_state=0).fit(E)
    return p.transform(E)

def ari_stability(E, y, k, reps=5, frac=0.8):
    rng = np.random.default_rng(0)
    aris = []
    for _ in range(reps):
        keep = np.concatenate([
            rng.choice(np.where(y==c)[0],
                       size=max(2, int(frac*np.sum(y==c))), replace=False)
            for c in np.unique(y)
        ])
        sublab,_ = per_class_kmeans(E[keep], y[keep], k)
        jitter = E[keep] + 1e-3*rng.normal(size=E[keep].shape)
        sublab2,_ = per_class_kmeans(jitter, y[keep], k)
        aris.append(adjusted_rand_score(sublab, sublab2))
    return float(np.mean(aris)) if len(aris) else float("nan")

# ---------------- Visual overlap (t-SNE plane) ----------------
def _ellipse_membership(X2d, mu, cov_inv, sigma2):
    """Mahalanobis membership: (x-mu)^T cov^{-1} (x-mu) <= sigma^2 ?"""
    d = X2d - mu
    m = np.einsum('...i,ij,...j->...', d, cov_inv, d)
    return m <= sigma2

def class_overlap_score_Z(Z, y_map, sublab_map, focus_class, sigma=2.0):
    """Return worst-pair symmetric overlap fraction for a class in the 2D map."""
    idx = np.where(y_map == focus_class)[0]
    if len(idx) < 4:
        return 1.0  # too small -> treat as bad
    labs = sublab_map[idx]
    uniq = np.unique(labs)
    if len(uniq) < 2:
        return 1.0
    pts = {sc: Z[idx[labs==sc]] for sc in uniq}
    # precompute ellipses
    ell = {}
    for sc in uniq:
        P = pts[sc]
        if len(P) < 3:
            return 1.0
        mu = P.mean(axis=0)
        C = np.cov(P.T)
        # regularize
        lam = 1e-6 * np.trace(C) / C.shape[0]
        try:
            inv = np.linalg.inv(C + lam*np.eye(2))
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(C + lam*np.eye(2))
        ell[sc] = (mu, inv)
    # compute symmetric overlap between all pairs
    sigma2 = float(sigma*sigma)
    worst = 0.0
    for i in uniq:
        for j in uniq:
            if i == j: continue
            mu_j, inv_j = ell[j]
            inside_ij = _ellipse_membership(pts[i], mu_j, inv_j, sigma2)
            frac_ij = float(np.mean(inside_ij)) if len(inside_ij) else 0.0
            mu_i, inv_i = ell[i]
            inside_ji = _ellipse_membership(pts[j], mu_i, inv_i, sigma2)
            frac_ji = float(np.mean(inside_ji)) if len(inside_ji) else 0.0
            sym = max(frac_ij, frac_ji)  # worst direction
            worst = max(worst, sym)
    return worst  # lower is better

# ---------------- Composite figure ----------------
def make_composite_figure_with_picks(
    E_full, y_full, E_map, y_map, Z,
    subs_full, subs_map, cents_full,
    ks_metrics, ks_panels, chosen, out_path,
    fig_width_in=7.2, fig_height_in=None, aspect=0.50, dpi=600,
    legend_side="right", metrics_band="iqr",
):
    import matplotlib.gridspec as gridspec
    import numpy as _np

    # (A) Metrics on FULL data over ks_metrics
    sil_mean, band_lo, band_hi, med_supports = [], [], [], []
    for k in ks_metrics:
        vals = []
        sub_full = subs_full[k]
        for c in np.unique(y_full):
            idx = np.where(y_full==c)[0]
            labs = sub_full[idx]
            if len(idx)>=3 and len(np.unique(labs))>=2:
                try: vals.append(silhouette_score(E_full[idx], labs))
                except: pass

        if metrics_band == "iqr":
            m, p25, p75 = _mean_iqr_finite(vals)
            sil_mean.append(m); band_lo.append(p25); band_hi.append(p75)
        else:
            m, sd, ci, n = _mean_sd_ci(vals)
            sil_mean.append(m)
            if metrics_band == "sd":
                band_lo.append(m - sd); band_hi.append(m + sd)
            elif metrics_band == "ci":
                band_lo.append(m - ci); band_hi.append(m + ci)
            else:  # none
                band_lo.append(_np.nan); band_hi.append(_np.nan)

        med, _ = support_stats(y_full, sub_full, k)
        med_supports.append(med)
    import os as _os, pandas as _pd, numpy as _np

    out_dir = _os.path.dirname(out_path)
    panelA_df = _pd.DataFrame({
        "k": ks_metrics,
        "sil_mean": sil_mean,            # mean within-class silhouette (Panel A blue line)
        "sil_band_lo": band_lo,          # exactly the band we plot (IQR/SD/CI depending on --metrics_band)
        "sil_band_hi": band_hi,
        "median_support": med_supports,  # Panel A orange dashed line (right y-axis)
    })

    # Also mark where k=2 sits relative to the max silhouette
    try:
        sil_arr = _np.array(sil_mean, dtype=float)
        k_max = ks_metrics[int(_np.nanargmax(sil_arr))]
        k_min = ks_metrics[int(_np.nanargmin(sil_arr))]
        panelA_df["delta_to_max"] = panelA_df["sil_mean"] - _np.nanmax(sil_arr)
        panelA_df["is_k2"] = panelA_df["k"].eq(2).astype(int)
    except Exception:
        k_max = None; k_min = None
        panelA_df["delta_to_max"] = _np.nan
        panelA_df["is_k2"] = 0

    csv_path = _os.path.join(out_dir, "panelA_values.csv")
    panelA_df.to_csv(csv_path, index=False)

    # Console printout (exactly what is plotted)
    print("\nPanel A values (exactly as plotted):")
    for k, m, lo, hi, ms in zip(ks_metrics, sil_mean, band_lo, band_hi, med_supports):
        print(f"  k={k:>2} | sil_mean={m:.4f} | band=[{lo:.4f}, {hi:.4f}] | median_support={ms:.0f}")
    if k_max is not None:
        print(f"  -> silhouette max at k={k_max}; min at k={k_min}. CSV: {csv_path}\n")

    # choose classes
    all_classes = list(np.unique(y_full))
    chosen = (chosen + [c for c in all_classes if c not in chosen])[:2]
    c1, c2 = int(chosen[0]), int(chosen[1])

    # two ks for panels
    kA, kB = (ks_panels + ks_panels)[:2]
    kA, kB = int(kA), int(kB)

    if fig_height_in is None:
        fig_height_in = fig_width_in * aspect

    fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.05, 1, 1], height_ratios=[1, 1])

    # (A) metrics with dual y-axes
    axA = fig.add_subplot(gs[:,0])

    # Left axis: silhouette (blue)
    l1, = axA.plot(ks_metrics, sil_mean, marker='o', linewidth=1.7, color='C0',
                   label="silhouette (mean)")
    band = None
    if metrics_band != "none" and _np.isfinite(band_lo).any():
        band = axA.fill_between(ks_metrics, band_lo, band_hi, alpha=0.15,
                                color='C0', label=f"{metrics_band} band")
    axA.set_xlabel("k")
    axA.set_ylabel("Within-class silhouette", color='C0')
    axA.tick_params(axis='y', labelcolor='C0')
    axA.spines['left'].set_color('C0')

    # Right axis: median support (orange), integer ticks
    axA2 = axA.twinx()
    l2, = axA2.plot(ks_metrics, med_supports, marker='s', linestyle='--',
                    linewidth=1.6, color='C1', label="median support")
    axA2.set_ylabel("Median support (samples)", color='C1')
    axA2.tick_params(axis='y', labelcolor='C1')
    axA2.spines['right'].set_color('C1')
    axA2.yaxis.set_major_locator(MaxNLocator(integer=True))

    axA.set_title("A  Within-class separation & support vs k")
    axA.grid(False)

    # helper: focused class panel
    def draw_focus(ax, cls, k, title):
        sub_map = subs_map[k]
        m_other = (y_map != cls)
        ax.scatter(Z[m_other,0], Z[m_other,1], s=5.5, alpha=0.06, c='grey', zorder=1)
        idx = np.where(y_map == cls)[0]
        labs = sub_map[idx]
        cmap = plt.get_cmap("tab10")
        markers = ['o','^','s','D','P','v','X']
        for sc in sorted(np.unique(labs)):
            m = idx[labs==sc]
            col = cmap(int(sc) % 10)
            _plot_cov_ellipse(ax, Z[m], n_std=2.0, face=col, edge=col, lw=1.4, alpha=0.13, z=1.2)
            ax.scatter(Z[m,0], Z[m,1], s=10, alpha=0.9,
                       marker=markers[int(sc)%len(markers)], zorder=2,
                       edgecolors='white', linewidths=0.25, c=[col])
        cents_k = cents_full[k]
        if cls in cents_k and cents_k[cls].size:
            # Project centroids using the E_map->Z linear map
            Cproj = project_centroids_linear(E_map, Z, {cls: cents_k[cls]})[cls]
            for j in range(Cproj.shape[0]):
                col = plt.get_cmap("tab10")(j % 10)
                ax.scatter(Cproj[j,0], Cproj[j,1], s=70, marker='x',
                           linewidths=2.2, c=[col], zorder=4)
        ax.set_title(title, fontsize=9.2)
        ax.tick_params(labelbottom=False, labelleft=False)

    # (B–E)
    axB = fig.add_subplot(gs[0,1]); draw_focus(axB, c1, kA, f"B  class {c1}  k={kA}")
    axC = fig.add_subplot(gs[0,2]); draw_focus(axC, c1, kB, f"C  class {c1}  k={kB}")
    axD = fig.add_subplot(gs[1,1]); draw_focus(axD, c2, kA, f"D  class {c2}  k={kA}")
    axE = fig.add_subplot(gs[1,2]); draw_focus(axE, c2, kB, f"E  class {c2}  k={kB}")

    # RIGHT legend
    handles_metrics = [l1] + ([band] if band is not None else []) + [l2]
    labels_metrics  = ["silhouette (mean)"] + ([f"{metrics_band} band"] if band is not None else []) + ["median support"]
    sub_handle = mlines.Line2D([], [], linestyle='None', marker='o',
                               markeredgecolor='white', markerfacecolor='C0',
                               markeredgewidth=0.25, markersize=6.5, label='sub-cluster')
    centroid_handle = mlines.Line2D([], [], linestyle='None', marker='x',
                                    markersize=6.5, markeredgewidth=2.0,
                                    color='black', label='centroid')
    ellipse_handle  = mpatches.Ellipse((0,0), 1.0, 0.6, angle=0,
                                       facecolor='C0', edgecolor='C0', alpha=0.14, linewidth=1.4)
    handles_cluster = [sub_handle, centroid_handle, ellipse_handle]
    labels_cluster  = ["sub-cluster", "centroid", "2σ covariance"]

    fig.tight_layout(rect=[0, 0, 0.82, 1])  # leave room on the right
    fig.legend(
        handles_metrics + handles_cluster,
        labels_metrics + labels_cluster,
        loc="center left", bbox_to_anchor=(0.845, 0.5),
        ncol=1, frameon=True, fontsize=7, handlelength=2.4,
        borderaxespad=0.2, labelspacing=0.8
    )

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--embed_model", default="", help="single checkpoint (used if --embed_models is empty)")
    ap.add_argument("--embed_models", nargs="+", default=[],
                    help="List of checkpoints to sweep. If set, overrides --embed_model.")
    ap.add_argument("--pca_dims", nargs="+", type=int, default=[-1, 128])
    ap.add_argument("--stability_reps", type=int, default=5)
    ap.add_argument("--ks_metrics", nargs="+", type=int, default=[1,2,3,4,5,9,10,20])
    ap.add_argument("--ks_panels",  nargs="+", type=int, default=[2,3])
    ap.add_argument("--outdir", default="plots_k_sweep_fixed")
    ap.add_argument("--downsample_per_class", type=int, default=0)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--fig_width_in", type=float, default=8.2)
    ap.add_argument("--fig_height_in", type=float, default=None, help="Override height (inches). If not set, uses width*aspect.")
    ap.add_argument("--aspect", type=float, default=0.50, help="Height/width when fig_height_in not set (smaller -> shorter).")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--rep_classes", nargs="+", type=int, default=[])
    ap.add_argument("--legend_side", choices=["right"], default="right")
    ap.add_argument("--metrics_band", choices=["iqr","sd","ci","none"], default="iqr")
    # AUTO PICK
    ap.add_argument("--auto_pick", action="store_true",
                    help="Search classes and downsample_per_class to minimize 2σ ellipse overlap in t-SNE.")
    ap.add_argument("--ds_candidates", nargs="+", type=int, default=[40,60,80],
                    help="Candidates for downsample_per_class when --auto_pick.")
    ap.add_argument("--ellipse_sigma", type=float, default=2.0,
                    help="Sigma for ellipse overlap test (default 2.0).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    X, y = load_data(args.sequences, args.labels)

    # ---- Embedding sweep (models × PCA dims), rank with k=2/3 stats ----
    models = args.embed_models if len(args.embed_models) else ([args.embed_model] if args.embed_model else [])
    if not models:
        raise SystemExit("Provide --embed_model or --embed_models.")

    X_cf_full = np.transpose(X, (0,2,1))
    sweep_rows, Es = [], {}

    for mpath in models:
        E_base = extract_embeddings(mpath, X_cf_full)
        for d in args.pca_dims:
            E = postprocess(E_base, d); Es[(mpath, d)] = E
            sub2, cents2 = per_class_kmeans(E, y, 2)
            sub3, _      = per_class_kmeans(E, y, 3)
            df2 = per_class_stats(E, y, sub2, cents2, 2)
            sil_m   = float(np.nanmean(df2["silhouette"]))      if "silhouette"     in df2 else np.nan
            d_m     = float(np.nanmean(df2["cohens_d"]))        if "cohens_d"       in df2 else np.nan
            mahal_m = float(np.nanmean(df2["mahal_sq"]))        if "mahal_sq"       in df2 else np.nan
            bal_m   = float(np.nanmean(df2["balance_ratio"]))   if "balance_ratio"  in df2 else np.nan
            med2,_  = support_stats(y, sub2, 2)
            med3,_  = support_stats(y, sub3, 3)
            df3     = per_class_stats(E, y, sub3, {}, 3)
            d_sil   = float(np.nanmean(df3["silhouette"])) - sil_m
            stab    = ari_stability(E, y, 2, reps=args.stability_reps)
            sweep_rows.append({
                "model": os.path.basename(mpath), "pca_dim": d,
                "sil_k2_mean": sil_m, "cohensd_k2_mean": d_m,
                "mahal_k2_mean": mahal_m, "balance_k2_mean": bal_m,
                "stability_ari_k2": stab,
                "med_support_k2": med2, "med_support_k3": med3,
                "delta_sil_2to3": d_sil
            })

    rank = pd.DataFrame(sweep_rows)
    rank["frag_support_drop"]  = rank["med_support_k2"] - rank["med_support_k3"]
    rank["delta_sil_2to3_pos"] = rank["delta_sil_2to3"].clip(lower=0)
    pos_cols = ["sil_k2_mean","cohensd_k2_mean","mahal_k2_mean","balance_k2_mean","stability_ari_k2"]
    neg_cols = ["frag_support_drop","delta_sil_2to3_pos"]
    for c in pos_cols + neg_cols:
        v = rank[c].values.astype(float); sd = np.nanstd(v)
        rank[c+"_z"] = 0.0 if (not np.isfinite(sd) or sd<1e-12) else (rank[c]-np.nanmean(v))/sd
    rank["composite"] = rank[[c+"_z" for c in pos_cols]].sum(axis=1) - rank[[c+"_z" for c in neg_cols]].sum(axis=1)
    rank["prefer_raw"] = (rank["pca_dim"] == -1).astype(int)
    rank = rank.sort_values(["composite","prefer_raw","sil_k2_mean","med_support_k2"],
                            ascending=[False, False, False, False]).reset_index(drop=True)
    rank.to_csv(os.path.join(args.outdir, "embedding_ranking.csv"), index=False)
    print("Top embeddings:\n", rank.head(5))
    best_row  = rank.iloc[0]
    best_dim  = int(best_row["pca_dim"])
    best_path = next(p for p in (args.embed_models if len(args.embed_models) else [args.embed_model])
                     if os.path.basename(p) == best_row["model"])
    E = Es[(best_path, best_dim)]
    print(f"Selected embedding: {os.path.basename(best_path)}, PCA={best_dim} (composite={best_row['composite']:.3f})")

    # ---- Build labels/centroids for ALL k we need (independent of t-SNE) ----
    ks_needed = sorted(set(args.ks_metrics) | set(args.ks_panels))
    sublabs_by_k, cents_by_k = {}, {}
    for k in ks_needed:
        sublab, cents = per_class_kmeans(E, y, k)
        sublabs_by_k[k] = sublab
        cents_by_k[k]   = cents

    # Per-class stats across ALL needed k (for tables / E-space metrics)
    stats_rows = []
    for k in ks_needed:
        dfk = per_class_stats(E, y, sublabs_by_k[k], cents_by_k[k], k)
        stats_rows.append(dfk)
    stats_all = pd.concat(stats_rows, ignore_index=True)
    stats_all.to_csv(os.path.join(args.outdir, "cluster_stats_by_class.csv"), index=False)

    # ----- AUTO-PICK: choose downsample_per_class and classes to minimize overlap in Z
    if args.auto_pick:
        print("Auto-pick: scanning downsample & classes for minimal 2σ-ellipse overlap...")
        rng = np.random.default_rng(0)
        best = {"score": -1e9}

        # Precompute per-class E-space silhouettes for ks_panels
        sil_by_class_k = {}
        for k in args.ks_panels:
            dfk = stats_all[stats_all["k"]==k]
            sil_by_class_k[k] = dfk.set_index("class")["silhouette"].to_dict()

        for ds in args.ds_candidates:
            # stratified sample
            idxs = []
            for c in np.unique(y):
                cand = np.where(y == c)[0]
                take = min(ds, len(cand))
                idxs.extend(rng.choice(cand, size=take, replace=False))
            idx_tsne = np.array(sorted(idxs))

            E_map = E[idx_tsne]; y_map = y[idx_tsne]
            Z = TSNE(n_components=2, init="pca", perplexity=args.perplexity,
                     learning_rate='auto', random_state=0).fit_transform(E_map)

            # evaluate classes
            vis_scores = []
            for cls in np.unique(y_map):
                per_k_overlap = []
                per_k_sil = []
                for k in args.ks_panels:
                    sub_map = sublabs_by_k[k][idx_tsne]
                    ov = class_overlap_score_Z(Z, y_map, sub_map, cls, sigma=args.ellipse_sigma)
                    per_k_overlap.append(ov)
                    # E-space silhouette for this class/k
                    s = sil_by_class_k.get(k, {}).get(int(cls), np.nan)
                    per_k_sil.append(s if np.isfinite(s) else 0.0)
                worst_ov = float(np.max(per_k_overlap)) if len(per_k_overlap) else 1.0
                mean_sil = float(np.mean(per_k_sil)) if len(per_k_sil) else 0.0
                score = (1.0 - worst_ov) + 0.5 * max(0.0, mean_sil)
                # tiny classes get de-prioritized implicitly by overlap instability; still keep score >= -inf
                vis_scores.append((score, 1.0-worst_ov, mean_sil, int(cls)))

            # pick top-2 classes
            vis_scores.sort(reverse=True)  # by score
            top2 = [v[3] for v in vis_scores[:2]] if len(vis_scores)>=2 else [vis_scores[0][3]]*2

            # objective: sum of top-2 scores
            total = vis_scores[0][0] + vis_scores[1][0] if len(vis_scores)>=2 else vis_scores[0][0]
            if total > best["score"]:
                best = {"score": total, "ds": ds, "classes": top2, "Z": Z, "idx_tsne": idx_tsne}

        # adopt best choice
        idx_tsne = best["idx_tsne"]; Z = best["Z"]
        rep_classes = best["classes"]
        print(f"Auto-pick selected downsample_per_class={best['ds']}, classes={rep_classes}, objective={best['score']:.3f}")
    else:
        # manual / previous behavior
        if args.downsample_per_class and args.downsample_per_class > 0:
            rng = np.random.default_rng(0)
            idxs = []
            for c in np.unique(y):
                cand = np.where(y == c)[0]
                take = min(args.downsample_per_class, len(cand))
                idxs.extend(rng.choice(cand, size=take, replace=False))
            idx_tsne = np.array(sorted(idxs))
        else:
            idx_tsne = np.arange(len(X))
        E_map = E[idx_tsne]; y_map = y[idx_tsne]
        Z = TSNE(n_components=2, init="pca", perplexity=args.perplexity,
                 learning_rate='auto', random_state=0).fit_transform(E_map)
        # default class choice
        top_classes, _ = pick_top_classes(stats_all, topn=2, min_support=max(10, int(0.01*len(y))))
        rep_classes = args.rep_classes[:2] if args.rep_classes else top_classes

    # ---- Save per-k global PNGs and summary for metrics ks only (using chosen idx_tsne/Z)
    E_map = E[idx_tsne]; y_map = y[idx_tsne]
    rows = []
    for k in args.ks_metrics:
        sil_c = silhouette_within_class(E, y, sublabs_by_k[k])
        med_support, tiny_pct = support_stats(y, sublabs_by_k[k], k)
        Cproj = project_centroids_linear(E_map, Z, cents_by_k[k])
        sub_tsne = sublabs_by_k[k][idx_tsne]
        title = f"t-SNE (fixed embedding) — k={k} • within-class silhouette={sil_c:.3f} • median support={med_support:.0f} • tiny%={tiny_pct:.1f}"
        plt_fname = os.path.join(args.outdir, f"tsne_k{k}.png")
        plot_panel(Z, y_map, sub_tsne, Cproj, k, plt_fname, title)
        rows.append({"k": k, "within_class_silhouette": sil_c,
                     "median_support": med_support, "tiny_centroids_pct": tiny_pct})
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "summary_metrics.csv"), index=False)

    # ---- Composite figure (PNG + PDF)
    subs_map = {k: sublabs_by_k[k][idx_tsne] for k in ks_needed}
    composite_png = os.path.join(args.outdir, "figure_k_choice.png")
    composite_pdf = os.path.join(args.outdir, "figure_k_choice.pdf")
    make_composite_figure_with_picks(
        E_full=E, y_full=y, E_map=E_map, y_map=y_map, Z=Z,
        subs_full=sublabs_by_k, subs_map=subs_map, cents_full=cents_by_k,
        ks_metrics=args.ks_metrics, ks_panels=args.ks_panels, chosen=rep_classes,
        out_path=composite_png, fig_width_in=args.fig_width_in,
        fig_height_in=args.fig_height_in, aspect=args.aspect, dpi=args.dpi,
        legend_side=args.legend_side, metrics_band=args.metrics_band
    )
    make_composite_figure_with_picks(
        E_full=E, y_full=y, E_map=E_map, y_map=y_map, Z=Z,
        subs_full=sublabs_by_k, subs_map=subs_map, cents_full=cents_by_k,
        ks_metrics=args.ks_metrics, ks_panels=args.ks_panels, chosen=rep_classes,
        out_path=composite_pdf, fig_width_in=args.fig_width_in,
        fig_height_in=args.fig_height_in, aspect=args.aspect, dpi=args.dpi,
        legend_side=args.legend_side, metrics_band=args.metrics_band
    )
    print("Composite figure (PNG/PDF):", composite_png, " / ", composite_pdf)

if __name__ == "__main__":
    main()
