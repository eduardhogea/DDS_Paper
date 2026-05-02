#!/usr/bin/env python
import os
import sys
import random

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from torch.utils.data import DataLoader, TensorDataset
from transformer_cluj_evaluate import VisionTransformer
from sequence_generation import load_sequences

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "..", "config"))
import config_uoc as config

# reproducibility
seed = getattr(config, 'random_seed', 0)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# paths
sequences_directory   = "/home/ubuntu/dds_paper/DDS_Paper/data_uoc/output_sequences"
model_save_directory  = "/home/ubuntu/dds_paper/DDS_Paper/model_weights_cluj"
results_path_ltn      = config.results_path_ltn

def extract_embeddings(model, loader, device):
    model.eval()
    all_embs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Extracting embeddings"):
            xb = xb.to(device)
            embs = model.embed(xb).cpu().numpy()
            all_embs.append(embs)
            all_labels.append(yb.numpy())
    return np.vstack(all_embs), np.hstack(all_labels)

def visualize_kmeans(emb2d, k, out_path):
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(emb2d)
    plt.figure(figsize=(6,6))
    plt.scatter(emb2d[:,0], emb2d[:,1],
                c=km.labels_, cmap='tab10', s=15, alpha=0.8)
    centers = km.cluster_centers_
    plt.scatter(centers[:,0], centers[:,1],
                c='black', marker='x', s=60, linewidths=2)
    plt.title(f'KMeans (k={k}) on t-SNE embeddings')
    plt.xlabel('Dim 1'); plt.ylabel('Dim 2')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    out_dir = os.path.join(results_path_ltn, "embedding_viz")
    os.makedirs(out_dir, exist_ok=True)

    X_train, y_train = load_sequences(
        os.path.join(sequences_directory, "train_sequences.npy"),
        os.path.join(sequences_directory, "train_labels.npy"),
    )
    X_t = torch.tensor(X_train, dtype=torch.float32).permute(0,2,1)
    y_t = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_t, y_t),
                        batch_size=config.batch_size,
                        shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionTransformer().to(device)
    best_path = os.path.join(model_save_directory, "best_model.pt")
    model.load_state_dict(torch.load(best_path, map_location=device))

    embeddings, labels = extract_embeddings(model, loader, device)

    # 1) stronger exaggeration + more iterations
    tsne = TSNE(
        n_components=2,
        perplexity=50,
        early_exaggeration=30,    # boost inter-cluster repulsion
        learning_rate=200,
        n_iter=2000,              # more optimization
        init='pca',
        random_state=seed,
    )
    emb2d = tsne.fit_transform(embeddings)

    # 2) simple post-scaling to stretch clusters apart
    emb2d *= 1.5

    for k in (2, 3, 5):
        out_file = os.path.join(out_dir, f"tsne_k{k}.png")
        visualize_kmeans(emb2d, k, out_file)
        print(f"→ saved: {out_file}")

if __name__ == "__main__":
    main()
