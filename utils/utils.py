
import umap
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from safetensors.torch import save_file
from transformers import AutoTokenizer


def linear_cka(X, Y) -> float:
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    XtY = X.T @ Y
    XtX = X.T @ X
    YtY = Y.T @ Y
    num   = np.sum(XtY ** 2)
    denom = np.sqrt(np.sum(XtX ** 2) * np.sum(YtY ** 2))

    return float(num / (denom + 1e-10))

def pca_plot(s_mat, t_mat, epoch, arg, phase):
    combined = np.vstack([s_mat, t_mat])
    pca      = PCA(n_components=2)
    pca.fit(combined)
    s_2d = pca.transform(s_mat)
    t_2d = pca.transform(t_mat)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(t_2d[:, 0], t_2d[:, 1], alpha=0.35, s=8, label=f"Teacher ({arg.LARGE_MODEL_ID.split('/')[-1]})", color="#2196F3")
    ax.scatter(s_2d[:, 0], s_2d[:, 1], alpha=0.35, s=8, label="FlowNet x_hat", color="#FF5722")
    
    var = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax.set_title(f"Flow PCA — epoch {epoch:.1f}")
    ax.legend(markerscale=3)
    plt.tight_layout()
    
    path = Path(arg.work_dir) / f"{phase}_viz" / f"flow_pca_epoch_{epoch:.2f}.png"
    plt.savefig(path, dpi=130)
    plt.close()

    arg.logger(f"  PCA plot saved → {path}\n")

def umap_plot(s_mat, t_mat, epoch, arg, phase):
    combined    = np.vstack([s_mat, t_mat])
    reducer     = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    combined_2d = reducer.fit_transform(combined)
    n    = len(s_mat)
    s_2d = combined_2d[:n]
    t_2d = combined_2d[n:]

    _, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(t_2d[:, 0], t_2d[:, 1], alpha=0.35, s=8, label=f"Teacher ({arg.LARGE_MODEL_ID.split('/')[-1]})", color="#2196F3")
    ax.scatter(s_2d[:, 0], s_2d[:, 1], alpha=0.35, s=8, label="FlowNet x_hat", color="#FF5722")

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(f"Flow UMAP — epoch {epoch:.1f}")
    ax.legend(markerscale=3)
    plt.tight_layout()

    path = Path(arg.work_dir) / f"{phase}_viz" / f"flow_umap_epoch_{epoch:.2f}.png"
    plt.savefig(path, dpi=130)
    plt.close()

    arg.logger(f"  UMAP plot saved → {path}\n")

def save_hf_model(model, save_dir, base_model_name):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    trainable_state = {
        k: v for k, v in model.state_dict().items()
        if not k.startswith("model.")  # exclude frozen base model weights
    }
    save_file(trainable_state, save_path / "model.safetensors")

    model.config.save_pretrained(save_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(save_path)
