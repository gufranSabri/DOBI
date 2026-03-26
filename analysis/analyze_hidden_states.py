"""
analyze_hidden_states.py
Compares hidden state geometry between two LLMs on the smoltalk dataset.
Usage: python analyze_hidden_states.py --config configs/analysis.yaml
"""

import os
import yaml
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from datasets import load_dataset
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA


# ── Helpers ────────────────────────────────────────────────────────────────────

def set_rng_state(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA in feature space — O(d²), not O(N²)."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    XtY = X.T @ Y
    XtX = X.T @ X
    YtY = Y.T @ Y
    num   = np.sum(XtY ** 2)
    denom = np.sqrt(np.sum(XtX ** 2) * np.sum(YtY ** 2))
    return float(num / (denom + 1e-10))


def effective_rank(X: np.ndarray) -> float:
    """Effective rank via entropy of normalised singular values."""
    _, s, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    s = s[s > 1e-10]
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def intrinsic_dim_pca(X: np.ndarray, threshold: float = 0.90) -> int:
    """Number of PCs needed to explain `threshold` of variance."""
    X = X - X.mean(0)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    var_ratio = np.cumsum(s**2) / (s**2).sum()
    return int(np.searchsorted(var_ratio, threshold) + 1)


def mean_pairwise_l2(X: np.ndarray, n_sample: int = 2000) -> float:
    """Average L2 distance between random pairs within a cloud."""
    idx = np.random.choice(len(X), min(n_sample, len(X)), replace=False)
    X_s = X[idx]
    diff = X_s[:, None, :] - X_s[None, :, :]          # (N,N,d)
    dists = np.linalg.norm(diff, axis=-1)              # (N,N)
    mask  = np.triu(np.ones((len(X_s), len(X_s)), dtype=bool), k=1)
    return float(dists[mask].mean())


def principal_angle_similarity(X: np.ndarray, Y: np.ndarray, k: int = 50) -> float:
    """
    Subspace similarity via principal angles.
    Returns mean cos(θ) over the top-k principal angles — 1.0 = identical subspace.
    """
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    _, _, Vx = np.linalg.svd(X, full_matrices=False)
    _, _, Vy = np.linalg.svd(Y, full_matrices=False)
    Qx = Vx[:k].T   # (d, k)
    Qy = Vy[:k].T   # (d, k)
    M  = Qx.T @ Qy  # (k, k)
    sv = np.linalg.svd(M, compute_uv=False)
    return float(sv.mean())


# ── Data ───────────────────────────────────────────────────────────────────────

def build_dataloader(args, tokenizer):
    print("Loading smoltalk dataset …")
    raw = load_dataset(args.DATASET_ID, args.DATASET_SUBSET, split="train")
    raw = raw.shuffle(seed=42).select(range(args.MAX_SAMPLES))

    def format_and_tokenize(examples):
        texts = tokenizer.apply_chat_template(
            examples["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=args.MAX_LENGTH,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized = raw.map(
        format_and_tokenize,
        batched=True,
        remove_columns=raw.column_names,
        desc="Tokenizing",
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=None,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    loader = torch.utils.data.DataLoader(
        tokenized,
        batch_size=args.BATCH_SIZE,
        collate_fn=collator,
        shuffle=False,
        num_workers=0,
    )
    return loader


# ── Extraction ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_hidden_states(model, dataloader, device, max_tokens: int = 50_000):
    """
    Returns a (N, D) float32 numpy array of last-layer hidden states
    for non-padding tokens, capped at max_tokens.
    """
    model.eval()
    all_hidden = []
    collected  = 0

    for batch in dataloader:
        if collected >= max_tokens:
            break

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        h = out.hidden_states[-1].float()   # (B, T, D)

        if attention_mask is not None:
            mask = attention_mask.bool().cpu().numpy()
            h_np = h.cpu().numpy()
            for b in range(h_np.shape[0]):
                valid = h_np[b][mask[b]]    # (valid_tokens, D)
                all_hidden.append(valid)
                collected += len(valid)
        else:
            B, T, D = h.shape
            all_hidden.append(h.cpu().numpy().reshape(B * T, D))
            collected += B * T

    mat = np.vstack(all_hidden)
    if len(mat) > max_tokens:
        idx = np.random.choice(len(mat), max_tokens, replace=False)
        mat = mat[idx]
    return mat


# ── Analysis ───────────────────────────────────────────────────────────────────

def analyze(small_mat: np.ndarray, large_mat: np.ndarray, args):
    sep  = "=" * 65
    sep2 = "-" * 65

    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log(sep)
    log(f"  Hidden-State Geometry Report")
    log(f"  Small : {args.SMALL_MODEL_ID}")
    log(f"  Large : {args.LARGE_MODEL_ID}")
    log(f"  Tokens: {len(small_mat):,} (small)  {len(large_mat):,} (large)")
    log(sep)

    # ── 1. Per-model statistics ──────────────────────────────────────────────
    log("\n[1] Per-Model Statistics")
    log(sep2)
    for name, mat in [("Small", small_mat), ("Large", large_mat)]:
        norms  = np.linalg.norm(mat, axis=-1)
        log(f"  {name}")
        log(f"    dim              : {mat.shape[1]}")
        log(f"    mean norm        : {norms.mean():.4f}  ±  {norms.std():.4f}")
        log(f"    channel mean     : {mat.mean():.6f}")
        log(f"    channel std      : {mat.std():.6f}")
        log(f"    min / max        : {mat.min():.4f}  /  {mat.max():.4f}")
        log(f"    effective rank   : {effective_rank(mat):.1f}")
        log(f"    intrinsic dim    : {intrinsic_dim_pca(mat, threshold=0.90):.0f}  (90% var)")
        log(f"    mean pairwise L2 : {mean_pairwise_l2(mat):.4f}")
        log()

    # ── 2. Cross-model alignment metrics ────────────────────────────────────
    log("[2] Cross-Model Alignment  (token-paired where possible)")
    log(sep2)

    # Use the minimum token count for paired metrics
    n = min(len(small_mat), len(large_mat))
    s = small_mat[:n]
    l = large_mat[:n]

    # Cosine similarity (token-level)
    cos = F.cosine_similarity(
        torch.tensor(s), torch.tensor(l), dim=-1
    ).numpy()
    log(f"  Cosine similarity    : {cos.mean():.4f}  ±  {cos.std():.4f}  (1.0 = perfect)")

    # MSE
    mse = float(np.mean((s - l) ** 2))
    log(f"  MSE                 : {mse:.6f}  (0.0 = perfect)")

    # Mean L2 between centroids
    centroid_l2 = float(np.linalg.norm(s.mean(0) - l.mean(0)))
    log(f"  Centroid L2 dist    : {centroid_l2:.4f}  (0.0 = perfect)")

    # Mean token-level L2
    token_l2 = float(np.linalg.norm(s - l, axis=-1).mean())
    log(f"  Mean token L2       : {token_l2:.4f}  (0.0 = perfect)")

    # Linear CKA
    cka = linear_cka(s, l)
    log(f"  Linear CKA          : {cka:.4f}  (1.0 = perfect)")

    # Principal angle similarity (subspace overlap)
    k = min(50, s.shape[1] // 2, l.shape[1] // 2)
    pas = principal_angle_similarity(s, l, k=k)
    log(f"  Principal angle sim : {pas:.4f}  (1.0 = identical subspace, k={k})")

    # ── 3. Distribution shift ────────────────────────────────────────────────
    log()
    log("[3] Distribution Shift")
    log(sep2)

    # Mean / std shift per channel (summarised)
    mean_shift = np.abs(s.mean(0) - l.mean(0))
    std_shift  = np.abs(s.std(0)  - l.std(0))
    log(f"  |Δmean| per channel : {mean_shift.mean():.4f}  ±  {mean_shift.std():.4f}  (max {mean_shift.max():.4f})")
    log(f"  |Δstd|  per channel : {std_shift.mean():.4f}  ±  {std_shift.std():.4f}  (max {std_shift.max():.4f})")

    # Wasserstein-1 on projected 1-D (first PC of combined)
    combined = np.vstack([s, l])
    pca1 = PCA(n_components=1)
    pca1.fit(combined)
    s_1d = pca1.transform(s).ravel()
    l_1d = pca1.transform(l).ravel()
    w1   = wasserstein_distance(s_1d, l_1d)
    log(f"  Wasserstein-1 (PC1) : {w1:.4f}  (0.0 = identical distribution)")

    # Explained variance of first 5 PCs for each
    log()
    log("[4] PCA Variance Explained (top 5 components)")
    log(sep2)
    for name, mat in [("Small", s), ("Large", l)]:
        pca = PCA(n_components=min(5, mat.shape[1]))
        pca.fit(mat - mat.mean(0))
        ev  = pca.explained_variance_ratio_ * 100
        log(f"  {name}: " + "  ".join(f"PC{i+1}={v:.1f}%" for i, v in enumerate(ev)))

    # ── 4. Gaussian noise assumption check ───────────────────────────────────
    log()
    log("[5] Gaussian Noise Assumption (key for Diffusion Bridge)")
    log(sep2)
    residual = s - l   # (N, D) — if gap ≈ Gaussian, this should be ≈ N(μ, σ²I)
    res_mean = residual.mean(0)
    res_std  = residual.std(0)
    # Isotropy: std should be roughly equal across dims if Gaussian
    isotropy = res_std.std() / (res_std.mean() + 1e-8)   # lower = more isotropic
    log(f"  Residual (small - large):")
    log(f"    mean of means    : {res_mean.mean():.6f}  (≈0 if zero-mean noise)")
    log(f"    mean of stds     : {res_std.mean():.6f}")
    log(f"    isotropy index   : {isotropy:.4f}  (0.0 = perfectly isotropic / Gaussian)")
    log(f"  → {'Gaussian assumption plausible ✓' if isotropy < 0.5 else 'Gap is anisotropic — Gaussian assumption may not hold ✗'}")

    log()
    log(sep)
    log("  Done.")
    log(sep)

    # Save report
    report_path = os.path.join(args.work_dir, "hidden_state_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved → {report_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.work_dir, exist_ok=True)
    set_rng_state(args.seed)
    device = args.device

    # ── Tokenizer (use small model's tokenizer) ──────────────────────────────
    print(f"Loading tokenizer from {args.SMALL_MODEL_ID} …")
    tokenizer = AutoTokenizer.from_pretrained(args.SMALL_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataloader = build_dataloader(args, tokenizer)

    # ── Small model ──────────────────────────────────────────────────────────
    print(f"\nLoading small model: {args.SMALL_MODEL_ID} …")
    small_model = AutoModelForCausalLM.from_pretrained(
        args.SMALL_MODEL_ID,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)
    for p in small_model.parameters():
        p.requires_grad = False

    print("Extracting small model hidden states …")
    small_mat = extract_hidden_states(small_model, dataloader, device, max_tokens=args.MAX_TOKENS)
    del small_model
    torch.cuda.empty_cache()

    # ── Large model ──────────────────────────────────────────────────────────
    print(f"\nLoading large model: {args.LARGE_MODEL_ID} …")
    large_model = AutoModelForCausalLM.from_pretrained(
        args.LARGE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    for p in large_model.parameters():
        p.requires_grad = False

    print("Extracting large model hidden states …")
    large_mat = extract_hidden_states(large_model, dataloader, device, max_tokens=args.MAX_TOKENS)
    del large_model
    torch.cuda.empty_cache()

    # ── Analysis ─────────────────────────────────────────────────────────────
    print()
    analyze(small_mat, large_mat, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="work_dir/analysis")
    parser.add_argument("--config",   default="configs/analysis.yaml")
    parser.add_argument("--device",   default="cuda")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    for k, v in config.items():
        setattr(args, k, v)

    main(args)