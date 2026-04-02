
import umap
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from safetensors.torch import save_file
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset, concatenate_datasets


def set_rng_state(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_datasets(args, tokenizer):
    args.logger("Loading smoltalk subsets …")

    MAGPIE_SUBSET = "smol-magpie-ultra"
    MAGPIE_EXCLUDE_CATS = {
        "advice-seeking", "brainstorming", "creative-writing",
        "editing", "planning", "role-playing"
    }

    use_all_train = args.MAX_TRAIN_SAMPLES == -1

    train_shards, val_shards = [], []

    if not use_all_train:
        total_needed       = args.MAX_TRAIN_SAMPLES + args.MAX_VAL_SAMPLES
        n_per_subset       = total_needed // len(args.DATASET_SUBSETS)
        n_val_per_subset   = args.MAX_VAL_SAMPLES // len(args.DATASET_SUBSETS)
        n_train_per_subset = n_per_subset - n_val_per_subset
    else:
        n_val_per_subset   = args.MAX_VAL_SAMPLES // len(args.DATASET_SUBSETS)

    for subset in args.DATASET_SUBSETS:
        train = load_dataset(args.DATASET_ID, subset, split="train")
        test  = load_dataset(args.DATASET_ID, subset, split="test")

        # Filter magpie-ultra to allowed categories only
        if subset == MAGPIE_SUBSET:
            train = train.filter(lambda ex: ex.get("category") not in MAGPIE_EXCLUDE_CATS)
            test  = test.filter(lambda ex: ex.get("category") not in MAGPIE_EXCLUDE_CATS)

        if use_all_train: train = train.shuffle(seed=42)
        else: train = train.shuffle(seed=42).select(range(min(n_train_per_subset, len(train))))

        test = test.shuffle(seed=42).select(range(min(n_val_per_subset, len(test))))

        under_budget = (
            (not use_all_train and len(train) < n_train_per_subset) or
            len(test) < n_val_per_subset
        )

        args.logger(f"  {subset}: {len(train)} train / {len(test)} val"
                    + (" ⚠ subset smaller than target" if under_budget else ""))

        train_shards.append(train)
        val_shards.append(test)

    train_raw = concatenate_datasets(train_shards).shuffle(seed=42)
    val_raw   = concatenate_datasets(val_shards).shuffle(seed=42)

    if not use_all_train and len(train_raw) > args.MAX_TRAIN_SAMPLES:
        train_raw = train_raw.select(range(args.MAX_TRAIN_SAMPLES))
    if len(val_raw) > args.MAX_VAL_SAMPLES:
        val_raw = val_raw.select(range(args.MAX_VAL_SAMPLES))

    args.logger(f"  Total: {len(train_raw)} train / {len(val_raw)} val\n")

    def format_and_tokenize(examples):
        texts = tokenizer.apply_chat_template(
            examples["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        tokenized = tokenizer(
            texts,
            truncation=False,   # don't truncate — we discard instead
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    def filter_by_length(example):
        return len(example["input_ids"]) <= args.MAX_LENGTH

    train_tokenized = train_raw.map(
        format_and_tokenize,
        batched=True,
        remove_columns=train_raw.column_names,
        desc="Tokenizing train",
        load_from_cache_file=False,
        keep_in_memory=True,
    ).filter(filter_by_length, desc="Filtering train by length")

    val_tokenized = val_raw.map(
        format_and_tokenize,
        batched=True,
        remove_columns=val_raw.column_names,
        desc="Tokenizing val",
        load_from_cache_file=False,
        keep_in_memory=True,
    ).filter(filter_by_length, desc="Filtering val by length")

    args.logger(f"  After length filter (≤{args.MAX_LENGTH} tokens): "
                f"{len(train_tokenized)} train / {len(val_tokenized)} val\n")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=None,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    return train_tokenized, val_tokenized, data_collator

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


