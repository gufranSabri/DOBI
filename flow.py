import os
import os
import yaml
import shutil
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments
from datasets import load_dataset
from safetensors.torch import load_file

from components import FlowTrainer, FlowNet, AlignmentConfig, AlignedModel
from utils.logger import Logger


def set_rng_state(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_excitation_model(save_dir):
    save_path = Path(save_dir)

    config = AlignmentConfig.from_pretrained(save_path)
    model = AlignedModel(config)

    trained_weights = load_file(save_path / "model.safetensors")
    missing, unexpected = model.load_state_dict(trained_weights, strict=False)

    print(f"Missing keys (expected - these are frozen base weights): {len(missing)}")
    print(f"Unexpected keys: {unexpected}")
    print("LOADED EXCITATION MODEL\n\n")

    return model

def prep_model_comps(args):
    print(f"Using device: {args.device}")

    args.logger("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(args.LARGE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    args.logger(f"Loading teacher: {args.LARGE_MODEL_ID} …")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.LARGE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    # Resolve teacher hidden-state dimension
    t_cfg = teacher.config
    hidden_dim = getattr(t_cfg, "hidden_size", None) or getattr(t_cfg, "d_model", None)
    assert hidden_dim is not None, "Cannot infer hidden_size from teacher config"
    args.logger(f"  Teacher hidden dim: {hidden_dim}")

    # load student
    args.logger(f"Loading student: {args.SMALL_MODEL_ID} …")
    student = load_excitation_model(args.excited_model_path).to(args.device)
    for p in student.parameters():
        p.requires_grad = False
    student.eval()

    args.logger(f"Loading FlowNet …")
    flownet = FlowNet(hidden_dim=teacher.config.hidden_size, d_model=512, num_layers=4, dropout=0.0)
    flownet = flownet.to(args.device)

    trainable = sum(p.numel() for p in flownet.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in flownet.parameters())
    args.logger(f"  FlowNet trainable params: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")
    args.logger("\n")

    return tokenizer, teacher, student, flownet


def build_datasets(args, tokenizer):
    from datasets import concatenate_datasets

    args.logger("Loading smoltalk subsets …")

    train_shards, val_shards = [], []
    total_needed    = args.MAX_TRAIN_SAMPLES + args.MAX_VAL_SAMPLES
    n_per_subset    = total_needed // len(args.DATASET_SUBSETS)
    n_val_per_subset   = args.MAX_VAL_SAMPLES  // len(args.DATASET_SUBSETS)
    n_train_per_subset = n_per_subset - n_val_per_subset

    for subset in args.DATASET_SUBSETS:
        train = load_dataset(args.DATASET_ID, subset, split="train")
        test  = load_dataset(args.DATASET_ID, subset, split="test")

        # Take as many as are available, up to the per-subset target
        train = train.shuffle(seed=42).select(range(min(n_train_per_subset, len(train))))
        test  = test.shuffle(seed=42).select(range(min(n_val_per_subset,   len(test))))

        args.logger(f"  {subset}: {len(train)} train / {len(test)} val"
                    + (" ⚠ subset smaller than target" if len(train) < n_train_per_subset
                                                       or len(test)  < n_val_per_subset else ""))
        train_shards.append(train)
        val_shards.append(test)

    train_raw = concatenate_datasets(train_shards).shuffle(seed=42)
    val_raw   = concatenate_datasets(val_shards).shuffle(seed=42)

    # Clamp totals to MAX_* in case some subsets exceeded budget via rounding
    if len(train_raw) > args.MAX_TRAIN_SAMPLES:
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
            truncation=True,
            max_length=args.MAX_LENGTH,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    train_tokenized = train_raw.map(
        format_and_tokenize,
        batched=True,
        remove_columns=train_raw.column_names,
        desc="Tokenizing train",
    )
    val_tokenized = val_raw.map(
        format_and_tokenize,
        batched=True,
        remove_columns=val_raw.column_names,
        desc="Tokenizing val",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=None,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    return train_tokenized, val_tokenized, data_collator


def prep_trainer(args, teacher, student, flownet, train_ds, val_ds, data_collator):
    total_steps = (
        len(train_ds)
        // (args.PER_DEVICE_TRAIN_BATCH_SIZE * args.GRADIENT_ACCUMULATION_STEPS)
    ) * args.TRAIN_EPOCHS
    warmup_steps = int(0.05 * total_steps)

    args.logger(f"Total training steps: {total_steps}\n")
    args.logger("TRAINABLE PARAMETERS (FlowNet):\n")
    for k, v in flownet.named_parameters():
        if v.requires_grad:
            args.logger(f"  {k}")
    args.logger("\n")

    training_args = TrainingArguments(
        output_dir=args.work_dir,

        num_train_epochs=args.TRAIN_EPOCHS,
        per_device_train_batch_size=args.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=args.PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=args.GRADIENT_ACCUMULATION_STEPS,

        learning_rate=float(args.LR),
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        lr_scheduler_type="cosine",

        logging_steps=100 if args.slurm_mode else 5,
        eval_strategy="steps",
        eval_steps=total_steps // 8,
        save_strategy="no",

        load_best_model_at_end=False,
        metric_for_best_model="cka",
        greater_is_better=True,

        fp16=(args.device == "cuda"),
        bf16=False,

        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,

        disable_tqdm=args.slurm_mode,
        log_level="error" if args.slurm_mode else "info",
    )

    trainer = FlowTrainer(
        arg=args,
        teacher_model=teacher,
        student=student,
        model=flownet,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    return trainer


def main(args):
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "flow_viz"), exist_ok=True)

    shutil.copy("flow.py", os.path.join(args.work_dir, "flow.py"))
    shutil.copy("configs/flow.yaml", os.path.join(args.work_dir, "flow.yaml"))
    shutil.copytree("components", os.path.join(args.work_dir, "components"), dirs_exist_ok=True)

    set_rng_state(args.seed)
    setattr(args, "logger", Logger(os.path.join(args.work_dir, "flow.log")))

    tokenizer, teacher, student, flownet = prep_model_comps(args)
    train_ds, val_ds, data_collator = build_datasets(args, tokenizer)

    trainer = prep_trainer(args, teacher, student, flownet, train_ds, val_ds, data_collator)

    args.logger("\nStarting Flow training …\n")
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="work_dir/flow_test")
    parser.add_argument("--excited-model-path", default="work_dir/50k/aligned_best")
    parser.add_argument("--config", default="configs/flow.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--slurm-mode", action="store_true")

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for key, value in config.items():
        setattr(args, key, value)

    main(args)