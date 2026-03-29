import os
import json
import yaml
import shutil
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers import TrainingArguments
from datasets import load_dataset, concatenate_datasets

from components import AlignedModel, AlignmentConfig, AlignmentTrainer
from utils.logger import Logger

def set_rng_state(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prep_model_comps(args):
    print(f"Using device: {args.device}")
    
    args.logger("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.SMALL_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    args.logger(f"Loading teacher: {args.LARGE_MODEL_ID} ...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.LARGE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    args.logger(f"Loading student: {args.SMALL_MODEL_ID} ...")
    student_config = AlignmentConfig(base_model=args.SMALL_MODEL_ID, teacher_model=args.LARGE_MODEL_ID)
    student = AlignedModel(student_config)
    student = student.to(args.device)

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in student.parameters())
    args.logger(f"  Trainable params : {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")

    args.logger("\n")

    return tokenizer, teacher, student

def build_datasets(args, tokenizer):
    from datasets import concatenate_datasets

    args.logger("Loading smoltalk subsets …")

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

        if use_all_train:
            train = train.shuffle(seed=42)
        else:
            train = train.shuffle(seed=42).select(range(min(n_train_per_subset, len(train))))

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
        load_from_cache_file=False,
        keep_in_memory=True,
    )
    val_tokenized = val_raw.map(
        format_and_tokenize,
        batched=True,
        remove_columns=val_raw.column_names,
        desc="Tokenizing val",
        load_from_cache_file=False,
        keep_in_memory=True,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=None,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    return train_tokenized, val_tokenized, data_collator

def prep_trainer(args, teacher, student, train_ds, val_ds, data_collator, logger):
    total_steps = (len(train_ds) // (args.PER_DEVICE_TRAIN_BATCH_SIZE * args.GRADIENT_ACCUMULATION_STEPS)) * args.TRAIN_EPOCHS
    warmup_steps = int(0.05 * total_steps)

    logger(f"Total training steps: {total_steps}\n")

    logger("TRAINABLE PARAMETERS:\n")
    for k,v in teacher.named_parameters():
        if v.requires_grad:
            logger(f"Teacher param: {k} | requires_grad: {v.requires_grad}")
    
    for k,v in student.named_parameters():
        if v.requires_grad:
            logger(f"Student param: {k} | requires_grad: {v.requires_grad}")

    logger("\n")

    training_args = TrainingArguments(
        output_dir=args.work_dir,

        num_train_epochs=args.TRAIN_EPOCHS,
        per_device_train_batch_size=args.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=args.PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=args.GRADIENT_ACCUMULATION_STEPS,

        learning_rate=float(args.LR),       # higher LR is fine since only projector trains
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        lr_scheduler_type="cosine",

        logging_steps=25,
        eval_strategy="steps",                          # changed from "epoch"
        eval_steps=total_steps // 8,
        save_strategy="no",
        save_total_limit=2,

        load_best_model_at_end=False,
        metric_for_best_model="cka",
        greater_is_better=True,
        
        fp16=(args.device == "cuda"),
        bf16=False,

        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,

        disable_tqdm=args.slurm_mode,  # disable tqdm in slurm mode to avoid issues with multiple processes
        log_level="error" if args.slurm_mode else "info",
    )

    trainer = AlignmentTrainer(
        arg=args,
        teacher_model=teacher,
        temperature=args.TEMPERATURE,
        model=student,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    return trainer


def main(args):
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "alignment_viz"), exist_ok=True)

    shutil.copy("alignment.py", os.path.join(args.work_dir, "alignment.py"))
    shutil.copy("configs/alignment.yaml", os.path.join(args.work_dir, "alignment.yaml"))
    shutil.copytree("components", os.path.join(args.work_dir, "components"), dirs_exist_ok=True)

    set_rng_state(args.seed)
    setattr(args, "logger", Logger(os.path.join(args.work_dir, "alignment.log")))

    tokenizer, teacher, student = prep_model_comps(args)
    train_ds, val_ds, data_collator = build_datasets(args, tokenizer)

    trainer = prep_trainer(args, teacher, student, train_ds, val_ds, data_collator, args.logger)

    print("\nStarting training …")
    trainer.train()

    # config_out = {
    #     "small_model_id": args.SMALL_MODEL_ID,
    #     "large_model_id": args.LARGE_MODEL_ID,
    # }
    # with open(f"{args.work_dir}/alignment.json", "w") as f:
    #     json.dump(config_out, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="work_dir/test")
    parser.add_argument("--config", default="configs/alignment.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument('--slurm-mode', action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for key, value in config.items():
        setattr(args, key, value)

    main(args)