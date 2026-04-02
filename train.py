import os
import yaml
import shutil
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers import TrainingArguments

from components.f2l import F2L, F2L_Config
from components.trainers import F2L_Trainer
from utils.logger import Logger
from utils.utils import set_rng_state
from utils.data import build_datasets


def prep_model_comps(args):
    print(f"Using device: {args.device}")
    
    args.logger("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.SMALL_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    args.logger(f"Loading teacher: {args.LARGE_MODEL_ID} ...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.LARGE_MODEL_ID,
        torch_dtype=torch.float16,
        # device_map="auto",
        trust_remote_code=True,
    ).to(torch.float32).to(args.device)  # load in float32 and then move to device to avoid potential issues with mixed precision
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    args.logger(f"Loading student: {args.SMALL_MODEL_ID} ...")
    student_config = F2L_Config(
        base_model=args.SMALL_MODEL_ID, teacher_model=args.LARGE_MODEL_ID, 
        lorify=args.LORIFY, num_flow_steps=args.NUM_FLOW_STEPS
    )
    student = F2L(student_config)
    student = student.to(args.device)

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in student.parameters())
    args.logger(f"  Trainable params : {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")

    args.logger("\n")

    return tokenizer, teacher, student

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

        logging_steps=10,
        eval_strategy="steps",                          # changed from "epoch"
        eval_steps=total_steps // 10,
        save_strategy="no",
        save_total_limit=2,

        load_best_model_at_end=False,
        metric_for_best_model="ce",
        greater_is_better=False,
        
        fp16=(args.device == "cuda"),
        bf16=False,

        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,

        disable_tqdm=args.slurm_mode,  # disable tqdm in slurm mode to avoid issues with multiple processes
        log_level="error" if args.slurm_mode else "info",
    )

    trainer = F2L_Trainer(
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

    shutil.copy("train.py", os.path.join(args.work_dir, "train.py"))
    shutil.copy("configs/f2l.yaml", os.path.join(args.work_dir, "f2l.yaml"))
    shutil.copytree("components", os.path.join(args.work_dir, "components"), dirs_exist_ok=True)

    set_rng_state(args.seed)
    setattr(args, "logger", Logger(os.path.join(args.work_dir, "f2l.log")))

    tokenizer, teacher, student = prep_model_comps(args)
    train_ds, val_ds, data_collator = build_datasets(args, tokenizer)

    trainer = prep_trainer(args, teacher, student, train_ds, val_ds, data_collator, args.logger)

    print("\nStarting training …")
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="work_dir/test")
    parser.add_argument("--config", default="configs/f2l.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument('--slurm-mode', action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for key, value in config.items():
        setattr(args, key, value)

    main(args)