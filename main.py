import os
import yaml
import shutil
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from components import (
    FlowModel, FlowConfig, FlowTrainer,
    KL_LOSSES,
)
from utils.logger import Logger
from utils.utils import set_rng_state
from utils.data import build_datasets


def load_flow_model(args, teacher):
    config = FlowConfig(
        base_model=args.SMALL_MODEL_ID,
        teacher_model=args.LARGE_MODEL_ID,
        teacher_hidden_size=teacher.config.hidden_size,
        flownet_d_model=getattr(args, "FLOWNET_D_MODEL", 512),
        flownet_layers=getattr(args, "FLOWNET_LAYERS", 4),
        flownet_heads=getattr(args, "FLOWNET_HEADS", 8),
        num_flow_steps=getattr(args, "NUM_FLOW_STEPS", 10),
    )
    return FlowModel(config, teacher_lm_head=teacher.lm_head)


def prep_model_comps(args):
    print(f"Using device: {args.device}")

    args.logger("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(args.LARGE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Under torchrun each process owns exactly one GPU (args.device is already
    # cuda:{local_rank}) and holds its own full teacher copy — device_map="auto"
    # would instead split one teacher across every visible GPU per process, which
    # both fights DDP's placement and multiplies memory use by world size.
    args.logger(f"Loading teacher: {args.LARGE_MODEL_ID} …")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.LARGE_MODEL_ID,
        dtype=torch.float16,
        device_map={"": args.device},
        trust_remote_code=True,
    )
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()
    args.logger(f"  Teacher hidden dim: {teacher.config.hidden_size}")

    args.logger(f"Loading student ({args.MODEL_TYPE}): {args.SMALL_MODEL_ID} …")
    student = load_flow_model(args, teacher)
    student = student.to(args.device)

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student.parameters())
    args.logger(f"  Student trainable params: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")
    args.logger("\n")

    return tokenizer, teacher, student


def prep_trainer(args, teacher, student, train_ds, val_ds, data_collator):
    total_steps = (
        len(train_ds) // (args.PER_DEVICE_TRAIN_BATCH_SIZE * args.GRADIENT_ACCUMULATION_STEPS)
    ) * args.TRAIN_EPOCHS
    warmup_steps = int(0.03 * total_steps)

    args.logger(f"Total training steps: {total_steps}\n")
    args.logger("TRAINABLE PARAMETERS:\n")
    for k, v in student.named_parameters():
        if v.requires_grad:
            args.logger(f"  {k} : {v.numel():,} params")
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
        max_grad_norm=1.0,

        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,

        load_best_model_at_end=False,
        # Checkpointing criterion: top1 agreement with the teacher's predicted tokens
        # on the response span (higher is better).
        metric_for_best_model="top1_agreement",
        greater_is_better=True,

        fp16=str(args.device).startswith("cuda"),
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
        model=student,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )
    return trainer


def main(args):
    # Under `torchrun --nproc_per_node=N`, every rank runs this script; only rank 0
    # should touch shared work_dir files (log/config/code snapshot) to avoid races.
    if args.is_main_process:
        os.makedirs(args.work_dir, exist_ok=True)
        shutil.copy("main.py", os.path.join(args.work_dir, "main.py"))
        shutil.copy(args.config, os.path.join(args.work_dir, os.path.basename(args.config)))
        shutil.copytree("components", os.path.join(args.work_dir, "components"), dirs_exist_ok=True)
    if args.world_size > 1:
        torch.distributed.barrier()

    set_rng_state(args.seed)
    setattr(args, "logger", Logger(
        os.path.join(args.work_dir, f"{args.MODEL_TYPE}.log"),
        is_main_process=args.is_main_process,
    ))

    tokenizer, teacher, student = prep_model_comps(args)
    train_ds, val_ds, data_collator = build_datasets(args, tokenizer)

    trainer = prep_trainer(args, teacher, student, train_ds, val_ds, data_collator)

    # Resume support: if work_dir already holds a Trainer checkpoint (e.g. a prior run
    # was cut off by the SLURM time limit), pick up from the latest one instead of
    # training from scratch.
    last_checkpoint = get_last_checkpoint(args.work_dir)
    if last_checkpoint is not None:
        args.logger(f"Resuming from checkpoint: {last_checkpoint}\n")
    else:
        args.logger(f"\nStarting {args.MODEL_TYPE} training …\n")
    trainer.train(resume_from_checkpoint=last_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="./work_dir/flow_test")
    parser.add_argument("--config", default="configs/flow.yaml")
    parser.add_argument("--kl-loss", dest="KL_LOSS", default=None, choices=sorted(KL_LOSSES),
                        help="KL variant used as the second loss term. Overrides KL_LOSS in the config file.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--slurm-mode", action="store_true")

    args = parser.parse_args()
    args.MODEL_TYPE = "flow"

    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    cli_kl_override = args.KL_LOSS
    for key, value in config.items():
        setattr(args, key, value)
    if cli_kl_override is not None:
        args.KL_LOSS = cli_kl_override

    # torchrun sets these env vars for every rank; absent (single-process launch) they
    # default to a lone rank 0. GRADIENT_ACCUMULATION_STEPS is divided by world size so
    # the effective batch size (per-device batch × accum × world size) stays fixed as
    # GPU count changes — wall-clock drops ~linearly instead.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")
        args.device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
        args.GRADIENT_ACCUMULATION_STEPS = max(1, args.GRADIENT_ACCUMULATION_STEPS // world_size)

    args.local_rank = local_rank
    args.world_size = world_size
    args.is_main_process = local_rank == 0

    main(args)
