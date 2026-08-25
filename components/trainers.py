import json
import os
import time

import torch
import torch.nn.functional as F
from transformers import Trainer

from utils.utils import save_hf_model
from utils.losses import (
    forward_kl, reverse_kl, symmetric_kl, js_distance,
    tv_distance, skewed_forward_kl, skewed_reverse_kl,
)
from components.noise import response_mask_from_labels


KL_LOSSES = {
    "forward_kl": forward_kl,
    "reverse_kl": reverse_kl,
    "symmetric_kl": symmetric_kl,
    "js_distance": js_distance,
    "tv_distance": tv_distance,
    "skewed_forward_kl": skewed_forward_kl,
    "skewed_reverse_kl": skewed_reverse_kl,
}


def resolve_kl_loss(name):
    if name not in KL_LOSSES:
        raise ValueError(f"Unknown KL_LOSS '{name}'. Choose from: {sorted(KL_LOSSES)}")
    return KL_LOSSES[name]


def _label_mask(mask_bool):
    """Build a `no_model_batch['label']`-style tensor: -100 off the mask, 0 on it. The
    KL losses in utils/losses.py only ever test `label != -100`, so the on-mask value
    is arbitrary — this lets us reuse them unmodified here."""
    return torch.where(mask_bool, torch.zeros_like(mask_bool, dtype=torch.long), -100)


class FlowTrainer(Trainer):
    """Trains the FlowNet + projector with exactly two losses: the flow-matching MSE
    velocity loss (primary), and a KL(teacher || pred) term (CLI-selected variant),
    both scored on the true response span (resp_mask)."""

    def __init__(self, arg, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.arg = arg

        self.LARGE_MODEL_ID = arg.LARGE_MODEL_ID
        self.SMALL_MODEL_ID = arg.SMALL_MODEL_ID
        self.best_top1_path = os.path.join(arg.work_dir, "best_top1.json")
        self.best_top1 = self._load_best_top1()

        self.kl_fn = resolve_kl_loss(arg.KL_LOSS)

    def _load_best_top1(self):
        if os.path.exists(self.best_top1_path):
            with open(self.best_top1_path) as f:
                return json.load(f)["best_top1"]
        return 0.0

    def _save_best_top1(self):
        if self.args.process_index == 0:
            with open(self.best_top1_path, "w") as f:
                json.dump({"best_top1": self.best_top1}, f)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")

        with torch.no_grad():
            teacher_out = self.teacher(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, return_dict=True,
            )
        t_logits = teacher_out.logits
        t_hidden = teacher_out.hidden_states[-1]

        student_out = model(input_ids=input_ids, attention_mask=attention_mask, teacher_embeddings=t_hidden)
        s_logits = student_out.logits

        # ── primary: flow-matching velocity MSE (computed inside the model) ──
        loss_flow = student_out.loss

        # ── KL(teacher, pred) over the true response span ──
        resp_mask = response_mask_from_labels(labels)
        B, T, V = s_logits.shape
        no_model_batch = {"label": _label_mask(resp_mask)}
        loss_kl = self.kl_fn(
            s_logits.reshape(-1, V).float(), t_logits.reshape(-1, V).float(), no_model_batch
        )

        loss_flow = loss_flow * float(getattr(self.arg, "W_FLOW", 1.0))
        loss_kl = loss_kl * float(getattr(self.arg, "W_KL", 1.0))
        loss = loss_flow + loss_kl

        if self.state.global_step % self.args.logging_steps == 0:
            self.arg.logger(
                f"Step {self.state.global_step} — total: {loss.item():.4f} | "
                f"flow: {loss_flow.item():.4f} | kl[{self.arg.KL_LOSS}]: {loss_kl.item():.4f}"
            )

        if torch.isnan(loss):
            self.arg.logger("NaN loss detected! Skipping step.")
            loss = loss * 0.0

        return (loss, student_out) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self.arg.logger(f"\n{'='*60}")
        self.arg.logger(f"Running validation at epoch {self.state.epoch:.1f} …")
        self.arg.logger("=" * 60)

        model = self.model
        model.eval()
        self.teacher.eval()
        device = next(model.parameters()).device

        sums = {"kl": 0.0, "ce": 0.0, "top1": 0.0, "n": 0.0}
        dataloader = self.get_eval_dataloader(eval_dataset or self.eval_dataset)

        start_time = time.time()
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask")
                labels = batch.get("labels")

                teacher_out = self.teacher(
                    input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True, return_dict=True,
                )
                t_logits = teacher_out.logits.float()
                t_hidden = teacher_out.hidden_states[-1]

                student_out = model(input_ids=input_ids, attention_mask=attention_mask, teacher_embeddings=t_hidden)
                s_logits = student_out.logits.float()
                B, T, V = s_logits.shape

                resp_mask = response_mask_from_labels(labels)
                no_model_batch = {"label": _label_mask(resp_mask)}
                loss_kl = self.kl_fn(
                    s_logits.reshape(-1, V), t_logits.reshape(-1, V), no_model_batch
                )

                shift_logits = s_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_ce = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1), ignore_index=-100
                )

                # top1 agreement vs teacher: scored on the response span, comparing
                # greedy predictions directly.
                pred_flat = s_logits[resp_mask]
                teacher_flat = t_logits[resp_mask]
                sums["top1"] += (pred_flat.argmax(-1) == teacher_flat.argmax(-1)).float().mean().item()

                sums["kl"] += loss_kl.item()
                sums["ce"] += loss_ce.item()
                sums["n"] += 1

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            keys = list(sums.keys())
            stats = torch.tensor([sums[k] for k in keys], dtype=torch.float64, device=device)
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
            sums = dict(zip(keys, stats.tolist()))

        n = max(sums["n"], 1.0)
        avg = {k: v / n for k, v in sums.items() if k != "n"}

        self.arg.logger(f"  FLOW      KL(t‖p): {avg['kl']:.6f} | CE: {avg['ce']:.6f} | top1 vs teacher: {avg['top1']:.4f}")
        self.arg.logger(f"  Time: {(time.time()-start_time)/60:.2f} min")

        metrics = {
            f"{metric_key_prefix}_kl": avg["kl"],
            f"{metric_key_prefix}_ce": avg["ce"],
            f"{metric_key_prefix}_top1_agreement": avg["top1"],
        }
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        if avg["top1"] > self.best_top1:
            self.best_top1 = avg["top1"]
            self._save_best_top1()
            if self.args.process_index == 0:
                self.arg.logger(f"New best top-1 agreement: {avg['top1']:.4f} — saving …\n")
                save_hf_model(model, save_dir=f"{self.arg.work_dir}/flow_best", base_model_name=self.SMALL_MODEL_ID)

        model.train()
        return metrics
