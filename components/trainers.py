import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Trainer
from utils.utils import save_hf_model, linear_cka, pca_plot, umap_plot

class F2L_Trainer(Trainer):
    def __init__(
        self, arg,
        teacher_model: nn.Module,
        temperature: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.temperature = temperature
        self.arg = arg

        self.LARGE_MODEL_ID = arg.LARGE_MODEL_ID
        self.SMALL_MODEL_ID = arg.SMALL_MODEL_ID

        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

        self.best_ce = float("inf")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids      = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        labels         = inputs.get("labels")

        with torch.no_grad():
            teacher_out = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        t_logits = teacher_out.logits
        t_hidden = teacher_out.hidden_states[-1]

        student_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            teacher_embeddings=t_hidden, 
        )
        s_logits   = student_out.logits
        s_hidden   = student_out.hidden_states

        # B, T, V = s_logits.shape
        # s_soft = F.log_softmax(s_logits / self.temperature, dim=-1).view(B * T, V)
        # t_soft = F.softmax(t_logits / self.temperature, dim=-1).view(B * T, V)
        # loss_kl = self.kl_loss_fn(s_soft, t_soft) * (self.temperature ** 2)

        # shift_logits = s_logits[..., :-1, :].contiguous()
        # shift_labels = labels[..., 1:].contiguous()
        # loss_ce = self.ce_loss_fn(
        #     shift_logits.view(-1, shift_logits.size(-1)),
        #     shift_labels.view(-1),
        # )

        # cos_sim = F.cosine_similarity(s_hidden, t_hidden, dim=-1)   # (B,T)
        # loss_cos = (1.0 - cos_sim).mean()
        # loss_mse = F.mse_loss(s_hidden, t_hidden)

        # loss_kl = loss_kl if self.arg.INCLUDE_KLD else 0.0
        # loss_ce = loss_ce if self.arg.INCLUDE_CE else 0.0
        # loss_cos = loss_cos if self.arg.INCLUDE_COS else 0.0
        # loss_mse = loss_mse if self.arg.INCLUDE_MSE else 0.0
        # loss = loss_kl + loss_ce + loss_cos + loss_mse + student_out.loss

        # if self.state.global_step % self.args.logging_steps == 0:
        #     self._log_sub_losses(loss_kl, loss_ce, loss_cos, loss_mse)

        loss = student_out.loss
        if self.state.global_step % self.args.logging_steps == 0:
            self.arg.logger(f"Step {self.state.global_step} — Flow Loss: {loss.item():.4f}")

        return (loss, student_out) if return_outputs else loss

    # def _log_sub_losses(self, l_kl, l_ce, l_cos, l_mse):
    #     if self.state.global_step % self.args.logging_steps == 0:
    #         self.arg.logger(f"Step {self.state.global_step} — Total Loss: {(l_kl + l_ce + l_cos + l_mse).item():.4f} | "
    #             f"KL: {l_kl.item():.4f}, CE: {l_ce.item():.4f}, COS: {l_cos.item():.4f}, MSE: {l_mse.item():.4f}")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self.arg.logger(f"\n{'='*60}")
        self.arg.logger(f"Running validation at epoch {self.state.epoch:.1f} …")
        self.arg.logger('='*60)

        model = self.model
        model.eval()
        self.teacher.eval()

        device = next(model.parameters()).device

        total_kl, total_ce, total_loss = 0.0, 0.0, 0.0
        n_batches = 0

        val_dataset = eval_dataset or self.eval_dataset
        dataloader  = self.get_eval_dataloader(val_dataset)

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}

                input_ids      = batch["input_ids"]
                attention_mask = batch.get("attention_mask")
                labels         = batch.get("labels")

                # ── teacher ────────────────────────────────────────────────
                teacher_out = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                t_logits = teacher_out.logits
                t_hidden = teacher_out.hidden_states[-1]

                # ── student ────────────────────────────────────────────────
                student_out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    teacher_embeddings=t_hidden,
                )
                s_logits = student_out["logits"]

                B, T, V = s_logits.shape

                # ── KL loss ───────────────────────────────────────────────
                s_soft = F.log_softmax(s_logits / self.temperature, dim=-1).view(B * T, V)
                t_soft = F.softmax(t_logits / self.temperature, dim=-1).view(B * T, V)

                loss_kl = self.kl_loss_fn(s_soft, t_soft) * (self.temperature ** 2)

                # ── CE loss (SHIFTED) ─────────────────────────────────────
                shift_logits = s_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_ce = self.ce_loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

                # ── combine ───────────────────────────────────────────────
                loss_kl_val = loss_kl.item() if self.arg.INCLUDE_KLD else 0.0
                loss_ce_val = loss_ce.item() if self.arg.INCLUDE_CE else 0.0
                loss_val    = loss_kl_val + loss_ce_val

                total_kl   += loss_kl_val
                total_ce   += loss_ce_val
                total_loss += loss_val
                n_batches  += 1

        avg_kl   = total_kl / n_batches
        avg_ce   = total_ce / n_batches
        avg_loss = total_loss / n_batches

        self.arg.logger(f"Validation @ epoch {self.state.epoch:.1f}")
        self.arg.logger(f"  KL divergence : {avg_kl:.6f}")
        self.arg.logger(f"  CE loss       : {avg_ce:.6f}")
        self.arg.logger(f"  Total loss    : {avg_loss:.6f}")
        self.arg.logger("\n")

        metrics = {
            f"{metric_key_prefix}_kl":   avg_kl,
            f"{metric_key_prefix}_ce":   avg_ce,
            f"{metric_key_prefix}_loss": avg_loss,
        }

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        # ── checkpoint on best total loss ────────────────────────────────
        if avg_ce < self.best_ce:
            self.best_ce = avg_ce
            self.arg.logger(
                f"New best CE loss: {avg_ce:.6f} at epoch {self.state.epoch:.1f}"
                f" — saving model checkpoint …"
            )
            save_hf_model(
                model,
                save_dir=f"{self.arg.work_dir}/best",
                base_model_name=self.SMALL_MODEL_ID,
            )

        model.train()
        return metrics