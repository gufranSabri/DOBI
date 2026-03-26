import umap
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import Trainer
from utils.utils import save_hf_model, linear_cka, pca_plot, umap_plot

class AlignmentTrainer(Trainer):
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

        self.best_cos = -1.0  # track best CKA for checkpointing

    # ── forward + loss ────────────────────────────────────────────────────────
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids      = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        labels         = inputs.get("labels")

        # ── student forward ───────────────────────────────────────────────────
        student_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_projected=True,
        )
        s_logits   = student_out["logits"].float()            # (B, T, vocab)
        s_projected = student_out["projected_hidden"].float() # (B, T, d_large)

        # ── teacher forward (no grad) ─────────────────────────────────────────
        with torch.no_grad():
            teacher_out = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        t_logits = teacher_out.logits.float()                 # (B, T, vocab)
        t_hidden = teacher_out.hidden_states[-1].float()      # (B, T, d_large)

        # ── 1. Hidden-state alignment losses ─────────────────────────────────
        # MSE
        loss_mse = F.mse_loss(s_projected, t_hidden)

        # Cosine (we want similarity → 1, so loss = 1 - cos)
        cos_sim = F.cosine_similarity(s_projected, t_hidden, dim=-1)   # (B,T)
        loss_cos = (1.0 - cos_sim).mean()

        # s_norm = s_projected.norm(dim=-1).mean()
        # t_norm = t_hidden.norm(dim=-1).mean()
        # loss_scale = F.mse_loss(s_norm, t_norm)  # scalar
        loss_scale = F.mse_loss(
            s_projected.norm(dim=-1), 
            t_hidden.norm(dim=-1)
        )

        loss_hidden = loss_mse if self.arg.INCLUDE_MSE else 0.0
        loss_hidden += loss_cos if self.arg.INCLUDE_COSINE else 0.0
        loss_hidden += loss_scale*0.001 if self.arg.INCLUDE_SCALE else 0.0

        # ── 2. KL distillation loss ───────────────────────────────────────────
        B, T, V = s_logits.shape
        s_soft = F.log_softmax(s_logits / self.temperature, dim=-1).view(B * T, V)
        t_soft = F.softmax(t_logits / self.temperature, dim=-1).view(B * T, V)
        loss_kl = self.kl_loss_fn(s_soft, t_soft) * (self.temperature ** 2)

        # ── 3. CE loss on ground-truth tokens ─────────────────────────────────
        if labels is not None:
            shift_logits = s_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_ce = self.ce_loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        else:
            loss_ce = torch.tensor(0.0, device=s_logits.device)

        # ── total ─────────────────────────────────────────────────────────────
        loss_hidden = loss_hidden
        loss_kl     = loss_kl if self.arg.INCLUDE_KLD else 0.0
        loss_ce     = loss_ce if self.arg.INCLUDE_CE else 0.0
        loss = loss_hidden + loss_kl + loss_ce

        self._log_sub_losses(loss_hidden, loss_kl, loss_ce, cos_sim.mean())

        return (loss, student_out) if return_outputs else loss

    def _log_sub_losses(self, l_h, l_kl, l_ce, cos_mean):
        if self.state.global_step % self.args.logging_steps == 0:
            self.arg.logger(f"Step {self.state.global_step} — Total Loss: {(l_h + l_kl + l_ce).item():.4f} | "
                f"Hidden: {l_h.item():.4f}, KL: {l_kl.item():.4f}, CE: {l_ce.item():.4f}, CosSim: {cos_mean.item():.4f}")

    # ── validation with alignment metrics + PCA plot ──────────────────────────
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        self.arg.logger(f"\n{'='*60}")
        self.arg.logger(f"Running validation at epoch {self.state.epoch:.1f} …")
        self.arg.logger('='*60)

        model = self.model
        model.eval()
        self.teacher.eval()

        device = next(model.parameters()).device

        all_s_hidden, all_t_hidden = [], []
        total_cos, total_mse, total_ce, n_batches = 0.0, 0.0, 0.0, 0

        val_dataset = eval_dataset or self.eval_dataset
        dataloader  = self.get_eval_dataloader(val_dataset)

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}

                s_out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    return_projected=True,
                )
                t_out = self.teacher(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    output_hidden_states=True,
                    return_dict=True,
                )

                s_proj = s_out["projected_hidden"].float()   # (B,T,d_large)
                t_hid  = t_out.hidden_states[-1].float()

                cos  = F.cosine_similarity(s_proj, t_hid, dim=-1).mean().item()
                mse  = F.mse_loss(s_proj, t_hid).item()

                # CE
                if "labels" in batch:
                    shift_l = s_out["logits"][..., :-1, :].contiguous()
                    shift_y = batch["labels"][..., 1:].contiguous()
                    ce = self.ce_loss_fn(
                        shift_l.view(-1, shift_l.size(-1)),
                        shift_y.view(-1),
                    ).item()
                else:
                    ce = 0.0

                total_cos += cos
                total_mse += mse
                total_ce  += ce
                n_batches += 1

                # Collect flat hidden states for PCA (subsample to keep memory sane)
                if len(all_s_hidden) < 2000:
                    mask = batch.get("attention_mask")  # (B, T)
                    if mask is not None:
                        mask = mask.bool().cpu()
                        # only keep non-padded token positions
                        all_s_hidden.append(s_proj.cpu().numpy()[mask.numpy()])   # (valid_tokens, d)
                        all_t_hidden.append(t_hid.cpu().numpy()[mask.numpy()])
                    else:
                        B, T, d = s_proj.shape
                        all_s_hidden.append(s_proj.cpu().numpy().reshape(B * T, d))
                        all_t_hidden.append(t_hid.cpu().numpy().reshape(B * T, d))

        avg_cos = total_cos / n_batches
        avg_mse = total_mse / n_batches
        avg_ce  = total_ce  / n_batches

        # ── Centred Kernel Alignment (CKA) ─────────────────────────────────
        s_mat = np.vstack(all_s_hidden)   # (N, d_large)
        t_mat = np.vstack(all_t_hidden)

        max_tokens = 50_000
        if len(s_mat) > max_tokens:
            idx = np.random.choice(len(s_mat), max_tokens, replace=False)
            s_mat = s_mat[idx]
            t_mat = t_mat[idx]

        cka = linear_cka(s_mat, t_mat)

        # ── L2 distance between means ───────────────────────────────────────
        mean_l2 = np.linalg.norm(s_mat.mean(0) - t_mat.mean(0))

        self.arg.logger(f"Alignment Metrics @ epoch {self.state.epoch:.1f}")
        self.arg.logger(f"  Cosine similarity  : {avg_cos:.4f}  (1.0 = perfect)")
        self.arg.logger(f"  MSE                : {avg_mse:.6f}  (0.0 = perfect)")
        self.arg.logger(f"  Linear CKA         : {cka:.4f}  (1.0 = perfect)")
        self.arg.logger(f"  Mean L2 distance   : {mean_l2:.4f}")
        self.arg.logger(f"  CE loss            : {avg_ce:.4f}")

        self.arg.logger("\n")

        # ── Plots ─────────────────────────────────────────────────────────
        pca_plot(s_mat, t_mat, epoch=self.state.epoch, arg=self.arg)
        umap_plot(s_mat, t_mat, epoch=self.state.epoch, arg=self.arg)

        # Log to trainer state
        metrics = {
            f"{metric_key_prefix}_cos_sim": avg_cos,
            f"{metric_key_prefix}_mse":     avg_mse,
            f"{metric_key_prefix}_cka":     cka,
            f"{metric_key_prefix}_mean_l2": mean_l2,
            f"{metric_key_prefix}_ce_loss": avg_ce,
        }
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        if self.best_cos < avg_cos:
            self.best_cos = avg_cos
            self.arg.logger(f"New best cosine similarity: {avg_cos:.4f} at epoch {self.state.epoch:.1f} — saving model checkpoint ...")
            save_hf_model(model, save_dir=f"{self.arg.work_dir}/best_model", base_model_name=self.SMALL_MODEL_ID)

        model.train()
        return metrics

