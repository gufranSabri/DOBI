"""DiffusionModel — discrete mask-token diffusion distiller (Dream-style).

A frozen student LM supplies (a) per-position conditioning (its final hidden state) and
(b) the tied output-head embedding matrix. A 1D UNet (MaskUNet) denoises a length-T canvas
of token ids toward the TEACHER's predicted tokens: during training a fraction of the
response tokens are replaced by [MASK] and the UNet learns to recover them; at inference
the canvas is iteratively unmasked by confidence (few DDIM-like steps).

Compared with the old flow-matching AlignedModel:
  * NO teacher_head — the UNet predicts vocab logits directly (tied to the student
    embedding), so we never decode a continuous teacher hidden state.
  * NO per-dim normalization buffers — there is no continuous coordinate system; the
    diffusion variable is discrete tokens.
  * The probe baseline is retargeted to logits: how well can the student's own final
    hidden (a light MLP) recover the teacher's tokens? (calibration floor)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PretrainedConfig,
    GenerationMixin,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from components.unet import MaskUNet
from components.noise import (
    build_canvas,
    num_still_masked,
)


class DiffusionConfig(PretrainedConfig):
    model_type = "diffusion_excitation"

    def __init__(
        self,
        base_model: str = "",
        teacher_model: str = "",
        d_model: int = 768,
        unet_lengths=(2048, 1024, 512),
        unet_channels=(768, 1152, 1536),
        n_heads: int = 12,
        num_sampling_steps: int = 10,
        mask_schedule: str = "cosine",
        mask_id: int = 151936,
        tie_output_head: bool = True,
        max_seq_len: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.teacher_model = teacher_model
        self.d_model = d_model
        self.unet_lengths = list(unet_lengths)
        self.unet_channels = list(unet_channels)
        self.n_heads = n_heads
        self.num_sampling_steps = num_sampling_steps
        self.mask_schedule = mask_schedule
        self.mask_id = mask_id
        self.tie_output_head = tie_output_head
        self.max_seq_len = max_seq_len


class DiffusionModel(PreTrainedModel, GenerationMixin):
    config_class = DiffusionConfig
    main_input_name = "input_ids"

    def __init__(self, config: DiffusionConfig):
        super().__init__(config)

        self.num_sampling_steps = getattr(config, "num_sampling_steps", 10)

        # ── frozen student ────────────────────────────────────────────────
        self.model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        for p in self.model.parameters():
            p.requires_grad = False

        base_cfg = self.model.config
        vocab_size = base_cfg.vocab_size
        d_student = base_cfg.hidden_size
        self.vocab_size = vocab_size
        self.mask_id = vocab_size
        # Keep config.mask_id consistent with the actual vocab (guards a stale config).
        self.config.mask_id = vocab_size

        # ── UNet denoiser (the only large trainable module) ───────────────
        self.unet = MaskUNet(
            vocab_size=vocab_size,
            d_out=d_student,                       # tied head lives in student-embed space
            d_model=config.d_model,
            channels=tuple(config.unet_channels),
            lengths=tuple(config.unet_lengths),
            n_heads=config.n_heads,
            t_dim=config.d_model,
            cond_dim=d_student,
            max_seq_len=config.max_seq_len,
        )
        # The UNet derives its input token embedding from the frozen student embedding
        # at forward time (passed via _student_embedding()), so no separate init step.

        # ── probe baseline (independent params; calibration only) ─────────
        self.probe = nn.Sequential(
            nn.Linear(d_student, d_student),
            nn.GELU(),
            nn.Linear(d_student, d_student),
        )

        self.length_multiple = 2 ** (self.unet.n_stages - 1)

    # ── helpers ───────────────────────────────────────────────────────────

    def can_generate(self) -> bool:
        return True

    def _student_embedding(self):
        """The frozen student input embedding matrix (V, d_student). Qwen ties this to
        the lm_head, so it is the correct readout basis for the tied output head."""
        return self.model.get_input_embeddings().weight

    @torch.no_grad()
    def _student_hidden(self, input_ids, attention_mask=None):
        """One frozen-student forward → its final (post-norm) hidden state, (B, T, d)."""
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        return out.hidden_states[-1]

    def _pad_to_multiple(self, tok_ids, attention_mask, cond):
        """Right-pad the canvas / cond / mask so length is a multiple of the UNet's
        down-sampling factor. Returns (tok_ids, attention_mask, cond, orig_T)."""
        B, T = tok_ids.shape
        m = self.length_multiple
        pad = (m - (T % m)) % m
        if pad == 0:
            return tok_ids, attention_mask, cond, T
        tok_ids = F.pad(tok_ids, (0, pad), value=self.mask_id)
        if attention_mask is None:
            attention_mask = torch.ones(B, T, device=tok_ids.device, dtype=torch.long)
        attention_mask = F.pad(attention_mask, (0, pad), value=0)
        cond = F.pad(cond, (0, 0, 0, pad), value=0.0)
        return tok_ids, attention_mask, cond, T

    def _unet_logits(self, tok_ids, t, cond, attention_mask):
        """Run the UNet with padding-to-multiple handled; returns logits at ORIGINAL T."""
        orig_T = tok_ids.shape[1]
        tok_ids, attn, cond, _ = self._pad_to_multiple(tok_ids, attention_mask, cond)
        logits = self.unet(tok_ids, t, cond, attn, self._student_embedding())
        return logits[:, :orig_T, :]

    def probe_logits(self, student_hidden):
        pred = self.probe(student_hidden.to(self.probe[0].weight.dtype))
        return F.linear(pred, self._student_embedding().to(pred.dtype))

    # ── training / eval forward ─────────────────────────────────────────────

    def slot_logits(self, input_ids, attention_mask=None, mask_bool=None, t=None):
        """Token-AT-POSITION logits: slot_logits[t] is the distribution over the token
        that lives at slot t. This is the UNet's native convention, used by BOTH training
        and the trainer's internal eval (compared directly to the teacher's token-at-slot
        target — no shift). If mask_bool is None, every position is masked (predict each
        token from the rest of the bidirectional canvas) at noise level t=1.
        """
        cond = self._student_hidden(input_ids, attention_mask)
        B, T = input_ids.shape
        if mask_bool is None:
            mask_bool = torch.ones(B, T, dtype=torch.bool, device=input_ids.device)
        if t is None:
            t = torch.ones(B, device=input_ids.device, dtype=torch.float32)
        canvas = build_canvas(input_ids, mask_bool, self.mask_id)
        logits = self._unet_logits(canvas, t, cond, attention_mask)       # (B, T, V) slot-aligned
        return logits, cond

    def forward(
        self,
        input_ids,
        attention_mask=None,
        t=None,
        mask_bool=None,
        cont_spans=None,
        return_dict=True,
        **kwargs,
    ):
        if t is not None or mask_bool is not None:
            # ── Training path ─────────────────────────────────────────────
            # The trainer supplies t and the mask; return slot-aligned logits + the probe.
            logits, cond = self.slot_logits(input_ids, attention_mask, mask_bool, t)
            probe_logits = self.probe_logits(cond)
            return {"logits": logits, "probe_logits": probe_logits, "mask_bool": mask_bool}

        # ── Eval / lm-eval loglikelihood path ─────────────────────────────
        # lm-eval feeds [context + continuation] as input_ids and reads logits[t] as the
        # distribution over token t+1 (next-token convention). Our UNet is token-at-slot:
        # slot_logits[t] = P(token at slot t).
        #
        # Loglikelihood rule: mask ONLY what is scored (the continuation span), keep
        # everything else REAL. Masking all slots makes every multiple-choice candidate
        # score identically (all-masked canvas carries no information about which
        # candidate is being scored) and loglikelihood collapses to chance.
        B, T = input_ids.shape
        if cont_spans is not None:
            mask_bool = torch.zeros(B, T, dtype=torch.bool, device=input_ids.device)
            for b, (start, end) in enumerate(cont_spans):
                mask_bool[b, start:end] = True
        else:
            mask_bool = torch.ones(B, T, dtype=torch.bool, device=input_ids.device)

        # A token-at-slot denoiser has no slot to predict INTO for the last position (a
        # causal LM gets that for free off the end of its input). Append one extra
        # [MASK] slot so there is one, run the UNet over T+1, then drop the first column
        # so out[s] = slot[s+1] = P(token s+1) — lm-eval's next-token convention.
        pad_ids = F.pad(input_ids, (0, 1), value=self.mask_id)
        pad_mask_bool = F.pad(mask_bool, (0, 1), value=True)
        pad_attention_mask = None
        if attention_mask is not None:
            pad_attention_mask = F.pad(attention_mask, (0, 1), value=1)

        cond = self._student_hidden(input_ids, attention_mask)
        cond = F.pad(cond, (0, 0, 0, 1), value=0.0)
        t_level = torch.ones(B, device=input_ids.device, dtype=torch.float32)
        canvas = build_canvas(pad_ids, pad_mask_bool, self.mask_id)
        slot = self._unet_logits(canvas, t_level, cond, pad_attention_mask)  # (B, T+1, V)

        out = slot[:, 1:, :]
        return CausalLMOutputWithPast(loss=None, logits=out)

    # ── generation (DDIM-style confidence unmasking) ────────────────────────

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, num_steps=None, **kwargs):
        """Fill a block of `L` new positions after the context by iterative confidence-
        based unmasking (Dream / MaskGIT). Deterministic (argmax commit, fixed schedule).

        Because attention is bidirectional, we cannot decode one token at a time cheaply;
        instead we lay down a fixed block of [MASK]s and commit the highest-confidence
        positions over K steps. K frozen-student + K UNet forwards for the whole block.
        """
        K = num_steps or self.num_sampling_steps
        B, ctx_len = input_ids.shape
        device = input_ids.device

        max_length = kwargs.get("max_length")
        if max_length is None:
            max_new = kwargs.get("max_new_tokens", 256)
            max_length = ctx_len + max_new
        L = max(1, max_length - ctx_len)
        stopping_criteria = kwargs.get("stopping_criteria")

        seq = torch.cat(
            [input_ids, torch.full((B, L), self.mask_id, device=device, dtype=input_ids.dtype)],
            dim=1,
        )
        if attention_mask is None:
            attention_mask = torch.ones((B, ctx_len), device=device, dtype=torch.long)
        mask = torch.cat([attention_mask, torch.ones((B, L), device=device, dtype=torch.long)], dim=1)

        new_slice = slice(ctx_len, ctx_len + L)
        still_masked = torch.ones((B, L), dtype=torch.bool, device=device)

        for step in range(K):
            t_level = torch.full((B,), 1.0 - step / K, device=device, dtype=torch.float32)
            # The frozen student can't ingest the MASK id (out of its vocab); feed it a
            # harmless real id (0) at still-masked slots just to get a conditioning vector.
            student_in = seq.clone()
            student_in[:, new_slice] = torch.where(
                still_masked, torch.zeros_like(student_in[:, new_slice]), seq[:, new_slice]
            )
            cond = self._student_hidden(student_in, mask)
            logits = self._unet_logits(seq, t_level, cond, mask)          # (B, T, V)
            new_logits = logits[:, new_slice, :]                          # (B, L, V)
            probs = F.softmax(new_logits.float(), dim=-1)
            conf, pred = probs.max(dim=-1)                                # (B, L)

            keep_masked = num_still_masked(step, K, L)                    # after this step
            for b in range(B):
                cur = still_masked[b]
                n_cur = int(cur.sum().item())
                n_commit = max(0, n_cur - keep_masked)
                if n_commit == 0:
                    continue
                # Rank still-masked positions by confidence; commit the top n_commit.
                cand_conf = conf[b].masked_fill(~cur, -float("inf"))
                topk = torch.topk(cand_conf, k=n_commit).indices
                seq[b, ctx_len + topk] = pred[b, topk]
                still_masked[b, topk] = False

            if not still_masked.any():
                break
            if stopping_criteria is not None and stopping_criteria(seq, None):
                break

        # Commit anything still masked (safety; schedule should have emptied it).
        if still_masked.any():
            student_in = seq.clone()
            student_in[:, new_slice] = torch.where(
                still_masked, torch.zeros_like(student_in[:, new_slice]), seq[:, new_slice]
            )
            cond = self._student_hidden(student_in, mask)
            t_level = torch.zeros((B,), device=device, dtype=torch.float32)
            logits = self._unet_logits(seq, t_level, cond, mask)
            pred = logits[:, new_slice, :].argmax(dim=-1)
            seq[:, new_slice] = torch.where(still_masked, pred, seq[:, new_slice])

        return seq
