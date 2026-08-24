"""MaskUNet — a 1D UNet denoiser for discrete mask-token diffusion.

Operates on a length-T canvas of token ids (T up to 2048). Each token is embedded,
then a UNet over the sequence LENGTH (down-sample -> bottleneck -> up-sample with skip
connections) refines a per-position representation and predicts a distribution over the
vocabulary at every position. The forward (noising) process masks a fraction of the
tokens; this network reverses it.

Key properties (deliberately different from the old FlowNet):
  * FULL BIDIRECTIONAL self-attention across the (pooled) sequence — every position can
    see every other. The old "no token mixing" property is gone by design; that is what
    lets the model "plan ahead" and fill a masked canvas coherently (Dream-style).
  * adaLN timestep conditioning: the scalar noise level t modulates every block via
    (shift, scale, gate).
  * The frozen student's final hidden state conditions the network two ways: (a) it is
    projected and ADDED to each slot's input at full resolution (so slot t always carries
    h_s[t] directly, before any pooling), and (b) it is cross-attended at every stage.
    The direct per-slot injection is what lets the network actually use the conditioning
    — cross-attention over pooled cond alone was too weak (top-1 stuck near random).
  * Tied INPUT and OUTPUT embeddings: both the token input embedding and the vocab
    output head reuse the FROZEN student embedding matrix E (V x d_student). Inputs are
    E[tok] projected d_student -> d_model by a small trainable Linear (plus one learnable
    [MASK] vector); outputs are h @ E.T. So neither carries a V-sized trainable table —
    the only trainable weights are the denoiser trunk, and predictions live in the real
    vocab geometry from step 0.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    """Sinusoidal embedding of a scalar timestep t in [0, 1]. t: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)            # (B, half)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)   # (B, 2*half)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class AdaLN(nn.Module):
    """Produces (shift, scale, gate) from the timestep embedding for one sub-block.

    Applied as: out = gate * f(norm(h) * (1 + scale) + shift), added residually. The
    gate is zero-initialized so each block starts as an identity map (stable training).
    """

    def __init__(self, t_dim, channels):
        super().__init__()
        self.lin = nn.Linear(t_dim, 3 * channels)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, t_emb):
        shift, scale, gate = self.lin(t_emb).chunk(3, dim=-1)     # (B, C) each
        return shift, scale, gate


def _modulate(x, shift, scale):
    # x: (B, T, C); shift/scale: (B, C)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ResConvBlock(nn.Module):
    """Residual 1D conv block (kernel 3, over the length axis) with adaLN."""

    def __init__(self, channels, t_dim):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.ada = AdaLN(t_dim, channels)

    def forward(self, h, t_emb):
        # h: (B, T, C)
        shift, scale, gate = self.ada(t_emb)
        x = _modulate(self.norm(h), shift, scale)                 # (B, T, C)
        x = x.transpose(1, 2)                                     # (B, C, T)
        x = self.conv2(self.act(self.conv1(x)))
        x = x.transpose(1, 2)                                     # (B, T, C)
        return h + gate.unsqueeze(1) * x


class SelfAttnBlock(nn.Module):
    """Full bidirectional multi-head self-attention over the length axis, adaLN-gated."""

    def __init__(self, channels, n_heads, t_dim):
        super().__init__()
        assert channels % n_heads == 0, f"channels {channels} not divisible by n_heads {n_heads}"
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)
        self.ada = AdaLN(t_dim, channels)

    def forward(self, h, t_emb, key_padding_mask=None):
        # h: (B, T, C); key_padding_mask: (B, T) bool, True on PAD keys to ignore.
        B, T, C = h.shape
        shift, scale, gate = self.ada(t_emb)
        x = _modulate(self.norm(h), shift, scale)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)   # (B, H, T, hd)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            # (B, 1, 1, T) additive mask: -inf on pad keys.
            attn_mask = torch.zeros(B, 1, 1, T, device=h.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(
                key_padding_mask.view(B, 1, 1, T), torch.finfo(q.dtype).min
            )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # (B, H, T, hd)
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        out = torch.nan_to_num(out)   # a fully-padded row can NaN; it's masked from loss
        return h + gate.unsqueeze(1) * out


class CrossAttnBlock(nn.Module):
    """Cross-attention: queries from the running hidden, K/V from the conditioning
    (student final hidden projected to this stage's width). adaLN-gated."""

    def __init__(self, channels, n_heads, t_dim, cond_dim):
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.q = nn.Linear(channels, channels)
        self.kv = nn.Linear(cond_dim, 2 * channels)
        self.proj = nn.Linear(channels, channels)
        self.ada = AdaLN(t_dim, channels)

    def forward(self, h, t_emb, cond, cond_padding_mask=None):
        # h: (B, T, C); cond: (B, S, cond_dim); cond_padding_mask: (B, S) bool True on PAD.
        B, T, C = h.shape
        S = cond.shape[1]
        shift, scale, gate = self.ada(t_emb)
        x = _modulate(self.norm(h), shift, scale)
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(cond).chunk(2, dim=-1)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if cond_padding_mask is not None:
            attn_mask = torch.zeros(B, 1, 1, S, device=h.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(
                cond_padding_mask.view(B, 1, 1, S), torch.finfo(q.dtype).min
            )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        out = torch.nan_to_num(out)
        return h + gate.unsqueeze(1) * out


class Stage(nn.Module):
    """One resolution level: ResConv -> SelfAttn -> CrossAttn (all adaLN-gated).

    Conditioning enters at the stage's own channel width (the caller passes an already
    stage-width-projected cond tensor). Length is unchanged; the caller handles down/up
    sampling around the stage.
    """

    def __init__(self, channels, n_heads, t_dim):
        super().__init__()
        self.res = ResConvBlock(channels, t_dim)
        self.sa = SelfAttnBlock(channels, n_heads, t_dim)
        self.ca = CrossAttnBlock(channels, n_heads, t_dim, channels)

    def forward(self, h, t_emb, cond, key_padding_mask, cond_padding_mask):
        h = self.res(h, t_emb)
        h = self.sa(h, t_emb, key_padding_mask=key_padding_mask)
        h = self.ca(h, t_emb, cond, cond_padding_mask=cond_padding_mask)
        return h


def _pool_mask(mask, factor):
    """Down-pool a (B, T) bool PAD mask by `factor`: a pooled position is PAD only if ALL
    its source positions were PAD. Returns (B, T // factor)."""
    if mask is None:
        return None
    B, T = mask.shape
    T2 = T // factor
    m = mask[:, : T2 * factor].view(B, T2, factor)
    return m.all(dim=2)


class MaskUNet(nn.Module):
    """1D UNet over the sequence length for mask-token diffusion.

    Args:
        vocab_size:  V (real tokens). Row V of the embedding is the learnable [MASK].
        d_out:       output vector width per position; MUST equal the student embedding
                     dim so the tied head `h @ E.T` is valid (E is V x d_out).
        d_model:     base channel width (== channels[0]).
        channels:    per-stage widths, coarsening as length halves, e.g. [768,1152,1536].
        lengths:     per-stage lengths (informational / validation), e.g. [2048,1024,512].
        n_heads:     attention heads (each stage uses the same count; width must divide).
        t_dim:       timestep-embedding width.
        cond_dim:    student hidden size (conditioning input dim).
        max_seq_len: canvas cap (position-embedding rows).
    """

    def __init__(
        self,
        vocab_size=151936,
        d_out=1536,
        d_model=768,
        channels=(768, 1152, 1536),
        lengths=(2048, 1024, 512),
        n_heads=12,
        t_dim=768,
        cond_dim=1536,
        max_seq_len=2048,
    ):
        super().__init__()
        channels = list(channels)
        assert channels[0] == d_model, "channels[0] must equal d_model"
        self.vocab_size = vocab_size
        self.mask_id = vocab_size            # sentinel id for [MASK] positions
        self.d_out = d_out
        self.d_model = d_model
        self.channels = channels
        self.lengths = list(lengths)
        self.max_seq_len = max_seq_len
        self.n_stages = len(channels)

        # Input token embedding is DERIVED from the frozen student embedding (passed in at
        # forward time) via a trainable projection d_out -> d_model; only the single
        # [MASK] vector is a fresh learnable parameter. No V-sized trainable table.
        self.embed_proj = nn.Linear(d_out, d_model, bias=False)
        self.mask_vec = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_vec, std=0.02)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Per-slot conditioning injection: the student's final hidden at slot t is
        # projected and ADDED to the input at slot t (in addition to the cross-attention
        # below). This guarantees slot t always carries its own h_s[t] — the exact signal
        # a trivial probe on h_s uses — surviving the UNet's pooling and adaLN gate-opening
        # (cross-attention alone, over pooled cond, was too weak to route it).
        self.cond_in = nn.Linear(cond_dim, d_model)

        # Timestep MLP.
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )
        self.t_dim = t_dim

        # Per-stage conditioning projections from the student hidden (cond_dim) to each
        # stage's channel width. Down and up stages at the same level share a projection.
        self.cond_proj = nn.ModuleList([nn.Linear(cond_dim, c) for c in channels])

        # Down path: stage at channels[i], then downsample to channels[i+1] & half length.
        self.down_stages = nn.ModuleList(
            [Stage(channels[i], n_heads, t_dim) for i in range(self.n_stages - 1)]
        )
        self.downsample = nn.ModuleList(
            [nn.Conv1d(channels[i], channels[i + 1], kernel_size=2, stride=2)
             for i in range(self.n_stages - 1)]
        )

        # Bottleneck at the coarsest level (channels[-1]).
        self.bottleneck = nn.ModuleList([
            Stage(channels[-1], n_heads, t_dim),
            Stage(channels[-1], n_heads, t_dim),
        ])

        # Up path: upsample channels[i+1] -> channels[i] & double length, concat skip
        # (channels[i]) -> fuse back to channels[i], then a stage.
        self.upsample = nn.ModuleList(
            [nn.ConvTranspose1d(channels[i + 1], channels[i], kernel_size=2, stride=2)
             for i in range(self.n_stages - 1)]
        )
        self.fuse = nn.ModuleList(
            [nn.Linear(2 * channels[i], channels[i]) for i in range(self.n_stages - 1)]
        )
        self.up_stages = nn.ModuleList(
            [Stage(channels[i], n_heads, t_dim) for i in range(self.n_stages - 1)]
        )

        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_out)   # -> tied-head space

    def forward(self, tok_ids, t, cond, attention_mask, embed_weight):
        """
        Args:
            tok_ids:       (B, T) long — canvas token ids ([MASK] == vocab_size).
            t:             (B,) float in [0, 1] — noise level.
            cond:          (B, T, cond_dim) — student final hidden (conditioning).
            attention_mask:(B, T) 1/0 or None — 1 on real tokens, 0 on PAD.
            embed_weight:  (V, d_out) FROZEN student embedding for the tied output head.

        Returns:
            logits: (B, T, V)
        """
        B, T = tok_ids.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds MaskUNet max_seq_len {self.max_seq_len}; "
                f"truncate inputs (e.g. HFLM(max_length={self.max_seq_len}))."
            )
        # UNet halves the length (n_stages - 1) times; require divisibility so pooling
        # is exact. Callers pad to a multiple of 2**(n_stages-1).
        factor = 2 ** (self.n_stages - 1)
        if T % factor != 0:
            raise ValueError(
                f"Sequence length {T} must be a multiple of {factor} for the UNet's "
                f"{self.n_stages - 1} down/up-samplings; pad the canvas first."
            )

        # Build the input embedding from the FROZEN student embedding: E[tok] projected
        # to d_model, with masked positions replaced by the learnable mask_vec. tok_ids
        # at masked positions hold the sentinel id `vocab_size` (out of range for E), so
        # clamp them to 0 before the lookup and overwrite with mask_vec afterwards.
        is_mask = tok_ids >= self.vocab_size                               # (B, T)
        safe_ids = tok_ids.clamp(max=self.vocab_size - 1)
        e = F.embedding(safe_ids, embed_weight)                           # (B, T, d_out)
        h = self.embed_proj(e.to(self.embed_proj.weight.dtype))          # (B, T, d_model)
        h = torch.where(is_mask.unsqueeze(-1), self.mask_vec.to(h.dtype), h)

        cond = cond.to(h.dtype)
        # Inject the per-slot student hidden directly at its own position (full resolution,
        # before any pooling): slot t's input carries h_s[t].
        h = h + self.cond_in(cond)

        pos = torch.arange(T, device=tok_ids.device)
        h = h + self.pos_embed(pos).unsqueeze(0)

        t_emb = self.t_mlp(timestep_embedding(t, self.t_dim).to(h.dtype))  # (B, t_dim)

        pad = None
        if attention_mask is not None:
            pad = attention_mask == 0                                      # (B, T) True on PAD

        skips = []
        pad_level = pad
        cond_cur = cond
        # ── Down path ─────────────────────────────────────────────────────
        for i in range(self.n_stages - 1):
            c_proj = self.cond_proj[i](cond_cur)                           # (B, T_i, C_i)
            h = self.down_stages[i](h, t_emb, c_proj, pad_level, pad_level)
            skips.append((h, pad_level))
            # downsample length /2, widen channels
            h = self.downsample[i](h.transpose(1, 2)).transpose(1, 2)      # (B, T_i/2, C_{i+1})
            pad_level = _pool_mask(pad_level, 2)
            cond_cur = self._pool_cond(cond_cur, 2)

        # ── Bottleneck ────────────────────────────────────────────────────
        c_proj = self.cond_proj[-1](cond_cur)
        for blk in self.bottleneck:
            h = blk(h, t_emb, c_proj, pad_level, pad_level)

        # ── Up path ───────────────────────────────────────────────────────
        for i in reversed(range(self.n_stages - 1)):
            h = self.upsample[i](h.transpose(1, 2)).transpose(1, 2)        # (B, T_i, C_i)
            skip_h, pad_level = skips.pop()
            h = self.fuse[i](torch.cat([h, skip_h], dim=-1))               # (B, T_i, C_i)
            cond_cur = self._unpool_cond_to(cond, skip_h.shape[1])
            c_proj = self.cond_proj[i](cond_cur)
            h = self.up_stages[i](h, t_emb, c_proj, pad_level, pad_level)

        h = self.out_norm(h)
        h = self.out_proj(h)                                               # (B, T, d_out)

        # Tied head: multiply by the frozen student embedding matrix.
        logits = F.linear(h.to(embed_weight.dtype), embed_weight)         # (B, T, V)
        return logits

    @staticmethod
    def _pool_cond(cond, factor):
        # Average-pool the conditioning along length to match a downsampled stage.
        B, T, D = cond.shape
        T2 = T // factor
        return cond[:, : T2 * factor].view(B, T2, factor, D).mean(dim=2)

    @staticmethod
    def _unpool_cond_to(cond, target_len):
        # Upsample the (full-length) cond to a target length by nearest-index striding.
        B, T, D = cond.shape
        if T == target_len:
            return cond
        idx = torch.linspace(0, T - 1, steps=target_len, device=cond.device).round().long()
        return cond[:, idx, :]
