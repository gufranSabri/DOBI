import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class AdaLN(nn.Module):
    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, d_model * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(cond).chunk(2, dim=-1)  # each [B, d_model]
        x = self.norm(x)
        x = x * (1.0 + scale[:, None]) + shift[:, None]
        return x
    

class FlowBlock(nn.Module):
    def __init__(self, d_model, num_heads, cond_dim, mlp_ratio, dropout):
        super().__init__()

        self.adaLN_self = AdaLN(d_model, cond_dim)
        self.self_attn  = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_self = nn.Dropout(dropout)

        self.adaLN_cross = AdaLN(d_model, cond_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop_cross = nn.Dropout(dropout)

        self.adaLN_ffn = AdaLN(d_model, cond_dim)
        dim_ffn = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ffn, d_model),
        )
        self.drop_ffn = nn.Dropout(dropout)

    def forward(self, x, t_emb, self_attn_mask, context=None):
        h      = self.adaLN_self(x, t_emb)
        
        sa, _  = self.self_attn(h, h, h, key_padding_mask=self_attn_mask)
        x      = x + self.drop_self(sa)

        if context is not None:
            h      = self.adaLN_cross(x, t_emb)
            ca, _  = self.cross_attn(h, context, context)
            x      = x + self.drop_cross(ca)

        h      = self.adaLN_ffn(x, t_emb)
        x      = x + self.drop_ffn(self.ffn(h))

        return x


class FlowNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        d_model:    int   = 1024,
        num_layers: int   = 6,
        num_heads:  int   = 8,
        mlp_ratio:  float = 4.0,
        dropout:    float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.d_model    = d_model

        self.input_proj = (
            nn.Linear(hidden_dim, d_model, bias=False)
            if hidden_dim != d_model else nn.Identity()
        )
        self.src_proj = (
            nn.Linear(hidden_dim, d_model, bias=False)
            if hidden_dim != d_model else nn.Identity()
        )

        self.timestep_emb = TimestepEmbedding(d_model)
        self.blocks = nn.ModuleList([
            FlowBlock(
                d_model=d_model, num_heads=num_heads,
                cond_dim=d_model, mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm_out    = nn.LayerNorm(d_model)
        self.output_proj = (
            nn.Linear(d_model, hidden_dim)
            if hidden_dim != d_model else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.elementwise_affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

        if isinstance(self.output_proj, nn.Linear):
            nn.init.zeros_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)

    def forward(self, x_tau, t, attention_mask, context=None):
        x   = x_tau.float()

        kpm = None
        if attention_mask is not None:
            kpm = (attention_mask == 0)  # [B, T]

        x   = self.input_proj(x)    # [B, T, d_model]
        context = self.src_proj(context) if context is not None else None  # [B, T, d_model]
        t_emb = self.timestep_emb(t)  # [B, d_model]

        for block in self.blocks:
            x = block(x, t_emb, self_attn_mask=kpm, context=context)

        x = self.norm_out(x)
        v = self.output_proj(x)      # [B, T, D]

        return v


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, T, D = 2, 64, 2048

    model = FlowNet(
        hidden_dim=D,
        d_model=512,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
    )

    x_tau          = torch.randn(B, T, D)
    t              = torch.randint(0, 1000, (B,))
    s_hidden       = torch.randn(B, T, D)
    attention_mask = torch.ones(B, T, dtype=torch.long)
    attention_mask[0, -10:] = 0  # simulate padding

    v = model(
        x_tau,
        t=t,
        attention_mask=attention_mask,
    )

    assert v.shape == (B, T, D), f"Expected {(B, T, D)}, got {v.shape}"
    print(f"FlowNet output shape : {v.shape}  ✓")
    print(f"Params               : {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")