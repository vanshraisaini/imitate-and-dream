import math

import torch
import torch.nn as nn


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class SinusoidalEmbedding(nn.Module):
    """Embeds the flow time t in [0, 1] into a dim-sized vector."""

    def __init__(self, dim, max_period=10000):
        super().__init__()
        assert dim % 2 == 0
        self.dim, self.max_period = dim, max_period

    def forward(self, t):
        t = t.float() * 1000.0
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class AdaLNBlock(nn.Module):
    """Self-attn over action tokens -> cross-attn to observation tokens -> MLP.

    Each sub-layer is modulated by the flow-time embedding (AdaLN-Zero:
    shift / scale / gate, with gates initialised to 0).
    """

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, mem, c):
        s1, sc1, g1, s2, sc2, g2, s3, sc3, g3 = self.ada(c).unsqueeze(1).chunk(9, dim=-1)
        h = modulate(self.norm1(x), s1, sc1)
        x = x + g1 * self.self_attn(h, h, h, need_weights=False)[0]
        h = modulate(self.norm2(x), s2, sc2)
        x = x + g2 * self.cross_attn(h, mem, mem, need_weights=False)[0]
        h = modulate(self.norm3(x), s3, sc3)
        return x + g3 * self.mlp(h)


class ActionTransformer(nn.Module):
    """Velocity network v_theta(A_t, t, obs).

    a_t: (B, horizon, action_dim)  noisy action chunk
    t  : (B,)                      flow time in [0, 1]
    mem: (B, N, dim)               observation tokens
    out: (B, horizon, action_dim)  predicted velocity
    """

    def __init__(self, action_dim=7, horizon=16, dim=256, depth=6, heads=8):
        super().__init__()
        self.in_proj = nn.Linear(action_dim, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, horizon, dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.blocks = nn.ModuleList([AdaLNBlock(dim, heads) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out_proj = nn.Linear(dim, action_dim)
        for m in (self.final_ada[1], self.out_proj):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, a_t, t, mem):
        x = self.in_proj(a_t) + self.pos_embed
        c = self.time_mlp(t)
        for blk in self.blocks:
            x = blk(x, mem, c)
        shift, scale = self.final_ada(c).unsqueeze(1).chunk(2, dim=-1)
        return self.out_proj(modulate(self.final_norm(x), shift, scale))