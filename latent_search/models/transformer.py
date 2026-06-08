from __future__ import annotations

import math

import torch
from torch import nn


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim: int, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.net = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32, device=timesteps.device)
            / max(half, 1)
        )
        args = timesteps.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if emb.shape[-1] < self.frequency_dim:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.net(emb)


class AdaLNTransformerBlock(nn.Module):
    """Transformer block with adaptive LayerNorm conditioning.

    This is a compact version of the DiT idea used in LaDi-WM: noisy latent
    tokens are processed with self-attention, while timestep/action/text
    conditioning modulates the block.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.ada(cond).chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out
        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int, dropout: float = 0.0) -> nn.Sequential:
    if depth < 1:
        raise ValueError("MLP depth must be >= 1")
    layers: list[nn.Module] = []
    dim = input_dim
    for _ in range(depth - 1):
        layers.extend([nn.Linear(dim, hidden_dim), nn.GELU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        dim = hidden_dim
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)
