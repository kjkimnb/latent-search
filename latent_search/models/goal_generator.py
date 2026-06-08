from __future__ import annotations

from typing import Any

import torch
from torch import nn

from latent_search.models.progress import pool_state
from latent_search.models.transformer import build_mlp


class GoalStateGenerator(nn.Module):
    """Optional language-conditioned goal latent generator.

    It predicts a diagonal Gaussian over final pooled DINO/SigLIP states.  This
    module is intentionally separate from the progress-planning path because a
    single language instruction can admit multiple valid final configurations.
    """

    def __init__(
        self,
        dino_dim: int = 768,
        siglip_dim: int = 768,
        text_dim: int = 768,
        history: int = 4,
        hidden_dim: int = 512,
        depth: int = 3,
        min_log_std: float = -5.0,
        max_log_std: float = 2.0,
    ):
        super().__init__()
        self.dino_dim = dino_dim
        self.siglip_dim = siglip_dim
        self.text_dim = text_dim
        self.history = history
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        state_dim = dino_dim + siglip_dim
        self.net = build_mlp(history * state_dim + text_dim, hidden_dim, 2 * state_dim, depth=depth)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "GoalStateGenerator":
        model_cfg = cfg.get("model", {})
        data_cfg = cfg.get("data", {})
        return cls(
            dino_dim=model_cfg.get("dino_dim", 768),
            siglip_dim=model_cfg.get("siglip_dim", 768),
            text_dim=model_cfg.get("text_dim", 768),
            history=data_cfg.get("history", 4),
            hidden_dim=model_cfg.get("hidden_dim", 512),
            depth=3,
        )

    def forward(
        self,
        history_dino: torch.Tensor,
        history_siglip: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history_dino.shape[1] != self.history or history_siglip.shape[1] != self.history:
            raise ValueError(f"Expected history length {self.history}.")
        dino = pool_sequence(history_dino)
        siglip = pool_sequence(history_siglip)
        x = torch.cat([dino, siglip], dim=-1).flatten(start_dim=1)
        mean, log_std = self.net(torch.cat([x, text_emb], dim=-1)).chunk(2, dim=-1)
        return mean, log_std.clamp(self.min_log_std, self.max_log_std)

    def sample(
        self,
        history_dino: torch.Tensor,
        history_siglip: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        mean, log_std = self(history_dino, history_siglip, text_emb)
        return mean + torch.randn_like(mean) * log_std.exp()


def pool_sequence(latent: torch.Tensor) -> torch.Tensor:
    """Pool `(B,T,D)` or `(B,T,N,D)` into `(B,T,D)`."""

    if latent.ndim == 3:
        return latent
    if latent.ndim == 4:
        states = [pool_state(latent[:, idx]) for idx in range(latent.shape[1])]
        return torch.stack(states, dim=1)
    raise ValueError(f"Expected sequence latent with 3 or 4 dims, got {tuple(latent.shape)}")
