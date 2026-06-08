from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from latent_search.models.transformer import build_mlp


class LanguageConditionedProgressComparator(nn.Module):
    """Task progress function C_omega(z, c).

    The score is not calibrated as a probability; it is an ordinal progress
    score trained so that later states in successful demonstrations rank above
    earlier states for the same instruction.
    """

    def __init__(
        self,
        dino_dim: int = 768,
        siglip_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 512,
        depth: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dino_dim = dino_dim
        self.siglip_dim = siglip_dim
        self.text_dim = text_dim
        input_dim = dino_dim + siglip_dim + text_dim
        self.net = build_mlp(input_dim, hidden_dim, 1, depth=depth, dropout=dropout)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "LanguageConditionedProgressComparator":
        model_cfg = cfg.get("model", {})
        progress_cfg = cfg.get("progress", {})
        return cls(
            dino_dim=model_cfg.get("dino_dim", 768),
            siglip_dim=model_cfg.get("siglip_dim", 768),
            text_dim=model_cfg.get("text_dim", 768),
            hidden_dim=progress_cfg.get("hidden_dim", 512),
            depth=progress_cfg.get("depth", 3),
            dropout=model_cfg.get("dropout", 0.1),
        )

    def forward(self, dino: torch.Tensor, siglip: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        dino_vec = pool_state(dino)
        siglip_vec = pool_state(siglip)
        if dino_vec.shape[-1] != self.dino_dim:
            raise ValueError(f"Expected DINO dim {self.dino_dim}, got {dino_vec.shape[-1]}.")
        if siglip_vec.shape[-1] != self.siglip_dim:
            raise ValueError(f"Expected SigLIP dim {self.siglip_dim}, got {siglip_vec.shape[-1]}.")
        x = torch.cat([dino_vec, siglip_vec, text_emb], dim=-1)
        return self.net(x).squeeze(-1)

    def ranking_loss(
        self,
        early_dino: torch.Tensor,
        early_siglip: torch.Tensor,
        late_dino: torch.Tensor,
        late_siglip: torch.Tensor,
        text_emb: torch.Tensor,
        margin: float = 0.2,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        early = self(early_dino, early_siglip, text_emb)
        late = self(late_dino, late_siglip, text_emb)
        loss = F.relu(early - late + margin).mean()
        accuracy = (late > early).float().mean()
        return loss, {
            "rank_loss": loss.detach(),
            "rank_acc": accuracy.detach(),
            "score_gap": (late - early).mean().detach(),
        }


def pool_state(latent: torch.Tensor) -> torch.Tensor:
    """Pool a single-state latent tensor to `(B,D)`.

    Accepts `(B,D)` pooled features or `(B,N,D)` token features.
    """

    if latent.ndim == 2:
        return latent
    if latent.ndim == 3:
        return latent.mean(dim=1)
    raise ValueError(f"Expected latent state shape (B,D) or (B,N,D), got {tuple(latent.shape)}")
