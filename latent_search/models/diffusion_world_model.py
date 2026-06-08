from __future__ import annotations

from typing import Any

import torch
from torch import nn

from latent_search.models.diffusion import DiffusionSchedule
from latent_search.models.latent import LatentSpec, flatten_latents, split_flat_latents
from latent_search.models.transformer import AdaLNTransformerBlock, TimestepEmbedder


class LatentDiffusionWorldModel(nn.Module):
    """LaDi-WM-style action-conditioned diffusion model over latent tokens.

    DINO and SigLIP latents are kept as token sequences instead of being reduced
    to pixels or decoded images.  The model predicts DDPM noise for a future
    latent trajectory conditioned on latent history, future actions, and the
    language embedding.
    """

    def __init__(
        self,
        dino_dim: int = 768,
        siglip_dim: int = 768,
        text_dim: int = 768,
        action_dim: int = 7,
        hidden_dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        diffusion_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        history: int = 4,
        horizon: int = 6,
        dino_tokens: int | None = None,
        siglip_tokens: int | None = None,
    ):
        super().__init__()
        self.dino_tokens = dino_tokens or 1
        self.siglip_tokens = siglip_tokens or 1
        dino_shape = (self.dino_tokens, dino_dim) if dino_tokens is not None else (dino_dim,)
        siglip_shape = (self.siglip_tokens, siglip_dim) if siglip_tokens is not None else (siglip_dim,)
        self.spec = LatentSpec(dino_shape=dino_shape, siglip_shape=siglip_shape)
        self.history = history
        self.horizon = horizon
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.dino_in = nn.Linear(dino_dim, hidden_dim)
        self.siglip_in = nn.Linear(siglip_dim, hidden_dim)
        self.dino_out = nn.Linear(hidden_dim, dino_dim)
        self.siglip_out = nn.Linear(hidden_dim, siglip_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.time_embed = TimestepEmbedder(hidden_dim)

        self.stream_embed = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)
        self.type_embed = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)
        self.temporal_embed = nn.Parameter(torch.randn(1, history + horizon, hidden_dim) * 0.02)
        self.dino_token_embed = nn.Parameter(torch.randn(1, self.dino_tokens, hidden_dim) * 0.02)
        self.siglip_token_embed = nn.Parameter(torch.randn(1, self.siglip_tokens, hidden_dim) * 0.02)

        self.blocks = nn.ModuleList(
            [AdaLNTransformerBlock(hidden_dim, num_heads, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.schedule = DiffusionSchedule(diffusion_steps, beta_start, beta_end)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "LatentDiffusionWorldModel":
        model_cfg = dict(cfg.get("model", cfg))
        data_cfg = cfg.get("data", {})
        model_cfg.setdefault("history", data_cfg.get("history", 4))
        model_cfg.setdefault("horizon", data_cfg.get("horizon", 6))
        return cls(**model_cfg)

    def loss(
        self,
        history_dino: torch.Tensor,
        history_siglip: torch.Tensor,
        future_dino: torch.Tensor,
        future_siglip: torch.Tensor,
        actions: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        future = flatten_latents(future_dino, future_siglip)
        timesteps = torch.randint(0, self.schedule.steps, (future.shape[0],), device=future.device)
        noisy_future, noise = self.schedule.add_noise(future, timesteps)
        predicted_noise = self.forward(history_dino, history_siglip, noisy_future, actions, text_emb, timesteps)
        loss = self.schedule.loss(predicted_noise, noise)
        return loss, {"wm_loss": loss.detach()}

    def forward(
        self,
        history_dino: torch.Tensor,
        history_siglip: torch.Tensor,
        noisy_future_flat: torch.Tensor,
        actions: torch.Tensor,
        text_emb: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        future_dino, future_siglip = split_flat_latents(noisy_future_flat, self.spec)
        history_tokens = self._embed_latents(history_dino, history_siglip, time_offset=0, is_future=False)
        future_tokens = self._embed_latents(
            future_dino,
            future_siglip,
            time_offset=self.history,
            is_future=True,
            actions=actions,
        )
        tokens = torch.cat([history_tokens, future_tokens], dim=1)
        cond = self.text_proj(text_emb) + self.time_embed(timesteps)
        for block in self.blocks:
            tokens = block(tokens, cond)
        future_tokens = self.norm(tokens[:, history_tokens.shape[1] :])
        return self._decode_future_tokens(future_tokens)

    @torch.no_grad()
    def sample(
        self,
        history_dino: torch.Tensor,
        history_siglip: torch.Tensor,
        actions: torch.Tensor,
        text_emb: torch.Tensor,
        steps: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = actions.shape[0]
        horizon = actions.shape[1]
        if horizon != self.horizon:
            raise ValueError(f"Expected action horizon {self.horizon}, got {horizon}.")
        x = torch.randn(batch, horizon, self.spec.flat_dim, device=actions.device, dtype=actions.dtype)
        max_steps = self.schedule.steps if steps is None else min(steps, self.schedule.steps)
        indices = torch.linspace(self.schedule.steps - 1, 0, max_steps, device=actions.device).round().long()
        for idx in indices:
            t = torch.full((batch,), int(idx.item()), device=actions.device, dtype=torch.long)
            noise = self.forward(history_dino, history_siglip, x, actions, text_emb, t)
            x = self.schedule.denoise_step(x, t, noise)
        return split_flat_latents(x, self.spec)

    @torch.no_grad()
    def rollout_flat(
        self,
        history_dino: torch.Tensor,
        history_siglip: torch.Tensor,
        actions: torch.Tensor,
        text_emb: torch.Tensor,
        steps: int | None = None,
    ) -> torch.Tensor:
        pred_dino, pred_siglip = self.sample(history_dino, history_siglip, actions, text_emb, steps=steps)
        return flatten_latents(pred_dino, pred_siglip)

    def _embed_latents(
        self,
        dino: torch.Tensor,
        siglip: torch.Tensor,
        time_offset: int,
        is_future: bool,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dino = self._ensure_tokens(dino, self.dino_tokens, self.spec.dino_shape[-1], "dino")
        siglip = self._ensure_tokens(siglip, self.siglip_tokens, self.spec.siglip_shape[-1], "siglip")
        if dino.shape[:2] != siglip.shape[:2]:
            raise ValueError("DINO and SigLIP sequences must share batch/time dimensions.")
        batch, steps = dino.shape[:2]
        if time_offset + steps > self.temporal_embed.shape[1]:
            raise ValueError("Sequence exceeds configured history+horizon length.")

        dino_tokens = self.dino_in(dino) + self.stream_embed[0] + self.dino_token_embed[:, : dino.shape[2]].unsqueeze(1)
        siglip_tokens = self.siglip_in(siglip) + self.stream_embed[1] + self.siglip_token_embed[:, : siglip.shape[2]].unsqueeze(1)
        tokens = torch.cat([dino_tokens, siglip_tokens], dim=2)
        tokens = tokens + self.type_embed[1 if is_future else 0]
        tokens = tokens + self.temporal_embed[:, time_offset : time_offset + steps].unsqueeze(2)
        if is_future:
            if actions is None:
                raise ValueError("Future latent tokens require actions.")
            action_emb = self.action_proj(actions).reshape(batch, steps, 1, self.hidden_dim)
            tokens = tokens + action_emb
        return tokens.flatten(start_dim=1, end_dim=2)

    def _decode_future_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        batch = tokens.shape[0]
        per_step = self.dino_tokens + self.siglip_tokens
        expected = self.horizon * per_step
        if tokens.shape[1] != expected:
            raise ValueError(f"Expected {expected} future tokens, got {tokens.shape[1]}.")
        tokens = tokens.reshape(batch, self.horizon, per_step, self.hidden_dim)
        dino_tokens = tokens[:, :, : self.dino_tokens]
        siglip_tokens = tokens[:, :, self.dino_tokens :]
        dino = self.dino_out(dino_tokens)
        siglip = self.siglip_out(siglip_tokens)
        if len(self.spec.dino_shape) == 1:
            dino = dino.squeeze(2)
        if len(self.spec.siglip_shape) == 1:
            siglip = siglip.squeeze(2)
        return flatten_latents(dino, siglip)

    @staticmethod
    def _ensure_tokens(latent: torch.Tensor, tokens: int, dim: int, name: str) -> torch.Tensor:
        if latent.ndim == 3:
            latent = latent.unsqueeze(2)
        if latent.ndim != 4:
            raise ValueError(f"{name} must have shape (B,T,D) or (B,T,N,D), got {tuple(latent.shape)}")
        if latent.shape[2] != tokens or latent.shape[3] != dim:
            raise ValueError(f"{name} expected token shape ({tokens},{dim}), got {tuple(latent.shape[2:])}")
        return latent
