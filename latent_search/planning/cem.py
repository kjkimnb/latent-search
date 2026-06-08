from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from latent_search.models.diffusion_world_model import LatentDiffusionWorldModel
from latent_search.models.progress import LanguageConditionedProgressComparator


@dataclass
class PlanResult:
    action: torch.Tensor
    action_sequence: torch.Tensor
    value: torch.Tensor
    info: dict[str, Any]


class CEMPlanner:
    """Latent-space MPC planner optimized with the cross-entropy method."""

    def __init__(
        self,
        world_model: LatentDiffusionWorldModel,
        comparator: LanguageConditionedProgressComparator,
        horizon: int,
        action_dim: int = 7,
        num_samples: int = 256,
        elite_frac: float = 0.1,
        iterations: int = 4,
        gamma: float = 0.98,
        action_low: list[float] | torch.Tensor | None = None,
        action_high: list[float] | torch.Tensor | None = None,
        terminal_weight: float = 1.0,
        trajectory_weight: float = 1.0,
        diffusion_steps: int | None = 10,
    ):
        self.world_model = world_model
        self.comparator = comparator
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_samples = num_samples
        self.elite_count = max(1, int(num_samples * elite_frac))
        self.iterations = iterations
        self.gamma = gamma
        self.terminal_weight = terminal_weight
        self.trajectory_weight = trajectory_weight
        self.diffusion_steps = diffusion_steps
        low = -torch.ones(action_dim) if action_low is None else torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.ones(action_dim) if action_high is None else torch.as_tensor(action_high, dtype=torch.float32)
        self.registered_low = low
        self.registered_high = high

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        world_model: LatentDiffusionWorldModel,
        comparator: LanguageConditionedProgressComparator,
    ) -> "CEMPlanner":
        planner_cfg = dict(cfg.get("planner", {}))
        planner_cfg.pop("kind", None)
        return cls(world_model=world_model, comparator=comparator, **planner_cfg)

    @torch.no_grad()
    def plan(
        self,
        history_dino: torch.Tensor,
        history_siglip: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> PlanResult:
        if history_dino.shape[0] != 1 or history_siglip.shape[0] != 1 or text_emb.shape[0] != 1:
            raise ValueError("CEMPlanner.plan currently expects batch size 1.")
        device = history_dino.device
        low = self.registered_low.to(device)
        high = self.registered_high.to(device)
        mean = (low + high).mul(0.5).expand(self.horizon, self.action_dim).clone()
        std = (high - low).mul(0.5).expand_as(mean).clone()

        best_actions = mean
        best_value = torch.tensor(float("-inf"), device=device)
        metrics: dict[str, Any] = {}

        self.world_model.eval()
        self.comparator.eval()
        for iteration in range(self.iterations):
            eps = torch.randn(self.num_samples, self.horizon, self.action_dim, device=device)
            actions = (mean + std * eps).clamp(low, high)
            values = self.evaluate_sequences(history_dino, history_siglip, text_emb, actions)
            elite_values, elite_idx = torch.topk(values, self.elite_count)
            elites = actions[elite_idx]
            mean = elites.mean(dim=0)
            std = elites.std(dim=0).clamp_min(1e-4)
            if elite_values[0] > best_value:
                best_value = elite_values[0]
                best_actions = elites[0]
            metrics[f"iter_{iteration}_best"] = float(elite_values[0].item())
            metrics[f"iter_{iteration}_mean"] = float(values.mean().item())

        return PlanResult(
            action=best_actions[0],
            action_sequence=best_actions,
            value=best_value,
            info=metrics,
        )

    @torch.no_grad()
    def evaluate_sequences(
        self,
        history_dino: torch.Tensor,
        history_siglip: torch.Tensor,
        text_emb: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        batch = actions.shape[0]
        hist_dino = history_dino.expand(batch, *history_dino.shape[1:])
        hist_siglip = history_siglip.expand(batch, *history_siglip.shape[1:])
        text = text_emb.expand(batch, -1)
        pred_dino, pred_siglip = self.world_model.sample(
            hist_dino,
            hist_siglip,
            actions,
            text,
            steps=self.diffusion_steps,
        )
        discounts = torch.tensor(
            [self.gamma**idx for idx in range(self.horizon)],
            device=actions.device,
            dtype=actions.dtype,
        )
        trajectory_scores = []
        for idx in range(self.horizon):
            trajectory_scores.append(self.comparator(pred_dino[:, idx], pred_siglip[:, idx], text))
        scores = torch.stack(trajectory_scores, dim=1)
        value = self.trajectory_weight * (scores * discounts).sum(dim=1)
        value = value + self.terminal_weight * scores[:, -1]
        return value
