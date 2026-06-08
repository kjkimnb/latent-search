from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch

from latent_search.models.diffusion_world_model import LatentDiffusionWorldModel
from latent_search.models.progress import LanguageConditionedProgressComparator
from latent_search.planning.cem import PlanResult


ActionSampler = Callable[[int, torch.device], torch.Tensor]


@dataclass
class Node:
    history_dino: torch.Tensor
    history_siglip: torch.Tensor
    depth: int
    visits: int = 0
    value_sum: float = 0.0
    children: list["Edge"] = field(default_factory=list)

    @property
    def value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


@dataclass
class Edge:
    action: torch.Tensor
    child: Node
    visits: int = 0
    value_sum: float = 0.0

    @property
    def value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


class ProgressiveWideningMCTS:
    """Continuous-action MCTS with progressive widening.

    This is primarily a research baseline.  For LIBERO-scale 7D continuous
    actions, CEM is usually the stronger default.
    """

    def __init__(
        self,
        world_model: LatentDiffusionWorldModel,
        comparator: LanguageConditionedProgressComparator,
        horizon: int,
        action_dim: int = 7,
        simulations: int = 128,
        exploration: float = 1.0,
        widening_k: float = 2.0,
        widening_alpha: float = 0.5,
        action_low: list[float] | torch.Tensor | None = None,
        action_high: list[float] | torch.Tensor | None = None,
        diffusion_steps: int | None = 5,
        action_sampler: ActionSampler | None = None,
    ):
        self.world_model = world_model
        self.comparator = comparator
        self.horizon = horizon
        self.action_dim = action_dim
        self.simulations = simulations
        self.exploration = exploration
        self.widening_k = widening_k
        self.widening_alpha = widening_alpha
        self.diffusion_steps = diffusion_steps
        self.low = -torch.ones(action_dim) if action_low is None else torch.as_tensor(action_low, dtype=torch.float32)
        self.high = torch.ones(action_dim) if action_high is None else torch.as_tensor(action_high, dtype=torch.float32)
        self.action_sampler = action_sampler

    @torch.no_grad()
    def plan(
        self,
        history_dino: torch.Tensor,
        history_siglip: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> PlanResult:
        if history_dino.shape[0] != 1:
            raise ValueError("ProgressiveWideningMCTS.plan expects batch size 1.")
        root = Node(history_dino=history_dino, history_siglip=history_siglip, depth=0)
        self.world_model.eval()
        self.comparator.eval()
        for _ in range(self.simulations):
            path: list[Edge] = []
            node = root
            while node.depth < self.horizon:
                if self._can_expand(node):
                    edge = self._expand(node, text_emb)
                    path.append(edge)
                    node = edge.child
                    break
                edge = self._select(node)
                path.append(edge)
                node = edge.child
            value = self._evaluate(node, text_emb)
            self._backpropagate(root, path, value)

        if not root.children:
            action = self._sample_actions(1, history_dino.device)[0]
            return PlanResult(action=action, action_sequence=action.unsqueeze(0), value=torch.tensor(0.0), info={})
        best = max(root.children, key=lambda edge: edge.visits)
        return PlanResult(
            action=best.action,
            action_sequence=best.action.unsqueeze(0),
            value=torch.tensor(best.value, device=history_dino.device),
            info={"root_visits": root.visits, "children": len(root.children)},
        )

    def _can_expand(self, node: Node) -> bool:
        limit = self.widening_k * max(node.visits, 1) ** self.widening_alpha
        return len(node.children) < int(math.ceil(limit))

    def _select(self, node: Node) -> Edge:
        log_n = math.log(max(node.visits, 1))

        def uct(edge: Edge) -> float:
            bonus = self.exploration * math.sqrt(log_n / (1 + edge.visits))
            return edge.value + bonus

        return max(node.children, key=uct)

    @torch.no_grad()
    def _expand(self, node: Node, text_emb: torch.Tensor) -> Edge:
        action = self._sample_actions(1, node.history_dino.device)[0]
        child_dino, child_siglip = self._one_step(node.history_dino, node.history_siglip, action, text_emb)
        edge = Edge(action=action, child=Node(child_dino, child_siglip, depth=node.depth + 1))
        node.children.append(edge)
        return edge

    @torch.no_grad()
    def _one_step(
        self,
        history_dino: torch.Tensor,
        history_siglip: torch.Tensor,
        action: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = torch.zeros(1, self.world_model.horizon, self.action_dim, device=action.device, dtype=action.dtype)
        actions[:, 0] = action
        pred_dino, pred_siglip = self.world_model.sample(
            history_dino,
            history_siglip,
            actions,
            text_emb,
            steps=self.diffusion_steps,
        )
        next_dino = pred_dino[:, 0]
        next_siglip = pred_siglip[:, 0]
        new_hist_dino = torch.cat([history_dino[:, 1:], next_dino.unsqueeze(1)], dim=1)
        new_hist_siglip = torch.cat([history_siglip[:, 1:], next_siglip.unsqueeze(1)], dim=1)
        return new_hist_dino, new_hist_siglip

    @torch.no_grad()
    def _evaluate(self, node: Node, text_emb: torch.Tensor) -> float:
        score = self.comparator(node.history_dino[:, -1], node.history_siglip[:, -1], text_emb)
        return float(score.item())

    def _backpropagate(self, root: Node, path: list[Edge], value: float) -> None:
        root.visits += 1
        root.value_sum += value
        for edge in path:
            edge.visits += 1
            edge.value_sum += value
            edge.child.visits += 1
            edge.child.value_sum += value

    def _sample_actions(self, count: int, device: torch.device) -> torch.Tensor:
        if self.action_sampler is not None:
            return self.action_sampler(count, device)
        low = self.low.to(device)
        high = self.high.to(device)
        return low + torch.rand(count, self.action_dim, device=device) * (high - low)
