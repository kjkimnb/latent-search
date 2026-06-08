#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from latent_search.data.libero_latent import load_episode, select_text_embedding, to_float_tensor
from latent_search.models import LatentDiffusionWorldModel, LanguageConditionedProgressComparator
from latent_search.planning import CEMPlanner, ProgressiveWideningMCTS
from latent_search.training.utils import load_config, load_model_state, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan one latent-space action from a cached LIBERO episode.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--world-model", required=True)
    parser.add_argument("--comparator", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--planner", choices=["cem", "mcts"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(str(cfg.get("device", "cuda")))
    world_model = LatentDiffusionWorldModel.from_config(cfg).to(device)
    comparator = LanguageConditionedProgressComparator.from_config(cfg).to(device)
    load_model_state(args.world_model, world_model)
    load_model_state(args.comparator, comparator)

    keys = {
        "dino": cfg["data"].get("dino_key", "dino"),
        "siglip": cfg["data"].get("siglip_key", "siglip"),
        "actions": cfg["data"].get("action_key", "actions"),
        "text": cfg["data"].get("text_key", "text_emb"),
    }
    episode = load_episode(Path(args.episode), keys)
    history = int(cfg["data"]["history"])
    start = int(args.start)
    end = start + history
    history_dino = to_float_tensor(episode["dino"][start:end]).unsqueeze(0).to(device)
    history_siglip = to_float_tensor(episode["siglip"][start:end]).unsqueeze(0).to(device)
    text_emb = to_float_tensor(select_text_embedding(episode["text"], end - 1)).unsqueeze(0).to(device)

    planner_kind = args.planner or cfg.get("planner", {}).get("kind", "cem")
    if planner_kind == "cem":
        planner = CEMPlanner.from_config(cfg, world_model, comparator)
    else:
        planner_cfg = dict(cfg.get("planner", {}))
        planner_cfg.pop("kind", None)
        allowed = {
            "horizon",
            "action_dim",
            "simulations",
            "exploration",
            "widening_k",
            "widening_alpha",
            "action_low",
            "action_high",
            "diffusion_steps",
        }
        planner_cfg = {key: value for key, value in planner_cfg.items() if key in allowed}
        planner = ProgressiveWideningMCTS(world_model, comparator, **planner_cfg)
    result = planner.plan(history_dino, history_siglip, text_emb)
    print({"action": result.action.detach().cpu().tolist(), "value": float(result.value.item()), "info": result.info})


if __name__ == "__main__":
    main()
