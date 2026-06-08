#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from latent_search.models import LatentDiffusionWorldModel, LanguageConditionedProgressComparator
from latent_search.models.encoders import TransformerTextEncoder, VisualFoundationEncoder
from latent_search.planning import CEMPlanner, ProgressiveWideningMCTS
from latent_search.training.utils import load_config, load_model_state, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate latent planner in online LIBERO environments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--world-model", required=True)
    parser.add_argument("--comparator", required=True)
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--camera-key", default="agentview_image")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--render-gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--planner", choices=["cem", "mcts"], default=None)
    parser.add_argument("--output-dir", default="runs/eval_libero")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--dino-model", default="facebook/dinov2-base")
    parser.add_argument("--siglip-model", default="google/siglip-base-patch16-224")
    parser.add_argument("--text-model", default="google/siglip-base-patch16-224")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoImageProcessor
    except ImportError as exc:
        raise ImportError("Install transformers before online LIBERO evaluation.") from exc

    cfg = load_config(args.config)
    device = resolve_device(str(cfg.get("device", "cuda")))
    output_dir = Path(args.output_dir) / args.suite / f"task_{args.task_id:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    world_model = LatentDiffusionWorldModel.from_config(cfg).to(device)
    comparator = LanguageConditionedProgressComparator.from_config(cfg).to(device)
    load_model_state(args.world_model, world_model)
    load_model_state(args.comparator, comparator)

    dino_processor = AutoImageProcessor.from_pretrained(args.dino_model)
    siglip_processor = AutoImageProcessor.from_pretrained(args.siglip_model)
    visual_encoder = VisualFoundationEncoder(args.dino_model, args.siglip_model).to(device)
    text_encoder = TransformerTextEncoder(args.text_model).to(device)

    env, task_language, init_states = build_libero_env(args.suite, args.task_id, args.image_size, args.render_gpu_id)
    text_emb = text_encoder.encode([task_language], device=device)
    planner = build_planner(args, cfg, world_model, comparator)

    episode_results = []
    for episode_idx in range(args.episodes):
        obs = reset_to_init_state(env, init_states, episode_idx, args.seed)
        history = deque(maxlen=int(cfg["data"]["history"]))
        frame = get_frame(obs, args.camera_key)
        dino, siglip = encode_frame(frame, dino_processor, siglip_processor, visual_encoder, device)
        for _ in range(history.maxlen):
            history.append((dino, siglip))
        frames = [frame]
        total_reward = 0.0
        success = False
        for step in range(args.max_steps):
            history_dino = torch.stack([item[0] for item in history], dim=1)
            history_siglip = torch.stack([item[1] for item in history], dim=1)
            result = planner.plan(history_dino, history_siglip, text_emb)
            action = result.action.detach().cpu().numpy().astype(np.float32)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            frame = get_frame(obs, args.camera_key)
            frames.append(frame)
            dino, siglip = encode_frame(frame, dino_processor, siglip_processor, visual_encoder, device)
            history.append((dino, siglip))
            success = success or is_success(reward, info)
            if done or success:
                break
        record = {
            "episode": episode_idx,
            "success": bool(success),
            "return": total_reward,
            "steps": step + 1,
            "task_language": task_language,
        }
        episode_results.append(record)
        print(record)
        if args.save_video:
            save_video(output_dir / f"episode_{episode_idx:03d}.mp4", frames)

    summary = {
        "suite": args.suite,
        "task_id": args.task_id,
        "task_language": task_language,
        "episodes": len(episode_results),
        "success_rate": float(np.mean([item["success"] for item in episode_results])),
        "mean_return": float(np.mean([item["return"] for item in episode_results])),
        "results": episode_results,
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    env.close()


def build_planner(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    world_model: LatentDiffusionWorldModel,
    comparator: LanguageConditionedProgressComparator,
) -> CEMPlanner | ProgressiveWideningMCTS:
    planner_kind = args.planner or cfg.get("planner", {}).get("kind", "cem")
    if planner_kind == "cem":
        return CEMPlanner.from_config(cfg, world_model, comparator)
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
    return ProgressiveWideningMCTS(world_model, comparator, **{k: v for k, v in planner_cfg.items() if k in allowed})


def build_libero_env(suite: str, task_id: int, image_size: int, render_gpu_id: int) -> tuple[Any, str, np.ndarray]:
    try:
        from libero import benchmark, get_libero_path
        from libero.envs import OffScreenRenderEnv
    except ImportError:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite]()
    task = task_suite.get_task(task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=image_size,
        camera_widths=image_size,
        render_gpu_device_id=render_gpu_id,
    )
    init_states = task_suite.get_task_init_states(task_id)
    print(f"LIBERO task: suite={suite}, task_id={task_id}, language={task.language}")
    return env, task.language, init_states


def reset_to_init_state(env: Any, init_states: np.ndarray, episode_idx: int, seed: int) -> dict[str, Any]:
    env.seed(seed + episode_idx)
    env.reset()
    obs = env.set_init_state(init_states[episode_idx % len(init_states)])
    dummy = np.zeros(7, dtype=np.float32)
    dummy[-1] = -1.0
    for _ in range(5):
        obs, _, _, _ = env.step(dummy)
    return obs


@torch.no_grad()
def encode_frame(
    frame: np.ndarray,
    dino_processor: Any,
    siglip_processor: Any,
    encoder: VisualFoundationEncoder,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = Image.fromarray(frame.astype(np.uint8))
    dino_pixels = dino_processor(images=[image], return_tensors="pt")["pixel_values"].to(device)
    siglip_pixels = siglip_processor(images=[image], return_tensors="pt")["pixel_values"].to(device)
    dino, siglip = encoder(dino_pixels, siglip_pixels)
    return dino, siglip


def get_frame(obs: dict[str, Any], camera_key: str) -> np.ndarray:
    candidates = [camera_key, camera_key.replace("_image", "_rgb"), "agentview_image", "agentview_rgb"]
    for key in candidates:
        if key in obs:
            frame = np.asarray(obs[key])
            if frame.ndim == 4:
                frame = frame[0]
            if frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
                frame = np.transpose(frame, (1, 2, 0))
            return np.clip(frame, 0, 255).astype(np.uint8)
    raise KeyError(f"Could not find camera frame. Available obs keys: {sorted(obs.keys())}")


def is_success(reward: float, info: Any) -> bool:
    if float(reward) > 0.0:
        return True
    if isinstance(info, dict):
        for key in ("success", "is_success", "done"):
            if key in info and bool(np.asarray(info[key]).any()):
                return True
    return False


def save_video(path: Path, frames: list[np.ndarray]) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError("Install imageio to use --save-video.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=20)


if __name__ == "__main__":
    main()
