#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from latent_search.data.libero_latent import discover_episode_files, load_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate preprocessed LIBERO latent feature episodes.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--dino-key", default="dino")
    parser.add_argument("--siglip-key", default="siglip")
    parser.add_argument("--action-key", default="actions")
    parser.add_argument("--text-key", default="text_emb")
    parser.add_argument("--max-files", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = discover_episode_files(Path(args.data_root))
    checked = 0
    total_windows = 0
    dino_shapes: set[tuple[int, ...]] = set()
    siglip_shapes: set[tuple[int, ...]] = set()
    action_shapes: set[tuple[int, ...]] = set()
    text_shapes: set[tuple[int, ...]] = set()
    errors: list[str] = []
    keys = {
        "dino": args.dino_key,
        "siglip": args.siglip_key,
        "actions": args.action_key,
        "text": args.text_key,
    }

    for path in files[: args.max_files]:
        try:
            episode = load_episode(path, keys)
            dino = np.asarray(episode["dino"])
            siglip = np.asarray(episode["siglip"])
            actions = np.asarray(episode["actions"])
            text = np.asarray(episode["text"])
            validate_episode(path, dino, siglip, actions, text, args.history, args.horizon)
            length = actions.shape[0]
            total_windows += max(length - args.history - args.horizon + 1, 0)
            dino_shapes.add(tuple(dino.shape[1:]))
            siglip_shapes.add(tuple(siglip.shape[1:]))
            action_shapes.add(tuple(actions.shape[1:]))
            text_shapes.add(tuple(text.shape))
            checked += 1
        except Exception as exc:  # noqa: BLE001 - CLI should collect all failures.
            errors.append(f"{path}: {exc}")

    print(f"found_files: {len(files)}")
    print(f"checked_files: {checked}")
    print(f"train_windows_in_checked_files: {total_windows}")
    print(f"dino_per_state_shapes: {sorted(dino_shapes)}")
    print(f"siglip_per_state_shapes: {sorted(siglip_shapes)}")
    print(f"action_shapes: {sorted(action_shapes)}")
    print(f"text_emb_shapes: {sorted(text_shapes)}")
    if len(dino_shapes) == 1 and len(siglip_shapes) == 1:
        dino_shape = next(iter(dino_shapes))
        siglip_shape = next(iter(siglip_shapes))
        print("suggested_config:")
        print(f"  model.dino_dim: {dino_shape[-1]}")
        print(f"  model.siglip_dim: {siglip_shape[-1]}")
        print(f"  model.dino_tokens: {dino_shape[0] if len(dino_shape) == 2 else 'null'}")
        print(f"  model.siglip_tokens: {siglip_shape[0] if len(siglip_shape) == 2 else 'null'}")
    if errors:
        print("errors:")
        for err in errors[:20]:
            print(f"  - {err}")
        raise SystemExit(1)
    print("status: OK")


def validate_episode(
    path: Path,
    dino: np.ndarray,
    siglip: np.ndarray,
    actions: np.ndarray,
    text: np.ndarray,
    history: int,
    horizon: int,
) -> None:
    if dino.ndim not in (2, 3):
        raise ValueError(f"dino must be (T,D) or (T,N,D), got {dino.shape}")
    if siglip.ndim not in (2, 3):
        raise ValueError(f"siglip must be (T,D) or (T,N,D), got {siglip.shape}")
    if actions.ndim != 2 or actions.shape[-1] != 7:
        raise ValueError(f"actions must be (T,7), got {actions.shape}")
    length = actions.shape[0]
    if dino.shape[0] != length or siglip.shape[0] != length:
        raise ValueError(f"length mismatch: dino={dino.shape[0]}, siglip={siglip.shape[0]}, actions={length}")
    if length < history + horizon:
        raise ValueError(f"episode too short for history+horizon={history+horizon}: {length}")
    if text.ndim not in (1, 2):
        raise ValueError(f"text_emb must be (C,) or (T,C), got {text.shape}")
    if text.ndim == 2 and text.shape[0] not in (1, length):
        raise ValueError(f"text_emb first dim must be 1 or T={length}, got {text.shape}")
    if not np.isfinite(dino).all() or not np.isfinite(siglip).all() or not np.isfinite(actions).all():
        raise ValueError(f"non-finite numeric values in {path}")


if __name__ == "__main__":
    main()
