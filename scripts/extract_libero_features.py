#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from latent_search.models.encoders import TransformerTextEncoder, VisualFoundationEncoder
from latent_search.training.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DINO/SigLIP latent features from raw LIBERO HDF5 demos.")
    parser.add_argument("--raw-root", required=True, help="Directory containing raw LIBERO .hdf5 files.")
    parser.add_argument("--output-root", required=True, help="Ignored output directory, e.g. data/libero_features.")
    parser.add_argument("--suite", default="libero_90", help="Subfolder name to place extracted episodes under.")
    parser.add_argument("--camera-key", default="agentview_rgb", help="Observation key in the raw demo HDF5 file.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dino-model", default="facebook/dinov2-base")
    parser.add_argument("--siglip-model", default="google/siglip-base-patch16-224")
    parser.add_argument("--text-model", default="google/siglip-base-patch16-224")
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import h5py
        from transformers import AutoImageProcessor
    except ImportError as exc:
        raise ImportError("Install h5py and transformers before feature extraction.") from exc

    device = resolve_device(args.device)
    raw_files = sorted(Path(args.raw_root).rglob("*.hdf5")) + sorted(Path(args.raw_root).rglob("*.h5"))
    if args.max_files is not None:
        raw_files = raw_files[: args.max_files]
    if not raw_files:
        raise FileNotFoundError(f"No raw LIBERO HDF5 files found under {args.raw_root}.")

    output_root = Path(args.output_root) / args.suite
    output_root.mkdir(parents=True, exist_ok=True)

    dino_processor = AutoImageProcessor.from_pretrained(args.dino_model)
    siglip_processor = AutoImageProcessor.from_pretrained(args.siglip_model)
    visual_encoder = VisualFoundationEncoder(args.dino_model, args.siglip_model).to(device)
    text_encoder = TransformerTextEncoder(args.text_model).to(device)

    total_episodes = 0
    for raw_path in tqdm(raw_files, desc="raw files"):
        instruction = infer_instruction(raw_path)
        text_emb = text_encoder.encode([instruction], device=device).cpu().numpy()[0]
        with h5py.File(raw_path, "r") as handle:
            demo_group = handle["data"] if "data" in handle else handle
            for demo_name in demo_group:
                demo = demo_group[demo_name]
                obs = demo["obs"] if "obs" in demo else demo
                if args.camera_key not in obs:
                    raise KeyError(f"{raw_path}:{demo_name} does not contain obs/{args.camera_key}.")
                images = np.asarray(obs[args.camera_key])
                images = normalize_images(images)
                actions = np.asarray(demo["actions"] if "actions" in demo else handle["actions"])
                dino, siglip = encode_video(images, dino_processor, siglip_processor, visual_encoder, device, args.batch_size)
                task_dir = output_root / f"{raw_path.stem}_demo"
                task_dir.mkdir(parents=True, exist_ok=True)
                out_path = task_dir / f"{demo_name}.npz"
                np.savez_compressed(
                    out_path,
                    dino=dino.astype(np.float32),
                    siglip=siglip.astype(np.float32),
                    actions=actions.astype(np.float32),
                    text_emb=text_emb.astype(np.float32),
                    instruction=np.asarray(instruction),
                )
                total_episodes += 1

    print(f"Extracted {total_episodes} episodes under {output_root}")
    print("Next: run scripts/verify_libero_features.py on the output directory.")


@torch.no_grad()
def encode_video(
    images: np.ndarray,
    dino_processor: Any,
    siglip_processor: Any,
    encoder: VisualFoundationEncoder,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    dino_chunks: list[np.ndarray] = []
    siglip_chunks: list[np.ndarray] = []
    pil_images = [Image.fromarray(frame) for frame in images]
    for start in range(0, len(pil_images), batch_size):
        batch = pil_images[start : start + batch_size]
        dino_inputs = dino_processor(images=batch, return_tensors="pt")
        siglip_inputs = siglip_processor(images=batch, return_tensors="pt")
        dino_pixels = dino_inputs["pixel_values"].to(device)
        siglip_pixels = siglip_inputs["pixel_values"].to(device)
        dino_tokens, siglip_tokens = encoder(dino_pixels, siglip_pixels)
        dino_chunks.append(dino_tokens.cpu().numpy())
        siglip_chunks.append(siglip_tokens.cpu().numpy())
    return np.concatenate(dino_chunks, axis=0), np.concatenate(siglip_chunks, axis=0)


def normalize_images(images: np.ndarray) -> np.ndarray:
    if images.ndim != 4:
        raise ValueError(f"Expected image array with shape (T,H,W,C) or (T,C,H,W), got {images.shape}.")
    if images.shape[1] in (1, 3) and images.shape[-1] not in (1, 3):
        images = np.transpose(images, (0, 2, 3, 1))
    if images.dtype != np.uint8:
        images = np.clip(images, 0, 255).astype(np.uint8)
    # LIBERO raw image dumps are often vertically flipped.
    return images[:, ::-1].copy()


def infer_instruction(path: Path) -> str:
    name = path.stem.replace("_demo", "")
    if "SCENE" in name:
        match = re.search(r"SCENE\d+_(.*)", name)
        if match:
            name = match.group(1)
    return name.replace("_", " ").strip().lower()


if __name__ == "__main__":
    main()
