#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from latent_search.data import LiberoLatentSequenceDataset, discover_episode_files, split_files
from latent_search.models import LatentDiffusionWorldModel
from latent_search.training.utils import (
    load_config,
    move_batch_to_device,
    resolve_device,
    save_checkpoint,
    save_config,
    set_seed,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LaDi-WM-style latent diffusion world model on LIBERO features.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 0)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "config.yaml")
    device = resolve_device(str(cfg.get("device", "cuda")))

    files = discover_episode_files(Path(args.data_root))
    train_files, val_files = split_files(files, float(cfg["data"].get("val_fraction", 0.05)), int(cfg.get("seed", 0)))
    train_ds = LiberoLatentSequenceDataset(args.data_root, files=train_files, **dataset_kwargs(cfg))
    val_ds = LiberoLatentSequenceDataset(args.data_root, files=val_files, **dataset_kwargs(cfg)) if val_files else None

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["train"].get("num_workers", 0)),
            pin_memory=device.type == "cuda",
        )

    model = LatentDiffusionWorldModel.from_config(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    start_epoch = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("step", 0)) + 1

    best_val = float("inf")
    epochs = int(cfg["train"]["epochs"])
    last_metrics = {}
    for epoch in range(start_epoch, epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg)
        metrics = {"epoch": epoch, "train_loss": train_loss}
        if val_loader is not None:
            metrics["val_loss"] = evaluate(model, val_loader, device)
            if metrics["val_loss"] < best_val:
                best_val = metrics["val_loss"]
                save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, cfg, metrics)
        if epoch % int(cfg["train"].get("save_every", 5)) == 0:
            save_checkpoint(output_dir / f"epoch_{epoch}.pt", model, optimizer, epoch, cfg, metrics)
        write_jsonl(output_dir / "metrics.jsonl", metrics)
        print(metrics)
        last_metrics = metrics

    save_checkpoint(output_dir / "final.pt", model, optimizer, max(epochs - 1, start_epoch), cfg, last_metrics)


def dataset_kwargs(cfg: dict) -> dict:
    data = cfg["data"]
    return {
        "history": int(data["history"]),
        "horizon": int(data["horizon"]),
        "stride": int(data.get("stride", 1)),
        "dino_key": data.get("dino_key", "dino"),
        "siglip_key": data.get("siglip_key", "siglip"),
        "action_key": data.get("action_key", "actions"),
        "text_key": data.get("text_key", "text_emb"),
    }


def train_one_epoch(
    model: LatentDiffusionWorldModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: dict,
) -> float:
    model.train()
    total = 0.0
    count = 0
    for step, batch in enumerate(tqdm(loader, desc="train world model")):
        batch = move_batch_to_device(batch, device)
        loss, _ = model.loss(
            batch["history_dino"],
            batch["history_siglip"],
            batch["future_dino"],
            batch["future_siglip"],
            batch["actions"],
            batch["text_emb"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 1.0)))
        optimizer.step()
        total += float(loss.item()) * batch["actions"].shape[0]
        count += batch["actions"].shape[0]
        if step % int(cfg["train"].get("log_every", 50)) == 0:
            tqdm.write(f"step={step} loss={loss.item():.6f}")
    return total / max(count, 1)


@torch.no_grad()
def evaluate(model: LatentDiffusionWorldModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in tqdm(loader, desc="validate world model"):
        batch = move_batch_to_device(batch, device)
        loss, _ = model.loss(
            batch["history_dino"],
            batch["history_siglip"],
            batch["future_dino"],
            batch["future_siglip"],
            batch["actions"],
            batch["text_emb"],
        )
        total += float(loss.item()) * batch["actions"].shape[0]
        count += batch["actions"].shape[0]
    return total / max(count, 1)


if __name__ == "__main__":
    main()
