#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from latent_search.data import ProgressPairDataset, discover_episode_files, split_files
from latent_search.models import LanguageConditionedProgressComparator
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
    parser = argparse.ArgumentParser(description="Train a language-conditioned progress comparator on LIBERO features.")
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
    train_ds = ProgressPairDataset(args.data_root, files=train_files, **dataset_kwargs(cfg))
    val_ds = ProgressPairDataset(args.data_root, files=val_files, **dataset_kwargs(cfg)) if val_files else None

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

    model = LanguageConditionedProgressComparator.from_config(cfg).to(device)
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
    metrics = {}
    for epoch in range(start_epoch, epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, device, cfg)
        metrics = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}}
        if val_loader is not None:
            val_metrics = run_epoch(model, val_loader, None, device, cfg)
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, cfg, metrics)
        if epoch % int(cfg["train"].get("save_every", 5)) == 0:
            save_checkpoint(output_dir / f"epoch_{epoch}.pt", model, optimizer, epoch, cfg, metrics)
        write_jsonl(output_dir / "metrics.jsonl", metrics)
        print(metrics)

    save_checkpoint(output_dir / "final.pt", model, optimizer, max(epochs - 1, start_epoch), cfg, metrics)


def dataset_kwargs(cfg: dict) -> dict:
    data = cfg["data"]
    progress = cfg["progress"]
    return {
        "pair_gap_min": int(progress.get("pair_gap_min", 1)),
        "pairs_per_episode": int(progress.get("pairs_per_window", 4)) * 256,
        "dino_key": data.get("dino_key", "dino"),
        "siglip_key": data.get("siglip_key", "siglip"),
        "action_key": data.get("action_key", "actions"),
        "text_key": data.get("text_key", "text_emb"),
    }


def run_epoch(
    model: LanguageConditionedProgressComparator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    cfg: dict,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    total_gap = 0.0
    count = 0
    margin = float(cfg["progress"].get("margin", 0.2))
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for step, batch in enumerate(tqdm(loader, desc="train progress" if train else "validate progress")):
            batch = move_batch_to_device(batch, device)
            loss, logs = model.ranking_loss(
                batch["early_dino"],
                batch["early_siglip"],
                batch["late_dino"],
                batch["late_siglip"],
                batch["text_emb"],
                margin=margin,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 1.0)))
                optimizer.step()
            batch_size = batch["text_emb"].shape[0]
            total_loss += float(loss.item()) * batch_size
            total_acc += float(logs["rank_acc"].item()) * batch_size
            total_gap += float(logs["score_gap"].item()) * batch_size
            count += batch_size
            if train and step % int(cfg["train"].get("log_every", 50)) == 0:
                tqdm.write(f"step={step} loss={loss.item():.6f} acc={logs['rank_acc'].item():.3f}")
    denom = max(count, 1)
    return {"loss": total_loss / denom, "acc": total_acc / denom, "score_gap": total_gap / denom}


if __name__ == "__main__":
    main()
