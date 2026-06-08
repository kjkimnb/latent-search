from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LatentSpec:
    """Shape metadata for a DINO/SigLIP latent pair.

    The models internally flatten token and channel dimensions into a vector per
    time step.  `LatentSpec` records how to recover the original token layout.
    """

    dino_shape: tuple[int, ...]
    siglip_shape: tuple[int, ...]

    @property
    def dino_flat_dim(self) -> int:
        return _flat_dim(self.dino_shape)

    @property
    def siglip_flat_dim(self) -> int:
        return _flat_dim(self.siglip_shape)

    @property
    def flat_dim(self) -> int:
        return self.dino_flat_dim + self.siglip_flat_dim


def _flat_dim(shape: tuple[int, ...]) -> int:
    size = 1
    for dim in shape:
        size *= dim
    return size


def infer_latent_spec(dino: torch.Tensor, siglip: torch.Tensor) -> LatentSpec:
    """Infer per-timestep latent shapes from batched tensors.

    Accepted input shapes are `(B,T,D)` and `(B,T,N,D)`.
    """

    if dino.ndim not in (3, 4):
        raise ValueError(f"dino must have shape (B,T,D) or (B,T,N,D), got {tuple(dino.shape)}")
    if siglip.ndim not in (3, 4):
        raise ValueError(f"siglip must have shape (B,T,D) or (B,T,N,D), got {tuple(siglip.shape)}")
    return LatentSpec(dino_shape=tuple(dino.shape[2:]), siglip_shape=tuple(siglip.shape[2:]))


def flatten_latents(dino: torch.Tensor, siglip: torch.Tensor) -> torch.Tensor:
    """Concatenate DINO and SigLIP latents into `(B,T,F)` vectors."""

    if dino.shape[:2] != siglip.shape[:2]:
        raise ValueError("DINO and SigLIP latents must share batch/time dimensions.")
    dino_flat = dino.flatten(start_dim=2)
    siglip_flat = siglip.flatten(start_dim=2)
    return torch.cat([dino_flat, siglip_flat], dim=-1)


def split_flat_latents(flat: torch.Tensor, spec: LatentSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Split `(B,T,F)` vectors back into DINO and SigLIP tensors."""

    if flat.shape[-1] != spec.flat_dim:
        raise ValueError(f"Expected last dim {spec.flat_dim}, got {flat.shape[-1]}.")
    dino_flat, siglip_flat = flat.split([spec.dino_flat_dim, spec.siglip_flat_dim], dim=-1)
    dino = dino_flat.reshape(*flat.shape[:2], *spec.dino_shape)
    siglip = siglip_flat.reshape(*flat.shape[:2], *spec.siglip_shape)
    return dino, siglip


def last_state(dino: torch.Tensor, siglip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the final time step from a latent history while preserving tokens."""

    return dino[:, -1], siglip[:, -1]
