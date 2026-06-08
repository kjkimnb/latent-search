from __future__ import annotations

import torch
from torch import nn


class VisualFoundationEncoder(nn.Module):
    """Lazy HuggingFace wrapper for DINOv2 and SigLIP patch latents.

    Training is expected to use cached features.  This module is provided for
    feature extraction jobs and online evaluation.  It intentionally avoids
    modifying transformer source code: patch tokens are read from
    `last_hidden_state`/`vision_model` outputs directly.
    """

    def __init__(
        self,
        dino_name: str = "facebook/dinov2-base",
        siglip_name: str = "google/siglip-base-patch16-224",
        freeze: bool = True,
    ):
        super().__init__()
        try:
            from transformers import AutoModel, SiglipModel
        except ImportError as exc:
            raise ImportError("Install transformers to use VisualFoundationEncoder.") from exc
        self.dino = AutoModel.from_pretrained(dino_name)
        self.siglip = SiglipModel.from_pretrained(siglip_name)
        if freeze:
            self.requires_grad_(False)
            self.eval()

    @torch.no_grad()
    def forward(
        self,
        dino_pixel_values: torch.Tensor,
        siglip_pixel_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if siglip_pixel_values is None:
            siglip_pixel_values = dino_pixel_values
        try:
            dino_out = self.dino(pixel_values=dino_pixel_values, interpolate_pos_encoding=True)
        except TypeError:
            dino_out = self.dino(pixel_values=dino_pixel_values)
        dino_tokens = dino_out.last_hidden_state
        if dino_tokens.shape[1] > 1:
            dino_tokens = dino_tokens[:, 1:]

        siglip_out = self.siglip.vision_model(pixel_values=siglip_pixel_values, output_hidden_states=True)
        siglip_tokens = siglip_out.last_hidden_state
        return dino_tokens, siglip_tokens


class TransformerTextEncoder(nn.Module):
    """Lazy text encoder that returns a pooled language embedding."""

    def __init__(self, model_name: str = "google/siglip-base-patch16-224", freeze: bool = True):
        super().__init__()
        try:
            from transformers import AutoTokenizer, SiglipModel
        except ImportError as exc:
            raise ImportError("Install transformers to use TransformerTextEncoder.") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = SiglipModel.from_pretrained(model_name)
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str], device: torch.device | None = None) -> torch.Tensor:
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        if device is not None:
            tokens = {key: value.to(device) for key, value in tokens.items()}
            self.model.to(device)
        out = self.model.text_model(**tokens)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        return out.last_hidden_state.mean(dim=1)
