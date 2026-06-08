from __future__ import annotations

import torch
import torch.nn.functional as F


class DiffusionSchedule(torch.nn.Module):
    """DDPM-style schedule for latent diffusion."""

    def __init__(self, steps: int = 100, beta_start: float = 1e-4, beta_end: float = 2e-2):
        super().__init__()
        if steps < 2:
            raise ValueError("diffusion steps must be >= 2")
        betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.steps = steps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        posterior_variance = betas * (1.0 - prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-20))

    def add_noise(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = _extract(self.sqrt_alphas_cumprod, timesteps, x_start.ndim)
        sqrt_om_alpha = _extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.ndim)
        return sqrt_alpha * x_start + sqrt_om_alpha * noise, noise

    def denoise_step(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        beta_t = _extract(self.betas, timesteps, x_t.ndim)
        alpha_t = _extract(self.alphas, timesteps, x_t.ndim)
        alpha_bar_t = _extract(self.alphas_cumprod, timesteps, x_t.ndim)
        mean = (x_t - beta_t * predicted_noise / torch.sqrt(1.0 - alpha_bar_t)) / torch.sqrt(alpha_t)
        noise = torch.randn_like(x_t)
        variance = _extract(self.posterior_variance, timesteps, x_t.ndim)
        nonzero = (timesteps > 0).float().reshape(x_t.shape[0], *((1,) * (x_t.ndim - 1)))
        return mean + nonzero * torch.sqrt(variance) * noise

    def loss(self, predicted_noise: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(predicted_noise, target_noise)


def _extract(values: torch.Tensor, timesteps: torch.Tensor, ndim: int) -> torch.Tensor:
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (ndim - 1)))
