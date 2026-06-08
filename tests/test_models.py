import torch

from latent_search.models.diffusion_world_model import LatentDiffusionWorldModel
from latent_search.models.latent import LatentSpec, flatten_latents, split_flat_latents
from latent_search.models.progress import LanguageConditionedProgressComparator


def test_flatten_and_split_latents():
    dino = torch.randn(2, 3, 4, 5)
    siglip = torch.randn(2, 3, 6)
    spec = LatentSpec(dino_shape=(4, 5), siglip_shape=(6,))
    flat = flatten_latents(dino, siglip)
    out_dino, out_siglip = split_flat_latents(flat, spec)
    assert torch.allclose(out_dino, dino)
    assert torch.allclose(out_siglip, siglip)


def test_progress_ranking_loss_shapes():
    model = LanguageConditionedProgressComparator(dino_dim=4, siglip_dim=5, text_dim=6, hidden_dim=16, depth=2)
    early_dino = torch.randn(8, 4)
    early_siglip = torch.randn(8, 5)
    late_dino = torch.randn(8, 4)
    late_siglip = torch.randn(8, 5)
    text = torch.randn(8, 6)
    loss, logs = model.ranking_loss(early_dino, early_siglip, late_dino, late_siglip, text)
    assert loss.ndim == 0
    assert set(logs) == {"rank_loss", "rank_acc", "score_gap"}


def test_world_model_loss_and_sample_shapes():
    model = LatentDiffusionWorldModel(
        dino_dim=4,
        siglip_dim=5,
        text_dim=6,
        hidden_dim=32,
        depth=2,
        num_heads=4,
        diffusion_steps=8,
        history=3,
        horizon=2,
    )
    history_dino = torch.randn(2, 3, 4)
    history_siglip = torch.randn(2, 3, 5)
    future_dino = torch.randn(2, 2, 4)
    future_siglip = torch.randn(2, 2, 5)
    actions = torch.randn(2, 2, 7)
    text = torch.randn(2, 6)
    loss, logs = model.loss(history_dino, history_siglip, future_dino, future_siglip, actions, text)
    assert loss.ndim == 0
    assert "wm_loss" in logs
    pred_dino, pred_siglip = model.sample(history_dino, history_siglip, actions, text, steps=2)
    assert pred_dino.shape == future_dino.shape
    assert pred_siglip.shape == future_siglip.shape
