# latent-search

Language-conditioned progress learning and latent world-model planning for
robotic manipulation.  The code is organized to train on preprocessed LIBERO
demonstrations with LaDi-WM-style latent states:

```text
z_t = [z_t^DINO ; z_t^SigLIP]
```

The main research loop implemented here is:

1. encode observations into DINO/SigLIP latent states;
2. train an action-conditioned latent diffusion world model;
3. train a language-conditioned task progress comparator from successful
   trajectory orderings;
4. plan in latent space with MPC/CEM or continuous-action MCTS and execute only
   the first action.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For LIBERO simulation you also need the LIBERO/robosuite stack installed in the
same environment.  This repository only assumes preprocessed demonstration files
for training; it does not vendor LIBERO itself.

## Expected LIBERO feature format

The training scripts consume either `.npz` or `.hdf5/.h5` episodes.  Each episode
should contain:

| key | shape | description |
| --- | --- | --- |
| `dino` | `(T, N_d, D_d)` or `(T, D_d)` | geometric latent tokens |
| `siglip` | `(T, N_s, D_s)` or `(T, D_s)` | semantic latent tokens |
| `actions` | `(T, 7)` | robot actions |
| `text_emb` | `(C,)` or `(T, C)` | language embedding |
| `instruction` | scalar string, optional | task text for bookkeeping |

If you already have LaDi-WM/ATM-style cached features, write a small converter
that maps their keys to the names above, or pass alternate key names through the
dataset config.

## Training

World model:

```bash
python3 scripts/train_world_model_libero.py \
  --config configs/libero_progress_world_model.yaml \
  --data-root /path/to/preprocessed/libero_90 \
  --output-dir runs/world_model
```

Progress comparator:

```bash
python3 scripts/train_progress_comparator_libero.py \
  --config configs/libero_progress_world_model.yaml \
  --data-root /path/to/preprocessed/libero_90 \
  --output-dir runs/progress
```

Offline planning smoke test over a cached episode:

```bash
python3 scripts/plan_latent_libero.py \
  --config configs/libero_progress_world_model.yaml \
  --world-model runs/world_model/best.pt \
  --comparator runs/progress/best.pt \
  --episode /path/to/episode.npz
```

## Relation to LaDi-WM

LaDi-WM predicts future states in the latent space of visual foundation models
instead of pixels, with action conditioning and a diffusion objective.  This
repository implements the same world-model interface at a compact scale:

```python
future = world_model.sample(history, actions, text_emb)
loss = world_model.loss(history, future, actions, text_emb)
```

The additional contribution implemented here is a language-conditioned progress
function:

```python
score = C_omega(z, c)
```

It is trained by ranking later states in successful demonstrations above earlier
states, then used as the objective for latent-space model predictive planning.
