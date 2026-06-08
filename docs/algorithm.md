# Progress-guided latent world-model planning

## State and language

For an RGB observation `I_t`, cached visual foundation model features define the
latent state:

```math
z_t = [z_t^D ; z_t^S]
```

where `z_t^D` is a DINO-style geometric feature and `z_t^S` is a SigLIP-style
semantic feature.  A language instruction `tau` is encoded as:

```math
c = TextEnc(tau)
```

## Language-conditioned progress function

The core learned objective is:

```math
C_\omega(z, c): Z \times C \rightarrow R
```

Given a successful demonstration trajectory
`z_0, z_1, ..., z_T`, ranking pairs are sampled with `i < j`.  The comparator is
trained with:

```math
L_rank = max(0, C_\omega(z_i,c) - C_\omega(z_j,c) + delta)
```

The implementation intentionally treats `C_omega` as an ordinal task progress
score, not a calibrated success probability or Euclidean goal distance.

## LaDi-WM-style latent diffusion world model

Given recent latent history, actions, and language:

```math
z_{t-H:t}, a_{t:t+K-1}, c
```

the world model learns a conditional latent distribution:

```math
z_{t+1:t+K} ~ p_\theta(. | z_{t-H:t}, a_{t:t+K-1}, c)
```

The training objective is a DDPM noise-prediction loss over future latent
trajectories:

```math
L_wm = || epsilon - epsilon_\theta(z^k_{t+1:t+K}, k, z_{t-H:t}, a_{t:t+K-1}, c) ||_2^2
```

where `k` is the diffusion timestep.

## Latent-space MPC objective

For a sampled action sequence `A = (a_t, ..., a_{t+K-1})`, the world model
predicts a future latent rollout:

```math
\hat z_{t+1:t+K} = M_\theta(z_{t-H:t}, A, c)
```

The planner scores the rollout by progress:

```math
J(A) = w_traj \sum_{k=1}^{K} gamma^{k-1} C_\omega(\hat z_{t+k}, c)
       + w_terminal C_\omega(\hat z_{t+K}, c)
```

The chosen sequence is:

```math
A^* = argmax_A J(A)
```

Only the first action `a_t^* = A^*[0]` is executed, then the real environment is
observed again and planning repeats.

## Continuous action search

Because LIBERO robot actions are 7D continuous vectors, the default planner uses
the cross-entropy method (CEM).  Progressive widening MCTS is included as a
baseline, but raw MCTS without widening is not appropriate for this action space.
