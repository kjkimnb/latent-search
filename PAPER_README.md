# Paper README: Language-conditioned Progress-guided Latent World Model Planning

本文档给后续写论文、写 related work、补实验或让 AI 继续扩展项目时使用。目标是让写作者快速理解：

1. 这篇工作的核心贡献是什么；
2. 数学对象如何定义；
3. 每个数学模块对应代码在哪里；
4. 论文实验应该如何组织。

---

## 1. 论文定位

推荐论文标题方向：

```text
Language-conditioned Progress Learning for Latent World-Model Planning in Robotic Manipulation
```

或：

```text
Progress-guided Latent World Model Planning for Language-conditioned Manipulation
```

这不是一篇标准 VLA 论文，因为系统不是直接学习：

```math
\pi(a_t | I_t, \tau)
```

而是学习：

```math
C_\omega(z_t, c)
```

作为任务进度函数，并用世界模型进行规划：

```math
a_t^* = Planner(M_\theta, C_\omega, z_{t-H:t}, c)
```

更准确的定位：

```text
Language-conditioned Progress Learning
+ Latent Diffusion World Model
+ Model Predictive Planning
```

投稿包装建议：

- CoRL / ICRA / IROS：机器人操作、模型式规划、语言条件泛化；
- RSS：需要强调 `C_omega(z,c)` 作为无需人工奖励的通用规划目标，并给出强消融；
- NeurIPS/ICML：需要更强理论或跨 benchmark 泛化，目前仅凭机器人系统组合可能不足。

---

## 2. 核心观点

传统 VLA：

```math
(I_t, \tau) \rightarrow a_t
```

本项目：

```math
I_t \rightarrow z_t
```

```math
\tau \rightarrow c
```

```math
C_\omega(z_t, c) \rightarrow \text{task progress score}
```

```math
M_\theta(z_{t-H:t}, a_{t:t+K-1}, c) \rightarrow \hat z_{t+1:t+K}
```

```math
a_t^* = \arg\max_{a_{t:t+K-1}} J(a_{t:t+K-1})
```

其中 `C_omega` 是论文最值得强调的贡献。世界模型和 CEM/MCTS 是验证平台。

一句话摘要可以写成：

> We learn a language-conditioned task progress function solely from successful demonstrations and use it as a reward-free planning objective for latent world-model-based robotic manipulation.

---

## 3. 数学形式

### 3.1 观测、编码和语言

RGB 观测：

```math
I_t
```

视觉 latent：

```math
z_t = [z_t^D ; z_t^S]
```

其中：

- `z_t^D`：DINO geometric latent；
- `z_t^S`：SigLIP semantic latent。

语言指令：

```math
\tau
```

语言 embedding：

```math
c = TextEnc(\tau)
```

代码位置：

| 数学对象 | 代码 |
| --- | --- |
| `z_t^D, z_t^S` | `latent_search/models/encoders.py` |
| latent flatten/split | `latent_search/models/latent.py` |
| `c = TextEnc(tau)` | `TransformerTextEncoder` in `latent_search/models/encoders.py` |
| LIBERO feature 提取 | `scripts/extract_libero_features.py` |

---

### 3.2 Progress Comparator

定义语言条件任务进度函数：

```math
C_\omega(z, c): \mathcal{Z} \times \mathcal{C} \rightarrow \mathbb{R}
```

输出值不是严格概率，也不是欧氏距离，而是 ordinal progress score。

成功轨迹：

```math
\zeta = (z_0, z_1, ..., z_T)
```

构造排序对：

```math
i < j
```

训练假设：

```math
C_\omega(z_j, c) > C_\omega(z_i, c)
```

ranking loss：

```math
\mathcal{L}_{rank}
= \max(0, C_\omega(z_i,c) - C_\omega(z_j,c) + \delta)
```

代码位置：

| 功能 | 代码 |
| --- | --- |
| Comparator 网络 | `latent_search/models/progress.py` |
| `ranking_loss` | `LanguageConditionedProgressComparator.ranking_loss` |
| 采样 `(z_i,z_j,c)` | `ProgressPairDataset` in `latent_search/data/libero_latent.py` |
| 训练脚本 | `scripts/train_progress_comparator_libero.py` |

论文中需要注意：

- 时间顺序不是严格任务进度单调，这是一项弱监督假设；
- 可在实验中通过 gap sampling、hard pair mining、subgoal-aware sampling 改进；
- 需要和 binary success classifier、time-to-goal regression、goal-conditioned value 做消融。

---

### 3.3 Latent Diffusion World Model

给定历史 latent、未来动作和语言：

```math
z_{t-H:t},\quad a_{t:t+K-1},\quad c
```

学习未来 latent 分布：

```math
z_{t+1:t+K}
\sim
p_\theta(\cdot \mid z_{t-H:t}, a_{t:t+K-1}, c)
```

DDPM 训练形式：

```math
z^k = \sqrt{\bar\alpha_k} z + \sqrt{1-\bar\alpha_k}\epsilon
```

```math
\mathcal{L}_{wm}
=
\left\|
\epsilon -
\epsilon_\theta(z^k_{t+1:t+K}, k, z_{t-H:t}, a_{t:t+K-1}, c)
\right\|_2^2
```

代码位置：

| 功能 | 代码 |
| --- | --- |
| diffusion schedule | `latent_search/models/diffusion.py` |
| token-level world model | `latent_search/models/diffusion_world_model.py` |
| DiT/AdaLN block | `latent_search/models/transformer.py` |
| world model 训练 | `scripts/train_world_model_libero.py` |

和 LaDi-WM 的关系：

- 相同点：都在视觉 foundation model latent space 预测未来，而不是预测 RGB 像素；
- 相同点：都使用 action-conditioned diffusion objective；
- 本项目新增：预测结果不直接给 policy，而是交给 `C_omega(z,c)` 做规划价值；
- 当前实现是 compact research implementation，不是完整复刻 LaDi-WM 大型工程。

---

### 3.4 Latent-space Planning

采样动作序列：

```math
A = (a_t, a_{t+1}, ..., a_{t+K-1})
```

世界模型 rollout：

```math
\hat z_{t+1:t+K}
=
M_\theta(z_{t-H:t}, A, c)
```

轨迹价值：

```math
J(A)
=
w_{traj}
\sum_{k=1}^{K}
\gamma^{k-1}
C_\omega(\hat z_{t+k}, c)
+
w_{terminal}
C_\omega(\hat z_{t+K}, c)
```

最优动作序列：

```math
A^* = \arg\max_A J(A)
```

执行第一步：

```math
a_t^* = A^*[0]
```

然后环境返回新观测，重复规划。

代码位置：

| 功能 | 代码 |
| --- | --- |
| CEM planner | `latent_search/planning/cem.py` |
| progressive widening MCTS | `latent_search/planning/mcts.py` |
| 离线 planning | `scripts/plan_latent_libero.py` |
| 在线 LIBERO eval | `scripts/eval_libero_planner.py` |

为什么默认 CEM：

- LIBERO 动作空间是 7 维连续动作；
- 原生 MCTS 对连续动作扩展会爆炸；
- CEM/PETS/PlaNet/Dreamer 系列更适合连续控制；
- MCTS 版本必须使用 progressive widening 才合理。

---

## 4. 代码架构总览

```text
latent_search/
  data/
    libero_latent.py
      LiberoLatentSequenceDataset
      ProgressPairDataset

  models/
    encoders.py
      VisualFoundationEncoder
      TransformerTextEncoder

    latent.py
      LatentSpec
      flatten_latents
      split_flat_latents

    diffusion.py
      DiffusionSchedule

    transformer.py
      TimestepEmbedder
      AdaLNTransformerBlock

    diffusion_world_model.py
      LatentDiffusionWorldModel

    progress.py
      LanguageConditionedProgressComparator

    goal_generator.py
      GoalStateGenerator

  planning/
    cem.py
      CEMPlanner

    mcts.py
      ProgressiveWideningMCTS

scripts/
  extract_libero_features.py
  verify_libero_features.py
  train_world_model_libero.py
  train_progress_comparator_libero.py
  plan_latent_libero.py
  eval_libero_planner.py
```

---

## 5. 论文实验建议

### 5.1 主实验

Benchmark：

- LIBERO-90：训练 world model 和 progress comparator；
- LIBERO-10 / LIBERO-LONG：测试长时程语言条件操作；
- 可选：LIBERO-Spatial、Object、Goal 做泛化。

指标：

- success rate；
- average return；
- average episode length；
- planning time per step；
- world model latent prediction MSE/noise loss；
- progress ranking accuracy。

### 5.2 Baselines

建议至少包括：

1. Behavior Cloning policy；
2. Open-loop action replay nearest neighbor；
3. World model + terminal-only progress；
4. World model + hand-designed sparse success reward；
5. World model + binary success classifier；
6. World model + random shooting；
7. World model + CEM；
8. World model + progressive widening MCTS。

如果资源允许：

- RT-1 / Octo / OpenVLA 风格 policy baseline；
- LaDi-WM policy baseline；
- Dreamer/PlaNet-style latent MPC baseline。

### 5.3 Ablation

强烈建议做以下消融：

| 消融 | 目的 |
| --- | --- |
| 去掉语言条件 `c` | 证明 `C(z,c)` 而不是普通 `V(z)` |
| 只用 DINO | 几何特征贡献 |
| 只用 SigLIP | 语义特征贡献 |
| DINO + SigLIP | 完整 latent |
| terminal-only `C(z_K,c)` | 和 trajectory progress objective 比较 |
| 不同 ranking gap | 验证时序监督强弱 |
| CEM vs MCTS | 连续动作规划器选择 |
| different horizon K | 长时程规划能力 |
| world model one-step vs multi-step | 误差累积影响 |

### 5.4 关键图表

建议论文包含：

1. 系统框图：language -> progress function；latent world model；planner；
2. 数学模块图：`M_theta` rollout + `C_omega` scoring；
3. LIBERO success rate 表；
4. 消融表；
5. progress score 随轨迹时间变化曲线；
6. world model predicted latent nearest-neighbor visualization；
7. 成功/失败视频帧序列。

---

## 6. 当前实现的限制

写论文时需要诚实说明或通过实验补强：

1. `i < j` 的时序排序不一定严格等价于任务进度单调；
2. world model 多步 rollout 会累积误差；
3. CEM 规划需要多次 world model sampling，在线速度可能慢；
4. HuggingFace DINO/SigLIP token 数和 LaDi-WM 官方预处理 token 数可能不同；
5. 当前 online LIBERO eval 是通用脚本，真实 benchmark 复现还需要固定 init states、camera、action normalization 等细节。

---

## 7. 可以继续扩展的方向

### 7.1 更强的 Progress Comparator

- hard negative pairs；
- temporal contrastive learning；
- success/failure mixed ranking；
- pairwise Bradley-Terry objective；
- calibration to success probability；
- uncertainty-aware progress score。

### 7.2 Goal Generator

当前已有可选模块：

```text
latent_search/models/goal_generator.py
```

可扩展为：

```math
q_\psi(z_{goal} | z_{t-H:t}, c)
```

并与 `C_\omega(z, z_goal, c)` 结合。

### 7.3 Hierarchical Planning

可以把语言任务分成子目标：

```math
\tau \rightarrow (\tau_1, ..., \tau_m)
```

每个子任务有自己的 progress score。

### 7.4 Faster Planning

- distill CEM planner into a policy；
- use value network warm start；
- use latent rollout caching；
- use deterministic world model for first-stage search, diffusion model for reranking。

---

## 8. 推荐摘要草稿

```text
Language-conditioned robotic manipulation often relies on direct action prediction
or hand-designed rewards. We propose a progress-guided latent world-model planning
framework that learns a language-conditioned task progress function from successful
demonstrations only. The learned progress function assigns an ordinal completion
score to latent visual states conditioned on language, and serves as a reward-free
objective for model predictive planning. A LaDi-WM-style latent diffusion world
model predicts future DINO/SigLIP latent states under candidate action sequences,
while a CEM planner selects actions that maximize predicted task progress. Experiments
on LIBERO evaluate whether language-conditioned progress learning can drive
long-horizon manipulation without manual reward design or direct policy imitation.
```

---

## 9. 推荐 citation 方向

Related work 应该覆盖：

- LaDi-WM；
- PlaNet / Dreamer；
- MuZero；
- PETS / CEM model-based control；
- VIP / visual progress or value learning；
- hindsight relabeling / goal-conditioned value learning；
- RT-1 / RT-2 / Octo / OpenVLA；
- LIBERO benchmark。
