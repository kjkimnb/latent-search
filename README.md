# latent-search

本项目实现一套用于机械臂操作的 **Language-conditioned Progress Learning +
Latent World Model Planning** 框架。核心思想是：

```text
RGB 图像 I_t -> DINO/SigLIP latent z_t
语言指令 tau -> 文本 embedding c
世界模型 M_theta 在 latent space 里预测未来
进度函数 C_omega(z, c) 评价任务完成进度
规划器在 latent space 搜索最优 7 维机械臂动作
```

它不是直接输出动作的 VLA policy，而是：

```text
(I_t, tau) -> z_t, c
M_theta(z_{t-H:t}, a_{t:t+K-1}, c) -> predicted future latents
C_omega(predicted latent, c) -> progress score
planner argmax progress -> action
```

默认训练和测试数据都放在 `data/`，训练输出放在 `runs/`。这些目录已经写入
`.gitignore`，不会被提交到 Git。

---

## 0. 下载本 GitHub 项目到本地电脑

```bash
git clone https://github.com/kjkimnb/latent-search.git
cd latent-search
```

如果你使用 SSH：

```bash
git clone git@github.com:kjkimnb/latent-search.git
cd latent-search
```

建议的本地目录结构：

```text
latent-search/
  configs/
  scripts/
  latent_search/
  data/                 # 数据集、本地缓存；已 gitignore
    libero_raw/          # 原始 LIBERO hdf5
    libero_features/     # 提取后的 dino/siglip latent
  runs/                 # 训练、评测输出；已 gitignore
  external/             # 外部仓库，例如 LIBERO；已 gitignore
```

---

## 1. 数据集下载、预处理和验证

本项目训练脚本不直接读取原始图像 HDF5，而是读取预处理后的 episode feature：

| key | shape | 含义 |
| --- | --- | --- |
| `dino` | `(T, N_d, D_d)` 或 `(T, D_d)` | DINO 几何 latent |
| `siglip` | `(T, N_s, D_s)` 或 `(T, D_s)` | SigLIP 语义 latent |
| `actions` | `(T, 7)` | LIBERO 7 维动作 |
| `text_emb` | `(C,)` 或 `(T, C)` | 语言指令 embedding |
| `instruction` | string，可选 | 原始语言指令 |

### 1.1 安装并下载 LIBERO 原始数据

先把官方 LIBERO 仓库放到 `external/`：

```bash
mkdir -p external data/libero_raw data/libero_features
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git external/LIBERO
```

下载 LIBERO 数据。参考官方 LIBERO 和 LaDi-WM 的做法，推荐先下载
`libero_100`，其中包含用于预训练的 LIBERO-90 和用于测试/下游的 LIBERO-10。

```bash
python external/LIBERO/benchmark_scripts/download_libero_datasets.py \
  --download-dir data/libero_raw \
  --datasets libero_100 \
  --use-huggingface
```

如果你的 LIBERO 版本没有 `--use-huggingface` 参数，去掉该参数：

```bash
python external/LIBERO/benchmark_scripts/download_libero_datasets.py \
  --download-dir data/libero_raw \
  --datasets libero_100
```

也可以下载全部数据：

```bash
python external/LIBERO/benchmark_scripts/download_libero_datasets.py \
  --download-dir data/libero_raw \
  --datasets all \
  --use-huggingface
```

### 1.2 检查原始数据是否下载成功

```bash
python - <<'PY'
from pathlib import Path
root = Path("data/libero_raw")
files = sorted(list(root.rglob("*.hdf5")) + list(root.rglob("*.h5")))
print("hdf5 files:", len(files))
print("first files:")
for path in files[:5]:
    print(" ", path)
assert len(files) > 0, "没有找到 LIBERO hdf5 文件，请检查下载目录"
PY
```

### 1.3 从原始 LIBERO HDF5 提取 DINO/SigLIP latent

下面命令会把原始 LIBERO episode 转成训练需要的 `.npz` feature 文件。

训练 world model 通常用 LIBERO-90：

```bash
python scripts/extract_libero_features.py \
  --raw-root data/libero_raw/libero_90 \
  --output-root data/libero_features \
  --suite libero_90 \
  --camera-key agentview_rgb \
  --batch-size 32 \
  --device cuda
```

如果你的下载目录里没有 `libero_90` 子目录，先查看实际目录：

```bash
python - <<'PY'
from pathlib import Path
for p in Path("data/libero_raw").iterdir():
    print(p)
PY
```

然后把 `--raw-root` 改成实际路径。

提取 LIBERO-10 测试/调试 feature：

```bash
python scripts/extract_libero_features.py \
  --raw-root data/libero_raw/libero_10 \
  --output-root data/libero_features \
  --suite libero_10 \
  --camera-key agentview_rgb \
  --batch-size 32 \
  --device cuda
```

> 注意：LaDi-WM 使用 DINO/SigLIP latent，而不是像素预测。不同视觉编码器和
> image processor 可能产生不同 token 数。提取后一定要运行下一步验证，并把
> `suggested_config` 中的 token 数写入配置。

### 1.4 验证预处理后的 feature 数据

```bash
python scripts/verify_libero_features.py \
  --data-root data/libero_features/libero_90 \
  --history 4 \
  --horizon 6
```

成功时会看到类似输出：

```text
found_files: 1000
checked_files: 100
train_windows_in_checked_files: ...
dino_per_state_shapes: [(256, 768)]
siglip_per_state_shapes: [(196, 768)]
action_shapes: [(7,)]
text_emb_shapes: [(768,)]
suggested_config:
  model.dino_dim: 768
  model.siglip_dim: 768
  model.dino_tokens: 256
  model.siglip_tokens: 196
status: OK
```

然后打开 `configs/libero_progress_world_model.yaml`，确认这些字段和验证结果一致：

```yaml
model:
  dino_dim: 768
  siglip_dim: 768
  dino_tokens: 256      # 按 verify 输出修改
  siglip_tokens: 196    # 按 verify 输出修改
```

如果你使用的是 LaDi-WM/ATM 已经缓存好的 9x9 latent，通常是：

```yaml
model:
  dino_tokens: 81
  siglip_tokens: 81
```

---

## 2. Python 环境和依赖安装

### 2.1 创建 conda 环境

推荐 Python 3.10：

```bash
conda create -n latent-search python=3.10 -y
conda activate latent-search
```

如果不用 conda，也可以：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2.2 安装 PyTorch

CUDA 12.1 示例：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

CUDA 11.8 示例：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

只做 CPU 调试：

```bash
pip install torch torchvision torchaudio
```

### 2.3 安装本项目依赖

```bash
pip install -e ".[dev]"
```

验证基础依赖：

```bash
python - <<'PY'
import torch, h5py, transformers, yaml
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("ok")
PY
```

运行单元测试：

```bash
pytest
```

### 2.4 安装 LIBERO 环境

在线 LIBERO 测试需要官方 LIBERO、robosuite、mujoco 等依赖：

```bash
pip install -r external/LIBERO/requirements.txt
pip install -e external/LIBERO
```

如果你遇到 MuJoCo/EGL 渲染问题，先设置：

```bash
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
```

---

## 3. 各阶段训练命令

所有训练输出默认建议放在 `runs/`，该目录已 gitignore。

### 3.1 训练 latent diffusion world model

```bash
python scripts/train_world_model_libero.py \
  --config configs/libero_progress_world_model.yaml \
  --data-root data/libero_features/libero_90 \
  --output-dir runs/world_model/libero_90
```

输出文件：

```text
runs/world_model/libero_90/
  config.yaml
  metrics.jsonl
  best.pt
  final.pt
  epoch_*.pt
```

查看训练曲线文本：

```bash
tail -n 5 runs/world_model/libero_90/metrics.jsonl
```

### 3.2 训练 Progress Comparator

```bash
python scripts/train_progress_comparator_libero.py \
  --config configs/libero_progress_world_model.yaml \
  --data-root data/libero_features/libero_90 \
  --output-dir runs/progress/libero_90
```

输出文件：

```text
runs/progress/libero_90/
  config.yaml
  metrics.jsonl
  best.pt
  final.pt
  epoch_*.pt
```

查看 ranking accuracy：

```bash
tail -n 5 runs/progress/libero_90/metrics.jsonl
```

### 3.3 离线 latent planning smoke test

先随便找一个预处理后的 episode：

```bash
EP=$(find data/libero_features/libero_90 -name "*.npz" | head -n 1)
echo $EP
```

运行离线规划：

```bash
python scripts/plan_latent_libero.py \
  --config configs/libero_progress_world_model.yaml \
  --world-model runs/world_model/libero_90/best.pt \
  --comparator runs/progress/libero_90/best.pt \
  --episode "$EP" \
  --start 0 \
  --planner cem
```

输出会包含：

```text
{
  "action": [ ... 7 numbers ... ],
  "value": ...,
  "info": ...
}
```

如果这一步报 shape 错误，通常是 `configs/libero_progress_world_model.yaml`
里的 `dino_tokens/siglip_tokens/dino_dim/siglip_dim` 和数据验证输出不一致。

---

## 4. 在 LIBERO 中在线测试

在线测试会：

1. 创建 LIBERO 环境；
2. 读取当前 RGB 观测；
3. 实时提取 DINO/SigLIP latent；
4. 用 world model + progress comparator + CEM/MCTS 规划动作；
5. 执行动作并循环；
6. 保存 `results.json`，可选保存视频。

### 4.1 单个任务测试

```bash
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0

python scripts/eval_libero_planner.py \
  --config configs/libero_progress_world_model.yaml \
  --world-model runs/world_model/libero_90/best.pt \
  --comparator runs/progress/libero_90/best.pt \
  --suite libero_10 \
  --task-id 0 \
  --episodes 10 \
  --max-steps 300 \
  --camera-key agentview_image \
  --output-dir runs/eval_libero \
  --save-video
```

结果保存位置：

```text
runs/eval_libero/libero_10/task_000/
  results.json
  episode_000.mp4
  episode_001.mp4
  ...
```

查看成功率：

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path("runs/eval_libero/libero_10/task_000/results.json")
data = json.loads(path.read_text())
print("success_rate:", data["success_rate"])
print("mean_return:", data["mean_return"])
PY
```

### 4.2 跑 LIBERO-10 全部任务

```bash
for TASK_ID in $(seq 0 9); do
  python scripts/eval_libero_planner.py \
    --config configs/libero_progress_world_model.yaml \
    --world-model runs/world_model/libero_90/best.pt \
    --comparator runs/progress/libero_90/best.pt \
    --suite libero_10 \
    --task-id ${TASK_ID} \
    --episodes 10 \
    --max-steps 300 \
    --camera-key agentview_image \
    --output-dir runs/eval_libero \
    --save-video
done
```

汇总所有任务：

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("runs/eval_libero/libero_10")
rates = []
for path in sorted(root.glob("task_*/results.json")):
    data = json.loads(path.read_text())
    rates.append(data["success_rate"])
    print(path.parent.name, data["success_rate"])
print("mean_success_rate:", sum(rates) / max(len(rates), 1))
PY
```

### 4.3 如何看结果

最重要的文件是：

```text
runs/eval_libero/<suite>/task_<id>/results.json
```

其中：

| 字段 | 含义 |
| --- | --- |
| `success_rate` | 多个 episode 的平均成功率 |
| `mean_return` | 平均 sparse reward |
| `results[i].success` | 第 i 个 episode 是否成功 |
| `results[i].steps` | 第 i 个 episode 执行步数 |
| `results[i].task_language` | LIBERO 语言任务 |

如果加了 `--save-video`，同目录下的 `episode_*.mp4` 可以直接查看机械臂轨迹。

---

## 5. 常见问题

### Q1: `Expected token shape ...` 报错怎么办？

运行：

```bash
python scripts/verify_libero_features.py --data-root data/libero_features/libero_90
```

把输出的：

```text
model.dino_dim
model.siglip_dim
model.dino_tokens
model.siglip_tokens
```

同步到 `configs/libero_progress_world_model.yaml`。

### Q2: 当前默认配置为什么是 `dino_tokens: 81`、`siglip_tokens: 81`？

这是为了贴近 LaDi-WM 的 9x9 latent token 设定。如果你用本项目的
`extract_libero_features.py` 直接从 HuggingFace 视觉模型提取特征，token 数可能是
256、196 或其他值，请以验证脚本输出为准。

### Q3: 这是 VLA 论文代码吗？

不是直接 VLA policy。它更准确地说是：

```text
Language-conditioned Progress Learning
+ Latent Diffusion World Model
+ Model Predictive Planning
```

### Q4: 数据、模型、视频会不会被提交到 Git？

不会。以下目录已在 `.gitignore`：

```text
data/
runs/
outputs/
wandb/
checkpoints/
external/
```

---

## 6. 文件速查

```text
configs/libero_progress_world_model.yaml       # 默认训练/规划配置
scripts/extract_libero_features.py             # 原始 LIBERO -> latent feature
scripts/verify_libero_features.py              # 验证 feature 数据
scripts/train_world_model_libero.py            # 训练世界模型
scripts/train_progress_comparator_libero.py    # 训练进度比较器
scripts/plan_latent_libero.py                  # 离线 latent planning
scripts/eval_libero_planner.py                 # 在线 LIBERO 测试
latent_search/models/diffusion_world_model.py  # LaDi-WM 风格世界模型
latent_search/models/progress.py               # C_omega(z,c)
latent_search/planning/cem.py                  # CEM/MPC planner
latent_search/planning/mcts.py                 # progressive widening MCTS
PAPER_README.md                                # 给写论文/后续 AI 的架构与数学说明
```
