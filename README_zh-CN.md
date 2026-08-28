[English](README.md) | [**简体中文**](README_zh-CN.md)

# SMP / CMoE / AME：Unitree G1 运动控制复现与迁移

本仓库将三个面向人形机器人运动控制的项目统一迁移到
[mjlab](https://github.com/mujocolab/mjlab) 与 **Unitree G1** 上，并共用同一套安装、训练和播放入口。
本工作为课程项目复现与工程迁移，不是三个原项目的官方实现。

## 项目来源

本仓库的三个部分分别来源于以下开源项目：

| 项目 | 上游代码 | 本仓库中的工作 | 当前状态 |
|---|---|---|---|
| **SMP**（Score-Matching Motion Priors） | [SUZ-tsinghua/smp](https://github.com/SUZ-tsinghua/smp) | 直接使用该项目在 Unitree G1 上实现的运动特征、扩散先验与下游任务 | **已完成** |
| **CMoE**（Contrastive Mixture of Experts） | [Fudan-MAGIC-Lab/CMoE](https://github.com/Fudan-MAGIC-Lab/CMoE) | 将五专家复杂地形运动策略迁移到 MuJoCo / mjlab | **已完成迁移** |
| **AME**（Attention-Based Map Encoding） | [SII-FUSC/AME_Locomotion](https://github.com/SII-FUSC/AME_Locomotion) | 迁移基于高程图与注意力编码的两阶段地形运动方法 | **未完成** |

> [!IMPORTANT]
> **AME 是尚未完成的任务。**

## 安装与环境

### 环境要求

- Linux；
- NVIDIA GPU 和可正常工作的驱动；
- `curl` 或其他可安装 `uv` 的方式。

训练在实际使用中需要 GPU。项目使用 `uv` 管理依赖，固定的 mjlab Git 提交与其他依赖均记录在 `uv.lock`。

### 安装 `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

### 安装项目

```bash
cd /path/to/smp
uv sync --frozen
```

项目通过 `.python-version` 指定 Python 3.13。本文命令均使用 `uv run`，无需手动激活虚拟环境。

### 验证环境

```bash
nvidia-smi

uv run python - <<'PY'
import torch
import mjlab
import smp

print("PyTorch:", torch.__version__)
print("mjlab:", mjlab.__file__)
print("smp:", smp.__file__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
  print("GPU:", torch.cuda.get_device_name(0))
PY
```

确认任务注册和训练入口可以加载：

```bash
uv run scripts/train.py Smp-Forward-G1 --help
uv run scripts/train.py CMoE-G1 --help
```

## 项目一：SMP

### 方法概述

SMP 来自论文 *Reusable Score-Matching Motion Priors for Physics-Based Character Control*
（Mu 等，2025）。本仓库的 SMP 部分直接使用
[SUZ-tsinghua/smp](https://github.com/SUZ-tsinghua/smp) 项目提供的 Unitree G1 运动特征、
扩散模型、生成式状态初始化、强化学习任务和奖励实现。

SMP 的流程分为三步：

1. 将动作数据转换为固定长度的运动特征窗口；
2. 在运动窗口上预训练一个小型 DDPM，并冻结其分数函数；
3. 在 PPO 训练中将分数函数的误差转化为 SDS 风格的运动先验奖励。

这样，每个下游任务不需要单独提供参考动作片段或训练对抗式判别器，也能借助同一个运动先验学习自然动作。

### 已实现任务

| Task ID | 演示 | 说明 |
|---|:---:|---|
| `Smp-Forward-G1` | <img src="https://raw.githubusercontent.com/SUZ-tsinghua/smp/assets/forward.gif" width="200"/> | 按指定的 `+x` 速度行走、慢跑或奔跑 |
| `Smp-Steering-G1` | <img src="https://raw.githubusercontent.com/SUZ-tsinghua/smp/assets/steering.gif" width="200"/> | 跟踪指定移动速度和面朝方向 |
| `Smp-Location-G1` | <img src="https://raw.githubusercontent.com/SUZ-tsinghua/smp/assets/location.gif" width="200"/> | 移动到世界坐标系中的 xy 目标位置 |
| `Smp-Getup-G1` | <img src="https://raw.githubusercontent.com/SUZ-tsinghua/smp/assets/getup.gif" width="200"/> | 从倒地姿态恢复站立 |

### 仓库提供的预训练先验

`datasets/pretrain_ckpt/` 中包含三个扩散先验，可以跳过预训练直接开始强化学习。
各任务的 `init_smp_state` 配置已指向对应文件。

| Checkpoint | 训练数据 | 使用任务 |
|---|---|---|
| `pretrained_loco.pt` | 行走 / 慢跑 / 奔跑 | `Smp-Forward-G1` |
| `pretrained_lafan_run.pt` | LAFAN 奔跑子集 | `Smp-Steering-G1`、`Smp-Location-G1` |
| `pretrained_getup_f2s2.pt` | 起身（倒地到站立） | `Smp-Getup-G1` |

### 数据处理

输入动作采用 [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
中 `g1` 数据相同的逐帧 CSV 格式。文件无表头、以逗号分隔、帧率为 30 FPS，每帧共 36 列：

| 列 | 字段 | 说明 |
|---|---|---|
| 0–2 | 根位置 `x y z` | 世界坐标系，单位为米 |
| 3–6 | 根姿态四元数 | 顺序为 `x y z w` |
| 7–35 | G1 的 29 个关节角 | 单位为弧度，关节顺序见 `scripts/csv_to_npz.py` 中的 `JOINT_NAMES` |

仓库不包含原始 CSV。可以使用 Hugging Face CLI 下载：

```bash
hf download lvhaidong/LAFAN1_Retargeting_Dataset --repo-type dataset \
  --include "g1/*.csv" --local-dir datasets/csv/_lafan_dl
mv datasets/csv/_lafan_dl/g1/*.csv datasets/csv/lafan/
```

`csv_to_npz.py` 不会递归查找文件，因此 CSV 必须直接位于 `--input-dir` 下。

将 CSV 转换为窗口化 NPZ：

```bash
uv run scripts/csv_to_npz.py \
  --input-dir datasets/csv/lafan \
  --output-dir datasets/npz/lafan
```

脚本会在 G1 仿真中回放动作，通过正向运动学计算末端位置，将 30 FPS 插值到 50 FPS，
再切分成 `(N, window_size, 59)` 的窗口。每帧的 59 维特征为：

```text
[root_pos(3), root_rot(6), joint_pos(29), ee_pos(15), root_lin_vel(3), root_ang_vel(3)]
```

所有空间量均锚定到窗口最后一帧的骨盆位置，并旋转到仅保留偏航角的局部坐标系。
常用参数包括 `--window-size`、`--stride`、`--input-fps`、`--output-fps`，以及用于并行处理的
`--shard-index / --num-shards`。

计算归一化统计：

```bash
uv run scripts/compute_norm_stats.py \
  --input-dir datasets/npz/lafan \
  --output datasets/norm_stats.npz
```

脚本计算各特征的 q01/q99 分位数，并将特征映射到 `[-1, 1]`。仓库已经提供基于完整 LAFAN G1
数据集生成的 `datasets/norm_stats.npz`。除非修改了特征布局，否则建议直接复用该文件。
较宽的数据分布能够减少 PPO 探索到分布外状态时的特征饱和，避免分数估计退化。

### 扩散先验预训练

以前进运动先验为例：

```bash
uv run scripts/csv_to_npz.py \
  --input-dir datasets/csv/forward \
  --output-dir datasets/npz/forward

uv run scripts/pretrain.py \
  --data-dir datasets/npz/forward/ \
  --num-layers 2 \
  --no-use-ema \
  --save-interval 5000 \
  --num-epochs 10000 \
  --train-split 1.0 \
  --d-model 128
```

### 奖励设计：`task × SMP`

本仓库将任务奖励与 SMP 奖励相乘：

```text
r = (Σᵢ wᵢ · taskᵢ(s)) × r_smp(s)

r_smp = exp(−wₛ/|K| · Σ_{i∈K} ‖ε̂_i − ε_i‖²)
```

这与原始 SMP 方法的加法组合不同：

```text
# 原方法
r = task_reward_weight · task + smp_reward_weight · r_smp

# 本仓库
r = task · r_smp
```

乘法形式要求任务表现和动作自然度同时较高：只优化其中一项时，总奖励仍会接近零；同时也省去了
`task_reward_weight : smp_reward_weight` 的权重平衡。四个任务的任务奖励分别为：

- **Forward**：跟踪 `+x` 方向 0.5–5 m/s 的目标速度；反向运动时奖励置零。
- **Steering**：组合速度跟踪与面朝方向对齐，目标速度为 0.5–2 m/s。
- **Location**：跟踪周期性重采样的世界坐标系 xy 目标。
- **Get-up**：组合头部向上速度与头部高度跟踪，从倒地状态开始。

### 生成式状态初始化

每次环境重置时，从冻结先验预采样的运动窗口池中选择一个窗口。最后一帧用于设置仿真状态，
完整窗口用于填充在线运动特征缓冲区，使 `r_smp` 从第 0 步起就有效。缓冲区采用相对环境原点的表示，
因此奖励不受并行环境在世界网格中位置的影响。

### 训练与播放

```bash
# 训练
uv run scripts/train.py Smp-Forward-G1 --env.scene.num-envs=4096

# 查看训练指标
uv run tensorboard --logdir logs

# 播放本地 checkpoint
uv run scripts/play.py Smp-Forward-G1 \
  --checkpoint-file logs/rsl_rl/smp_forward_g1/<run>/model_500.pt \
  --num-envs 4
```

## 项目二：CMoE

### 方法概述

CMoE 来自 *Contrastive Mixture of Experts for Motion Control and Terrain Adaptation of Humanoid Robots*。
它通过五个专家对不同运动模式进行建模，并使用原型对比目标促进专家分工，从而提升复杂地形上的运动控制和适应能力。

本仓库对 CMoE 进行了独立的 MuJoCo / mjlab 迁移，保留了以下核心设计：

- 12 自由度下肢控制；
- 10 帧本体感知历史；
- 77 点地形高度扫描；
- actor / critic 非对称观测；
- 五专家策略、状态估计器与地形估计器；
- 原型对比学习目标与 CMoE PPO 损失；
- 地形课程、域随机化和九类复杂地形。

CMoE 从头端到端训练，不读取 SMP 的预训练 checkpoint。

### 训练

```bash
uv run scripts/train.py CMoE-G1 --env.scene.num-envs=4096
```

### 播放复杂地形路线

将 `src/smp/rl/tasks/cmoe/__init__.py` 中的 `play_env_cfg` 切换为
`g1_cmoe_course_env_cfg(difficulty=0.5)`，即可沿 x 轴依次播放九类 CMoE 地形：

```bash
uv run scripts/play.py CMoE-G1 \
  --checkpoint-file logs/rsl_rl/g1_cmoe/<run>/model_<iteration>.pt \
  --num-envs 4
```

`--num-envs` 同时决定独立路线的行数，统一难度由任务注册中的 `difficulty` 参数设置。

## 项目三：AME（未完成）

### 方法概述

AME 来源于 `SII-FUSC/AME_Locomotion` 对 *Attention-Based Map Encoding for Learning Generalized
Legged Locomotion* 的 Unitree G1 复现。方法先使用 CNN 提取局部地形特征，再让由本体感知状态构成的
query 通过多头注意力选择当前运动最相关的高程图区域，最后将地形编码与本体特征送入 actor / critic。

本仓库计划将原 Isaac Lab 实现迁移到 MuJoCo / mjlab。目前已经迁入：

- G1 关节顺序、默认姿态、PD 参数与力矩限制；
- 33 × 21 × 3 高程图、噪声和地形扫描观测；
- CNN、MHA、本体感知编码器和可选全局上下文编码器；
- actor / critic 观测、奖励、终止条件、地形课程与域随机化；
- 第一阶段和第二阶段地形配置；
- AME PPO 更新逻辑及原项目 `model_state_dict` checkpoint 格式；
- attention weights 的保存与可视化工具。

但 AME 的端到端训练、仿真验证、性能对齐和最终结果复现仍未完成。因此该部分当前应视为迁移中的开发代码，
不建议将其用于结果对比或下游研究基线。更详细的迁移边界见
[`src/smp/rl/tasks/ame/MIGRATION.md`](src/smp/rl/tasks/ame/MIGRATION.md)。

### 已注册的实验任务

| Task ID | 计划用途 | 状态 |
|---|---|---|
| `AME-G1` | 第一阶段地形运动训练 | 未验证 |
| `AME-G1-Global` | 带全局上下文编码器的第一阶段模型 | 未验证 |
| `AME-G1-Finetune` | 第二阶段地形微调 | 未验证 |

### 开发用命令

以下命令仅保留给后续开发和验证。

```bash
# 第一阶段
uv run scripts/train.py AME-G1 --env.scene.num-envs=4096

# 第二阶段：从第一阶段最新运行继续
uv run scripts/train.py AME-G1-Finetune \
  --env.scene.num-envs=4096 \
  --agent.resume \
  --agent.load-run='.*_ame$' \
  --agent.load-checkpoint='model_.*\.pt'
```

指定 GPU 时，`--gpu-ids` 需要以列表形式传入：

```bash
uv run scripts/train.py AME-G1 --env.scene.num-envs=4096 --gpu-ids '[2]'
```

多卡可使用 `--gpu-ids '[0,1]'`，全部可见 GPU 可使用 `--gpu-ids all`。

播放迁移后的 checkpoint：

```bash
uv run scripts/play.py AME-G1 \
  --checkpoint-file logs/rsl_rl/g1_ame/<run>/model_<iteration>.pt \
  --num-envs 1
```

原 AME_Locomotion 的 `ame1.pt` 与 `ame2.pt` 保持原始 checkpoint 布局，计划分别通过
`AME-G1` 和 `AME-G1-Global` 加载。兼容代码已经迁入，但尚未完成行为一致性验证。

保存并绘制 MHA attention weights：

```bash
uv run scripts/play.py AME-G1 \
  --checkpoint-file logs/rsl_rl/g1_ame/<run>/model_<iteration>.pt \
  --num-envs 1 \
  --num-steps 300 \
  --attention-file attention_weights.npy

uv run python -m smp.rl.ame.plot_attention \
  --attention-file attention_weights.npy \
  --output-dir attn_vis
```

## 任务速查

导入 `smp.rl.tasks` 时，以下任务会注册到 `mjlab.tasks.registry`：

| 项目 | Task ID | 状态 |
|---|---|---|
| SMP | `Smp-Forward-G1` | 已完成 |
| SMP | `Smp-Steering-G1` | 已完成 |
| SMP | `Smp-Location-G1` | 已完成 |
| SMP | `Smp-Getup-G1` | 已完成 |
| CMoE | `CMoE-G1` | 已完成迁移 |
| AME | `AME-G1` | 未完成 / 未验证 |
| AME | `AME-G1-Global` | 未完成 / 未验证 |
| AME | `AME-G1-Finetune` | 未完成 / 未验证 |

## 引用

如果本仓库对你的工作有帮助，请优先引用三个项目各自的原始论文，并注明本仓库使用了对应开源实现：

- **SMP** — Mu 等，*Reusable Score-Matching Motion Priors for Physics-Based Character Control*，2025。
  [论文](https://arxiv.org/abs/2512.03028) · [项目主页](https://yxmu.foo/smp-page/) ·
  [本仓库使用的 SMP 项目](https://github.com/SUZ-tsinghua/smp)
- **CMoE** — Ma 等，*Contrastive Mixture of Experts for Motion Control and Terrain Adaptation of Humanoid Robots*，ICRA 2026。
  [论文](https://arxiv.org/abs/2603.03067) · [代码](https://github.com/Fudan-MAGIC-Lab/CMoE)
- **AME** — He 等，*Attention-Based Map Encoding for Learning Generalized Legged Locomotion*，Science Robotics，2025。
  [论文](https://arxiv.org/abs/2506.09588) · [G1 复现代码](https://github.com/SII-FUSC/AME_Locomotion)

## 致谢

感谢 SMP、CMoE 和 AME 的论文作者公开研究成果，也感谢以下项目和数据资源为本仓库提供实现基础：

- [SUZ-tsinghua/smp](https://github.com/SUZ-tsinghua/smp)：本仓库直接使用的 Unitree G1 SMP 项目；
- [CMoE](https://github.com/Fudan-MAGIC-Lab/CMoE)：复杂地形五专家策略的原始实现；
- [AME_Locomotion](https://github.com/SII-FUSC/AME_Locomotion)：AME 在 Unitree G1 上的开源复现；
- [mjlab](https://github.com/mujocolab/mjlab)：本仓库统一使用的 MuJoCo 强化学习环境与训练入口；
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl)：PPO 与 on-policy 训练基础；
- [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)：
  SMP 运动先验所使用的 G1 重定向动作数据。

CMoE 迁移代码保留原项目的 BSD-3-Clause 声明，详见
[`LICENSES/CMoE.txt`](LICENSES/CMoE.txt) 与 [`NOTICE`](NOTICE)。使用本仓库时，请同时遵循各上游项目、
依赖与数据集的许可证和引用要求。
