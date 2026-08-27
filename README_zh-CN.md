[English](README.md) | [**简体中文**](README_zh-CN.md)

# SMP——分数匹配运动先验（复现）

本项目在 **Unitree G1** 人形机器人上复现 **SMP：用于物理角色控制的可复用分数匹配运动先验**（Mu 等，2025）。原始 MimicKit 实现没有提供 G1 配置，因此本仓库完成了从运动特征、先验模型到任务与奖励的完整 G1 移植。

项目先在运动窗口上预训练一个小型扩散模型（DDPM），再将其**冻结的分数函数**作为 SDS 风格的引导奖励用于 PPO。这样，策略无需针对每个任务提供动作片段或对抗式判别器，也能学习自然的下游任务动作。

这是一个课程项目复现，基于 [**mjlab**](https://github.com/mujocolab/mjlab) 实现，并复用其 `ManagerBasedRlEnv` 以及 `mjlab.scripts.train` / `play` 入口。原始方法和参考实现如下：

- **论文：** Mu 等，SMP，2025——[arXiv:2512.03028](https://arxiv.org/abs/2512.03028) · [项目主页](https://yxmu.foo/smp-page/)
- **原始代码：** [`xbpeng/MimicKit`](https://github.com/xbpeng/MimicKit)（参见 `docs/README_SMP.md`）

> 本项目与原方法最主要的有意差异是奖励组合方式，详见下文的[奖励设计](#奖励设计task--smp)。

## 仓库提供的预训练先验

为了跳过预训练并直接运行强化学习，仓库在 `datasets/pretrain_ckpt/` 中提供了三个预训练扩散先验。各任务的环境配置已经通过 `init_smp_state` 事件指向对应文件，无需额外设置。

| Checkpoint | 训练数据 | 使用任务 |
|---|---|---|
| `pretrained_loco.pt` | 行走 / 慢跑 / 奔跑 | `Smp-Forward-G1` |
| `pretrained_lafan_run.pt` | LAFAN 奔跑子集 | `Smp-Steering-G1`、`Smp-Location-G1` |
| `pretrained_getup_f2s2.pt` | 起身（倒地→站立） | `Smp-Getup-G1` |

## 安装

项目统一使用 [`uv`](https://docs.astral.sh/uv/) 管理软件包，包括固定版本的 `mjlab` Git 提交在内，所有依赖均锁定在 `uv.lock` 中。

### 环境要求

- Linux、NVIDIA GPU 和可正常工作的 NVIDIA 驱动（`nvidia-smi` 应能成功运行）。实际训练需要 GPU。
- `curl`，或其他能够安装 `uv` 的方式。

### 安装 `uv`

在 Linux 上使用官方独立安装脚本：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

### 安装项目

在仓库根目录创建 `.venv`，并严格按照锁文件安装依赖：

```bash
cd /path/to/smp
uv sync --frozen
```

项目通过 `.python-version` 指定 Python 3.13；如果系统中没有合适的解释器，`uv` 会自动获取。首次同步需要下载 PyTorch 和 GPU 仿真组件，可能需要几分钟。

本文命令均使用 `uv run`，不需要手动激活虚拟环境。如需进入虚拟环境，可运行：

```bash
source .venv/bin/activate
```

### 验证环境

首先检查驱动能否识别 GPU：

```bash
nvidia-smi
```

然后验证项目导入以及 PyTorch 的 CUDA 支持：

```bash
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

在 NVIDIA 训练机器上，`CUDA available` 应为 `True`。最后确认 SMP 任务已经注册，且训练命令可以正常加载：

```bash
uv run scripts/train.py Smp-Forward-G1 --help
```

## 流程

1. **数据处理**：CSV → 窗口化 NPZ → 归一化统计。
2. **扩散模型预训练**：在运动窗口上训练 DDPM ε 预测器。使用仓库提供的 checkpoint 时可以完全跳过这一步。
3. **强化学习**：使用冻结的运动先验作为引导奖励训练 PPO。

---

## 数据处理

第一阶段将原始运动 CSV 转换为扩散先验使用的窗口特征张量，同时计算预训练和强化学习共享的归一化统计。两个脚本均由 `tyro` 驱动，请在项目根目录通过 `uv` 运行。

### 输入 CSV（LAFAN1 重定向格式）

输入必须采用与 **[LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)** 的 `g1` 数据相同的逐帧 CSV 格式，这也是 `mjlab` 的 `MotionLoader` 所读取的格式。每个文件无表头、以逗号分隔，每帧一行，帧率为 **30 FPS**，共 **36 列**：

| 列 | 字段 | 说明 |
|---|---|---|
| 0–2 | 根位置 `x y z` | 世界坐标系，单位为米 |
| 3–6 | 根姿态四元数 | 顺序为 **`x y z w`** |
| 7–35 | G1 的 29 个关节角 | 单位为弧度；关节顺序见 `scripts/csv_to_npz.py` 中的 `JOINT_NAMES` |

仓库**不提供这些 CSV 文件**，需要自行下载。完整 G1 数据可将数据集中的 [`g1/`](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/tree/main/g1) 目录下载至 `datasets/csv/lafan/`：

```bash
# 示例：使用 Hugging Face CLI，先执行 pip install -U huggingface_hub
hf download lvhaidong/LAFAN1_Retargeting_Dataset --repo-type dataset \
  --include "g1/*.csv" --local-dir datasets/csv/_lafan_dl
mv datasets/csv/_lafan_dl/g1/*.csv datasets/csv/lafan/
```

> `csv_to_npz.py` 只会在当前目录匹配 `*.csv`，不会递归查找，因此 CSV 必须直接放在 `--input-dir` 下，不能继续嵌套在 `g1/` 目录中。

### CSV → 窗口化 NPZ

```bash
uv run scripts/csv_to_npz.py \
  --input-dir datasets/csv/lafan \
  --output-dir datasets/npz/lafan
```

脚本会在 G1 仿真中回放每个 CSV，通过正向运动学计算被跟踪末端的位置，将帧率从 30 FPS 插值到 50 FPS，再切分成形状为 `(N, window_size, 59)` 的**骨盆锚定、仅保留偏航角**的窗口，每个动作片段生成一个 `.npz`。每帧 59 维特征的布局为：

```text
[root_pos(3), root_rot(6), joint_pos(29), ee_pos(15), root_lin_vel(3), root_ang_vel(3)]
```

常用参数包括：`--window-size`（默认 `10`）、`--stride`（默认 `1`）、`--input-fps`（默认 `30`）、`--output-fps`（默认 `50`），以及用于并行切分大型数据集的 `--shard-index / --num-shards`。

### 归一化统计

```bash
uv run scripts/compute_norm_stats.py \
  --input-dir datasets/npz/lafan \
  --output datasets/norm_stats.npz
```

该脚本拼接 `--input-dir` 下的所有窗口，并写入每个特征的 **q01/q99 分位数**（`q_low` / `q_high`，可通过 `--q-low / --q-high` 调整），用于将特征映射到 `[-1, 1]`。

仓库已经包含基于完整 LAFAN G1 数据集计算的 `datasets/norm_stats.npz`，预训练默认通过 `--norm-stats-file datasets/norm_stats.npz` 使用它。重新训练先验时，应优先复用这个文件，而不是在通常较窄的自定义动作片段上重新计算；只有改变特征布局时才需要重新计算。

> **归一化统计应来自多样化的数据集（完整 LAFAN），而不是狭窄的数据子集。** q01/q99 范围定义了特征归一化，并被写入预训练 checkpoint，也就是冻结去噪器在强化学习阶段看到的精确映射。PPO 策略经常会进入预训练动作分布之外的状态；如果归一化器只在少量动作上拟合，例如仅使用行走、慢跑和奔跑片段，超出范围的特征会向 ±1 饱和，导致去噪器接收到分布外输入，使分数估计和 SMP 引导奖励在最需要它们的状态上退化。较宽的归一化范围能让分数函数覆盖强化学习实际访问的状态。

---

## 扩散模型预训练

### 将 CSV 数据集转换为 NPZ

首先将对应 CSV 数据集转换为窗口化 NPZ。以前进先验为例：

```bash
uv run scripts/csv_to_npz.py \
  --input-dir datasets/csv/forward \
  --output-dir datasets/npz/forward
```

### 训练前进先验

```bash
uv run scripts/pretrain.py --data-dir datasets/npz/forward/ --num-layers 2 --no-use-ema --save-interval 5000 --num-epochs 10000 --train-split 1.0 --d-model 128
```

---

## 强化学习

项目通过 `mjlab.tasks.registry` 注册五个下游任务。导入 `smp.rl.tasks` 时会自动完成注册。

| Task | 演示 | 说明 |
|---|:---:|---|
| `Smp-Forward-G1` | <img src="https://raw.githubusercontent.com/SUZ-tsinghua/smp/assets/forward.gif" width="200"/> | 按指定的 `+x` 速度行走、慢跑或奔跑 |
| `Smp-Steering-G1` | <img src="https://raw.githubusercontent.com/SUZ-tsinghua/smp/assets/steering.gif" width="200"/> | 跟踪指定速度和面朝方向 |
| `Smp-Location-G1` | <img src="https://raw.githubusercontent.com/SUZ-tsinghua/smp/assets/location.gif" width="200"/> | 移动到世界坐标系中的 xy 目标位置 |
| `Smp-Getup-G1` | <img src="https://raw.githubusercontent.com/SUZ-tsinghua/smp/assets/getup.gif" width="200"/> | 从倒地姿态恢复站立 |
| `CMoE-G1` | — | 使用五个对比专家完成复杂地形运动 |

### 训练与播放

```bash
# 训练，checkpoint 保存在 logs/ 下
uv run scripts/train.py Smp-Forward-G1 --env.scene.num-envs=4096

# 查看训练指标
uv run tensorboard --logdir logs

# 使用本地 checkpoint 播放训练后的策略
uv run scripts/play.py Smp-Forward-G1 \
  --checkpoint-file logs/rsl_rl/smp_forward_g1/<run>/model_500.pt \
  --num-envs 4
```

四个 `Smp-*` 任务使用仓库内置的运动先验。`CMoE-G1` 是
[CMoE](https://github.com/Fudan-MAGIC-Lab/CMoE) 的独立完整迁移，会端到端训练五专家策略、
状态/地形估计器和原型对比目标：

```bash
uv run scripts/train.py CMoE-G1 --env.scene.num-envs=4096
```

该任务保留原始的 12 自由度下肢控制、10 帧本体感知历史、77 点高度扫描、非对称 critic
观测、地形课程、域随机化与 CMoE PPO 损失，不使用 SMP 先验 checkpoint。

将 `src/smp/rl/tasks/cmoe/__init__.py` 中的 `play_env_cfg` 切换为
`g1_cmoe_course_env_cfg(difficulty=0.5)` 后，可沿 x 轴依次播放九种 CMoE
地形，每个环境占用一条独立路线：

```bash
uv run scripts/play.py CMoE-G1 \
  --checkpoint-file logs/rsl_rl/g1_cmoe/<run>/model_<iteration>.pt \
  --num-envs 4
```

`--num-envs` 同时决定路线行数，统一地形难度由任务注册中的 `difficulty`
参数设置。

### 奖励设计：`task × SMP`

每个任务只使用一个**乘法组合**奖励项 `task_smp_product`：

```text
r  =  ( Σᵢ wᵢ · taskᵢ(s) )  ×  r_smp(s)
```

其中：

```text
r_smp = exp(−wₛ/|K| · Σ_{i∈K} ‖ε̂_i − ε_i‖²)
```

`r_smp` 是 SDS 引导奖励，即冻结去噪器在固定扩散时间步集合 `K` 上的 ε 预测误差，并按时间步归一化。

这是本项目与原始 SMP / MimicKit 的**关键差异**。原方法以加法组合任务奖励和 SMP 奖励，并分别设置权重 `task_reward_weight` 与 `smp_reward_weight`：

```text
# 原方法（加法）：r = task_reward_weight · task + smp_reward_weight · r_smp
# 本项目（乘法）：r = task · r_smp
```

目标是让策略在**完成任务的同时保持较高的 SMP 奖励**。乘积只有在两个部分都较高时才会较大，任意一项降低都会使总奖励趋近于 0，因此奖励调参更简单、更稳定：

- **无需平衡任务和先验的权重。** 加法形式需要调整 `task_reward_weight : smp_reward_weight`，其最佳比例会随任务和训练阶段变化；乘法形式直接去掉了这一参数。
- **无法只利用其中一项刷取奖励。** 使用加法时，策略可能只最大化其中一项，例如保持静止以获得自然姿态的高先验奖励却不完成任务，或者以不自然动作冲向目标。使用乘法时，两种失败模式的奖励都接近 0，策略必须同时完成任务并保持在运动流形上。

各任务的 `taskᵢ` 分量先加权求和，再由 `r_smp` 门控：

- **Forward**：仅跟踪速度，奖励为 `exp(−s·‖v_cmd − v_xy‖²)`；当速度在目标方向上的投影为负时奖励置零。方向固定为 `+x`，命令速度为 0.5–5 m/s。
- **Steering**：`0.5·` 速度跟踪 `+ 0.5·` 朝向对齐 `max(face_dir · heading, 0)`；随机目标方向和面朝方向，速度为 0.5–2 m/s。
- **Location**：仅跟踪位置，奖励为 `exp(−s·‖xy_goal − xy_robot‖)`，目标是周期性重新采样的世界坐标系位置，使用 `ws=4`。
- **Get-up**：`0.7·` 头部向上速度 `+ 0.3·` 头部高度跟踪，两项均采用 `exp(−s·max(target − ·, 0)²)`，初始状态来自倒地 GSI。

### 生成式状态初始化（GSI）

每次重置时，从冻结先验预采样得到的运动窗口池中抽取一个初始状态。窗口最后一帧用于设置仿真状态，整个窗口用于初始化在线特征缓冲区，因此从第 0 步开始 `r_smp` 就具有意义。每个环境都重置到各自的场景原点，同时特征缓冲区保持为**相对于环境原点**的表示，使引导奖励不受环境在世界网格中位置的影响。

### 运动特征

引导奖励对在线重建的滚动运动特征窗口进行评分。该窗口由 `smp.rl.utils.MotionFeatureBuffer` 维护，与预训练布局一致。G1 每帧为 59 维，并锚定到最后一帧仅含偏航角的局部坐标系：

```text
[root_pos(3), root_rot(6), joint_pos(29), ee_pos(15), root_lin_vel(3), root_ang_vel(3)]
```

## 引用与致谢

本仓库复现了 SMP。使用本项目时，请引用原始工作并注明参考实现：

- **SMP**——Mu 等，*Reusable Score-Matching Motion Priors for Physics-Based Character Control*，2025。[arXiv:2512.03028](https://arxiv.org/abs/2512.03028)
- **CMoE**——Ma 等，*Contrastive Mixture of Experts for Motion Control and Terrain Adaptation of Humanoid Robots*，ICRA 2026。[arXiv:2603.03067](https://arxiv.org/abs/2603.03067)
- **MimicKit**——SMP 原始实现：<https://github.com/xbpeng/MimicKit>
- **mjlab**——强化学习环境基础框架：<https://github.com/mujocolab/mjlab>

迁移后的 CMoE 组件保留 BSD-3-Clause 许可，详见
[`LICENSES/CMoE.txt`](LICENSES/CMoE.txt) 与 [`NOTICE`](NOTICE)。
