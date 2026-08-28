[**English**](README.md) | [简体中文](README_zh-CN.md)

# SMP / CMoE / AME: Unitree G1 Motion-Control Reproductions and Ports

This repository brings three humanoid motion-control projects onto
[mjlab](https://github.com/mujocolab/mjlab) and the **Unitree G1**, with a shared set of installation,
training, and playback entry points. This is a course-project reproduction and engineering port, not an
official implementation of any of the three upstream projects.

## Project sources

The three parts of this repository come from the following open-source projects:

| Project | Upstream code | Work included here | Status |
|---|---|---|---|
| **SMP** (Score-Matching Motion Priors) | [SUZ-tsinghua/smp](https://github.com/SUZ-tsinghua/smp) | Directly uses its Unitree G1 motion features, diffusion priors, and downstream tasks | **Complete** |
| **CMoE** (Contrastive Mixture of Experts) | [Fudan-MAGIC-Lab/CMoE](https://github.com/Fudan-MAGIC-Lab/CMoE) | Ports the five-expert complex-terrain locomotion policy to MuJoCo / mjlab | **Port complete** |
| **AME** (Attention-Based Map Encoding) | [SII-FUSC/AME_Locomotion](https://github.com/SII-FUSC/AME_Locomotion) | Ports the two-stage elevation-map and attention-based locomotion method | **Incomplete** |

> [!IMPORTANT]
> **AME is an unfinished task.**

## Setup

### Requirements

- Linux;
- an NVIDIA GPU and a working NVIDIA driver;
- `curl`, or another way to install `uv`.

Training requires a GPU in practice. The project uses `uv` to manage dependencies; the pinned mjlab Git
revision and all other dependencies are recorded in `uv.lock`.

### Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

### Install the project

```bash
cd /path/to/smp
uv sync --frozen
```

The project selects Python 3.13 through `.python-version`. All commands below use `uv run`, so manually
activating the virtual environment is unnecessary.

### Verify the environment

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

Confirm that the task registry and training entry points load:

```bash
uv run scripts/train.py Smp-Forward-G1 --help
uv run scripts/train.py CMoE-G1 --help
```

## Project 1: SMP

### Overview

SMP comes from *Reusable Score-Matching Motion Priors for Physics-Based Character Control* (Mu et al.,
2025). The SMP portion of this repository directly uses the Unitree G1 motion features, diffusion model,
Generative State Initialization, reinforcement-learning tasks, and reward implementation provided by
[SUZ-tsinghua/smp](https://github.com/SUZ-tsinghua/smp).

The SMP pipeline has three stages:

1. Convert motion data into fixed-length motion-feature windows.
2. Pretrain a small DDPM on the windows and freeze its score function.
3. Turn the score error into an SDS-style motion-prior reward during PPO training.

This allows downstream policies to learn natural motion from a reusable prior without a task-specific
reference clip or adversarial discriminator.

### Implemented tasks

| Task ID | Demo | Description |
|---|:---:|---|
| `Smp-Forward-G1` | <img src="https://raw.githubusercontent.com/SUZ-tsinghua/smp/assets/forward.gif" width="200"/> | Walk, jog, or run at a commanded `+x` speed |
| `Smp-Steering-G1` | <img src="https://raw.githubusercontent.com/SUZ-tsinghua/smp/assets/steering.gif" width="200"/> | Track a commanded velocity and facing direction |
| `Smp-Location-G1` | <img src="https://raw.githubusercontent.com/SUZ-tsinghua/smp/assets/location.gif" width="200"/> | Move to a world-frame xy target |
| `Smp-Getup-G1` | <img src="https://raw.githubusercontent.com/SUZ-tsinghua/smp/assets/getup.gif" width="200"/> | Stand up from a fallen pose |

### Included pretrained priors

Three diffusion priors are included under `datasets/pretrain_ckpt/`, allowing RL training to start without
pretraining. Each task's `init_smp_state` configuration already points to the corresponding file.

| Checkpoint | Training data | Used by |
|---|---|---|
| `pretrained_loco.pt` | Walk / jog / run | `Smp-Forward-G1` |
| `pretrained_lafan_run.pt` | LAFAN running subset | `Smp-Steering-G1`, `Smp-Location-G1` |
| `pretrained_getup_f2s2.pt` | Get-up (fallen to standing) | `Smp-Getup-G1` |

### Data processing

Motion input follows the per-frame CSV format used by the `g1` split of the
[LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset).
Each file is headerless and comma-separated, sampled at 30 FPS, and contains 36 columns per frame:

| Columns | Field | Description |
|---|---|---|
| 0–2 | Root position `x y z` | World frame, in metres |
| 3–6 | Root-orientation quaternion | Ordered as `x y z w` |
| 7–35 | 29 G1 joint angles | Radians; see `JOINT_NAMES` in `scripts/csv_to_npz.py` for the order |

Raw CSV files are not included. They can be downloaded with the Hugging Face CLI:

```bash
hf download lvhaidong/LAFAN1_Retargeting_Dataset --repo-type dataset \
  --include "g1/*.csv" --local-dir datasets/csv/_lafan_dl
mv datasets/csv/_lafan_dl/g1/*.csv datasets/csv/lafan/
```

`csv_to_npz.py` does not search recursively, so CSV files must be placed directly under `--input-dir`.

Convert CSV files into windowed NPZ files:

```bash
uv run scripts/csv_to_npz.py \
  --input-dir datasets/csv/lafan \
  --output-dir datasets/npz/lafan
```

The script replays the motion in the G1 simulation, computes end-effector positions through forward
kinematics, interpolates from 30 to 50 FPS, and slices the result into `(N, window_size, 59)` windows.
Each frame contains 59 features:

```text
[root_pos(3), root_rot(6), joint_pos(29), ee_pos(15), root_lin_vel(3), root_ang_vel(3)]
```

All spatial quantities are anchored to the pelvis position of the final window frame and rotated into a
yaw-only local frame. Useful options include `--window-size`, `--stride`, `--input-fps`, `--output-fps`,
and `--shard-index / --num-shards` for parallel processing.

Compute normalization statistics:

```bash
uv run scripts/compute_norm_stats.py \
  --input-dir datasets/npz/lafan \
  --output datasets/norm_stats.npz
```

The script calculates the q01/q99 quantiles for each feature and maps features into `[-1, 1]`. The
repository already includes `datasets/norm_stats.npz`, computed from the full LAFAN G1 dataset. Reuse it
unless the feature layout changes. Statistics from a broad motion distribution reduce feature saturation
when PPO explores out-of-distribution states and help prevent score-estimation degradation.

### Diffusion-prior pretraining

For example, to train the forward-motion prior:

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

### Reward design: `task × SMP`

This repository multiplies the task reward by the SMP reward:

```text
r = (Σᵢ wᵢ · taskᵢ(s)) × r_smp(s)

r_smp = exp(−wₛ/|K| · Σ_{i∈K} ‖ε̂_i − ε_i‖²)
```

This differs from the additive composition in the original SMP method:

```text
# Original method
r = task_reward_weight · task + smp_reward_weight · r_smp

# This repository
r = task · r_smp
```

The product requires both task performance and motion naturalness to be high: optimizing only one term
still produces a near-zero total reward. It also removes the need to balance
`task_reward_weight : smp_reward_weight`. The four task rewards are:

- **Forward:** track a 0.5–5 m/s target speed along `+x`; reward is zero for backward motion.
- **Steering:** combine velocity tracking with facing-direction alignment at 0.5–2 m/s.
- **Location:** track a periodically resampled world-frame xy target.
- **Get-up:** combine upward head velocity and head-height tracking from a fallen state.

### Generative State Initialization

At every reset, a window is selected from a motion pool presampled from the frozen prior. Its final frame
sets the simulation state, while the full window fills the online feature buffer so that `r_smp` is valid
from step 0. The buffer is represented relative to each environment origin, making the reward independent
of the environment's position in the world grid.

### Train and play

```bash
# Train
uv run scripts/train.py Smp-Forward-G1 --env.scene.num-envs=4096

# Inspect training metrics
uv run tensorboard --logdir logs

# Play a local checkpoint
uv run scripts/play.py Smp-Forward-G1 \
  --checkpoint-file logs/rsl_rl/smp_forward_g1/<run>/model_500.pt \
  --num-envs 4
```

## Project 2: CMoE

### Overview

CMoE comes from *Contrastive Mixture of Experts for Motion Control and Terrain Adaptation of Humanoid
Robots*. It models different motion modes with five experts and promotes expert specialization through a
prototype contrastive objective, improving motion control and adaptation on complex terrain.

This repository provides an independent MuJoCo / mjlab port that retains the following core design:

- 12-DoF lower-body control;
- a 10-frame proprioceptive history;
- a 77-point terrain-height scan;
- asymmetric actor / critic observations;
- a five-expert policy with state and terrain estimators;
- the prototype contrastive objective and CMoE PPO losses;
- terrain curriculum, domain randomization, and nine complex terrain types.

CMoE is trained end to end from scratch and does not load an SMP prior checkpoint.

### Train

```bash
uv run scripts/train.py CMoE-G1 --env.scene.num-envs=4096
```

### Play the complex-terrain course

Switch `play_env_cfg` in `src/smp/rl/tasks/cmoe/__init__.py` to
`g1_cmoe_course_env_cfg(difficulty=0.5)` to play all nine CMoE terrain types sequentially along the x-axis:

```bash
uv run scripts/play.py CMoE-G1 \
  --checkpoint-file logs/rsl_rl/g1_cmoe/<run>/model_<iteration>.pt \
  --num-envs 4
```

`--num-envs` also determines the number of independent course lanes. Shared terrain difficulty is set by
the `difficulty` argument in task registration.

## Project 3: AME (incomplete)

### Overview

AME is based on the `SII-FUSC/AME_Locomotion` Unitree G1 reproduction of *Attention-Based Map Encoding
for Learning Generalized Legged Locomotion*. It first extracts local terrain features with a CNN, then uses
a query built from proprioception to select motion-relevant elevation-map regions through multi-head
attention. The resulting terrain encoding and proprioceptive features are passed to the actor and critic.

This repository intends to port the original Isaac Lab implementation to MuJoCo / mjlab. The following
parts have been brought into the repository:

- G1 joint order, default pose, PD parameters, and effort limits;
- a 33 × 21 × 3 elevation map, noise, and terrain-scan observations;
- CNN, MHA, proprioceptive encoder, and optional global-context encoder;
- actor / critic observations, rewards, terminations, terrain curriculum, and domain randomization;
- stage-one and stage-two terrain configurations;
- the AME PPO update and original `model_state_dict` checkpoint layout;
- attention-weight recording and visualization utilities.

End-to-end AME training, simulation validation, performance alignment, and final result reproduction are
not complete. Treat this section as work-in-progress migration code; it is not suitable as a validated
baseline for comparison or downstream research. See
[`src/smp/rl/tasks/ame/MIGRATION.md`](src/smp/rl/tasks/ame/MIGRATION.md) for detailed migration boundaries.

### Registered experimental tasks

| Task ID | Intended use | Status |
|---|---|---|
| `AME-G1` | Stage-one terrain locomotion training | Unverified |
| `AME-G1-Global` | Stage one with the global-context encoder | Unverified |
| `AME-G1-Finetune` | Stage-two terrain finetuning | Unverified |

### Development commands

The following commands are retained only for continued development and validation.

```bash
# Stage one
uv run scripts/train.py AME-G1 --env.scene.num-envs=4096

# Stage two: resume from the latest stage-one run
uv run scripts/train.py AME-G1-Finetune \
  --env.scene.num-envs=4096 \
  --agent.resume \
  --agent.load-run='.*_ame$' \
  --agent.load-checkpoint='model_.*\.pt'
```

Pass GPU IDs as a list:

```bash
uv run scripts/train.py AME-G1 --env.scene.num-envs=4096 --gpu-ids '[2]'
```

Use `--gpu-ids '[0,1]'` for multiple GPUs or `--gpu-ids all` for all visible GPUs.

Play a migrated checkpoint:

```bash
uv run scripts/play.py AME-G1 \
  --checkpoint-file logs/rsl_rl/g1_ame/<run>/model_<iteration>.pt \
  --num-envs 1
```

The `ame1.pt` and `ame2.pt` files from AME_Locomotion retain the original checkpoint layout and are
intended to load through `AME-G1` and `AME-G1-Global`, respectively. Compatibility code is present, but
behavioral equivalence has not yet been validated.

Save and plot MHA attention weights:

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

## Task reference

Importing `smp.rl.tasks` registers the following tasks with `mjlab.tasks.registry`:

| Project | Task ID | Status |
|---|---|---|
| SMP | `Smp-Forward-G1` | Complete |
| SMP | `Smp-Steering-G1` | Complete |
| SMP | `Smp-Location-G1` | Complete |
| SMP | `Smp-Getup-G1` | Complete |
| CMoE | `CMoE-G1` | Port complete |
| AME | `AME-G1` | Incomplete / unverified |
| AME | `AME-G1-Global` | Incomplete / unverified |
| AME | `AME-G1-Finetune` | Incomplete / unverified |

## Citation

If this repository is useful to your work, please cite the original papers for all three projects and
credit the corresponding open-source implementations used here:

- **SMP** — Mu et al., *Reusable Score-Matching Motion Priors for Physics-Based Character Control*, 2025.
  [Paper](https://arxiv.org/abs/2512.03028) · [Project page](https://yxmu.foo/smp-page/) ·
  [SMP project used by this repository](https://github.com/SUZ-tsinghua/smp)
- **CMoE** — Ma et al., *Contrastive Mixture of Experts for Motion Control and Terrain Adaptation of
  Humanoid Robots*, ICRA 2026. [Paper](https://arxiv.org/abs/2603.03067) ·
  [Code](https://github.com/Fudan-MAGIC-Lab/CMoE)
- **AME** — He et al., *Attention-Based Map Encoding for Learning Generalized Legged Locomotion*,
  Science Robotics, 2025. [Paper](https://arxiv.org/abs/2506.09588) ·
  [G1 reproduction](https://github.com/SII-FUSC/AME_Locomotion)

## Acknowledgements

We thank the authors of SMP, CMoE, and AME for publishing their research, as well as the following projects
and datasets on which this repository builds:

- [SUZ-tsinghua/smp](https://github.com/SUZ-tsinghua/smp): the Unitree G1 SMP project used directly here;
- [CMoE](https://github.com/Fudan-MAGIC-Lab/CMoE): the original five-expert complex-terrain implementation;
- [AME_Locomotion](https://github.com/SII-FUSC/AME_Locomotion): the open-source Unitree G1 AME reproduction;
- [mjlab](https://github.com/mujocolab/mjlab): the shared MuJoCo RL environment and training entry points;
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl): the PPO and on-policy training foundation;
- [LAFAN1 Retargeting Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset):
  the retargeted G1 motion data used by the SMP priors.

The ported CMoE code retains the upstream BSD-3-Clause notice; see
[`LICENSES/CMoE.txt`](LICENSES/CMoE.txt) and [`NOTICE`](NOTICE). When using this repository, also follow the
license and citation requirements of each upstream project, dependency, and dataset.
