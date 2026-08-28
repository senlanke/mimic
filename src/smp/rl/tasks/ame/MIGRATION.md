# AME migration contract

Source project: `/home/ksl/HL/AME_Locomotion`.

## Placement

- Task registration, environment, MDP, assets and terrains are under
  `src/smp/rl/tasks/ame`.
- The RSL-RL 5 AME model and PPO are under `src/smp/rl/ame`.
- The source `FINETUNE` branch is registered separately as
  `AME-G1-Finetune`; the first-stage task is `AME-G1`. The released
  `attach_global=True` model is registered as `AME-G1-Global`.

## Migration table

| Source item | SMP location | Migration method | Status |
|---|---|---|---|
| G1 joint order and default pose | `assets/robots/unitree.py` | API translation onto MJLab G1 MJCF | Preserved |
| PD gains, effort limits and armature | `assets/robots/unitree.py` | API translation | Preserved |
| Actor/critic observation order | `ame_env_cfg.py` | API translation | Preserved |
| 33x21x3 elevation map and noise | `mdp/observations.py` | API translation | Preserved |
| Command, reset and event ranges | `ame_env_cfg.py` | API translation | Preserved |
| Reward formulas, weights and contact history | `mdp/rewards.py` | Direct logic copy + API translation | Preserved |
| Termination and terrain curriculum | `ame_env_cfg.py`, `mdp/commands.py` | API translation | Preserved |
| Stage-one terrain composition | `terrains/terrain_cfg.py` | Direct parameters + engine translation | Preserved |
| Stage-two terrain composition | `terrains/finetune_terrain_cfg.py` | Direct parameters + engine translation | Preserved |
| gaps/stakes/stonebridge/stepping-stones formulas | `terrains/loco_hf_terrains.py` | Direct formula copy | Preserved |
| rails formula | `terrains/rails_terrain_cfg.py` | Direct formula copy using MuJoCo boxes | Preserved |
| Stage-matched play terrains and fixed play commands | `ame_env_cfg.py` | SMP task separation | Adjusted |
| CNN, MHA and proprio embeddings | `src/smp/rl/ame/actor_critic_encoder.py` | Direct topology copy + RSL-RL 5 interface | Preserved |
| Direct learned action std | `src/smp/rl/ame/actor_critic_encoder.py` | Direct behavior copy; no std clamp | Preserved |
| AME PPO update loop and shared encoder ownership | `src/smp/rl/ame/ppo.py` | Direct logic copy + RSL-RL 5 API translation | Preserved |
| Original `model_state_dict` checkpoint layout | `src/smp/rl/ame/runner.py`, `ppo.py` | Direct layout preservation | Preserved |
| MHA attention recording and plotting | `src/smp/rl/ame/play.py`, `plot_attention.py` | Direct feature migration | Preserved |
| PPO scalar parameters | `ame_rl_cfg.py` | Direct parameters | Preserved |

## Explicit engine and API boundaries

- The Isaac Lab USD is replaced by MJLab's full-collision G1 MJCF. The MJCF
  supplies MuJoCo collision geometry, mass and inertia.
- The original 5 cm AME height arrays and surface meshes are retained. As in
  the CMoE migration, MuJoCo collision heightfields use a 10 cm stride while
  the policy ray grid remains 5 cm.
- Isaac static and dynamic friction collapse to MuJoCo sliding friction. The
  original `[0.3, 1.0]` range and 64-bucket per-geometry assignment are kept.
  MuJoCo has no direct rigid-material restitution field equivalent to Isaac's
  `restitution_range=(0.0, 0.1)`, so no substitute parameter is introduced.
- Isaac's actuator `velocity_limit_sim` has no direct MuJoCo actuator field.
  Effort limits and PD control are preserved; no torque-speed approximation or
  artificial velocity clamp is added.
- `AMEPPO` expresses the source combined `ActorCriticEncoder` as RSL-RL 5 actor
  and critic models. The actor owns CNN, MHA, global encoder and query projector
  modules and the critic directly references them; proprio embeddings and MLP
  heads remain separate.
- `AMEPPO` owns the source PPO update loop. Actor, critic and shared encoder
  parameters form one optimizer parameter set and receive one global gradient
  clipping operation per mini-batch.
- Each task's play configuration retains its own training terrain generator:
  `AME-G1` uses stage-one terrains and `AME-G1-Finetune` uses stage-two terrains.
- AME keeps the source combined `model_state_dict` layout as its only checkpoint
  format. The supplied `pretrained/ame1.pt` therefore loads directly through
  `AMERunner`; `ame2.pt` uses the explicit `AME-G1-Global` task. No
  actor/critic-format compatibility path is retained.
- The solver uses SMP/CMoE's MuJoCo settings: 5 ms simulation step, 10 Newton
  iterations and 20 line-search iterations.

No training, simulation or import validation was run, as requested.
