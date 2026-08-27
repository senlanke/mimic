# AME migration contract

Source project: `/home/ksl/HL/AME_Locomotion`.

## Placement

- Task registration, environment, MDP, assets and terrains are under
  `src/smp/rl/tasks/ame`.
- The RSL-RL 5 AME model is under `src/smp/rl/ame`.
- The source `FINETUNE` branch is registered separately as
  `AME-G1-Finetune`; the first-stage task is `AME-G1`.

## Migration table

| Source item | SMP location | Migration method | Status |
|---|---|---|---|
| 29DoF joint order and default pose | `assets/robots/unitree.py` | API translation onto MJLab G1 MJCF | Preserved |
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
| Play single-stakes terrain and commands | `terrains/play_terrain_cfg.py`, `ame_env_cfg.py` | Direct parameters | Preserved |
| CNN, MHA and proprio embeddings | `src/smp/rl/ame/actor_critic_encoder.py` | Direct topology copy + RSL-RL 5 interface | Preserved |
| Direct learned action std | `src/smp/rl/ame/actor_critic_encoder.py` | Direct behavior copy; no std clamp | Preserved |
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
- The source combined `ActorCriticEncoder` is expressed as RSL-RL 5 actor and
  critic models. CNN and MHA modules are shared; proprio embeddings and MLP
  heads remain separate.
- Source RSL-RL checkpoints use the old combined `model_state_dict` layout and
  are not silently remapped to the RSL-RL 5 actor/critic checkpoint layout.
- The solver uses SMP/CMoE's MuJoCo settings: 5 ms simulation step, 10 Newton
  iterations and 20 line-search iterations.

No training, simulation or import validation was run, as requested.
