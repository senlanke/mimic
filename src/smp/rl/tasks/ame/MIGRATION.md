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
| G1 default pose | `assets/robots/unitree.py` | API translation onto MJLab G1 MJCF | Preserved |
| PD gains, effort limits and armature | `assets/robots/unitree.py`, `actuators.py` | MuJoCo position control; pre-step PD effort recording | Preserved |
| Actor/critic observation terms | `ame_env_cfg.py`, `mdp/observations.py` | Root COM linear velocity, original term order | Preserved |
| 33x21x3 elevation map and noise | `mdp/observations.py` | 20 m ray-start offset; per-reset height bias | Preserved |
| Command, reset and event ranges | `ame_env_cfg.py` | API translation | Preserved |
| Torque and acceleration penalties | `mdp/rewards.py`, `assets/robots/actuators.py` | Saved pre-step PD efforts and per-physics-step velocity differences | Source measurement timing |
| Reward weights and contact history | `mdp/rewards.py`, `mdp/contacts.py` | World-frame net normal forces and 1 N contact timing | Source force semantics |
| Termination and terrain curriculum | `ame_env_cfg.py`, `mdp/commands.py` | API translation | Preserved |
| Stage-one terrain composition | `terrains/terrain_cfg.py`, `columns.py` | Original proportions expanded into 20 independent columns | Preserved |
| Stage-two terrain composition | `terrains/finetune_terrain_cfg.py`, `columns.py` | Original proportions expanded into 20 independent columns | Preserved |
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
- AME generates a complete 5 cm heightfield for raycasting (161 samples per
  axis on an 8 m tile), with `contype=0` and `conaffinity=0` so it cannot produce
  contacts. A second heightfield samples every other point for 10 cm collision
  geometry (81 samples per axis). The collision geometry uses group 4, excluded
  by the scanner's group-0 filter. Both fields share the same XY extent; the
  spawn origin comes from the full-resolution field. Border pixel counts follow
  the source heightfield decorator. The unused surface mesh and height-array
  cache were removed. As explicitly selected for this migration, steep edges retain
  MuJoCo heightfield triangulation; Isaac's `slope_threshold` vertex shifts
  into vertical faces are not represented.
- Terrain columns follow the source cumulative-proportion allocation. Each
  column is a separate MJLab sub-terrain entry, preserving 20 independent
  columns under MJLab's one-column-per-entry curriculum generator.
- Joint reset uses zero position and velocity offsets. This equals the source
  default pose multiplied by 1 and default zero velocity multiplied by [-1, 1].
- Ray starts are 20 m above the yaw-aligned torso grid. Map coordinates remain
  relative to the torso, matching the source observation frame. No missing-ray
  replacement or NaN sanitization is added.
- Each actuator records its computed and motor-limit-clipped PD efforts before
  every physics step. Torque penalties use the final substep's saved efforts.
  Acceleration penalties use the velocity difference across that physics step,
  divided by the physics timestep. MuJoCo's position actuator still applies
  the control; the saved PD effort is Isaac's reward estimate.
- Critic base linear velocity, linear tracking and foot sliding use body COM
  velocities, matching Isaac Lab's root/body velocity aliases.
- AME contact sensors accumulate only the normal component of each solved
  contact in world coordinates, excluding tangential friction. A net normal
  force norm greater than 1 N starts contact timing. Existing three-substep
  force history and air/contact time updates consume these values. The sensors
  read the solver contacts directly without adding native contact sensors.
- Isaac static and dynamic friction collapse to MuJoCo sliding friction. The
  original `[0.3, 1.0]` range and 64-bucket per-geometry assignment are kept.
  Robot-local geom indices are mapped through `asset.indexing.geom_ids` before
  writing the global model; terrain friction is not randomized by this event.
  MuJoCo has no direct rigid-material restitution field equivalent to Isaac's
  `restitution_range=(0.0, 0.1)`, so no substitute parameter is introduced.
- Finetune torso mass randomization also scales the default principal inertia
  by `new_mass / default_mass`, matching Isaac Lab's default
  `recompute_inertia=True`. Body indices are mapped into the global model and
  MuJoCo constants are recomputed after the event.
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
- Optimizer parameters follow the source module's `parameters()` order: direct
  action standard deviation first, followed by the registered child modules.
  SMP checkpoints saved with the previous std-last order must not restore their
  optimizer state with this implementation. Start a fresh run or load model
  weights only; no optimizer-order compatibility branch is added.
- The solver uses SMP/CMoE's MuJoCo settings: 5 ms simulation step, 10 Newton
  iterations and 20 line-search iterations.
- AME has no RND or symmetry implementation. Their unused constructor arguments
  and config resolvers were removed; supplying these algorithm options now
  produces Python's normal unexpected-keyword error. The runner's RND logging
  metadata remains `None`.

## Validation

- Before separating scan and collision geometry, generated every stage-one and
  stage-two column and verified 20 columns per stage and 161x161 height samples.
- Verified torque penalties with deliberately shuffled actuator ordering:
  zero below effort limits and the exact excess above the limits.
- Verified per-reset map bias writes back only to selected environment rows.
- Ran a 16-environment GPU rollout and one AMEPPO update on stage-one terrain
  with one difficulty row. Observations were finite and every scan ray hit.
- Ran a 20-environment Finetune GPU smoke check covering startup randomization,
  zero reset velocities, observations, rewards, ray hits and partial resets.
- These checks validate execution and the corrected mappings, not convergence.
- No further validation is performed for the scan/collision split, as requested.
- The subsequent COM velocity, normal-force sensor, substep reward measurement
  and optimizer-order changes have not been executed or tested, as requested.
  The earlier checks above do not validate these changes or training convergence.

## Lessons checked against CMoE

See `CMOE_MJLAB_MIGRATION_NOTES_zh-CN.md` at the repository root. Its 10 cm
collision grid addresses observed MuJoCo Warp heightfield collision overflow;
it is not an algorithm change. CMoE separately reads its original 5 cm height
arrays in `cmoe_scan_heights`. AME's raycasting does not use that array sampler,
so merely retaining an unused surface mesh does not preserve fine observations.
Short smoke checks do not establish collision capacity or long-run stability.

CMoE's action delay and command-before-reward step order are specific to its
source task. AME retains its own source's no-delay control and command update
after reward/reset. No CMoE-specific PPO tuning, noise clamps or NaN replacement
is introduced.
