# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University).

"""The fixed-waist Unitree G1 asset used by the original CMoE task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco_warp as mjwarp
import torch
from mjlab.actuator import Actuator, ActuatorCfg, ActuatorCmd
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec import create_position_actuator
from mjlab.utils.spec_config import CollisionCfg

_G1_URDF = (
  Path(__file__).parent
  / "assets"
  / "g1"
  / "29dof_urdf"
  / "g1_29dof_with_hand_fixed_modify_collision.urdf"
)
_G1_MESHES = _G1_URDF.parent.parent / "meshes"

LOWER_BODY_JOINTS = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
)
LOWER_VELOCITY_LIMITS = (32.0, 20.0, 32.0, 20.0, 37.0, 37.0) * 2
LOWER_TORQUE_LIMITS = (88.0, 139.0, 88.0, 139.0, 50.0, 50.0) * 2
LOWER_STIFFNESS = (100.0, 100.0, 100.0, 150.0, 40.0, 40.0) * 2
LOWER_DAMPING = (2.0, 2.0, 2.0, 4.0, 2.0, 2.0) * 2
LOWER_DEFAULT_TARGETS = (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0) * 2


@dataclass(kw_only=True)
class CMoEPositionActuatorCfg(ActuatorCfg):
  stiffness: tuple[float, ...]
  damping: tuple[float, ...]
  effort_limit: tuple[float, ...]
  default_target: tuple[float, ...]
  decimation: int

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> "CMoEPositionActuator":
    return CMoEPositionActuator(self, entity, target_ids, target_names)


class CMoEPositionActuator(Actuator[CMoEPositionActuatorCfg]):
  """Apply CMoE's action delay within each four-substep control interval."""

  def __init__(
    self,
    cfg: CMoEPositionActuatorCfg,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    order = [cfg.target_names_expr.index(name) for name in target_names]
    self._stiffness = tuple(cfg.stiffness[index] for index in order)
    self._damping = tuple(cfg.damping[index] for index in order)
    self._effort_limit = tuple(cfg.effort_limit[index] for index in order)
    self._default_target = tuple(cfg.default_target[index] for index in order)
    self._substep = 0
    self._delay_steps: torch.Tensor | None = None
    self._previous_target: torch.Tensor | None = None
    self._current_target: torch.Tensor | None = None

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    for name, stiffness, damping, effort_limit in zip(
      target_names,
      self._stiffness,
      self._damping,
      self._effort_limit,
      strict=True,
    ):
      self._mjs_actuators.append(
        create_position_actuator(
          spec,
          name,
          stiffness=stiffness,
          damping=damping,
          effort_limit=effort_limit,
          transmission_type=self.cfg.transmission_type,
        )
      )

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)
    defaults = torch.tensor(self._default_target, device=device)
    self._delay_steps = torch.zeros(data.nworld, dtype=torch.long, device=device)
    self._previous_target = defaults.repeat(data.nworld, 1)
    self._current_target = self._previous_target.clone()

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self._delay_steps is not None
    assert self._previous_target is not None
    assert self._current_target is not None
    if self._substep == 0:
      self._current_target.copy_(cmd.position_target)
      self._delay_steps.random_(0, self.cfg.decimation)
    use_current = self._substep >= self._delay_steps
    return torch.where(
      use_current[:, None], self._current_target, self._previous_target
    )

  def update(self, dt: float) -> None:
    del dt
    self._substep += 1
    if self._substep == self.cfg.decimation:
      assert self._previous_target is not None
      assert self._current_target is not None
      self._previous_target.copy_(self._current_target)
      self._substep = 0

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    assert self._previous_target is not None
    assert self._current_target is not None
    defaults = torch.tensor(self._default_target, device=self._previous_target.device)
    indices = slice(None) if env_ids is None else env_ids
    self._previous_target[indices] = defaults
    self._current_target[indices] = defaults


def get_cmoe_g1_spec() -> mujoco.MjSpec:
  """Load the 12-DoF URDF and add the CMoE sensor attachment sites."""
  spec = mujoco.MjSpec.from_file(str(_G1_URDF))
  spec.compiler.meshdir = str(_G1_MESHES)
  spec.body("pelvis").add_freejoint(name="freejoint")

  collision_index = 0
  for geom in spec.geoms:
    if geom.contype:
      geom.name = f"{geom.parent.name}_collision_{collision_index}"
      collision_index += 1

  spec.body("pelvis").add_site(name="cmoe_scan_frame", pos=(0.4, 0.0, 0.0))
  sample_positions = (
    (0.03, 0.0, -0.035),
    (0.12, 0.0, -0.035),
    (-0.05, 0.0, -0.035),
    (0.06, 0.03, -0.035),
    (0.06, -0.03, -0.035),
  )
  for side in ("left", "right"):
    foot = spec.body(f"{side}_ankle_roll_link")
    foot.add_site(name=f"{side}_foot", pos=(0.0, 0.0, 0.0))
    for index, position in enumerate(sample_positions, 1):
      foot.add_site(name=f"cmoe_{side}_foot_sample_point{index}", pos=position)
  return spec


def get_cmoe_g1_robot_cfg() -> EntityCfg:
  """Return the original CMoE G1: fixed waist/arms and 12 actuated joints."""
  actuators = (
    CMoEPositionActuatorCfg(
      target_names_expr=LOWER_BODY_JOINTS,
      stiffness=LOWER_STIFFNESS,
      damping=LOWER_DAMPING,
      effort_limit=LOWER_TORQUE_LIMITS,
      default_target=LOWER_DEFAULT_TARGETS,
      decimation=4,
    ),
  )
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.8),
      joint_pos={
        ".*_hip_pitch_joint": -0.1,
        ".*_knee_joint": 0.3,
        ".*_ankle_pitch_joint": -0.2,
        ".*": 0.0,
      },
      joint_vel={".*": 0.0},
    ),
    spec_fn=get_cmoe_g1_spec,
    collisions=(
      CollisionCfg(
        geom_names_expr=(r".*_collision_.*",),
        condim={r"^(left|right)_ankle_roll_link_collision_.*$": 3, ".*": 3},
      ),
    ),
    articulation=EntityArticulationInfoCfg(
      actuators=actuators,
      soft_joint_pos_limit_factor=0.9,
    ),
  )


__all__ = [
  "LOWER_BODY_JOINTS",
  "LOWER_TORQUE_LIMITS",
  "LOWER_VELOCITY_LIMITS",
  "get_cmoe_g1_robot_cfg",
  "get_cmoe_g1_spec",
]
