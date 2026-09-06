"""Implicit position control with AME's per-substep reward measurements."""

from dataclasses import dataclass

import torch
from mjlab.actuator import Actuator, ActuatorCfg, ActuatorCmd
from mjlab.utils.spec import create_position_actuator


@dataclass(kw_only=True)
class AMEPositionActuatorCfg(ActuatorCfg):
  stiffness: float
  damping: float
  effort_limit: float
  armature: float

  def build(self, entity, target_ids, target_names):
    return AMEPositionActuator(self, entity, target_ids, target_names)


class AMEPositionActuator(Actuator[AMEPositionActuatorCfg]):
  @property
  def command_field(self):
    return "position"

  def edit_spec(self, spec, target_names):
    for name in target_names:
      self._mjs_actuators.append(create_position_actuator(
        spec, name,
        stiffness=self.cfg.stiffness,
        damping=self.cfg.damping,
        effort_limit=self.cfg.effort_limit,
        armature=self.cfg.armature,
        transmission_type=self.cfg.transmission_type,
      ))

  def initialize(self, mj_model, model, data, device):
    super().initialize(mj_model, model, data, device)
    shape = (data.nworld, len(self._target_ids_list))
    self.computed_effort = torch.zeros(shape, device=device)
    self.applied_effort = torch.zeros(shape, device=device)
    self.joint_acc = torch.zeros(shape, device=device)
    self._previous_vel = torch.zeros(shape, device=device)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    self.computed_effort.copy_(
      self.cfg.stiffness * (cmd.position_target - cmd.pos)
      + self.cfg.damping * (cmd.velocity_target - cmd.vel)
      + cmd.effort_target
    )
    self.applied_effort.copy_(self.computed_effort.clamp(
      -self.cfg.effort_limit, self.cfg.effort_limit
    ))
    self._previous_vel.copy_(cmd.vel)
    return cmd.position_target

  def update(self, dt: float):
    velocity = self.entity.data.joint_vel[:, self.target_ids]
    self.joint_acc.copy_((velocity - self._previous_vel) / dt)

  def reset(self, env_ids=None):
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self.computed_effort[ids] = 0.0
    self.applied_effort[ids] = 0.0
    self.joint_acc[ids] = 0.0
    self._previous_vel[ids] = 0.0
