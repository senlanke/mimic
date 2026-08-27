"""AME-specific reward functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def feet_stumble(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  z_force = force[..., 2].abs()
  xy_force = torch.linalg.norm(force[..., :2], dim=-1)
  return (xy_force > 4.0 * z_force).any(dim=-1).float()


def illegal_contact(
  env: "ManagerBasedRlEnv", sensor_name: str, force_threshold: float
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force_norm = torch.linalg.norm(sensor.data.force_history, dim=-1)
  return (force_norm.max(dim=2)[0] > force_threshold).any(dim=1)


def feet_too_near(
  env: "ManagerBasedRlEnv", threshold: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  positions = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
  return (threshold - torch.linalg.norm(positions[:, 0] - positions[:, 1], dim=-1)).clamp(min=0.0)


def air_time_variance_penalty(
  env: "ManagerBasedRlEnv", sensor_name: str
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  air = sensor.data.last_air_time
  contact = sensor.data.last_contact_time
  return torch.var(air.clamp(max=0.5), dim=1) + torch.var(
    contact.clamp(max=0.5), dim=1
  )


def feet_air_time_positive_biped(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  command_name: str,
  threshold: float,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  air = sensor.data.current_air_time
  contact = sensor.data.current_contact_time
  in_contact = contact > 0.0
  in_mode_time = torch.where(in_contact, contact, air)
  single_stance = torch.sum(in_contact.int(), dim=1) == 1
  reward = torch.min(
    torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
  )[0]
  reward = torch.clamp(reward, max=threshold)
  command = env.command_manager.get_command(command_name)
  return reward * (torch.linalg.norm(command[:, :2], dim=1) > 0.1)


class joint_coordination_rel:
  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self.joint_ids = [
      (asset.find_joints(pair[0])[0][0], asset.find_joints(pair[1])[0][0])
      for pair in cfg.params["coord_joints"]
    ]

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg,
    coord_joints: tuple[tuple[str, str], ...],
    coord_signs: tuple[tuple[float, float], ...],
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    default = asset.data.default_joint_pos
    result = torch.zeros(env.num_envs, device=env.device)
    for (first, second), signs in zip(self.joint_ids, coord_signs, strict=True):
      first_rel = (asset.data.joint_pos[:, first] - default[:, first]) * signs[0]
      second_rel = (asset.data.joint_pos[:, second] - default[:, second]) * signs[1]
      result += torch.square(first_rel - second_rel)
    return result / len(coord_joints)


def applied_torque_limits(
  env: "ManagerBasedRlEnv",
  limits: tuple[float, ...],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  effort = asset.data.actuator_force[:, asset_cfg.actuator_ids].abs()
  limit = torch.tensor(limits, device=env.device, dtype=effort.dtype)
  return (effort - limit).clamp(min=0.0).sum(dim=1)


def undesired_contacts(
  env: "ManagerBasedRlEnv", sensor_name: str, threshold: float
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force_norm = torch.linalg.norm(sensor.data.force_history, dim=-1)
  return (force_norm.max(dim=2)[0] > threshold).sum(dim=1)


def track_lin_vel_xy_yaw_frame_exp(
  env: "ManagerBasedRlEnv", command_name: str, std: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  velocity = quat_apply_inverse(
    yaw_quat(asset.data.root_link_quat_w), asset.data.root_link_lin_vel_w
  )
  error = torch.square(command[:, :2] - velocity[:, :2]).sum(dim=1)
  return torch.exp(-error / std**2)


def track_ang_vel_z_world_exp(
  env: "ManagerBasedRlEnv", command_name: str, std: float,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_w[:, 2])
  return torch.exp(-error / std**2)


def ang_vel_xy_l2(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_ang_vel_b[:, :2]).sum(dim=1)


def feet_slide(
  env: "ManagerBasedRlEnv", sensor_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  velocity = torch.linalg.norm(
    asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1
  )
  contacts = torch.linalg.norm(sensor.data.force_history, dim=-1).max(dim=2)[0] > 1.0
  return (velocity * contacts).sum(dim=1)


def joint_deviation_l1(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  default = asset.data.default_joint_pos
  ids = asset_cfg.joint_ids
  return torch.abs(asset.data.joint_pos[:, ids] - default[:, ids]).sum(dim=1)


__all__ = [
  "air_time_variance_penalty",
  "applied_torque_limits",
  "feet_air_time_positive_biped",
  "feet_stumble",
  "feet_too_near",
  "joint_coordination_rel",
  "illegal_contact",
  "joint_deviation_l1",
  "ang_vel_xy_l2",
  "feet_slide",
  "track_ang_vel_z_world_exp",
  "track_lin_vel_xy_yaw_frame_exp",
  "undesired_contacts",
]
