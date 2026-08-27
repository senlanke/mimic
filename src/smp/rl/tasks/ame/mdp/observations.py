"""AME terrain-map observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import RayCastSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class elevation_map:
  """Return 33x21 terrain hit points as flattened sensor-local xyz values."""

  def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedRlEnv"):
    self.offset = torch.zeros(env.num_envs, 1, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice) -> None:
    self.offset[env_ids].uniform_(-0.05, 0.05)

  def __call__(
    self, env: "ManagerBasedRlEnv", sensor_name: str, noise: bool = False
  ) -> torch.Tensor:
    sensor: RayCastSensor = env.scene[sensor_name]
    data = sensor.data
    relative_w = data.hit_pos_w - data.pos_w.unsqueeze(1)
    quat = yaw_quat(data.quat_w)
    num_envs, num_rays, _ = relative_w.shape
    local = quat_apply_inverse(
      quat.unsqueeze(1).expand(-1, num_rays, -1).reshape(-1, 4),
      relative_w.reshape(-1, 3),
    ).reshape(num_envs, num_rays, 3)
    if noise:
      local[..., 2] += torch.randn_like(local[..., 2]) * 0.03
      local[..., 2] += self.offset
    local[..., 2].clamp_(min=-1.2, max=0.0)
    return local.reshape(num_envs, num_rays * 3)


__all__ = ["elevation_map"]
