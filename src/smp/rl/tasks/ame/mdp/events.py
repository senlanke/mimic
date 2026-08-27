"""AME domain-randomization events."""

from __future__ import annotations

import torch
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg


@requires_model_fields("geom_friction")
def randomize_geom_friction_buckets(
  env,
  env_ids: torch.Tensor | None,
  friction_range: tuple[float, float],
  num_buckets: int,
  asset_cfg: SceneEntityCfg,
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  geom_ids = torch.as_tensor(asset_cfg.geom_ids, device=env.device)
  buckets = torch.empty(num_buckets, device=env.device).uniform_(*friction_range)
  bucket_ids = torch.randint(
    num_buckets, (env_ids.shape[0], geom_ids.shape[0]), device=env.device
  )
  env_grid, geom_grid = torch.meshgrid(env_ids, geom_ids, indexing="ij")
  env.sim.model.geom_friction[env_grid, geom_grid, 0] = buckets[bucket_ids]


__all__ = ["randomize_geom_friction_buckets"]
