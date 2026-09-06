"""AME domain-randomization events."""

from __future__ import annotations

import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
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
  geom_ids = env.scene[asset_cfg.name].indexing.geom_ids[asset_cfg.geom_ids]
  buckets = torch.empty(num_buckets, device=env.device).uniform_(*friction_range)
  bucket_ids = torch.randint(
    num_buckets, (env_ids.shape[0], geom_ids.shape[0]), device=env.device
  )
  env.sim.model.geom_friction[env_ids[:, None], geom_ids, 0] = buckets[bucket_ids]


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def randomize_body_mass(
  env,
  env_ids: torch.Tensor | None,
  mass_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
) -> None:
  """Add payload mass and scale inertia as Isaac Lab's mass event does."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  body_ids = env.scene[asset_cfg.name].indexing.body_ids[asset_cfg.body_ids]
  mass = env.sim.get_default_field("body_mass")[body_ids]
  inertia = env.sim.get_default_field("body_inertia")[body_ids]
  randomized_mass = mass + torch.empty(
    len(env_ids), len(body_ids), device=env.device
  ).uniform_(*mass_range)
  env.sim.model.body_mass[env_ids[:, None], body_ids] = randomized_mass
  env.sim.model.body_inertia[env_ids[:, None], body_ids] = (
    inertia * (randomized_mass / mass).unsqueeze(-1)
  )


__all__ = ["randomize_body_mass", "randomize_geom_friction_buckets"]
