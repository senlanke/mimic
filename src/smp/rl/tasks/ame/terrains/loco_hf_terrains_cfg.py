"""Configuration classes for AME's original height-field terrains."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import TerrainOutput

from smp.rl.tasks.cmoe.height_field.hf_terrains_cfg import (
  HfTerrainBaseCfg,
  _height_field_to_hfield_surface_mesh,
  _height_field_to_output,
)

from . import loco_hf_terrains


@dataclass(kw_only=True)
class _AMEHeightFieldCfg(HfTerrainBaseCfg):
  horizontal_scale: float = 0.05
  vertical_scale: float = 0.005
  slope_threshold: float = 0.75
  _rng: np.random.Generator = field(init=False, repr=False)
  height_fields: list[np.ndarray] = field(default_factory=list, init=False, repr=False)

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    self._rng = rng
    width_pixels = int(self.size[0] / self.horizontal_scale)
    length_pixels = int(self.size[1] / self.horizontal_scale)
    border_pixels = int(self.border_width / self.horizontal_scale)
    raw = np.zeros((width_pixels, length_pixels), dtype=np.int16)
    cfg_for_gen = copy.deepcopy(self)
    cfg_for_gen.size = (
      (width_pixels - 2 * border_pixels) * self.horizontal_scale,
      (length_pixels - 2 * border_pixels) * self.horizontal_scale,
    )
    generated = self._generate_height_field(difficulty, cfg_for_gen)
    raw[border_pixels:-border_pixels, border_pixels:-border_pixels] = generated
    self.height_fields.append(raw)

    collision_cfg = replace(self, horizontal_scale=0.1)
    output = _height_field_to_output(
      heights=raw[::2, ::2].T,
      cfg=collision_cfg,
      spec=spec,
      rng=rng,
    )
    output.instinct_surface_mesh = _height_field_to_hfield_surface_mesh(raw.T, self)
    return output

  def _generate_height_field(self, difficulty: float, cfg_for_gen) -> np.ndarray:
    return self.generate(difficulty, cfg_for_gen, self._rng)


@dataclass(kw_only=True)
class HfStonesBridgeTerrainCfg(_AMEHeightFieldCfg):
  stone_height_max: float
  stone_width_range: tuple[float, float]
  stone_length_range: tuple[float, float]
  stone_distance_range: tuple[float, float]
  stone_lateral_distance_range: tuple[float, float]
  holes_depth: float = -10.0
  platform_width: float = 1.0
  generate = staticmethod(loco_hf_terrains.stones_bridge_terrain)


@dataclass(kw_only=True)
class HfRandomUniformTerrainCfg(_AMEHeightFieldCfg):
  noise_range: tuple[float, float]
  noise_step: float = 0.005
  downsampled_scale: float = 0.05
  generate = staticmethod(loco_hf_terrains.random_uniform_terrain)


@dataclass(kw_only=True)
class HfPyramidSlopedTerrainCfg(_AMEHeightFieldCfg):
  slope_range: tuple[float, float]
  platform_width: float = 1.0
  inverted: bool = False
  generate = staticmethod(loco_hf_terrains.pyramid_sloped_terrain)


@dataclass(kw_only=True)
class HfSteppingStonesTerrainCfg(_AMEHeightFieldCfg):
  stone_height_max: float
  stone_width_range: tuple[float, float]
  stone_distance_range: tuple[float, float]
  holes_depth: float = -10.0
  platform_width: float = 1.0
  generate = staticmethod(loco_hf_terrains.stepping_stones_terrain)


@dataclass(kw_only=True)
class HfConcentricGapTerrainCfg(_AMEHeightFieldCfg):
  gap_width_range: tuple[float, float]
  ground_width_range: tuple[float, float]
  ground_height_max: float
  gap_depth: float = -2.0
  platform_width: float = 1.0
  generate = staticmethod(loco_hf_terrains.concentric_gap_terrain)


@dataclass(kw_only=True)
class HfDoubleColumnStakesTerrainCfg(_AMEHeightFieldCfg):
  stake_height_max: float
  stake_side_range: tuple[float, float]
  stake_gap_range: tuple[float, float]
  column_gap_range: tuple[float, float]
  column_jitter: float = 0.0
  holes_depth: float = -2.0
  platform_width: float = 1.0
  generate = staticmethod(loco_hf_terrains.double_column_stakes_terrain)


@dataclass(kw_only=True)
class HfAlternateColumnStakesTerrainCfg(HfDoubleColumnStakesTerrainCfg):
  generate = staticmethod(loco_hf_terrains.alternate_column_stakes_terrain)


__all__ = ["HfAlternateColumnStakesTerrainCfg", "HfConcentricGapTerrainCfg", "HfDoubleColumnStakesTerrainCfg", "HfPyramidSlopedTerrainCfg", "HfRandomUniformTerrainCfg", "HfSteppingStonesTerrainCfg", "HfStonesBridgeTerrainCfg"]
