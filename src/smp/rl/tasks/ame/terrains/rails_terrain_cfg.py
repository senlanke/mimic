"""MJLab translation of Isaac Lab's AME rails terrain."""

from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_border, make_plane


@dataclass(kw_only=True)
class BoxRailsTerrainCfg(SubTerrainCfg):
  rail_thickness_range: tuple[float, float]
  rail_height_range: tuple[float, float]
  platform_width: float = 1.0

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng
    rail_height = self.rail_height_range[1] - difficulty * (
      self.rail_height_range[1] - self.rail_height_range[0]
    )
    rail_1_thickness, rail_2_thickness = self.rail_thickness_range
    rail_center = (self.size[0] * 0.5, self.size[1] * 0.5, rail_height * 0.5)
    body = spec.body("terrain")
    geometries = []

    rail_1_inner = (self.platform_width, self.platform_width)
    rail_1_outer = (
      self.platform_width + 2.0 * rail_1_thickness,
      self.platform_width + 2.0 * rail_1_thickness,
    )
    geometries.extend(
      TerrainGeometry(geom=geom)
      for geom in make_border(body, rail_1_outer, rail_1_inner, rail_height, rail_center)
    )

    rail_2_inner = (
      self.platform_width + (self.size[0] - self.platform_width) * 0.6,
      self.platform_width + (self.size[1] - self.platform_width) * 0.6,
    )
    rail_2_outer = (
      rail_2_inner[0] + 2.0 * rail_2_thickness,
      rail_2_inner[1] + 2.0 * rail_2_thickness,
    )
    geometries.extend(
      TerrainGeometry(geom=geom)
      for geom in make_border(body, rail_2_outer, rail_2_inner, rail_height, rail_center)
    )
    geometries.extend(
      TerrainGeometry(geom=geom)
      for geom in make_plane(body, self.size, 0.0, center_zero=False)
    )
    origin = np.array((self.size[0] * 0.5, self.size[1] * 0.5, 0.0))
    return TerrainOutput(origin=origin, geometries=geometries)


__all__ = ["BoxRailsTerrainCfg"]
