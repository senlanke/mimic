"""Stage-one AME rough terrain configuration."""

from mjlab import terrains as terrain_gen
from mjlab.terrains import TerrainGeneratorCfg

from .loco_hf_terrains_cfg import (
  HfConcentricGapTerrainCfg,
  HfPyramidSlopedTerrainCfg,
  HfRandomUniformTerrainCfg,
  HfSteppingStonesTerrainCfg,
)

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=50.0,
  num_rows=10,
  num_cols=20,
  curriculum=True,
  sub_terrains={
    "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.1, step_height_range=(0.05, 0.2), step_width=0.3,
      platform_width=3.0, border_width=1.0,
    ),
    "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.1, step_height_range=(0.05, 0.2), step_width=0.3,
      platform_width=3.0, border_width=1.0,
    ),
    "boxes": terrain_gen.BoxRandomGridTerrainCfg(
      proportion=0.1, grid_width=0.45, grid_height_range=(0.05, 0.2),
      platform_width=2.0,
    ),
    "random_rough": HfRandomUniformTerrainCfg(
      proportion=0.1, noise_range=(0.02, 0.10), noise_step=0.02,
      downsampled_scale=0.1, border_width=0.25,
    ),
    "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
      proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0,
      border_width=0.25,
    ),
    "hf_pyramid_slope_inv": HfPyramidSlopedTerrainCfg(
      proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0,
      border_width=0.25, inverted=True,
    ),
    "hf_steppingstones": HfSteppingStonesTerrainCfg(
      proportion=0.2, stone_height_max=0.05,
      stone_width_range=(0.25, 0.5), stone_distance_range=(0.05, 0.25),
      platform_width=2.0, holes_depth=-2.0, border_width=0.25,
    ),
    "hf_gaps": HfConcentricGapTerrainCfg(
      proportion=0.2, gap_width_range=(0.1, 0.5), ground_width_range=(0.5, 0.5),
      ground_height_max=0.025, gap_depth=-2.0, platform_width=2.0,
      border_width=0.25,
    ),
  },
)

__all__ = ["ROUGH_TERRAINS_CFG"]
