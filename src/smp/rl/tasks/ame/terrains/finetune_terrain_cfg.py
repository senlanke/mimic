"""Stage-two AME fine-tuning terrain configuration."""

from mjlab import terrains as terrain_gen
from mjlab.terrains import TerrainGeneratorCfg

from .loco_hf_terrains_cfg import (
  HfAlternateColumnStakesTerrainCfg,
  HfConcentricGapTerrainCfg,
  HfDoubleColumnStakesTerrainCfg,
  HfStonesBridgeTerrainCfg,
)
from .rails_terrain_cfg import BoxRailsTerrainCfg

FINETUNE_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0), border_width=50.0, num_rows=10, num_cols=20,
  curriculum=True,
  sub_terrains={
    "pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.1, step_height_range=(0.05, 0.25), step_width=0.3,
      platform_width=3.0, border_width=1.0,
    ),
    "pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.1, step_height_range=(0.05, 0.25), step_width=0.3,
      platform_width=3.0, border_width=1.0,
    ),
    "stakes1": HfDoubleColumnStakesTerrainCfg(
      proportion=0.1, stake_height_max=0.03, stake_side_range=(0.20, 0.40),
      stake_gap_range=(0.1, 0.3), column_gap_range=(0.1, 0.1),
      column_jitter=0.0, holes_depth=-2.0, platform_width=2.0,
      border_width=0.25,
    ),
    "stakes2": HfAlternateColumnStakesTerrainCfg(
      proportion=0.2, stake_height_max=0.03, stake_side_range=(0.20, 0.40),
      stake_gap_range=(0.05, 0.15), column_gap_range=(0.0, 0.2),
      column_jitter=0.0, holes_depth=-2.0, platform_width=2.0,
      border_width=0.25,
    ),
    "stakes3": HfAlternateColumnStakesTerrainCfg(
      proportion=0.2, stake_height_max=0.03, stake_side_range=(0.20, 0.40),
      stake_gap_range=(0.05, 0.25), column_gap_range=(0.3, 0.2),
      column_jitter=0.0, holes_depth=-2.0, platform_width=2.0,
      border_width=0.25,
    ),
    "hf_gaps": HfConcentricGapTerrainCfg(
      proportion=0.1, gap_width_range=(0.2, 0.6), ground_width_range=(0.5, 0.5),
      ground_height_max=0.03, gap_depth=-2.0, platform_width=2.0,
      border_width=0.25,
    ),
    "stonebridge": HfStonesBridgeTerrainCfg(
      proportion=0.1, stone_height_max=0.03, stone_width_range=(0.25, 0.35),
      stone_length_range=(0.6, 1.0), stone_distance_range=(0.3, 0.5),
      stone_lateral_distance_range=(0.0, 0.0), holes_depth=-2.0,
      platform_width=2.0, border_width=0.25,
    ),
    "rails": BoxRailsTerrainCfg(
      proportion=0.1, rail_height_range=(0.25, 0.05),
      rail_thickness_range=(0.1, 0.3), platform_width=2.0,
    ),
  },
)

__all__ = ["FINETUNE_ROUGH_TERRAINS_CFG"]
