"""AME play terrain matching the source play configuration."""

from mjlab.terrains import TerrainGeneratorCfg

from .loco_hf_terrains_cfg import HfAlternateColumnStakesTerrainCfg

PLAY_TERRAIN_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=50.0,
  num_rows=1,
  num_cols=1,
  curriculum=False,
  sub_terrains={
    "stakes": HfAlternateColumnStakesTerrainCfg(
      proportion=0.5,
      stake_height_max=0.0,
      stake_side_range=(0.2, 0.2),
      stake_gap_range=(0.3, 0.3),
      column_gap_range=(0.3, 0.3),
      column_jitter=0.0,
      holes_depth=-2.0,
      platform_width=2.0,
      border_width=0.25,
    )
  },
)

__all__ = ["PLAY_TERRAIN_CFG"]
