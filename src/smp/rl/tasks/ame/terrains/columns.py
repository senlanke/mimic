"""Map Isaac Lab's terrain columns onto MJLab's one-column-per-entry generator."""

import copy

import numpy as np
from mjlab.terrains import TerrainGeneratorCfg


def expand_terrain_columns(cfg: TerrainGeneratorCfg) -> None:
  terrains = list(cfg.sub_terrains.items())
  proportions = np.array([terrain.proportion for _, terrain in terrains])
  cumulative = np.cumsum(proportions / proportions.sum())
  columns = {}
  for column in range(cfg.num_cols):
    index = np.searchsorted(cumulative, column / cfg.num_cols + 0.001, side="right")
    name, terrain = terrains[index]
    columns[f"{name}_{column}"] = copy.deepcopy(terrain)
    columns[f"{name}_{column}"].proportion = 1.0 / cfg.num_cols
  cfg.sub_terrains = columns
