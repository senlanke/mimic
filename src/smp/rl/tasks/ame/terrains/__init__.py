"""AME locomotion terrain configurations."""

from .finetune_terrain_cfg import FINETUNE_ROUGH_TERRAINS_CFG
from .loco_hf_terrains_cfg import *
from .rails_terrain_cfg import BoxRailsTerrainCfg
from .terrain_cfg import ROUGH_TERRAINS_CFG

__all__ = ["FINETUNE_ROUGH_TERRAINS_CFG", "ROUGH_TERRAINS_CFG"]
