# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University).
#
# The terrain formulas below are adapted from CMoE/legged_gym and the
# BSD-licensed Isaac Gym terrain utilities.

"""The original CMoE heightfield terrain set for MJLab."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import mujoco
import numpy as np
import torch
from mjlab.terrains.terrain_entity import TerrainEntityCfg
from mjlab.terrains.terrain_generator import (
  TerrainGeneratorCfg,
  TerrainOutput,
)
from scipy import interpolate, ndimage

from smp.rl.tasks.cmoe.height_field.hf_terrains_cfg import (
  HfTerrainBaseCfg,
  _height_field_to_hfield_surface_mesh,
  _height_field_to_output,
)

_HORIZONTAL_SCALE = 0.05
_COLLISION_HORIZONTAL_SCALE = 0.1
_COLLISION_STRIDE = 2
_VERTICAL_SCALE = 0.005
_DOWNSAMPLED_SCALE = 0.075
_TERRAIN_SIZE = (10.0, 10.0)
_ROUGH_HEIGHT = (0.01, 0.03)

# The original 40 terrain columns.  The first four columns are rough slopes,
# followed by the four stair-up, four stair-down, four discrete, twelve gap,
# four hurdle, four mixed-obstacle and four narrow-stair columns.
CMOE_COLUMN_KINDS = (
  ("rough_neg",) * 2
  + ("rough_pos",) * 2
  + ("stairs_up",) * 4
  + ("stairs_down",) * 4
  + ("discrete",) * 4
  + ("parkour_gap",) * 12
  + ("parkour_hurdle",) * 4
  + ("mix",) * 4
  + ("narrow_stairs",) * 4
)
CMOE_PLAY_COLUMN_KINDS = (
  "rough_neg",
  "stairs_up",
  "stairs_down",
  "discrete",
  "parkour_gap",
  "parkour_gap",
  "parkour_gap",
  "parkour_hurdle",
  "mix",
  "narrow_stairs",
)
CMOE_COURSE_KINDS = (
  "rough_neg",
  "rough_pos",
  "stairs_up",
  "stairs_down",
  "discrete",
  "parkour_gap",
  "parkour_hurdle",
  "mix",
  "narrow_stairs",
)
CMOE_TERRAIN_CLASS = {
  "rough_neg": 1,
  "rough_pos": 1,
  "stairs_up": 2,
  "stairs_down": 3,
  "discrete": 4,
  "parkour_gap": 5,
  "parkour_hurdle": 8,
  "mix": 9,
  "narrow_stairs": 10,
}


def _random_uniform(
  raw: np.ndarray, difficulty: float, rng: np.random.Generator
) -> None:
  max_height = (_ROUGH_HEIGHT[1] - _ROUGH_HEIGHT[0]) * difficulty + _ROUGH_HEIGHT[0]
  height = rng.uniform(_ROUGH_HEIGHT[0], max_height)
  min_height = int(-height / _VERTICAL_SCALE)
  max_height = int(height / _VERTICAL_SCALE)
  step = int(0.005 / _VERTICAL_SCALE)
  heights = np.arange(min_height, max_height + step, step)
  downsampled = rng.choice(
    heights,
    size=(
      int(raw.shape[0] * _HORIZONTAL_SCALE / _DOWNSAMPLED_SCALE),
      int(raw.shape[1] * _HORIZONTAL_SCALE / _DOWNSAMPLED_SCALE),
    ),
  )
  x = np.linspace(0, raw.shape[0] * _HORIZONTAL_SCALE, downsampled.shape[0])
  y = np.linspace(0, raw.shape[1] * _HORIZONTAL_SCALE, downsampled.shape[1])
  spline = interpolate.RectBivariateSpline(x, y, downsampled, kx=1, ky=1)
  xx = np.linspace(0, raw.shape[0] * _HORIZONTAL_SCALE, raw.shape[0])
  yy = np.linspace(0, raw.shape[1] * _HORIZONTAL_SCALE, raw.shape[1])
  raw += np.rint(spline(xx, yy)).astype(np.int16)


def _pyramid_slope(raw: np.ndarray, slope: float) -> None:
  width, length = raw.shape
  x = np.arange(width)
  y = np.arange(length)
  center_x = width // 2
  center_y = length // 2
  xx, yy = np.meshgrid(x, y, sparse=True)
  xx = ((center_x - np.abs(center_x - xx)) / center_x).reshape(width, 1)
  yy = ((center_y - np.abs(center_y - yy)) / center_y).reshape(1, length)
  max_height = int(slope * (_HORIZONTAL_SCALE / _VERTICAL_SCALE) * (width / 2))
  raw += (max_height * xx * yy).astype(raw.dtype)
  platform = int(3.0 / _HORIZONTAL_SCALE / 2)
  x1 = width // 2 - platform
  y1 = length // 2 - platform
  raw[:] = np.clip(raw, min(raw[x1, y1], 0), max(raw[x1, y1], 0))


def _pyramid_stairs(raw: np.ndarray, step_height: float) -> None:
  step_width = int(0.30 / _HORIZONTAL_SCALE)
  step_height = int(step_height / _VERTICAL_SCALE)
  platform = int(3.0 / _HORIZONTAL_SCALE)
  border = int(0.5 / _HORIZONTAL_SCALE)
  start_x, stop_x = border, raw.shape[0] - border
  start_y, stop_y = border, raw.shape[1] - border
  height = 0
  while (stop_x - start_x) > platform and (stop_y - start_y) > platform:
    start_x += step_width
    stop_x -= step_width
    start_y += step_width
    stop_y -= step_width
    height += step_height
    raw[start_x:stop_x, start_y:stop_y] = height


def _discrete_obstacles(
  raw: np.ndarray, difficulty: float, rng: np.random.Generator
) -> None:
  max_height = int((0.05 + difficulty * 0.1) / _VERTICAL_SCALE)
  min_size = int(1.0 / _HORIZONTAL_SCALE)
  max_size = int(2.0 / _HORIZONTAL_SCALE)
  width_range = np.arange(min_size, max_size, 4)
  height_range = np.array([-max_height, -max_height // 2, max_height // 2, max_height])
  for _ in range(20):
    width = int(rng.choice(width_range))
    length = int(rng.choice(width_range))
    start_i = int(rng.choice(np.arange(0, raw.shape[0] - width, 4)))
    start_j = int(rng.choice(np.arange(0, raw.shape[1] - length, 4)))
    raw[start_i : start_i + width, start_j : start_j + length] = rng.choice(
      height_range
    )
  platform = int(3.0 / _HORIZONTAL_SCALE)
  x1, x2 = (raw.shape[0] - platform) // 2, (raw.shape[0] + platform) // 2
  y1, y2 = (raw.shape[1] - platform) // 2, (raw.shape[1] + platform) // 2
  raw[x1:x2, y1:y2] = 0


def _parkour_gap(raw: np.ndarray, difficulty: float, rng: np.random.Generator) -> None:
  mid_y = raw.shape[1] // 2
  platform_len = int(1.0 / _HORIZONTAL_SCALE)
  gap_depth = -round(rng.uniform(0.5, 1.5) / _VERTICAL_SCALE)
  half_valid_width = round(
    rng.uniform(1 - 0.5 * difficulty, 1.5 - 0.5 * difficulty) / _HORIZONTAL_SCALE
  )
  raw[:platform_len, :] = 0
  gap_size = round((0.1 + 0.7 * difficulty) / _HORIZONTAL_SCALE)
  dis_x_min = round(0.8 / _HORIZONTAL_SCALE) + gap_size
  dis_x_max = round(1.4 / _HORIZONTAL_SCALE) + gap_size
  dis_x = platform_len
  last_dis_x = dis_x
  for _ in range(4):
    dis_x += int(rng.integers(dis_x_min, dis_x_max))
    raw[dis_x - gap_size // 2 : dis_x + gap_size // 2, :] = gap_depth
    rand_y = int(rng.integers(-2, 2))
    raw[last_dis_x:dis_x, : mid_y + rand_y - half_valid_width] = gap_depth
    raw[last_dis_x:dis_x, mid_y + rand_y + half_valid_width :] = gap_depth
    last_dis_x = dis_x
  pad = int(0.1 // _HORIZONTAL_SCALE)
  raw[:, :pad] = 0
  raw[:, -pad:] = 0
  raw[:pad, :] = 0
  raw[-pad:, :] = 0


def _parkour_hurdle(
  raw: np.ndarray, difficulty: float, rng: np.random.Generator
) -> None:
  platform_len = round(2.0 / _HORIZONTAL_SCALE)
  stone_len = round((0.1 + 0.2 * difficulty) / _HORIZONTAL_SCALE)
  height_min = round((0.2 * difficulty) / _VERTICAL_SCALE)
  height_max = round((0.15 + 0.25 * difficulty) / _VERTICAL_SCALE)
  dis_x_min = round(1.2 / _HORIZONTAL_SCALE)
  dis_x_max = round(2.0 / _HORIZONTAL_SCALE)
  raw[:platform_len, :] = 0
  dis_x = platform_len
  for _ in range(4):
    dis_x += int(rng.integers(dis_x_min, dis_x_max))
    raw[dis_x - stone_len // 2 : dis_x + stone_len // 2, :] = rng.integers(
      height_min, height_max
    )
  pad = int(0.1 // _HORIZONTAL_SCALE)
  raw[:, :pad] = 0
  raw[:, -pad:] = 0
  raw[:pad, :] = 0
  raw[-pad:, :] = 0


def _mix_obstacles(
  raw: np.ndarray, difficulty: float, rng: np.random.Generator
) -> None:
  diff = difficulty * 1.1
  gap_depth = -int(rng.integers(100, 300))
  raw[:40, :] = 0
  raw[30:36, :] = int(30 * diff)
  raw[36:42, :] = int(60 * diff)
  raw[42:48, :] = int(90 * diff)
  raw[48:60, :] = int(120 * diff)
  gap_start = 60
  gap_end = 72 - round(10 - diff * 10)
  raw[gap_start:gap_end, :] = gap_depth
  raw[gap_end:84, :] = int(120 * diff)
  raw[86:96, :] = int(96 * diff)
  raw[96:99, :] = int(170 * diff)
  raw[99:111, :] = int(120 * diff)
  gap_start = 111
  gap_end = 123 - round(10 - diff * 10)
  raw[gap_start:gap_end, :] = gap_depth
  raw[gap_end:140, :] = int(120 * diff)
  raw[140:160, :] = int(60 * diff)
  mid_y = raw.shape[1] // 2
  raw[:, mid_y + 20 :] = gap_depth
  raw[:, : mid_y - 20] = gap_depth


def _narrow_stairs(
  raw: np.ndarray, difficulty: float, rng: np.random.Generator
) -> None:
  mid_y = raw.shape[1] // 2
  num_stones = 24
  step_height = round(0.25 * difficulty / _VERTICAL_SCALE)
  half_valid_width = round((1.0 - 0.5 * difficulty) / _HORIZONTAL_SCALE)
  platform_len = round(2.5 / _HORIZONTAL_SCALE)
  raw[:platform_len, :] = 0
  dis_x = platform_len
  stair_height = 0
  gap_depth = -int(rng.integers(10, 300))
  for i in range(num_stones):
    rand_x = 6
    if i < num_stones // 2 - 2:
      stair_height += step_height
    elif i > num_stones // 2 + 2:
      stair_height -= step_height
    raw[dis_x : dis_x + rand_x, :] = stair_height
    raw[dis_x : dis_x + rand_x, : mid_y - half_valid_width] = gap_depth
    raw[dis_x : dis_x + rand_x, mid_y + half_valid_width :] = gap_depth
    dis_x += rand_x
  pad = int(0.1 // _HORIZONTAL_SCALE)
  raw[:, :pad] = 0
  raw[:, -pad:] = 0
  raw[:pad, :] = 0
  raw[-pad:, :] = 0


def _make_raw(kind: str, difficulty: float, rng: np.random.Generator) -> np.ndarray:
  raw = np.zeros(
    (
      int(_TERRAIN_SIZE[0] / _HORIZONTAL_SCALE),
      int(_TERRAIN_SIZE[1] / _HORIZONTAL_SCALE),
    ),
    dtype=np.int16,
  )
  if kind.startswith("rough"):
    _pyramid_slope(raw, (-1 if kind == "rough_neg" else 1) * 0.4 * difficulty)
    _random_uniform(raw, difficulty, rng)
  elif kind == "stairs_up":
    _pyramid_stairs(raw, -(0.05 + 0.18 * difficulty))
    _random_uniform(raw, difficulty, rng)
  elif kind == "stairs_down":
    _pyramid_stairs(raw, 0.05 + 0.18 * difficulty)
    _random_uniform(raw, difficulty, rng)
  elif kind == "discrete":
    _discrete_obstacles(raw, difficulty, rng)
    _random_uniform(raw, difficulty, rng)
  elif kind == "parkour_gap":
    _parkour_gap(raw, difficulty, rng)
    _random_uniform(raw, difficulty, rng)
  elif kind == "parkour_hurdle":
    _parkour_hurdle(raw, difficulty, rng)
    _random_uniform(raw, difficulty, rng)
  elif kind == "mix":
    _mix_obstacles(raw, difficulty, rng)
    _random_uniform(raw, difficulty, rng)
  elif kind == "narrow_stairs":
    _narrow_stairs(raw, difficulty, rng)
    _random_uniform(raw, difficulty, rng)
  return raw


@dataclass(kw_only=True)
class CMoETerrainCfg(HfTerrainBaseCfg):
  kind: str
  horizontal_scale: float = _HORIZONTAL_SCALE
  vertical_scale: float = _VERTICAL_SCALE
  slope_threshold: float = 1.5
  height_fields: list[np.ndarray] = field(default_factory=list, init=False, repr=False)

  def function(
    self,
    difficulty: float,
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
  ) -> TerrainOutput:
    difficulty = np.floor(difficulty * 10.0) / 10.0
    raw = _make_raw(self.kind, difficulty, rng)
    self.height_fields.append(raw)
    collision_cfg = replace(
      self,
      horizontal_scale=_COLLISION_HORIZONTAL_SCALE,
    )
    output = _height_field_to_output(
      heights=raw[::_COLLISION_STRIDE, ::_COLLISION_STRIDE].T,
      cfg=collision_cfg,
      spec=spec,
      rng=rng,
    )
    output.instinct_surface_mesh = _height_field_to_hfield_surface_mesh(raw.T, self)
    parkour = self.kind in {
      "parkour_gap",
      "parkour_hurdle",
      "mix",
      "narrow_stairs",
    }
    if parkour:
      output.origin = np.array([0.75, self.size[1] * 0.5, 0.0])
    else:
      x1 = int((self.size[0] * 0.5 - 1.0) / self.horizontal_scale)
      x2 = int((self.size[0] * 0.5 + 1.0) / self.horizontal_scale)
      y1 = int((self.size[1] * 0.5 - 1.0) / self.horizontal_scale)
      y2 = int((self.size[1] * 0.5 + 1.0) / self.horizontal_scale)
      output.origin = np.array(
        [
          self.size[0] * 0.5,
          self.size[1] * 0.5,
          raw[x1:x2, y1:y2].max() * self.vertical_scale,
        ]
      )
    return output


@dataclass(kw_only=True)
class CMoEPlayCourseCfg(HfTerrainBaseCfg):
  difficulty: float = 0.5
  horizontal_scale: float = _HORIZONTAL_SCALE
  vertical_scale: float = _VERTICAL_SCALE
  slope_threshold: float = 1.5
  height_fields: list[np.ndarray] = field(default_factory=list, init=False, repr=False)

  def function(
    self,
    difficulty: float,
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
  ) -> TerrainOutput:
    del difficulty
    difficulty = np.floor(self.difficulty * 10.0) / 10.0
    raw = np.concatenate(
      [_make_raw(kind, difficulty, rng) for kind in CMOE_COURSE_KINDS], axis=0
    )
    self.height_fields.append(raw)
    collision_cfg = replace(
      self,
      horizontal_scale=_COLLISION_HORIZONTAL_SCALE,
    )
    output = _height_field_to_output(
      heights=raw[::_COLLISION_STRIDE, ::_COLLISION_STRIDE].T,
      cfg=collision_cfg,
      spec=spec,
      rng=rng,
    )
    output.instinct_surface_mesh = _height_field_to_hfield_surface_mesh(raw.T, self)
    output.origin = np.array([0.75, self.size[1] * 0.5, 0.0])
    return output


@dataclass(kw_only=True)
class CMoEPlayTerrainEntityCfg(TerrainEntityCfg):
  def __setattr__(self, name: str, value) -> None:
    super().__setattr__(name, value)
    if name == "num_envs" and self.terrain_generator is not None:
      self.terrain_generator.num_cols = value


def cmoe_terrain_generator_cfg(play: bool = False) -> TerrainGeneratorCfg:
  column_kinds = CMOE_PLAY_COLUMN_KINDS if play else CMOE_COLUMN_KINDS
  sub_terrains = {
    f"{kind}_{column:02d}": CMoETerrainCfg(
      proportion=1.0 / len(column_kinds),
      kind=kind,
    )
    for column, kind in enumerate(column_kinds)
  }
  return TerrainGeneratorCfg(
    seed=None,
    curriculum=True,
    size=_TERRAIN_SIZE,
    border_width=25.0,
    num_rows=10,
    num_cols=len(column_kinds),
    color_scheme="none",
    sub_terrains=sub_terrains,
    add_lights=False,
  )


def cmoe_play_course_terrain_cfg(
  num_envs: int = 1,
  difficulty: float = 0.5,
) -> CMoEPlayTerrainEntityCfg:
  course = CMoEPlayCourseCfg(
    proportion=1.0,
    difficulty=difficulty,
  )
  return CMoEPlayTerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      seed=None,
      curriculum=False,
      size=(_TERRAIN_SIZE[0] * len(CMOE_COURSE_KINDS), _TERRAIN_SIZE[1]),
      border_width=25.0,
      num_rows=1,
      num_cols=num_envs,
      color_scheme="none",
      sub_terrains={"course": course},
      difficulty_range=(difficulty, difficulty),
      add_lights=False,
    ),
  )


def cmoe_height_field(env) -> torch.Tensor:
  if not hasattr(env, "cmoe_terrain_data"):
    generator = env.cfg.scene.terrain.terrain_generator
    columns = [cfg.height_fields for cfg in generator.sub_terrains.values()]
    if len(columns) == 1 and isinstance(
      next(iter(generator.sub_terrains.values())), CMoEPlayCourseCfg
    ):
      terrain = np.concatenate(columns[0], axis=1)
    else:
      patches = np.asarray(columns).transpose(1, 2, 0, 3)
      terrain = patches.reshape(
        generator.num_rows * patches.shape[1], len(columns) * patches.shape[3]
      )
    border = round(generator.border_width / _HORIZONTAL_SCALE)
    terrain = np.pad(terrain, border)
    threshold = 1.5 * _HORIZONTAL_SCALE / _VERTICAL_SCALE
    move_x = np.zeros_like(terrain, dtype=np.int8)
    move_y = np.zeros_like(terrain, dtype=np.int8)
    move_corner = np.zeros_like(terrain, dtype=np.int8)
    move_x[:-1] += terrain[1:] - terrain[:-1] > threshold
    move_x[1:] -= terrain[:-1] - terrain[1:] > threshold
    move_y[:, :-1] += terrain[:, 1:] - terrain[:, :-1] > threshold
    move_y[:, 1:] -= terrain[:, :-1] - terrain[:, 1:] > threshold
    move_corner[:-1, :-1] += terrain[1:, 1:] - terrain[:-1, :-1] > threshold
    move_corner[1:, 1:] -= terrain[:-1, :-1] - terrain[1:, 1:] > threshold
    edge_mask = ndimage.binary_dilation(
      (move_x != 0) | (move_y != 0) | (move_corner != 0),
      structure=np.ones((3, 3)),
    )
    env.cmoe_terrain_data = (
      torch.as_tensor(terrain, dtype=torch.int16, device=env.device),
      torch.as_tensor(edge_mask, dtype=torch.bool, device=env.device),
    )
  return env.cmoe_terrain_data[0]


def cmoe_edge_mask(env) -> torch.Tensor:
  cmoe_height_field(env)
  return env.cmoe_terrain_data[1]


def cmoe_height_indices(env, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  generator = env.cfg.scene.terrain.terrain_generator
  border = generator.border_width
  grid_x = generator.num_rows * generator.size[0]
  grid_y = generator.num_cols * generator.size[1]
  px = ((points[..., 0] + grid_x / 2 + border) / _HORIZONTAL_SCALE).long()
  py = ((points[..., 1] + grid_y / 2 + border) / _HORIZONTAL_SCALE).long()
  height_field = cmoe_height_field(env)
  return px.clamp(0, height_field.shape[0] - 2), py.clamp(0, height_field.shape[1] - 2)


def cmoe_scan_heights(env, points: torch.Tensor) -> torch.Tensor:
  height_field = cmoe_height_field(env)
  px, py = cmoe_height_indices(env, points)
  heights = torch.stack(
    (
      height_field[px, py],
      height_field[px, py],
      height_field[px + 1, py],
      height_field[px - 1, py],
    )
  ).float()
  return heights.mean(dim=0) * _VERTICAL_SCALE


def cmoe_foot_heights(env, points: torch.Tensor) -> torch.Tensor:
  height_field = cmoe_height_field(env)
  px, py = cmoe_height_indices(env, points)
  heights = torch.stack(
    (
      height_field[px, py],
      height_field[px + 1, py],
      height_field[px, py + 1],
    )
  ).float()
  return points[..., 2] - heights.mean(dim=0) * _VERTICAL_SCALE


def cmoe_feet_at_edge(env, points: torch.Tensor) -> torch.Tensor:
  edge_mask = cmoe_edge_mask(env)
  generator = env.cfg.scene.terrain.terrain_generator
  border = generator.border_width
  grid_x = generator.num_rows * generator.size[0]
  grid_y = generator.num_cols * generator.size[1]
  px = ((points[..., 0] + grid_x / 2 + border) / _HORIZONTAL_SCALE).round().long()
  py = ((points[..., 1] + grid_y / 2 + border) / _HORIZONTAL_SCALE).round().long()
  px = px.clamp(0, edge_mask.shape[0] - 1)
  py = py.clamp(0, edge_mask.shape[1] - 1)
  return edge_mask[px, py]


def cmoe_terrain_class(env) -> torch.Tensor:
  generator = env.cfg.scene.terrain.terrain_generator
  sub_terrains = list(generator.sub_terrains.values())
  if len(sub_terrains) == 1 and isinstance(sub_terrains[0], CMoEPlayCourseCfg):
    classes = torch.tensor(
      [CMOE_TERRAIN_CLASS[kind] for kind in CMOE_COURSE_KINDS],
      device=env.device,
    )
    course_start = -generator.num_rows * generator.size[0] * 0.5
    segment_length = generator.size[0] / len(CMOE_COURSE_KINDS)
    segment = torch.floor(
      (env.scene["robot"].data.root_link_pos_w[:, 0] - course_start)
      / segment_length
    ).long()
    return classes[segment]

  columns = env.scene.terrain.terrain_types
  classes = torch.tensor(
    [CMOE_TERRAIN_CLASS[cfg.kind] for cfg in sub_terrains],
    device=env.device,
  )
  return classes[columns]


__all__ = [
  "CMOE_COLUMN_KINDS",
  "CMOE_PLAY_COLUMN_KINDS",
  "CMOE_TERRAIN_CLASS",
  "CMoETerrainCfg",
  "cmoe_edge_mask",
  "cmoe_feet_at_edge",
  "cmoe_foot_heights",
  "cmoe_height_field",
  "cmoe_scan_heights",
  "cmoe_terrain_class",
  "cmoe_terrain_generator_cfg",
]
