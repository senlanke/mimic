"""Original AME height-field terrain formulas for MJLab."""

from __future__ import annotations

import numpy as np
from scipy import interpolate


def random_uniform_terrain(difficulty: float, cfg, rng: np.random.Generator) -> np.ndarray:
  del difficulty
  width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
  length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
  width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
  length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)
  height_min = int(cfg.noise_range[0] / cfg.vertical_scale)
  height_max = int(cfg.noise_range[1] / cfg.vertical_scale)
  height_step = int(cfg.noise_step / cfg.vertical_scale)
  height_range = np.arange(height_min, height_max + height_step, height_step)
  height_field_downsampled = rng.choice(
    height_range, size=(width_downsampled, length_downsampled)
  )
  x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
  y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
  function = interpolate.RectBivariateSpline(x, y, height_field_downsampled)
  x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
  y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
  return np.rint(function(x_upsampled, y_upsampled)).astype(np.int16)


def pyramid_sloped_terrain(difficulty: float, cfg, rng: np.random.Generator) -> np.ndarray:
  del rng
  if cfg.inverted:
    slope = -cfg.slope_range[0] - difficulty * (
      cfg.slope_range[1] - cfg.slope_range[0]
    )
  else:
    slope = cfg.slope_range[0] + difficulty * (
      cfg.slope_range[1] - cfg.slope_range[0]
    )
  width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
  length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
  height_max = int(slope * cfg.size[0] / 2 / cfg.vertical_scale)
  center_x = int(width_pixels / 2)
  center_y = int(length_pixels / 2)
  x = np.arange(0, width_pixels)
  y = np.arange(0, length_pixels)
  xx, yy = np.meshgrid(x, y, sparse=True)
  xx = ((center_x - np.abs(center_x - xx)) / center_x).reshape(width_pixels, 1)
  yy = ((center_y - np.abs(center_y - yy)) / center_y).reshape(1, length_pixels)
  hf_raw = height_max * xx * yy
  platform_width = int(cfg.platform_width / cfg.horizontal_scale / 2)
  z_pf = hf_raw[width_pixels // 2 - platform_width, length_pixels // 2 - platform_width]
  return np.rint(np.clip(hf_raw, min(0, z_pf), max(0, z_pf))).astype(np.int16)


def stepping_stones_terrain(difficulty: float, cfg, rng: np.random.Generator) -> np.ndarray:
  stone_width = cfg.stone_width_range[1] - difficulty * (
    cfg.stone_width_range[1] - cfg.stone_width_range[0]
  )
  stone_distance = cfg.stone_distance_range[0] + difficulty * (
    cfg.stone_distance_range[1] - cfg.stone_distance_range[0]
  )
  width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
  length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
  stone_distance = int(stone_distance / cfg.horizontal_scale)
  stone_width = int(stone_width / cfg.horizontal_scale)
  stone_height_max = int(cfg.stone_height_max / cfg.vertical_scale)
  holes_depth = int(cfg.holes_depth / cfg.vertical_scale)
  platform_width = int(cfg.platform_width / cfg.horizontal_scale)
  stone_height_range = np.arange(-stone_height_max - 1, stone_height_max, step=1)
  hf_raw = np.full((width_pixels, length_pixels), holes_depth)
  start_x, start_y = 0, 0
  if length_pixels >= width_pixels:
    while start_y < length_pixels:
      stop_y = min(length_pixels, start_y + stone_width)
      start_x = int(rng.integers(0, stone_width))
      stop_x = max(0, start_x - stone_distance)
      hf_raw[0:stop_x, start_y:stop_y] = rng.choice(stone_height_range)
      while start_x < width_pixels:
        stop_x = min(width_pixels, start_x + stone_width)
        hf_raw[start_x:stop_x, start_y:stop_y] = rng.choice(stone_height_range)
        start_x += stone_width + stone_distance
      start_y += stone_width + stone_distance
  else:
    while start_x < width_pixels:
      stop_x = min(width_pixels, start_x + stone_width)
      start_y = int(rng.integers(0, stone_width))
      stop_y = max(0, start_y - stone_distance)
      hf_raw[start_x:stop_x, 0:stop_y] = rng.choice(stone_height_range)
      while start_y < length_pixels:
        stop_y = min(length_pixels, start_y + stone_width)
        hf_raw[start_x:stop_x, start_y:stop_y] = rng.choice(stone_height_range)
        start_y += stone_width + stone_distance
      start_x += stone_width + stone_distance
  x1 = (width_pixels - platform_width) // 2
  x2 = (width_pixels + platform_width) // 2
  y1 = (length_pixels - platform_width) // 2
  y2 = (length_pixels + platform_width) // 2
  hf_raw[x1:x2, y1:y2] = 0
  return np.rint(hf_raw).astype(np.int16)


def stones_bridge_terrain(difficulty: float, cfg, rng: np.random.Generator) -> np.ndarray:
  stone_width = cfg.stone_width_range[1] - difficulty * (cfg.stone_width_range[1] - cfg.stone_width_range[0])
  stone_length = cfg.stone_length_range[1] - difficulty * (cfg.stone_length_range[1] - cfg.stone_length_range[0])
  stone_distance = cfg.stone_distance_range[0] + difficulty * (cfg.stone_distance_range[1] - cfg.stone_distance_range[0])
  stone_lateral_distance = cfg.stone_lateral_distance_range[0] + difficulty * (
    cfg.stone_lateral_distance_range[1] - cfg.stone_lateral_distance_range[0]
  )
  width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
  length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
  stone_distance = int(stone_distance / cfg.horizontal_scale)
  stone_lateral_distance = int(stone_lateral_distance / cfg.horizontal_scale)
  stone_width = int(stone_width / cfg.horizontal_scale)
  stone_length = int(stone_length / cfg.horizontal_scale)
  stone_height_max = int(cfg.stone_height_max / cfg.vertical_scale)
  holes_depth = int(cfg.holes_depth / cfg.vertical_scale)
  platform_width = int(cfg.platform_width / cfg.horizontal_scale)
  stone_height_range = np.arange(-stone_height_max - 1, stone_height_max, step=1)
  hf_raw = np.full((width_pixels, length_pixels), holes_depth)

  start_x = stone_distance
  while start_x < width_pixels:
    stop_x = min(width_pixels, start_x + stone_width)
    start_y = (length_pixels - stone_length) // 2 + int(rng.choice((-stone_lateral_distance, stone_lateral_distance)))
    stop_y = start_y + stone_length
    hf_raw[start_x:stop_x, start_y:stop_y] = rng.choice(stone_height_range)
    start_x = stop_x + stone_distance

  start_y = stone_distance
  while start_y < length_pixels:
    stop_y = min(length_pixels, start_y + stone_width)
    start_x = (width_pixels - stone_length) // 2 + int(rng.choice((-stone_lateral_distance, stone_lateral_distance)))
    stop_x = start_x + stone_length
    hf_raw[start_x:stop_x, start_y:stop_y] = rng.choice(stone_height_range)
    start_y = stop_y + stone_distance

  x1 = (width_pixels - platform_width) // 2
  x2 = (width_pixels + platform_width) // 2
  y1 = (length_pixels - platform_width) // 2
  y2 = (length_pixels + platform_width) // 2
  hf_raw[x1:x2, y1:y2] = 0
  return np.rint(hf_raw).astype(np.int16)


def double_column_stakes_terrain(difficulty: float, cfg, rng: np.random.Generator) -> np.ndarray:
  stake_side = cfg.stake_side_range[1] - difficulty * (cfg.stake_side_range[1] - cfg.stake_side_range[0])
  stake_gap = cfg.stake_gap_range[0] + difficulty * (cfg.stake_gap_range[1] - cfg.stake_gap_range[0])
  column_gap = cfg.column_gap_range[0] + difficulty * (cfg.column_gap_range[1] - cfg.column_gap_range[0])
  width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
  length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
  stake_side_px = max(1, int(stake_side / cfg.horizontal_scale))
  stake_gap_px = max(0, int(stake_gap / cfg.horizontal_scale))
  column_gap_px = max(0, int(column_gap / cfg.horizontal_scale))
  column_jitter_px = max(0, int(cfg.column_jitter / cfg.horizontal_scale))
  stake_height_max_px = max(0, int(cfg.stake_height_max / cfg.vertical_scale))
  holes_depth_px = int(cfg.holes_depth / cfg.vertical_scale)
  platform_width_px = max(1, int(cfg.platform_width / cfg.horizontal_scale))
  hf_raw = np.full((width_pixels, length_pixels), holes_depth_px, dtype=float)
  half_lower = stake_side_px // 2
  half_upper = stake_side_px - half_lower
  center_offset_px = stake_side_px + column_gap_px
  center_x = width_pixels // 2
  center_y = length_pixels // 2
  stake_height_values = np.arange(-stake_height_max_px, stake_height_max_px + 1) if stake_height_max_px > 0 else np.array([0])

  def paint_square(cx: int, cy: int, value: int) -> None:
    if cx < 0 or cx >= width_pixels or cy < 0 or cy >= length_pixels:
      return
    hf_raw[
      max(0, cx - half_lower) : min(width_pixels, cx + half_upper),
      max(0, cy - half_lower) : min(length_pixels, cy + half_upper),
    ] = value

  def place_column_pair(primary_pos: int, along_x: bool) -> None:
    base_offset = max(center_offset_px // 2, half_lower)
    for sign in (-1, 1):
      jitter = rng.integers(-column_jitter_px, column_jitter_px + 1) if column_jitter_px > 0 else 0
      if along_x:
        cy = int(np.clip(center_y + sign * base_offset + jitter, half_lower, length_pixels - half_upper))
        paint_square(primary_pos, cy, int(rng.choice(stake_height_values)))
      else:
        cx = int(np.clip(center_x + sign * base_offset + jitter, half_lower, width_pixels - half_upper))
        paint_square(cx, primary_pos, int(rng.choice(stake_height_values)))

  def extend_from_edge(along_x: bool) -> None:
    start = 0
    step = stake_gap_px + stake_side_px
    while 0 <= start < width_pixels:
      place_column_pair(int(start), along_x)
      start += step

  extend_from_edge(along_x=True)
  extend_from_edge(along_x=False)

  x1 = (width_pixels - platform_width_px) // 2
  x2 = (width_pixels + platform_width_px) // 2
  y1 = (length_pixels - platform_width_px) // 2
  y2 = (length_pixels + platform_width_px) // 2
  hf_raw[x1:x2, y1:y2] = 0
  return np.rint(hf_raw).astype(np.int16)


def concentric_gap_terrain(difficulty: float, cfg, rng: np.random.Generator) -> np.ndarray:
  gap_depth = int(2.0 / cfg.vertical_scale)
  gap_width = int((cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])) / cfg.horizontal_scale)
  ground_width = int((cfg.ground_width_range[0] + (1.0 - difficulty) * (cfg.ground_width_range[1] - cfg.ground_width_range[0])) / cfg.horizontal_scale)
  ground_height_max = int(cfg.ground_height_max / cfg.vertical_scale)
  width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
  length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
  platform_width = int(cfg.platform_width / cfg.horizontal_scale)
  hf_raw = np.zeros((width_pixels, length_pixels))
  start_x, start_y = 0, 0
  stop_x, stop_y = width_pixels, length_pixels
  is_gap = True
  while (stop_x - start_x) > platform_width and (stop_y - start_y) > platform_width:
    if is_gap:
      hf_raw[start_x:stop_x, start_y:stop_y] = -gap_depth
      start_x += gap_width
      stop_x -= gap_width
      start_y += gap_width
      stop_y -= gap_width
    else:
      hf_raw[start_x:stop_x, start_y:stop_y] = rng.integers(-ground_height_max, ground_height_max + 1)
      start_x += ground_width
      stop_x -= ground_width
      start_y += ground_width
      stop_y -= ground_width
    is_gap = not is_gap
  x1 = (width_pixels - platform_width) // 2
  x2 = (width_pixels + platform_width) // 2
  y1 = (length_pixels - platform_width) // 2
  y2 = (length_pixels + platform_width) // 2
  hf_raw[x1:x2, y1:y2] = 0
  return np.rint(hf_raw).astype(np.int16)


def alternate_column_stakes_terrain(difficulty: float, cfg, rng: np.random.Generator) -> np.ndarray:
  stake_side = cfg.stake_side_range[1] - difficulty * (cfg.stake_side_range[1] - cfg.stake_side_range[0])
  stake_gap = cfg.stake_gap_range[0] + difficulty * (cfg.stake_gap_range[1] - cfg.stake_gap_range[0])
  column_gap = cfg.column_gap_range[1] - difficulty * (cfg.column_gap_range[1] - cfg.column_gap_range[0])
  width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
  length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
  stake_side_px = max(1, int(stake_side / cfg.horizontal_scale))
  stake_gap_px = max(0, int(stake_gap / cfg.horizontal_scale))
  column_gap_px = max(0, int(column_gap / cfg.horizontal_scale))
  column_jitter_px = max(0, int(cfg.column_jitter / cfg.horizontal_scale))
  stake_height_max_px = max(0, int(cfg.stake_height_max / cfg.vertical_scale))
  holes_depth_px = int(cfg.holes_depth / cfg.vertical_scale)
  platform_width_px = max(1, int(cfg.platform_width / cfg.horizontal_scale))
  hf_raw = np.full((width_pixels, length_pixels), holes_depth_px, dtype=float)
  half_lower = stake_side_px // 2
  half_upper = stake_side_px - half_lower
  stake_height_values = np.arange(-stake_height_max_px, stake_height_max_px + 1) if stake_height_max_px > 0 else np.array([0])

  def paint_square(cx: int, cy: int, value: int) -> None:
    if cx < 0 or cx >= width_pixels or cy < 0 or cy >= length_pixels:
      return
    hf_raw[
      max(0, cx - half_lower) : min(width_pixels, cx + half_upper),
      max(0, cy - half_lower) : min(length_pixels, cy + half_upper),
    ] = value

  def place_alternate_columns(start_pos: int, along_x: bool) -> None:
    offset = column_gap_px // 2
    step = stake_gap_px + stake_side_px
    while start_pos < (width_pixels if along_x else length_pixels):
      jitter = rng.integers(-column_jitter_px, column_jitter_px + 1) if column_jitter_px > 0 else 0
      height_value = int(rng.choice(stake_height_values))
      if along_x:
        paint_square(start_pos, length_pixels // 2 + offset + jitter, height_value)
      else:
        paint_square(width_pixels // 2 + offset + jitter, start_pos, height_value)
      offset = -offset
      start_pos += step

  place_alternate_columns(0, along_x=True)
  place_alternate_columns(0, along_x=False)
  x1 = (width_pixels - platform_width_px) // 2
  x2 = (width_pixels + platform_width_px) // 2
  y1 = (length_pixels - platform_width_px) // 2
  y2 = (length_pixels + platform_width_px) // 2
  hf_raw[x1:x2, y1:y2] = 0
  return np.rint(hf_raw).astype(np.int16)


__all__ = ["alternate_column_stakes_terrain", "concentric_gap_terrain", "double_column_stakes_terrain", "pyramid_sloped_terrain", "random_uniform_terrain", "stepping_stones_terrain", "stones_bridge_terrain"]
