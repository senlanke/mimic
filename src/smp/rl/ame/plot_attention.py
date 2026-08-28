"""Render recorded AME attention weights as terrain-grid images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro


@dataclass(frozen=True)
class PlotConfig:
  attention_file: str = "attention_weights.npy"
  output_dir: str = "attn_vis"
  map_height: int = 17
  map_width: int = 11


def main(cfg: PlotConfig) -> None:
  attention_weights = np.load(cfg.attention_file)
  output_dir = Path(cfg.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  for step, weights in enumerate(attention_weights):
    attention_map = weights[0, 0][::-1].reshape(
      cfg.map_height, cfg.map_width, order="F"
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(attention_map, cmap="viridis", interpolation="nearest")
    axis.set_title(f"AME attention step {step}")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / f"attn_step_{step:04d}.png")
    plt.close(figure)


if __name__ == "__main__":
  main(tyro.cli(PlotConfig))
