"""MJLab runner for AME's original checkpoint layout."""

from __future__ import annotations

import torch
from mjlab.rl.runner import MjlabOnPolicyRunner


class AMERunner(MjlabOnPolicyRunner):
  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    load_iteration = self.alg.load(checkpoint, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = checkpoint["iter"]
    return checkpoint["infos"]


__all__ = ["AMERunner"]
