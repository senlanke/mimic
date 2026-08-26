# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University).

"""CMoE environment step order on the MJLab manager stack."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs import types

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def cmoe_step(
  self: "ManagerBasedRlEnv", action: torch.Tensor
) -> types.VecEnvStepReturn:
  """Run command and perturbation callbacks before termination and reward."""
  if not self.cfg.auto_reset and torch.any(self._manual_reset_pending):
    pending_ids = self._manual_reset_pending.nonzero(as_tuple=False).squeeze(-1)
    raise RuntimeError(
      f"Environments {pending_ids.cpu().tolist()} must be reset via "
      "reset(env_ids=...) before calling step() again when auto_reset=False."
    )

  self.extras["log"] = dict()
  self.action_manager.process_action(action.to(self.device))

  for _ in range(self.cfg.decimation):
    self._sim_step_counter += 1
    self.action_manager.apply_action()
    self.scene.write_data_to_sim()
    self.sim.step()
    self.scene.update(dt=self.physics_dt)
    self.metrics_manager.compute_substep()

  self.episode_length_buf += 1
  self.common_step_counter += 1
  self.sim.forward()

  self.command_manager.compute(dt=self.step_dt)
  if "step" in self.event_manager.available_modes:
    self.event_manager.apply(mode="step", dt=self.step_dt)
  if "interval" in self.event_manager.available_modes:
    self.event_manager.apply(mode="interval", dt=self.step_dt)
  self.sim.forward()

  self.reset_buf = self.termination_manager.compute()
  self.reset_terminated = self.termination_manager.terminated
  self.reset_time_outs = self.termination_manager.time_outs
  self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
  self.metrics_manager.compute()

  reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
  if self.cfg.auto_reset and len(reset_env_ids) > 0:
    self.recorder_manager.record_pre_reset(reset_env_ids)
    self._reset_idx(reset_env_ids)
    self.scene.write_data_to_sim()
    self.sim.forward()

  self.sim.sense()
  self.obs_buf = self.observation_manager.compute(update_history=True)

  if self.cfg.auto_reset and len(reset_env_ids) > 0:
    self.recorder_manager.record_post_reset(reset_env_ids)
  elif len(reset_env_ids) > 0:
    self._manual_reset_pending[reset_env_ids] = True

  self.recorder_manager.record_post_step()
  return (
    self.obs_buf,
    self.reward_buf,
    self.reset_terminated,
    self.reset_time_outs,
    self.extras,
  )


__all__ = ["cmoe_step"]
