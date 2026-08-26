# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University).
# Adapted from rsl_rl (BSD-3-Clause, Copyright (c) 2021 ETH Zurich, Nikita Rudin
# and NVIDIA CORPORATION & AFFILIATES). See rsl_rl/LICENSE.

"""CMoE runner with explicit terminal-observation handling."""

from __future__ import annotations

import os
import time
from types import MethodType
from typing import Any

import torch
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.env import VecEnv
from tensordict import TensorDict


_PROPRIO_FIELDS = (
  ("command", 0, 3),
  ("base_ang_vel", 3, 6),
  ("projected_gravity", 6, 9),
  ("joint_pos", 9, 21),
  ("joint_vel", 21, 33),
  ("action", 33, 45),
)


def _check_observation_slice(
  tensor: torch.Tensor,
  name: str,
  start: int,
  end: int,
  stage: str,
  actions: torch.Tensor | None = None,
) -> None:
  value = tensor[:, start:end]
  invalid = ~torch.isfinite(value)
  if not invalid.any():
    return
  indices = invalid.nonzero(as_tuple=False)
  env_ids = indices[:, 0].unique().cpu().tolist()
  dimensions = indices[:, 1].unique().cpu().tolist()
  action_values = (
    f", actions={actions[env_ids].cpu().tolist()}" if actions is not None else ""
  )
  raise RuntimeError(
    f"CMoE non-finite rollout observation at {stage}: {name}: "
    f"env_ids={env_ids}, local_dims={dimensions}, "
    f"nan={torch.isnan(value).sum().item()}, inf={torch.isinf(value).sum().item()}"
    f"{action_values}"
  )


def _check_rollout_observations(
  obs: TensorDict, stage: str, actions: torch.Tensor | None = None
) -> None:
  actor = obs["actor"]
  for frame in range(10):
    offset = frame * 45
    for name, start, end in _PROPRIO_FIELDS:
      _check_observation_slice(
        actor,
        f"actor.history[{frame}].{name}",
        offset + start,
        offset + end,
        stage,
        actions,
      )
  _check_observation_slice(actor, "actor.height_scan", 450, 527, stage, actions)

  critic = obs["critic"]
  for name, start, end in _PROPRIO_FIELDS:
    _check_observation_slice(critic, f"critic.{name}", start, end, stage, actions)
  _check_observation_slice(critic, "critic.base_lin_vel", 45, 48, stage, actions)
  _check_observation_slice(
    critic, "critic.external_force", 48, 51, stage, actions
  )
  _check_observation_slice(critic, "critic.height_scan", 51, 128, stage, actions)


def _check_rollout_tensor(tensor: torch.Tensor, name: str, stage: str) -> None:
  invalid = ~torch.isfinite(tensor)
  if invalid.any():
    indices = invalid.nonzero(as_tuple=False)
    env_ids = indices[:, 0].unique().cpu().tolist()
    raise RuntimeError(
      f"CMoE non-finite rollout tensor at {stage}: {name}: "
      f"env_ids={env_ids}, nan={torch.isnan(tensor).sum().item()}, "
      f"inf={torch.isinf(tensor).sum().item()}"
    )


class CMoERunner(MjlabOnPolicyRunner):
  """Run the original CMoE rollout/update loop on an MJLab VecEnv.

  MJLab normally auto-resets terminated environments before returning from
  ``step``.  CMoE needs the terminal privileged observation for the estimator,
  so this runner disables auto-reset and resets only after copying the terminal
  critic observation into the transition.
  """

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict[str, Any],
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    from smp.rl.tasks.cmoe.env import cmoe_step

    env.unwrapped.step = MethodType(cmoe_step, env.unwrapped)
    super().__init__(env, train_cfg, log_dir, device)

  def _reset_done(self, terminal_obs: TensorDict, dones: torch.Tensor) -> TensorDict:
    done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
    if len(done_ids) == 0:
      return terminal_obs
    reset_obs, _ = self.env.unwrapped.reset(env_ids=done_ids)
    reset_obs = TensorDict(
      reset_obs,
      batch_size=[self.env.num_envs],
      device=self.device,
    )
    next_obs = terminal_obs.clone()
    next_obs[done_ids] = reset_obs[done_ids]
    return next_obs

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if "model_state_dict" not in checkpoint:
      return super().load(path, load_cfg, strict, map_location)

    model_state = checkpoint.pop("model_state_dict")
    parameter_names = [
      "distribution.std_param" if name == "std" else name for name in model_state
    ]
    model_state["distribution.std_param"] = model_state.pop("std")
    checkpoint["actor_state_dict"] = model_state
    checkpoint["critic_state_dict"] = model_state

    if load_cfg is None or load_cfg.get("optimizer", False):
      optimizer_state = checkpoint["optimizer_state_dict"]
      parameter_ids = optimizer_state["param_groups"][0]["params"]
      parameter_id_by_name = dict(zip(parameter_names, parameter_ids, strict=True))
      current_names = [name for name, _ in self.alg._raw_actor.named_parameters()]
      optimizer_state["param_groups"][0]["params"] = [
        parameter_id_by_name[name] for name in current_names
      ]

    load_iteration = self.alg.load(checkpoint, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = checkpoint["iter"]
    return checkpoint["infos"]

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    self.env.unwrapped.cfg.auto_reset = False
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations().to(self.device)
    _check_rollout_observations(obs, "initial observation")
    self.alg.train_mode()
    if self.is_distributed:
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()
    self.logger.init_logging_writer()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      start = time.time()
      with torch.inference_mode():
        for rollout_step in range(self.cfg["num_steps_per_env"]):
          stage = f"iteration={it}, rollout_step={rollout_step}, before action"
          _check_rollout_observations(obs, stage)
          actions = self.alg.act(obs)
          _check_rollout_tensor(actions, "actions", stage)
          terminal_obs, rewards, dones, extras = self.env.step(
            actions.to(self.env.device)
          )
          terminal_obs = terminal_obs.to(self.device)
          rewards = rewards.to(self.device)
          dones = dones.to(self.device)
          stage = f"iteration={it}, rollout_step={rollout_step}, after simulation"
          _check_rollout_observations(terminal_obs, stage, actions)
          _check_rollout_tensor(rewards, "rewards", stage)
          step_extras = dict(extras)
          terminal_critic_obs = self.alg._select_obs(terminal_obs, "critic")
          obs = self._reset_done(terminal_obs, dones)
          _check_rollout_observations(
            obs, f"iteration={it}, rollout_step={rollout_step}, after reset"
          )
          self.alg.process_env_step(
            obs,
            rewards,
            dones,
            step_extras,
            next_critic_obs=terminal_critic_obs,
          )
          intrinsic_rewards = None
          self.logger.process_env_step(rewards, dones, step_extras, intrinsic_rewards)

        stop = time.time()
        collect_time = stop - start
        start = stop
        self.alg.compute_returns(obs)

      loss_dict = self.alg.update()
      stop = time.time()
      self.current_learning_iteration = it
      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=stop - start,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=None,
      )
      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

    self.current_learning_iteration = total_it
    if self.logger.writer is not None:
      self.save(
        os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")
      )
      self.logger.stop_logging_writer()


__all__ = ["CMoERunner"]
