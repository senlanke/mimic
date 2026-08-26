# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University).
# Adapted from rsl_rl (BSD-3-Clause, Copyright (c) 2021 ETH Zurich, Nikita Rudin
# and NVIDIA CORPORATION & AFFILIATES). See rsl_rl/LICENSE.

"""The original CMoE PPO loop adapted to the current MJLab boundary."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import chain
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.env import VecEnv
from rsl_rl.utils import (
  compile_model,
  resolve_callable,
  resolve_obs_groups,
  resolve_optimizer,
)
from tensordict import TensorDict

from .storage import RolloutStorage


def _tensor_stats(tensor: torch.Tensor) -> str:
  finite = torch.isfinite(tensor)
  values = tensor[finite]
  bounds = (
    f"min={values.min().item():.6g}, max={values.max().item():.6g}"
    if values.numel()
    else "min=nan, max=nan"
  )
  return (
    f"shape={tuple(tensor.shape)}, {bounds}, "
    f"nan={torch.isnan(tensor).sum().item()}, "
    f"inf={torch.isinf(tensor).sum().item()}"
  )


def _check_tensor(name: str, tensor: torch.Tensor, stage: str) -> None:
  if not torch.isfinite(tensor).all():
    raise RuntimeError(f"CMoE non-finite at {stage}: {name}: {_tensor_stats(tensor)}")


def _check_tensordict(name: str, data: TensorDict, stage: str) -> None:
  for key, value in data.items():
    if isinstance(value, TensorDict):
      _check_tensordict(f"{name}.{key}", value, stage)
    else:
      _check_tensor(f"{name}.{key}", value, stage)


class CMoEPPO:
  """PPO with the CMoE estimators and prototype objective."""

  def __init__(
    self,
    actor: nn.Module,
    critic: nn.Module,
    storage: RolloutStorage,
    num_learning_epochs: int = 1,
    num_mini_batches: int = 1,
    clip_param: float = 0.2,
    gamma: float = 0.998,
    lam: float = 0.95,
    value_loss_coef: float = 1.0,
    entropy_coef: float = 0.0,
    learning_rate: float = 1e-3,
    max_grad_norm: float = 1.0,
    optimizer: str = "adam",
    use_clipped_value_loss: bool = True,
    schedule: str = "fixed",
    desired_kl: float = 0.01,
    device: str = "cpu",
    multi_gpu_cfg: dict[str, Any] | None = None,
    obs_groups: dict[str, list[str]] | None = None,
  ) -> None:
    self.device = device
    self.actor = actor.to(device)
    self.critic = critic.to(device)
    self.actor_critic = self.actor
    self._raw_actor = self.actor
    self._raw_critic = self.critic
    self.storage = storage
    self.transition = RolloutStorage.Transition()
    self.obs_groups = obs_groups or {"actor": ["actor"], "critic": ["critic"]}

    self.desired_kl = desired_kl
    self.schedule = schedule
    self.learning_rate = learning_rate
    self.clip_param = clip_param
    self.num_learning_epochs = num_learning_epochs
    self.num_mini_batches = num_mini_batches
    self.value_loss_coef = value_loss_coef
    self.entropy_coef = entropy_coef
    self.gamma = gamma
    self.lam = lam
    self.max_grad_norm = max_grad_norm
    self.use_clipped_value_loss = use_clipped_value_loss

    self.is_multi_gpu = multi_gpu_cfg is not None
    if self.is_multi_gpu:
      self.gpu_global_rank = multi_gpu_cfg["global_rank"]
      self.gpu_world_size = multi_gpu_cfg["world_size"]
    else:
      self.gpu_global_rank = 0
      self.gpu_world_size = 1

    parameters = self.actor.parameters()
    if self.critic is not self.actor:
      parameters = chain(parameters, self.critic.parameters())
    self.optimizer = resolve_optimizer(optimizer)(parameters, lr=learning_rate)
    self.rnd = None
    self._update_index = 0

  def _check_parameters(self, stage: str) -> None:
    for name, parameter in self._raw_actor.named_parameters():
      _check_tensor(name, parameter, stage)
    if self._raw_critic is not self._raw_actor:
      for name, parameter in self._raw_critic.named_parameters():
        _check_tensor(f"critic.{name}", parameter, stage)

  def _check_gradients(self, stage: str) -> None:
    for name, parameter in self._raw_actor.named_parameters():
      if parameter.grad is not None:
        _check_tensor(name, parameter.grad, stage)
    if self._raw_critic is not self._raw_actor:
      for name, parameter in self._raw_critic.named_parameters():
        if parameter.grad is not None:
          _check_tensor(f"critic.{name}", parameter.grad, stage)

  def _check_std(self, stage: str) -> None:
    std = self._raw_actor.distribution.std_param
    _check_tensor("distribution.std_param", std, stage)
    if (std < 0.0).any():
      raise RuntimeError(
        f"CMoE invalid std at {stage}: distribution.std_param: "
        f"{_tensor_stats(std)}"
      )

  def act(self, obs: TensorDict) -> torch.Tensor:
    """Compute and record the action and both observations for one step."""
    self.transition.actions = self.actor.act(obs).detach()
    self.transition.values = self.actor.evaluate(obs).detach()
    self.transition.actions_log_prob = self.actor.get_output_log_prob(
      self.transition.actions
    ).detach()
    self.transition.action_mean = self.actor.output_mean.detach()
    self.transition.action_sigma = self.actor.output_std.detach()
    self.transition.observations = obs
    self.transition.critic_observations = self._select_obs(obs, "critic")
    return self.transition.actions

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, Any],
    next_critic_obs: TensorDict | None = None,
  ) -> None:
    """Store a step, keeping terminal privileged data before reset."""
    self.transition.next_critic_observations = (
      next_critic_obs.clone()
      if next_critic_obs is not None
      else self._select_obs(obs, "critic").clone()
    )
    self.transition.rewards = rewards.clone()
    self.transition.dones = dones
    if "time_outs" in extras:
      self.transition.rewards += self.gamma * torch.squeeze(
        self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),
        1,
      )
    self.storage.add_transitions(self.transition)
    self.transition.clear()
    self.actor.reset(dones)
    if self.critic is not self.actor:
      self.critic.reset(dones)

  def compute_returns(self, critic_obs: TensorDict) -> None:
    last_values = self.actor.evaluate(self._select_obs(critic_obs, "critic")).detach()
    self.storage.compute_returns(last_values, self.gamma, self.lam)

  def update(self) -> dict[str, float]:
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_estimation_loss = 0.0
    mean_latent_loss = 0.0
    mean_recons_loss = 0.0
    mean_kld_loss = 0.0
    mean_estimation_loss2 = 0.0
    mean_latent_loss2 = 0.0
    mean_recons_loss2 = 0.0
    mean_kld_loss2 = 0.0
    mean_contrastive_loss = 0.0
    entropy_mean = 0.0

    for mini_batch_index, batch in enumerate(
      self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    ):
      stage = f"update={self._update_index}, minibatch={mini_batch_index}"
      self._check_parameters(f"{stage}, before forward")
      _check_tensordict("observations", batch.observations, stage)
      _check_tensordict("critic_observations", batch.critic_observations, stage)
      _check_tensordict(
        "next_critic_observations", batch.next_critic_observations, stage
      )
      for name in (
        "actions",
        "old_actions_log_prob",
        "old_mu",
        "old_sigma",
        "advantages",
        "returns",
        "target_values",
      ):
        _check_tensor(name, getattr(batch, name), stage)
      self._check_std(f"{stage}, before forward")

      self.actor.act(batch.observations)
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      value = self.actor.evaluate(batch.critic_observations)
      mu = self.actor.output_mean
      sigma = self.actor.output_std
      entropy = self.actor.output_entropy
      _check_tensor("actions_log_prob", actions_log_prob, stage)
      _check_tensor("value", value, stage)
      _check_tensor("mu", mu, stage)
      _check_tensor("sigma", sigma, stage)
      _check_tensor("entropy", entropy, stage)
      entropy_mean += entropy.mean().item()

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = torch.sum(
            torch.log(sigma / batch.old_sigma + 1.0e-5)
            + (torch.square(batch.old_sigma) + torch.square(batch.old_mu - mu))
            / (2.0 * torch.square(sigma))
            - 0.5,
            axis=-1,
          )
          kl_mean = torch.mean(kl)
          _check_tensor("kl", kl, stage)
          _check_tensor("kl_mean", kl_mean, stage)
          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()
          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      estimator_losses = self.actor.update_estimators(
        batch.observations,
        batch.critic_observations,
        batch.next_critic_observations,
        lr=self.learning_rate,
        gradient_sync=self.reduce_gradients if self.is_multi_gpu else None,
      )
      _check_tensor(
        "estimator_losses",
        torch.as_tensor(estimator_losses, device=self.device),
        stage,
      )
      self._check_parameters(f"{stage}, after estimator step")
      self._check_std(f"{stage}, after estimator step")
      contrastive_loss = self.actor.compute_contrastive_loss(batch.observations)
      _check_tensor("contrastive_loss", contrastive_loss, stage)

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
      surrogate = -torch.squeeze(batch.advantages) * ratio
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
      _check_tensor("ratio", ratio, stage)
      _check_tensor("surrogate_loss", surrogate_loss, stage)

      if self.use_clipped_value_loss:
        value_clipped = batch.target_values + (value - batch.target_values).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (value - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - value).pow(2).mean()
      _check_tensor("value_loss", value_loss, stage)

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy.mean()
        + contrastive_loss
      )
      _check_tensor("loss", loss, stage)
      self.optimizer.zero_grad()
      loss.backward()
      self._check_gradients(f"{stage}, after backward")
      if self.is_multi_gpu:
        self.reduce_parameters()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      if self.critic is not self.actor:
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()
      self._check_parameters(f"{stage}, after optimizer step")
      self._check_std(f"{stage}, after optimizer step")

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_estimation_loss += estimator_losses[0]
      mean_latent_loss += estimator_losses[1]
      mean_recons_loss += estimator_losses[2]
      mean_kld_loss += estimator_losses[3]
      mean_estimation_loss2 += estimator_losses[4]
      mean_latent_loss2 += estimator_losses[5]
      mean_recons_loss2 += estimator_losses[6]
      mean_kld_loss2 += estimator_losses[7]
      mean_contrastive_loss += contrastive_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    self.storage.clear()
    self._update_index += 1
    return {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "entropy": entropy_mean / num_updates,
      "estimation": mean_estimation_loss / num_updates,
      "latent": mean_latent_loss / num_updates,
      "reconstruction": mean_recons_loss / num_updates,
      "kld": mean_kld_loss / num_updates,
      "estimation_terrain": mean_estimation_loss2 / num_updates,
      "latent_terrain": mean_latent_loss2 / num_updates,
      "reconstruction_terrain": mean_recons_loss2 / num_updates,
      "kld_terrain": mean_kld_loss2 / num_updates,
      "contrastive": mean_contrastive_loss / num_updates,
    }

  def train_mode(self) -> None:
    self.actor.train()
    if self.critic is not self.actor:
      self.critic.train()

  def eval_mode(self) -> None:
    self.actor.eval()
    if self.critic is not self.actor:
      self.critic.eval()

  def test_mode(self) -> None:
    self.eval_mode()

  def save(self) -> dict[str, Any]:
    return {
      "actor_state_dict": self._raw_actor.state_dict(),
      "critic_state_dict": self._raw_critic.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
    }

  def load(
    self, loaded_dict: dict[str, Any], load_cfg: dict[str, bool] | None, strict: bool
  ) -> bool:
    if load_cfg is None:
      load_cfg = {
        "actor": True,
        "critic": True,
        "optimizer": True,
        "iteration": True,
      }
    if load_cfg.get("actor"):
      self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
    if load_cfg.get("critic") and self._raw_critic is not self._raw_actor:
      self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
    if load_cfg.get("optimizer"):
      self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
    return load_cfg.get("iteration", False)

  def get_policy(self) -> nn.Module:
    return self._raw_actor

  def compile(self, mode: str | None = None) -> None:
    self.actor = compile_model(self._raw_actor, mode)
    self.critic = (
      self.actor
      if self._raw_critic is self._raw_actor
      else compile_model(self._raw_critic, mode)
    )

  def broadcast_parameters(self) -> None:
    if not self.is_multi_gpu:
      return
    model_params = [self._raw_actor.state_dict()]
    if self._raw_critic is not self._raw_actor:
      model_params.append(self._raw_critic.state_dict())
    torch.distributed.broadcast_object_list(model_params, src=0)
    self._raw_actor.load_state_dict(model_params[0])
    if self._raw_critic is not self._raw_actor:
      self._raw_critic.load_state_dict(model_params[1])

  def reduce_parameters(self) -> None:
    parameters = list(self.actor.parameters())
    if self.critic is not self.actor:
      parameters.extend(self.critic.parameters())
    self.reduce_gradients(parameters)

  def reduce_gradients(self, parameters: Iterable[nn.Parameter]) -> None:
    parameters = list(parameters)
    gradients = [p.grad.view(-1) for p in parameters if p.grad is not None]
    all_gradients = torch.cat(gradients)
    torch.distributed.all_reduce(all_gradients, op=torch.distributed.ReduceOp.SUM)
    all_gradients /= self.gpu_world_size
    offset = 0
    for parameter in parameters:
      if parameter.grad is not None:
        numel = parameter.numel()
        parameter.grad.data.copy_(
          all_gradients[offset : offset + numel].view_as(parameter.grad.data)
        )
        offset += numel

  def _select_obs(self, obs: TensorDict, obs_set: str) -> TensorDict:
    return obs.select(*self.obs_groups[obs_set])

  @staticmethod
  def construct_algorithm(
    obs: TensorDict,
    env: VecEnv,
    cfg: dict[str, Any],
    device: str,
  ) -> "CMoEPPO":
    alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))
    actor_class = resolve_callable(cfg["actor"].pop("class_name"))
    cfg["critic"].pop("class_name")
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
    cfg["algorithm"].pop("share_cnn_encoders")
    cfg["algorithm"].pop("normalize_advantage_per_mini_batch")

    actor = actor_class(
      obs,
      cfg["obs_groups"],
      "actor",
      env.num_actions,
      **cfg["actor"],
    ).to(device)
    critic_obs = obs.select(*cfg["obs_groups"]["critic"])
    storage = RolloutStorage(
      env.num_envs,
      cfg["num_steps_per_env"],
      obs,
      critic_obs,
      [env.num_actions],
      device,
    )
    algorithm_cfg = cfg["algorithm"].copy()
    algorithm_cfg.pop("rnd_cfg")
    algorithm = alg_class(
      actor,
      actor,
      storage,
      device=device,
      obs_groups=cfg["obs_groups"],
      multi_gpu_cfg=cfg["multi_gpu"],
      **algorithm_cfg,
    )
    algorithm.compile(cfg.get("torch_compile_mode"))
    return algorithm


__all__ = ["CMoEPPO"]
