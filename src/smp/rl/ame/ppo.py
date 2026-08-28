"""Original AME PPO loop adapted to the RSL-RL 5 runner boundary."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer
from tensordict import TensorDict


class AMEPPO:
  """PPO used by AME with one shared CNN/MHA parameter set."""

  def __init__(
    self,
    actor: nn.Module,
    critic: nn.Module,
    storage: RolloutStorage,
    num_learning_epochs: int = 5,
    num_mini_batches: int = 4,
    clip_param: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    value_loss_coef: float = 1.0,
    entropy_coef: float = 0.01,
    learning_rate: float = 1.0e-3,
    max_grad_norm: float = 1.0,
    optimizer: str = "adam",
    use_clipped_value_loss: bool = True,
    schedule: str = "adaptive",
    desired_kl: float = 0.01,
    normalize_advantage_per_mini_batch: bool = False,
    device: str = "cpu",
    rnd_cfg: dict | None = None,
    symmetry_cfg: dict | None = None,
    multi_gpu_cfg: dict | None = None,
  ) -> None:
    del rnd_cfg, symmetry_cfg
    self.device = device
    self.is_multi_gpu = multi_gpu_cfg is not None
    if multi_gpu_cfg is not None:
      self.gpu_global_rank = multi_gpu_cfg["global_rank"]
      self.gpu_world_size = multi_gpu_cfg["world_size"]
    else:
      self.gpu_global_rank = 0
      self.gpu_world_size = 1

    self.actor = actor.to(device)
    self.critic = critic.to(device)
    self._raw_actor = self.actor
    self._raw_critic = self.critic
    self.storage = storage
    self.transition = RolloutStorage.Transition()
    self.parameters = self._ordered_parameters()
    self.optimizer = resolve_optimizer(optimizer)(self.parameters, lr=learning_rate)
    self.rnd = None

    self.clip_param = clip_param
    self.num_learning_epochs = num_learning_epochs
    self.num_mini_batches = num_mini_batches
    self.value_loss_coef = value_loss_coef
    self.entropy_coef = entropy_coef
    self.gamma = gamma
    self.lam = lam
    self.max_grad_norm = max_grad_norm
    self.use_clipped_value_loss = use_clipped_value_loss
    self.desired_kl = desired_kl
    self.schedule = schedule
    self.learning_rate = learning_rate
    self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

  def act(self, obs: TensorDict) -> torch.Tensor:
    self.transition.hidden_states = (
      self.actor.get_hidden_state(),
      self.critic.get_hidden_state(),
    )
    self.transition.actions = self.actor(obs, stochastic_output=True).detach()
    self.transition.values = self.critic(obs).detach()
    self.transition.actions_log_prob = self.actor.get_output_log_prob(
      self.transition.actions
    ).detach()
    self.transition.distribution_params = tuple(
      parameter.detach() for parameter in self.actor.output_distribution_params
    )
    self.transition.observations = obs
    return self.transition.actions

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    self.actor.update_normalization(obs)
    self.critic.update_normalization(obs)
    self.transition.rewards = rewards.clone()
    self.transition.dones = dones
    if "time_outs" in extras:
      self.transition.rewards += self.gamma * torch.squeeze(
        self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),
        1,
      )
    self.storage.add_transition(self.transition)
    self.transition.clear()
    self.actor.reset(dones)
    self.critic.reset(dones)

  def compute_returns(self, obs: TensorDict) -> None:
    last_values = self.critic(obs).detach()
    advantage = 0
    for step in reversed(range(self.storage.num_transitions_per_env)):
      next_values = (
        last_values
        if step == self.storage.num_transitions_per_env - 1
        else self.storage.values[step + 1]
      )
      next_is_not_terminal = 1.0 - self.storage.dones[step].float()
      delta = (
        self.storage.rewards[step]
        + next_is_not_terminal * self.gamma * next_values
        - self.storage.values[step]
      )
      advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
      self.storage.returns[step] = advantage + self.storage.values[step]
    self.storage.advantages = self.storage.returns - self.storage.values
    if not self.normalize_advantage_per_mini_batch:
      self.storage.advantages = (
        self.storage.advantages - self.storage.advantages.mean()
      ) / (self.storage.advantages.std() + 1.0e-8)

  def update(self) -> dict[str, float]:
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    for batch in generator:
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (
            batch.advantages - batch.advantages.mean()
          ) / (batch.advantages.std() + 1.0e-8)

      self.actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[1],
      )
      mean = self.actor.output_mean
      std = self.actor.output_std
      entropy = self.actor.output_entropy

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          old_mean, old_std = batch.old_distribution_params
          kl = torch.sum(
            torch.log(std / old_std + 1.0e-5)
            + (torch.square(old_std) + torch.square(old_mean - mean))
            / (2.0 * torch.square(std))
            - 0.5,
            dim=-1,
          )
          kl_mean = torch.mean(kl)
          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
          if self.is_multi_gpu:
            learning_rate = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(learning_rate, src=0)
            self.learning_rate = learning_rate.item()
          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
      surrogate = -torch.squeeze(batch.advantages) * ratio
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy.mean()
      )
      self.optimizer.zero_grad()
      loss.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()
      nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
      self.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    self.storage.clear()
    return {
      "value": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
    }

  def train_mode(self) -> None:
    self.actor.train()
    self.critic.train()

  def eval_mode(self) -> None:
    self.actor.eval()
    self.critic.eval()

  def save(self) -> dict:
    return {
      "model_state_dict": self._model_state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict(),
    }

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    if load_cfg is None:
      load_cfg = {
        "actor": True,
        "critic": True,
        "optimizer": True,
        "iteration": True,
      }
    model_state = loaded_dict["model_state_dict"]
    if load_cfg.get("actor") or load_cfg.get("critic"):
      self._raw_actor.cnns["map_cnn"].load_state_dict(
        self._submodule_state(model_state, "map_cnn"), strict=strict
      )
      self._raw_actor.cnns["mha"].load_state_dict(
        self._submodule_state(model_state, "mha"), strict=strict
      )
      if self._raw_actor.attach_global:
        self._raw_actor.global_encoder.load_state_dict(
          self._submodule_state(model_state, "global_encoder"), strict=strict
        )
        self._raw_actor.query_projector.load_state_dict(
          self._submodule_state(model_state, "query_projector"), strict=strict
        )
    if load_cfg.get("actor"):
      self._raw_actor.proprio_embedding.load_state_dict(
        self._submodule_state(model_state, "actor_proprio_embedding"), strict=strict
      )
      self._raw_actor.mlp.load_state_dict(
        self._submodule_state(model_state, "actor"), strict=strict
      )
      self._raw_actor.distribution.std_param.data.copy_(model_state["std"])
    if load_cfg.get("critic"):
      self._raw_critic.proprio_embedding.load_state_dict(
        self._submodule_state(model_state, "critic_proprio_embedding"), strict=strict
      )
      self._raw_critic.mlp.load_state_dict(
        self._submodule_state(model_state, "critic"), strict=strict
      )
    if load_cfg.get("optimizer"):
      self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
    return load_cfg.get("iteration", False)

  def _ordered_parameters(self) -> list[nn.Parameter]:
    modules = [
      self.actor.cnns["map_cnn"],
      self.actor.proprio_embedding,
      self.critic.proprio_embedding,
    ]
    if self.actor.attach_global:
      modules.extend([self.actor.global_encoder, self.actor.query_projector])
    modules.extend(
      [self.actor.cnns["mha"], self.actor.mlp, self.critic.mlp, self.actor.distribution]
    )
    return [parameter for module in modules for parameter in module.parameters()]

  @staticmethod
  def _submodule_state(state_dict: dict, prefix: str) -> dict:
    prefix = prefix + "."
    return {
      key[len(prefix) :]: value
      for key, value in state_dict.items()
      if key.startswith(prefix)
    }

  def _model_state_dict(self) -> dict:
    state_dict = {}
    modules = (
      ("map_cnn", self._raw_actor.cnns["map_cnn"]),
      ("actor_proprio_embedding", self._raw_actor.proprio_embedding),
      ("critic_proprio_embedding", self._raw_critic.proprio_embedding),
    )
    for prefix, module in modules:
      state_dict.update(
        {f"{prefix}.{key}": value for key, value in module.state_dict().items()}
      )
    if self._raw_actor.attach_global:
      for prefix, module in (
        ("global_encoder", self._raw_actor.global_encoder),
        ("query_projector", self._raw_actor.query_projector),
      ):
        state_dict.update(
          {f"{prefix}.{key}": value for key, value in module.state_dict().items()}
        )
    for prefix, module in (
      ("mha", self._raw_actor.cnns["mha"]),
      ("actor", self._raw_actor.mlp),
      ("critic", self._raw_critic.mlp),
    ):
      state_dict.update(
        {f"{prefix}.{key}": value for key, value in module.state_dict().items()}
      )
    state_dict["std"] = self._raw_actor.distribution.std_param
    return state_dict

  def get_policy(self) -> nn.Module:
    return self._raw_actor

  def compile(self, mode: str | None = None) -> None:
    self.actor = compile_model(self._raw_actor, mode)
    self.critic = compile_model(self._raw_critic, mode)

  def broadcast_parameters(self) -> None:
    model_parameters = [self._raw_actor.state_dict(), self._raw_critic.state_dict()]
    torch.distributed.broadcast_object_list(model_parameters, src=0)
    self._raw_actor.load_state_dict(model_parameters[0])
    self._raw_critic.load_state_dict(model_parameters[1])

  def reduce_parameters(self) -> None:
    gradients = [
      parameter.grad.view(-1)
      for parameter in self.parameters
      if parameter.grad is not None
    ]
    all_gradients = torch.cat(gradients)
    torch.distributed.all_reduce(all_gradients, op=torch.distributed.ReduceOp.SUM)
    all_gradients /= self.gpu_world_size
    offset = 0
    for parameter in self.parameters:
      if parameter.grad is not None:
        numel = parameter.numel()
        parameter.grad.data.copy_(
          all_gradients[offset : offset + numel].view_as(parameter.grad.data)
        )
        offset += numel

  @staticmethod
  def construct_algorithm(
    obs: TensorDict,
    env: VecEnv,
    cfg: dict[str, Any],
    device: str,
  ) -> "AMEPPO":
    algorithm_class = resolve_callable(cfg["algorithm"].pop("class_name"))
    actor_class = resolve_callable(cfg["actor"].pop("class_name"))
    critic_class = resolve_callable(cfg["critic"].pop("class_name"))
    cfg["obs_groups"] = resolve_obs_groups(
      obs, cfg["obs_groups"], ["actor", "critic"]
    )
    cfg["algorithm"] = resolve_rnd_config(
      cfg["algorithm"], obs, cfg["obs_groups"], env
    )
    cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)
    cfg["algorithm"].pop("share_cnn_encoders")

    actor = actor_class(
      obs,
      cfg["obs_groups"],
      "actor",
      env.num_actions,
      **cfg["actor"],
    ).to(device)
    critic = critic_class(
      obs,
      cfg["obs_groups"],
      "critic",
      1,
      cnns=actor.cnns,
      global_encoder=actor.global_encoder,
      query_projector=actor.query_projector,
      **cfg["critic"],
    ).to(device)
    print(f"Actor Model: {actor}")
    print(f"Critic Model: {critic}")

    storage = RolloutStorage(
      "rl",
      env.num_envs,
      cfg["num_steps_per_env"],
      obs,
      [env.num_actions],
      device,
    )
    algorithm = algorithm_class(
      actor,
      critic,
      storage,
      device=device,
      **cfg["algorithm"],
      multi_gpu_cfg=cfg["multi_gpu"],
    )
    algorithm.compile(cfg.get("torch_compile_mode"))
    return algorithm


__all__ = ["AMEPPO"]
