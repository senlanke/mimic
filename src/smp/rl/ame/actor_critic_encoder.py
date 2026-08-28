"""Attention-based map encoder from AME_Locomotion.

The CNN/MHA architecture and map layout are kept from the original
``ActorCriticEncoder``.  The surrounding actor/critic model protocol is the
one used by the RSL-RL version bundled with MJLab.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.modules import EmpiricalNormalization, HiddenState, MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories
from tensordict import TensorDict, TensorDictBase


class AMEDirectGaussianDistribution(Distribution):
  """Original AME Gaussian with a directly learned, unclamped standard deviation."""

  def __init__(self, output_dim: int, init_std: float = 1.0) -> None:
    super().__init__(output_dim)
    self.std_param = nn.Parameter(init_std * torch.ones(output_dim))
    self._distribution: Normal | None = None
    Normal.set_default_validate_args(False)

  @property
  def input_dim(self) -> int:
    return self.output_dim

  def update(self, mlp_output: torch.Tensor) -> None:
    self._distribution = Normal(mlp_output, self.std_param.expand_as(mlp_output))

  def sample(self) -> torch.Tensor:
    return self._distribution.sample()

  def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
    return mlp_output

  def as_deterministic_output_module(self) -> nn.Module:
    return nn.Identity()

  @property
  def mean(self) -> torch.Tensor:
    return self._distribution.mean

  @property
  def std(self) -> torch.Tensor:
    return self._distribution.stddev

  @property
  def entropy(self) -> torch.Tensor:
    return self._distribution.entropy().sum(dim=-1)

  @property
  def params(self) -> tuple[torch.Tensor, ...]:
    return self.mean, self.std

  def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    return self._distribution.log_prob(outputs).sum(dim=-1)

  def kl_divergence(
    self,
    old_params: tuple[torch.Tensor, ...],
    new_params: tuple[torch.Tensor, ...],
  ) -> torch.Tensor:
    old_distribution = Normal(*old_params)
    new_distribution = Normal(*new_params)
    return torch.distributions.kl_divergence(
      old_distribution, new_distribution
    ).sum(dim=-1)


class AMEModel(nn.Module):
  """Actor or critic head backed by the original AME terrain encoder."""

  is_recurrent = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict[str, Any] | None = None,
    map_scan_dim: tuple[int, int, int] = (33, 21, 3),
    mha_dim: int = 64,
    num_heads: int = 16,
    cnn_downsample: bool = True,
    attach_global: bool = False,
    cnns: nn.ModuleDict | None = None,
    global_encoder: nn.Module | None = None,
    query_projector: nn.Module | None = None,
  ) -> None:
    super().__init__()
    self.obs_groups = obs_groups[obs_set]
    self.obs_set = obs_set
    self.obs_dim = sum(obs[name].shape[-1] for name in self.obs_groups)
    self.map_scan_dim = map_scan_dim
    self.map_scan_size = map_scan_dim[0] * map_scan_dim[1] * map_scan_dim[2]
    self.proprio_dim = self.obs_dim - self.map_scan_size
    if self.proprio_dim <= 0:
      raise ValueError(
        f"AME {obs_set} observation has {self.obs_dim} values, but the map "
        f"requires {self.map_scan_size}."
      )

    self.mha_dim = mha_dim
    self.cnn_downsample = cnn_downsample
    self.attach_global = attach_global
    self.obs_normalization = obs_normalization
    self.obs_normalizer = (
      EmpiricalNormalization(self.obs_dim) if obs_normalization else nn.Identity()
    )

    if cnns is None:
      stride = 2 if cnn_downsample else 1
      self.cnns = nn.ModuleDict(
        {
          "map_cnn": nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2, stride=stride),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, mha_dim, kernel_size=3 if cnn_downsample else 5, padding=1 if cnn_downsample else 2),
            nn.ReLU(),
            nn.BatchNorm2d(mha_dim),
          ),
          "mha": nn.MultiheadAttention(
            embed_dim=mha_dim, num_heads=num_heads, batch_first=True
          ),
        }
      )
    else:
      self.__dict__["cnns"] = cnns

    self.proprio_embedding = nn.Linear(self.proprio_dim, mha_dim)
    if attach_global:
      if global_encoder is None:
        self.global_encoder = MLP(mha_dim, mha_dim, (256, 128), "elu")
        self.query_projector = nn.Linear(mha_dim * 2, mha_dim)
      else:
        self.__dict__["global_encoder"] = global_encoder
        self.__dict__["query_projector"] = query_projector
    else:
      self.global_encoder = None
      self.query_projector = None

    if distribution_cfg is not None:
      distribution_cfg = copy.deepcopy(distribution_cfg)
      dist_class: type[Distribution] = resolve_callable(
        distribution_cfg.pop("class_name")
      )
      self.distribution: Distribution | None = dist_class(
        output_dim, **distribution_cfg
      )
      head_output_dim = self.distribution.input_dim
    else:
      self.distribution = None
      head_output_dim = output_dim

    head_input_dim = self.proprio_dim + mha_dim * (2 if attach_global else 1)
    self.mlp = MLP(head_input_dim, head_output_dim, hidden_dims, activation)
    if self.distribution is not None:
      self.distribution.init_mlp_weights(self.mlp)

  def _observation_tensor(self, obs: TensorDictBase) -> torch.Tensor:
    return self.obs_normalizer(
      torch.cat([obs[name] for name in self.obs_groups], dim=-1)
    )

  def _encode(
    self, observation: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    length, width, coord_dim = self.map_scan_dim
    map_scan = observation[:, -self.map_scan_size :].reshape(
      -1, width, length, coord_dim
    )
    map_input = map_scan.permute(0, 3, 1, 2)
    cnn_features = self.cnns["map_cnn"](map_input)
    local_features = cnn_features.permute(0, 2, 3, 1).flatten(1, 2)

    proprio = observation[:, : -self.map_scan_size]
    query = self.proprio_embedding(proprio)
    global_feature = None
    if self.attach_global:
      global_feature = self.global_encoder(local_features).amax(dim=1)
      query = self.query_projector(torch.cat((global_feature, query), dim=-1))

    attended, attention_weights = self.cnns["mha"](
      query=query.unsqueeze(1), key=local_features, value=local_features
    )
    encoded = torch.cat((attended.squeeze(1), proprio), dim=-1)
    if global_feature is not None:
      encoded = torch.cat((global_feature, encoded), dim=-1)
    return encoded, attention_weights

  def get_latent(
    self,
    obs: TensorDictBase,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    del masks, hidden_state
    encoded, attention_weights = self._encode(self._observation_tensor(obs))
    del attention_weights
    return encoded

  def forward_with_attention(
    self,
    obs: TensorDictBase,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    encoded, attention_weights = self._encode(self._observation_tensor(obs))
    output = self.mlp(encoded)
    if self.distribution is not None:
      output = self.distribution.deterministic_output(output)
    return output, attention_weights

  def forward(
    self,
    obs: TensorDictBase,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    obs = unpad_trajectories(obs, masks) if masks is not None else obs
    output = self.mlp(self.get_latent(obs, None, hidden_state))
    if self.distribution is None:
      return output
    if stochastic_output:
      self.distribution.update(output)
      return self.distribution.sample()
    return self.distribution.deterministic_output(output)

  def reset(
    self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None
  ) -> None:
    del dones, hidden_state

  def get_hidden_state(self) -> HiddenState:
    return None

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    del dones

  def update_normalization(self, obs: TensorDictBase) -> None:
    if self.obs_normalization:
      value = torch.cat([obs[name] for name in self.obs_groups], dim=-1)
      self.obs_normalizer.update(value)

  @property
  def output_mean(self) -> torch.Tensor:
    return self.distribution.mean

  @property
  def output_std(self) -> torch.Tensor:
    return self.distribution.std

  @property
  def output_entropy(self) -> torch.Tensor:
    return self.distribution.entropy

  @property
  def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
    return self.distribution.params

  def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    return self.distribution.log_prob(outputs)

  def get_kl_divergence(
    self,
    old_params: tuple[torch.Tensor, ...],
    new_params: tuple[torch.Tensor, ...],
  ) -> torch.Tensor:
    return self.distribution.kl_divergence(old_params, new_params)

  def as_jit(self) -> nn.Module:
    return _AMEExport(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _AMEExport(self, verbose=verbose)


class _AMEExport(nn.Module):
  def __init__(self, model: AMEModel, verbose: bool = False) -> None:
    super().__init__()
    self.verbose = verbose
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.cnns = copy.deepcopy(model.cnns)
    self.proprio_embedding = copy.deepcopy(model.proprio_embedding)
    self.global_encoder = copy.deepcopy(model.global_encoder)
    self.query_projector = copy.deepcopy(model.query_projector)
    self.mlp = copy.deepcopy(model.mlp)
    self.map_scan_dim = model.map_scan_dim
    self.map_scan_size = model.map_scan_size
    self.attach_global = model.attach_global
    self.input_size = model.obs_dim

  def forward(self, observation: torch.Tensor) -> torch.Tensor:
    observation = self.obs_normalizer(observation)
    length, width, coord_dim = self.map_scan_dim
    map_scan = observation[:, -self.map_scan_size :].reshape(
      -1, width, length, coord_dim
    )
    local = self.cnns["map_cnn"](map_scan.permute(0, 3, 1, 2))
    local = local.permute(0, 2, 3, 1).flatten(1, 2)
    proprio = observation[:, : -self.map_scan_size]
    query = self.proprio_embedding(proprio)
    global_feature = None
    if self.attach_global:
      global_feature = self.global_encoder(local).amax(dim=1)
      query = self.query_projector(torch.cat((global_feature, query), dim=-1))
    attended, _ = self.cnns["mha"](
      query=query.unsqueeze(1), key=local, value=local
    )
    encoded = torch.cat((attended.squeeze(1), proprio), dim=-1)
    if global_feature is not None:
      encoded = torch.cat((global_feature, encoded), dim=-1)
    return self.mlp(encoded)

  @torch.jit.export
  def reset(self) -> None:
    pass

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]


__all__ = ["AMEDirectGaussianDistribution", "AMEModel"]
