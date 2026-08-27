"""AME PPO adapted to the RSL-RL 5 actor/critic boundary."""

from __future__ import annotations

from typing import Any

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict


class AMEPPO(PPO):
  """PPO construction for AME's actor/critic-shared terrain encoder."""

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
    return algorithm


__all__ = ["AMEPPO"]
