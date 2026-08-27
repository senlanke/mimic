"""RSL-RL configuration for the AME task."""

from dataclasses import dataclass

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class AMEModelCfg(RslRlModelCfg):
  class_name: str = "smp.rl.ame.actor_critic_encoder:AMEModel"
  hidden_dims: tuple[int, ...] = (512, 256, 128)
  activation: str = "elu"
  obs_normalization: bool = False
  map_scan_dim: tuple[int, int, int] = (33, 21, 3)
  mha_dim: int = 64
  num_heads: int = 16
  cnn_downsample: bool = True
  attach_global: bool = False


@dataclass
class AMEOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
  check_for_nan: bool = False


def g1_ame_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  actor = AMEModelCfg(
    distribution_cfg={
      "class_name": "smp.rl.ame.actor_critic_encoder:AMEDirectGaussianDistribution",
      "init_std": 1.0,
    }
  )
  critic = AMEModelCfg(distribution_cfg=None)
  return AMEOnPolicyRunnerCfg(
    actor=actor,
    critic=critic,
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.008,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      share_cnn_encoders=True,
    ),
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    experiment_name="g1_ame",
    run_name="ame",
    logger="tensorboard",
    upload_model=False,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10_000,
  )


__all__ = ["AMEModelCfg", "g1_ame_ppo_runner_cfg"]
