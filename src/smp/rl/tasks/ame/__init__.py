"""AME terrain tasks for the Unitree G1 robot."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.ame import AMERunner
from smp.rl.tasks.ame.ame_env_cfg import g1_ame_env_cfg
from smp.rl.tasks.ame.ame_rl_cfg import g1_ame_ppo_runner_cfg

register_mjlab_task(
  task_id="AME-G1",
  env_cfg=g1_ame_env_cfg(play=False, finetune=False),
  play_env_cfg=g1_ame_env_cfg(play=True, finetune=False),
  rl_cfg=g1_ame_ppo_runner_cfg(),
  runner_cls=AMERunner,
)

_global_rl_cfg = g1_ame_ppo_runner_cfg(attach_global=True)
_global_rl_cfg.run_name = "ame_global"

register_mjlab_task(
  task_id="AME-G1-Global",
  env_cfg=g1_ame_env_cfg(play=False, finetune=False),
  play_env_cfg=g1_ame_env_cfg(play=True, finetune=False),
  rl_cfg=_global_rl_cfg,
  runner_cls=AMERunner,
)

_finetune_rl_cfg = g1_ame_ppo_runner_cfg()
_finetune_rl_cfg.run_name = "ame_finetune"

register_mjlab_task(
  task_id="AME-G1-Finetune",
  env_cfg=g1_ame_env_cfg(play=False, finetune=True),
  play_env_cfg=g1_ame_env_cfg(play=True, finetune=True),
  rl_cfg=_finetune_rl_cfg,
  runner_cls=AMERunner,
)

__all__ = ["g1_ame_env_cfg", "g1_ame_ppo_runner_cfg"]
