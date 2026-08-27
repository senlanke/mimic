# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University).

"""CMoE terrain locomotion task — registers ``CMoE-G1`` on import."""

from mjlab.tasks.registry import register_mjlab_task

from smp.rl.cmoe import CMoERunner
from smp.rl.tasks.cmoe.cmoe_env_cfg import (
  g1_cmoe_course_env_cfg,
  g1_cmoe_env_cfg,
)
from smp.rl.tasks.cmoe.cmoe_rl_cfg import g1_cmoe_ppo_runner_cfg

register_mjlab_task(
  task_id="CMoE-G1",
  env_cfg=g1_cmoe_env_cfg(play=False),
  # Original CMoE play terrain.
  play_env_cfg=g1_cmoe_env_cfg(play=True),
  # Sequential course: uncomment this line and comment the line above.
  # play_env_cfg=g1_cmoe_course_env_cfg(difficulty=0.5),
  rl_cfg=g1_cmoe_ppo_runner_cfg(),
  runner_cls=CMoERunner,
)

__all__ = [
  "g1_cmoe_course_env_cfg",
  "g1_cmoe_env_cfg",
  "g1_cmoe_ppo_runner_cfg",
]
