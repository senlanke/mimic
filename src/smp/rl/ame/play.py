"""AME playback with MHA attention recording."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import mjlab
import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from smp.rl.ame.runner import AMERunner


@dataclass(frozen=True)
class AMEPlayConfig:
  checkpoint_file: str
  num_envs: int = 1
  device: str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  video: bool = False
  video_length: int = 300
  num_steps: int | None = None
  attention_file: str = "attention_weights.npy"


class AttentionPolicy:
  def __init__(self, actor, output_path: Path) -> None:
    self.actor = actor
    self.output_path = output_path
    self.weights: list[np.ndarray] = []

  def __call__(self, obs):
    actions, attention_weights = self.actor.forward_with_attention(obs)
    self.weights.append(attention_weights.detach().cpu().numpy())
    return actions

  def save(self) -> None:
    self.output_path.parent.mkdir(parents=True, exist_ok=True)
    weights = np.stack(self.weights)
    np.save(self.output_path, weights)
    print(f"[INFO] Attention weights saved to {self.output_path}, shape={weights.shape}")


def run_play(task_id: str, cfg: AMEPlayConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = cfg.num_envs

  checkpoint_path = Path(cfg.checkpoint_file).resolve()
  env = ManagerBasedRlEnv(
    cfg=env_cfg,
    device=device,
    render_mode="rgb_array" if cfg.video else None,
  )
  if cfg.video:
    env = VideoRecorder(
      env,
      video_folder=checkpoint_path.parent / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner = AMERunner(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint_path),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  attention_policy = AttentionPolicy(
    runner.get_inference_policy(device=device),
    Path(cfg.attention_file).resolve(),
  )

  if cfg.viewer == "auto":
    resolved_viewer = (
      "native"
      if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
      else "viser"
    )
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, attention_policy).run(num_steps=cfg.num_steps)
  else:
    ViserPlayViewer(env, attention_policy).run(num_steps=cfg.num_steps)

  attention_policy.save()
  env.close()


def main() -> None:
  all_tasks = [task for task in list_tasks() if task.startswith("AME-")]
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    AMEPlayConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
