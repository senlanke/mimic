# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 The CMoE Authors (Fudan University).

"""G1 CMoE terrain locomotion task."""

from __future__ import annotations

import mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  GridPatternCfg,
  ObjRef,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.nan_guard import NanGuardCfg
from mjlab.viewer import ViewerConfig

from smp.rl.tasks.cmoe import mdp
from smp.rl.tasks.cmoe.asset import (
  COLLISION_GEOM_PATTERN,
  LOWER_BODY_JOINTS,
  LOWER_TORQUE_LIMITS,
  LOWER_VELOCITY_LIMITS,
  get_cmoe_g1_robot_cfg,
)
from smp.rl.tasks.cmoe.terrain import cmoe_terrain_generator_cfg

HIP_JOINTS = (
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
)
FOOT_SITES = ("left_foot", "right_foot")
FOOT_SAMPLE_SITES = tuple(
  f"cmoe_{side}_foot_sample_point{index}"
  for side in ("left", "right")
  for index in range(1, 6)
)


def _cmoe_robot_cfg():
  return get_cmoe_g1_robot_cfg()


def _cmoe_terrain_cfg(play: bool):
  return cmoe_terrain_generator_cfg(play)


def _cmoe_spec_fn(spec: mujoco.MjSpec) -> None:
  spec.memory = 128_000_000


def g1_cmoe_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the CMoE-G1 environment configuration."""
  lower_cfg = SceneEntityCfg("robot", joint_names=LOWER_BODY_JOINTS)
  foot_cfg = SceneEntityCfg("robot", site_names=FOOT_SITES)
  foot_sample_cfg = SceneEntityCfg("robot", site_names=FOOT_SAMPLE_SITES)
  hip_cfg = SceneEntityCfg("robot", joint_names=HIP_JOINTS)
  proprio_params = {
    "command_name": "twist",
    "asset_cfg": lower_cfg,
    "command_scale": (2.0, 2.0, 0.25),
    "ang_vel_scale": 0.25,
    "joint_pos_scale": 1.0,
    "joint_vel_scale": 0.05,
    "noise_scales": (0.05, 0.05, 0.01, 0.075),
  }
  terrain_scan = mdp.CMoERayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="site", name="cmoe_scan_frame", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.0, 0.6), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
  )
  height_params = {
    "sensor_name": terrain_scan.name,
    "height_scale": 5.0,
    "noise_amplitude": 0.15,
    "extreme_points": 4,
  }
  foot_contact = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="body",
      pattern=("left_ankle_roll_link", "right_ankle_roll_link"),
      entity="robot",
    ),
    fields=("found", "force"),
    reduce="netforce",
    track_air_time=True,
  )
  collision_sensor = ContactSensorCfg(
    name="penalized_contact",
    primary=ContactMatch(mode="body", pattern=r".*(hip|knee).*", entity="robot"),
    fields=("found", "force"),
    reduce="netforce",
  )
  pelvis_sensor = ContactSensorCfg(
    name="pelvis_contact",
    primary=ContactMatch(mode="body", pattern=r".*pelvis.*", entity="robot"),
    fields=("found", "force"),
    reduce="netforce",
  )

  actor_terms = {
    "proprio": ObservationTermCfg(
      func=mdp.ProprioHistory,
      clip=(-100.0, 100.0),
      params={
        **proprio_params,
        "corrupt": True,
      },
    ),
    "height_scan": ObservationTermCfg(
      func=mdp.cmoe_height_scan,
      clip=(-100.0, 100.0),
      params={**height_params, "corrupt": True},
    ),
  }
  critic_terms = {
    "proprio": ObservationTermCfg(
      func=mdp.cmoe_proprio,
      clip=(-100.0, 100.0),
      params=proprio_params,
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.cmoe_base_lin_vel,
      clip=(-100.0, 100.0),
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "external_force": ObservationTermCfg(
      func=mdp.cmoe_external_force,
      clip=(-100.0, 100.0),
    ),
    "height_scan": ObservationTermCfg(
      func=mdp.cmoe_height_scan,
      clip=(-100.0, 100.0),
      params=height_params,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=LOWER_BODY_JOINTS,
      scale=0.25,
      use_default_offset=True,
    )
  }

  events = {
    "reset_scene": EventTermCfg(func=envs_mdp.reset_scene_to_default, mode="reset"),
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.3, 0.3),
          "y": (-0.3, 0.3),
        },
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.5, 0.5),
          "roll": (-0.5, 0.5),
          "pitch": (-0.5, 0.5),
          "yaw": (-0.5, 0.5),
        },
      },
    ),
    "reset_joints": EventTermCfg(
      func=mdp.reset_joints_by_scale,
      mode="reset",
      params={
        "position_range": (0.5, 1.5),
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "reset_height_scan": EventTermCfg(
      func=mdp.reset_height_scan,
      mode="reset",
      params={"sensor_name": terrain_scan.name},
    ),
    "reset_external_force": EventTermCfg(
      func=mdp.reset_external_force,
      mode="reset",
      params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")},
    ),
    "push_robot": EventTermCfg(
      func=mdp.push_robot,
      mode="interval",
      interval_range_s=(16.0, 16.0),
      is_global_time=True,
      params={"velocity_range": (-1.0, 1.0)},
    ),
    "clear_disturbance": EventTermCfg(
      func=mdp.clear_external_force,
      mode="step",
      params={"asset_cfg": SceneEntityCfg("robot", body_names="pelvis")},
    ),
    "disturbance": EventTermCfg(
      func=mdp.apply_external_force_local,
      mode="interval",
      interval_range_s=(0.16, 0.16),
      is_global_time=True,
      params={
        "force_range": (-30.0, 30.0),
        "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
      },
    ),
    "foot_friction": EventTermCfg(
      func=dr.geom_friction,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", geom_names=COLLISION_GEOM_PATTERN
        ),
        "operation": "abs",
        "ranges": (0.3, 1.0),
        "shared_random": True,
      },
    ),
    "base_inertial_properties": EventTermCfg(
      func=mdp.randomize_base_inertial_properties,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
        "payload_range": (-1.0, 2.0),
        "com_range": (-0.05, 0.05),
      },
    ),
    "pd_gains": EventTermCfg(
      func=mdp.randomize_pd_gains,
      mode="reset",
      params={
        "kp_range": (0.9, 1.1),
        "kd_range": (0.9, 1.1),
      },
    ),
  }

  rewards = {
    "tracking_lin_vel": RewardTermCfg(
      func=mdp.tracking_lin_vel,
      weight=2.0,
      params={"command_name": "twist", "sigma": 0.25},
    ),
    "tracking_yaw": RewardTermCfg(
      func=mdp.tracking_yaw,
      weight=2.0,
      params={"command_name": "twist"},
    ),
    "lin_vel_z": RewardTermCfg(func=mdp.lin_vel_z, weight=-1.0),
    "ang_vel_xy": RewardTermCfg(func=mdp.ang_vel_xy, weight=-0.05),
    "orientation": RewardTermCfg(func=mdp.orientation, weight=-2.0),
    "base_height": RewardTermCfg(
      func=mdp.base_height,
      weight=-15.0,
      params={
        "contact_sensor_name": foot_contact.name,
        "target_height": 0.75,
        "foot_height": 0.035,
        "asset_cfg": foot_cfg,
      },
    ),
    "feet_stumble": RewardTermCfg(
      func=mdp.feet_stumble, weight=-1.0, params={"sensor_name": foot_contact.name}
    ),
    "collision": RewardTermCfg(
      func=mdp.collision,
      weight=-15.0,
      params={"sensor_name": collision_sensor.name},
    ),
    "feet_lateral_distance": RewardTermCfg(
      func=mdp.feet_lateral_distance,
      weight=0.8,
      params={"min_distance": 0.18, "max_distance": 0.24, "asset_cfg": foot_cfg},
    ),
    "feet_air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=1.0,
      params={"sensor_name": foot_contact.name},
    ),
    "feet_ground_parallel": RewardTermCfg(
      func=mdp.feet_ground_parallel,
      weight=-0.02,
      params={"asset_cfg": foot_sample_cfg},
    ),
    "feet_edge": RewardTermCfg(
      func=mdp.feet_edge,
      weight=-1.0,
      params={
        "contact_sensor_name": foot_contact.name,
        "asset_cfg": foot_cfg,
      },
    ),
    "hip_dof_error": RewardTermCfg(
      func=mdp.hip_dof_error, weight=-0.5, params={"asset_cfg": hip_cfg}
    ),
    "dof_acc": RewardTermCfg(
      func=mdp.dof_acc, weight=-2.5e-7, params={"asset_cfg": lower_cfg}
    ),
    "dof_vel": RewardTermCfg(
      func=mdp.dof_vel, weight=-5.0e-4, params={"asset_cfg": lower_cfg}
    ),
    "torques": RewardTermCfg(
      func=mdp.torques, weight=-1.0e-5, params={"asset_cfg": lower_cfg}
    ),
    "action_rate": RewardTermCfg(func=mdp.action_rate, weight=-0.3),
    "dof_pos_limits": RewardTermCfg(
      func=mdp.dof_pos_limits, weight=-2.0, params={"asset_cfg": lower_cfg}
    ),
    "dof_vel_limits": RewardTermCfg(
      func=mdp.dof_vel_limits,
      weight=-1.0,
      params={
        "velocity_limits": LOWER_VELOCITY_LIMITS,
        "asset_cfg": lower_cfg,
      },
    ),
    "torque_limits": RewardTermCfg(
      func=mdp.torque_limits,
      weight=-1.0,
      params={"torque_limits": LOWER_TORQUE_LIMITS, "asset_cfg": lower_cfg},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "pelvis_contact": TerminationTermCfg(
      func=mdp.pelvis_contact, params={"sensor_name": pelvis_sensor.name}
    ),
    "bad_orientation": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    ),
    "base_too_low": TerminationTermCfg(
      func=mdp.root_height_below_on_terrain,
      params={"minimum_height": 0.5},
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=4096,
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=_cmoe_terrain_cfg(play),
        max_init_terrain_level=5,
      ),
      entities={"robot": _cmoe_robot_cfg()},
      sensors=(
        terrain_scan,
        foot_contact,
        collision_sensor,
        pelvis_sensor,
      ),
      spec_fn=_cmoe_spec_fn,
      extent=2.0,
    ),
    observations={
      "actor": ObservationGroupCfg(
        terms=actor_terms, concatenate_terms=True, enable_corruption=True
      ),
      "critic": ObservationGroupCfg(
        terms=critic_terms, concatenate_terms=True, enable_corruption=False
      ),
    },
    actions=actions,
    commands={
      "twist": mdp.CMoEVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(10.0, 10.0),
        heading_command=True,
        heading_control_stiffness=0.5,
        rel_heading_envs=1.0,
        ranges=UniformVelocityCommandCfg.Ranges(
          lin_vel_x=(-0.3, 1.0),
          lin_vel_y=(-0.3, 0.3),
          ang_vel_z=(-1.0, 1.0),
          heading=(-1.6, 1.6),
        ),
        hard_ranges=UniformVelocityCommandCfg.Ranges(
          lin_vel_x=(0.3, 1.0),
          lin_vel_y=(0.0, 0.0),
          ang_vel_z=(0.0, 0.0),
          heading=(0.0, 0.0),
        ),
      )
    },
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={
      "terrain_levels": CurriculumTermCfg(
        func=mdp.terrain_levels, params={"command_name": "twist"}
      )
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      mujoco=MujocoCfg(timestep=0.005),
      nan_guard=NanGuardCfg(enabled=True),
    ),
    decimation=4,
    episode_length_s=20.0,
    scale_rewards_by_dt=True,
  )

  if play:
    cfg.scene.num_envs = 150
    cfg.scene.terrain.max_init_terrain_level = 9
    cfg.events.pop("push_robot")
    cfg.events.pop("clear_disturbance")
    cfg.events.pop("disturbance")
    cfg.events["base_inertial_properties"].params["payload_range"] = (0.0, 0.0)
    cfg.observations["actor"].terms["proprio"].params["corrupt"] = False
    cfg.observations["actor"].terms["height_scan"].params["corrupt"] = False
    cfg.observations["critic"].terms["proprio"].params["corrupt"] = False
    cfg.observations["critic"].terms["height_scan"].params["corrupt"] = False
    command = cfg.commands["twist"]
    command.resampling_time_range = (60.0, 60.0)
    command.ranges = UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.8, 0.8),
      lin_vel_y=(0.0, 0.0),
      ang_vel_z=(0.0, 0.0),
      heading=(0.0, 0.0),
    )
    command.hard_ranges = command.ranges

  return cfg
