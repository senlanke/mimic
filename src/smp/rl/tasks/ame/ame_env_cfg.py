"""Unitree G1 29DoF AME terrain-locomotion task."""

from __future__ import annotations

import math

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
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from smp.rl.tasks.ame import mdp
from smp.rl.tasks.ame.assets.robots.unitree import get_unitree_g1_29dof_cfg
from smp.rl.tasks.ame.terrains.finetune_terrain_cfg import FINETUNE_ROUGH_TERRAINS_CFG
from smp.rl.tasks.ame.terrains.play_terrain_cfg import PLAY_TERRAIN_CFG
from smp.rl.tasks.ame.terrains.terrain_cfg import ROUGH_TERRAINS_CFG

FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
COLLISION_GEOMS = (".*_collision",)
EFFORT_LIMITS = (
  88.0, 139.0, 88.0, 139.0, 25.0, 25.0,
  88.0, 139.0, 88.0, 139.0, 25.0, 25.0,
  88.0, 25.0, 25.0,
  25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
  25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
)


def _spec_fn(spec: mujoco.MjSpec) -> None:
  spec.memory = 128_000_000
  for geom in spec.geoms:
    if geom.name.endswith("_collision"):
      geom.group = 3
      geom.priority = 1


def g1_ame_env_cfg(
  *, play: bool = False, finetune: bool = False
) -> ManagerBasedRlEnvCfg:
  """Build stage-one or stage-two AME environment configuration."""
  terrain_scan = RayCastSensorCfg(
    name="height_scanner",
    frame=ObjRef(type="body", name="torso_link", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.05),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
  )
  feet_contact = ContactSensorCfg(
    name="feet_contact",
    primary=ContactMatch(
      mode="body",
      pattern=("left_ankle_roll_link", "right_ankle_roll_link"),
      entity="robot",
    ),
    secondary=None,
    fields=("found", "force"),
    reduce="netforce",
    track_air_time=True,
    history_length=3,
  )
  undesired_contact = ContactSensorCfg(
    name="undesired_contact",
    primary=ContactMatch(
      mode="body",
      pattern=r"^(?!.*ankle.*).*$",
      entity="robot",
    ),
    secondary=None,
    fields=("found", "force"),
    reduce="netforce",
    history_length=3,
  )
  illegal_contact = ContactSensorCfg(
    name="illegal_contact",
    primary=ContactMatch(
      mode="body",
      pattern=r"^(torso_link|pelvis|.*_shoulder_.*_link|.*_hip_.*_link|.*_knee_link|.*_elbow_link|waist_.*_link)$",
      entity="robot",
    ),
    secondary=None,
    fields=("found", "force"),
    reduce="netforce",
    history_length=3,
  )

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      scale=0.2,
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "velocity_commands": ObservationTermCfg(
      func=envs_mdp.generated_commands, params={"command_name": "base_velocity"}
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      scale=0.05,
      noise=Unoise(n_min=-2.0, n_max=2.0),
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
    "height_scan": ObservationTermCfg(
      func=mdp.elevation_map,
      params={"sensor_name": terrain_scan.name, "noise": True},
    ),
  }
  critic_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      scale=0.2,
    ),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "velocity_commands": ObservationTermCfg(
      func=envs_mdp.generated_commands, params={"command_name": "base_velocity"}
    ),
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel, scale=0.05),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
    "height_scan": ObservationTermCfg(
      func=mdp.elevation_map,
      params={"sensor_name": terrain_scan.name, "noise": False},
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot", actuator_names=(".*",), scale=0.25,
      use_default_offset=True,
    )
  }
  commands = {
    "base_velocity": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(10.0, 10.0),
      rel_standing_envs=0.0,
      rel_heading_envs=1.0,
      heading_command=True,
      heading_control_stiffness=0.5,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(0.0, 1.5), lin_vel_y=(0.0, 0.0),
        ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi),
      ),
    )
  }
  events = {
    "reset_scene": EventTermCfg(func=envs_mdp.reset_scene_to_default, mode="reset"),
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0), "velocity_range": (-1.0, 1.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "physics_material": EventTermCfg(
      func=mdp.randomize_geom_friction_buckets,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=COLLISION_GEOMS),
        "friction_range": (0.3, 1.0),
        "num_buckets": 64,
      },
    ),
    "add_base_mass": EventTermCfg(
      func=dr.body_mass,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "operation": "add", "ranges": (-1.0, 3.0),
      },
    ),
    "base_com": EventTermCfg(
      func=dr.body_com_offset,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "operation": "add",
        "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.01, 0.01)},
      },
    ),
    "base_external_force_torque": EventTermCfg(
      func=envs_mdp.apply_external_force_torque,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "force_range": (0.0, 0.0),
        "torque_range": (0.0, 0.0),
      },
    ),
    "push_robot": EventTermCfg(
      func=envs_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(5.0, 10.0),
      params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    ),
  }

  rewards = {
    "termination_penalty": RewardTermCfg(func=envs_mdp.is_terminated, weight=-200.0),
    "track_lin_vel_xy_exp": RewardTermCfg(
      func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=2.0,
      params={"command_name": "base_velocity", "std": 0.25},
    ),
    "track_ang_vel_z_exp": RewardTermCfg(
      func=mdp.track_ang_vel_z_world_exp, weight=3.0,
      params={"command_name": "base_velocity", "std": 0.25},
    ),
    "ang_vel_xy_l2": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05),
    "undesired_contacts": RewardTermCfg(
      func=mdp.undesired_contacts, weight=-1.0,
      params={"sensor_name": undesired_contact.name, "threshold": 1.0},
    ),
    "dof_torques_l2": RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=-1.5e-7),
    "dof_acc_l2": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-1.25e-7),
    "dof_vel_l2": RewardTermCfg(func=envs_mdp.joint_vel_l2, weight=-0.001),
    "dof_pos_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-1.0),
    "dof_torques_limits": RewardTermCfg(
      func=mdp.applied_torque_limits, weight=-0.01,
      params={"limits": EFFORT_LIMITS},
    ),
    "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
    "flat_orientation_l2": RewardTermCfg(func=envs_mdp.flat_orientation_l2, weight=-2.0),
    "feet_air_time": RewardTermCfg(
      func=mdp.feet_air_time_positive_biped, weight=0.25,
      params={"sensor_name": feet_contact.name, "command_name": "base_velocity", "threshold": 0.6},
    ),
    "feet_air_time_variance": RewardTermCfg(
      func=mdp.air_time_variance_penalty, weight=-0.7,
      params={"sensor_name": feet_contact.name},
    ),
    "feet_slide": RewardTermCfg(
      func=mdp.feet_slide, weight=-0.1,
      params={"sensor_name": feet_contact.name, "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES)},
    ),
    "feet_stumble": RewardTermCfg(
      func=mdp.feet_stumble, weight=-2.0, params={"sensor_name": feet_contact.name},
    ),
    "feet_too_near": RewardTermCfg(
      func=mdp.feet_too_near, weight=-1.0,
      params={"threshold": 0.2, "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES)},
    ),
    "joint_coordination": RewardTermCfg(
      func=mdp.joint_coordination_rel, weight=-0.2,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "coord_joints": (("left_hip_pitch_joint", "right_shoulder_pitch_joint"), ("right_hip_pitch_joint", "left_shoulder_pitch_joint")),
        "coord_signs": ((1.0, 1.0), (1.0, 1.0)),
      },
    ),
    "joint_deviation_hip": RewardTermCfg(
      func=mdp.joint_deviation_l1, weight=-0.1,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_yaw_joint", ".*_hip_roll_joint"))},
    ),
    "joint_deviation_arms": RewardTermCfg(
      func=mdp.joint_deviation_l1, weight=-0.3,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*"))},
    ),
    "joint_deviation_waists": RewardTermCfg(
      func=mdp.joint_deviation_l1, weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=("waist.*",))},
    ),
  }
  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "base_contact": TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": illegal_contact.name, "force_threshold": 1.0},
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=FINETUNE_ROUGH_TERRAINS_CFG if finetune else ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
      ),
      entities={"robot": get_unitree_g1_29dof_cfg()},
      sensors=(terrain_scan, feet_contact, undesired_contact, illegal_contact),
      num_envs=4096,
      extent=2.5,
      spec_fn=_spec_fn,
    ),
    observations={
      "actor": ObservationGroupCfg(terms=actor_terms, concatenate_terms=True, enable_corruption=True),
      "critic": ObservationGroupCfg(terms=critic_terms, concatenate_terms=True, enable_corruption=False),
    },
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={
      "terrain_levels": CurriculumTermCfg(
        func=mdp.terrain_levels_vel, params={"command_name": "base_velocity"}
      )
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot", body_name="torso_link", distance=3.0,
      elevation=-5.0, azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=140, njmax=3000, contact_sensor_maxmatch=512,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,
    episode_length_s=20.0,
    scale_rewards_by_dt=True,
  )

  if finetune:
    cfg.events["reset_base"].params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    cfg.commands["base_velocity"].ranges.heading = (0.0, 0.0)
    cfg.rewards["dof_torques_limits"].weight = -0.05
    cfg.rewards["action_rate_l2"].weight = -0.05
    cfg.rewards["flat_orientation_l2"].weight = -5.0
    cfg.rewards["feet_air_time"].weight = 0.5
    cfg.rewards["feet_air_time_variance"].weight = -2.0
    cfg.rewards["feet_slide"].weight = -0.3
    cfg.rewards["feet_stumble"].weight = -5.0
    cfg.rewards["feet_too_near"].weight = -5.0
    cfg.rewards["joint_coordination"].weight = -0.5
  else:
    cfg.events.pop("push_robot")
    cfg.events.pop("add_base_mass")
    cfg.events.pop("base_com")
    cfg.observations["actor"].enable_corruption = False
    cfg.observations["actor"].terms["height_scan"].params["noise"] = False

  if play:
    cfg.scene.num_envs = 50
    cfg.episode_length_s = 40.0
    cfg.curriculum = {}
    cfg.events.pop("base_external_force_torque")
    if finetune:
      cfg.events.pop("push_robot")
    cfg.scene.terrain.terrain_generator = PLAY_TERRAIN_CFG
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.events["reset_base"].params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
    cfg.observations["actor"].enable_corruption = False
    cfg.observations["actor"].terms["height_scan"].params["noise"] = False
    command = cfg.commands["base_velocity"]
    command.ranges.lin_vel_x = (1.0, 1.0)
    command.ranges.lin_vel_y = (0.0, 0.0)
    command.ranges.heading = (0.0, 0.0)

  return cfg


__all__ = ["g1_ame_env_cfg"]
