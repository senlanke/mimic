"""Unitree G1 29DoF configuration used by AME_Locomotion."""

from __future__ import annotations

import copy

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import FULL_COLLISION, get_spec
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


G1_JOINT_SDK_NAMES = (
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
  "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
  "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

AME_G1_INITIAL_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.8),
  joint_pos={
    "left_hip_pitch_joint": -0.1,
    "right_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.3,
    ".*_ankle_pitch_joint": -0.2,
    ".*_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.25,
    "right_shoulder_roll_joint": -0.25,
    ".*_elbow_joint": 0.97,
    "left_wrist_roll_joint": 0.15,
    "right_wrist_roll_joint": -0.15,
  },
  joint_vel={".*": 0.0},
)

AME_G1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_hip_pitch_joint", ".*_hip_yaw_joint"),
      stiffness=100.0,
      damping=2.0,
      effort_limit=88.0,
      armature=0.01,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=("waist_yaw_joint",),
      stiffness=200.0,
      damping=5.0,
      effort_limit=88.0,
      armature=0.01,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_hip_roll_joint",),
      stiffness=100.0,
      damping=2.0,
      effort_limit=139.0,
      armature=0.01,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_knee_joint",),
      stiffness=150.0,
      damping=4.0,
      effort_limit=139.0,
      armature=0.01,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(
        ".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_roll_joint",
      ),
      stiffness=40.0,
      damping=10.0,
      effort_limit=25.0,
      armature=0.01,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_ankle_.*",),
      stiffness=40.0,
      damping=2.0,
      effort_limit=25.0,
      armature=0.01,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=("waist_roll_joint", "waist_pitch_joint"),
      stiffness=40.0,
      damping=5.0,
      effort_limit=25.0,
      armature=0.01,
    ),
    BuiltinPositionActuatorCfg(
      target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
      stiffness=40.0,
      damping=10.0,
      effort_limit=5.0,
      armature=0.01,
    ),
  ),
  soft_joint_pos_limit_factor=0.9,
)

UNITREE_G1_29DOF_CFG = EntityCfg(
  init_state=AME_G1_INITIAL_STATE,
  collisions=(FULL_COLLISION,),
  spec_fn=get_spec,
  articulation=AME_G1_ARTICULATION,
)


def get_unitree_g1_29dof_cfg() -> EntityCfg:
  return copy.deepcopy(UNITREE_G1_29DOF_CFG)


__all__ = [
  "AME_G1_INITIAL_STATE",
  "G1_JOINT_SDK_NAMES",
  "UNITREE_G1_29DOF_CFG",
  "get_unitree_g1_29dof_cfg",
]
