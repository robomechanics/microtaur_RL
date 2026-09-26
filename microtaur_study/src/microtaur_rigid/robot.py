"""Rigid Microtaur model: names, XL330 motor model, MjSpec patches, entity cfg.

Model: assets/rigid_microtaur/robot_modified.xml, byte-identical to upstream
aryan-chandra-cmu/Microtaur_RL eef9f92 (2026-09-20). Its root inertial is the
hardware-identified one (total mass 0.540 kg, Tests 21-22), and every joint's
armature / frictionloss / damping is already set in the XML. The only model
change made here is the motor joint range (stand +/- 0.75 rad).

The actuator model is entirely in Python (the XML has no <actuator> block).
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from microtaur_common.robot_constants import (  # noqa: F401  (re-exported)
  ASSET_DIR,
  XML_PATH,
  EXPECTED_TOTAL_MASS_KG,
  ROOT_BODY,
  LEG_JOINT_NAMES,
  PASSIVE_JOINT_NAMES,
  FOOT_SITE_NAMES,
  FOOT_GEOM_NAMES,
  BODY_COLLISION_GEOM_NAMES,
  STAND_A,
  STAND_E,
  LEG_SIGNS,
  STAND_JOINT_POS,
  JOINT_HALF_RANGE_RAD,
  XL330_SUPPLY_V,
  XL330_STALL_TORQUE_NM,
  XL330_STALL_CURRENT_A,
  XL330_NO_LOAD_SPEED_RAD_S,
  KP,
  KD,
  ARMATURE,
  EFFORT_LIMIT_NM,
  VELOCITY_LIMIT_RAD_S,
  FRICTIONLOSS_NM,
  XL330_R_OHM,
  XL330_KT_NM_PER_A,
  XL330_COPPER_W_PER_NM2,
  motor_joint_ranges,
)


def _load_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(XML_PATH))
  meshdir = ASSET_DIR / spec.meshdir
  spec.assets = {
    p.relative_to(ASSET_DIR).as_posix(): p.read_bytes()
    for p in meshdir.rglob("*")
    if p.is_file()
  } | {p.name: p.read_bytes() for p in meshdir.rglob("*") if p.is_file()}
  return spec


def make_spec() -> mujoco.MjSpec:
  """XML plus the motor-range patch, with a mass check."""
  spec = _load_spec()
  for name, (lo, hi) in motor_joint_ranges().items():
    joint = spec.joint(name)
    joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    joint.range[:] = np.array((lo, hi))
  for name in FOOT_SITE_NAMES:
    if spec.site(name) is None:
      raise ValueError(f"Missing foot site {name}")

  total = float(spec.compile().body_subtreemass[1])
  if abs(total - EXPECTED_TOTAL_MASS_KG) > 1e-6:
    raise ValueError(
      f"Rigid Microtaur total mass is {total:.6f} kg, expected "
      f"{EXPECTED_TOTAL_MASS_KG:.3f} kg (measured). Wrong XML?"
    )
  return spec


def make_robot_cfg() -> EntityCfg:
  motors = DcMotorActuatorCfg(
    target_names_expr=LEG_JOINT_NAMES,
    stiffness=KP,
    damping=KD,
    effort_limit=EFFORT_LIMIT_NM,
    saturation_effort=XL330_STALL_TORQUE_NM,
    velocity_limit=VELOCITY_LIMIT_RAD_S,
    armature=ARMATURE,
    frictionloss=FRICTIONLOSS_NM,
  )
  init_state = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.10),  # overwritten by the IK-consistent reset
    joint_pos=copy.deepcopy(STAND_JOINT_POS),
    joint_vel={".*": 0.0},
  )
  return EntityCfg(
    init_state=init_state,
    collisions=(),
    spec_fn=make_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=(motors,),
      soft_joint_pos_limit_factor=0.95,
    ),
  )
