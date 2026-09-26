"""IsaacLab articulation for the rigid Microtaur.

USD: assets/rigid_microtaur/microtaur_rigid.usda, built from the MJCF by
scripts/isaac_make_usd.py (microtaur_isaac.usd). Servo model identical to the
mjlab one (KP 1, KD 0, 0.129 N m cap, torque-speed envelope). Joint friction:
on Isaac Sim 5.x IsaacLab's `friction` is the static Coulomb torque only, so
`dynamic_friction` is set to the same value to reproduce MuJoCo's frictionloss
(measured 2026-09-25: with dynamic 0 a slipping joint has no friction at all).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from microtaur_common.kinematics import FULL_JOINT_NAMES
from microtaur_common.reset import NOMINAL_FULL_Q, NOMINAL_ROOT_HEIGHT_M
from microtaur_common.robot_constants import (
  ARMATURE, ASSET_DIR, EFFORT_LIMIT_NM, FRICTIONLOSS_NM, KD, KP, LEG_JOINT_NAMES, PASSIVE_ARMATURE,
  PASSIVE_DAMPING, PASSIVE_FRICTIONLOSS_NM, PASSIVE_JOINT_NAMES, VELOCITY_LIMIT_RAD_S, XL330_STALL_TORQUE_NM,
)

USD_PATH = ASSET_DIR / "microtaur_rigid.usda"

# Closed chain in PhysX: 16 position iterations and a 2.5 ms step keep the loop
# closure within MuJoCo's error on trot at KD 0.045 (spike 2026-09-25); at KD 0
# the p99 still passes but the max rises to ~5 mm (open).
SOLVER_POSITION_ITERATIONS = 16
SOLVER_VELOCITY_ITERATIONS = 0

MICROTAUR_RIGID_CFG = ArticulationCfg(
  prim_path="{ENV_REGEX_NS}/Robot",
  spawn=sim_utils.UsdFileCfg(
    usd_path=str(USD_PATH),
    activate_contact_sensors=True,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
      enabled_self_collisions=False,
      solver_position_iteration_count=SOLVER_POSITION_ITERATIONS,
      solver_velocity_iteration_count=SOLVER_VELOCITY_ITERATIONS,
    ),
    # The robot is 10 cm long with a 6.2 mm foot: default contact offsets are
    # centimetres, which would make feet collide in the air.
    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
  ),
  init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, NOMINAL_ROOT_HEIGHT_M),
    joint_pos=dict(zip(FULL_JOINT_NAMES, map(float, NOMINAL_FULL_Q.reshape(16)))),
    joint_vel={".*": 0.0},
  ),
  soft_joint_pos_limit_factor=0.95,
  actuators={
    "motors": DCMotorCfg(
      joint_names_expr=list(LEG_JOINT_NAMES),
      stiffness=KP,
      damping=KD,
      effort_limit=EFFORT_LIMIT_NM,
      saturation_effort=XL330_STALL_TORQUE_NM,
      velocity_limit=VELOCITY_LIMIT_RAD_S,
      armature=ARMATURE,
      friction=FRICTIONLOSS_NM,
      dynamic_friction=FRICTIONLOSS_NM,
      viscous_friction=0.0,
    ),
    "passive": ImplicitActuatorCfg(
      joint_names_expr=list(PASSIVE_JOINT_NAMES),
      stiffness=0.0,
      damping=PASSIVE_DAMPING,
      armature=PASSIVE_ARMATURE,
      friction=PASSIVE_FRICTIONLOSS_NM,
      dynamic_friction=PASSIVE_FRICTIONLOSS_NM,
      viscous_friction=0.0,
    ),
  },
)
