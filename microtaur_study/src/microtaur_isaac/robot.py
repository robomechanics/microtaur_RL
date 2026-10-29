"""IsaacLab articulation for the rigid Microtaur.

USD: assets/rigid_microtaur/microtaur_rigid.usda, built from the MJCF by
scripts/isaac_make_usd.py (microtaur_isaac.usd). Servo model identical to the
mjlab one (KP, KD, effort cap, torque-speed envelope and friction from
microtaur_common.robot_constants, measured 2026-10-28). Joint friction:
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
# Same asset plus the visual meshes (scripts/isaac_make_usd.py); used by the Play tasks.
VISUAL_USD_PATH = ASSET_DIR / "microtaur_rigid_visual.usd"

# Closed chain in PhysX: 16 position iterations and a 2.5 ms step keep the loop
# closure within MuJoCo's error on trot at KD 0.045 (spike 2026-09-25). Re-checked
# 2026-10-28 in the training env (scripts/isaac_closure_check.py: Teacher-Cur, DR +
# pushes, 256 envs x 400 steps of t2 trotting, effort 0.40 stall, friction 0.028):
# closing-joint gap p99 / max = 0.153 / 3.19 mm at KD 0, 0.136 / 1.91 mm at KD 0.0092,
# 0.121 / 0.89 mm at KD 0.045; no env above 5 mm at any of them.
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


# -----------------------------------------------------------------------------
# Active-yaw variant (assets/active_yaw_microtaur/robot_2imu.xml): the same legs and leg motors plus the
# spine XL330-M288. USD built by scripts/isaac_make_usd_yaw.py, which also checks the masses, inertias,
# closure, the spine inertia / natural frequency and standing against MuJoCo.
# -----------------------------------------------------------------------------
from microtaur_common.robot_constants import (  # noqa: E402
  SPINE_ARMATURE, SPINE_EFFORT_LIMIT_NM, SPINE_FRICTIONLOSS_NM, SPINE_JOINT_NAME, SPINE_KD, SPINE_KP,
  SPINE_VELOCITY_LIMIT_RAD_S, XL330_M288_STALL_TORQUE_NM, YAW_ASSET_DIR,
)

YAW_USD_PATH = YAW_ASSET_DIR / "microtaur_yaw.usda"
YAW_VISUAL_USD_PATH = YAW_ASSET_DIR / "microtaur_yaw_visual.usd"
# Root (front-half mount) height with the lowest foot sphere on the ground at the stand pose and spine 0
# (MuJoCo, robot_2imu.xml). The rear feet are 5.2 mm higher than the front ones at equal leg angles.
YAW_NOMINAL_ROOT_HEIGHT_M = 0.076492

MICROTAUR_YAW_CFG = MICROTAUR_RIGID_CFG.replace(
  spawn=MICROTAUR_RIGID_CFG.spawn.replace(usd_path=str(YAW_USD_PATH)),
  init_state=MICROTAUR_RIGID_CFG.init_state.replace(
    pos=(0.0, 0.0, YAW_NOMINAL_ROOT_HEIGHT_M),
    joint_pos={**MICROTAUR_RIGID_CFG.init_state.joint_pos, SPINE_JOINT_NAME: 0.0},
  ),
  actuators={
    **MICROTAUR_RIGID_CFG.actuators,
    "spine": DCMotorCfg(
      joint_names_expr=[SPINE_JOINT_NAME],
      stiffness=SPINE_KP,
      damping=SPINE_KD,
      effort_limit=SPINE_EFFORT_LIMIT_NM,
      saturation_effort=XL330_M288_STALL_TORQUE_NM,
      velocity_limit=SPINE_VELOCITY_LIMIT_RAD_S,
      armature=SPINE_ARMATURE,
      friction=SPINE_FRICTIONLOSS_NM,
      dynamic_friction=SPINE_FRICTIONLOSS_NM,
      viscous_friction=0.0,
    ),
  },
)
