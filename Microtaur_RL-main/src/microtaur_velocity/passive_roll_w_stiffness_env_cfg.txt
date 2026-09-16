from __future__ import annotations

ENV_CFG_REVISION = "2026-08-04-olympus-to-microtaur-physical-scaling-v6"

import copy
import math
import os
from dataclasses import MISSING, replace

import mujoco
import numpy as np
import torch

try:
  from mjlab.actuator import DcMotorActuatorCfg
except ImportError:
  from mjlab.actuator.dc_actuator import DcMotorActuatorCfg

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sensor.builtin_sensor import ObjRef
from mjlab.sensor.raycast_sensor import RayCastSensorCfg, RingPatternCfg
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

try:
  from mjlab.utils import configclass
except Exception:
  from dataclasses import dataclass as configclass

from microtaur_velocity.microtaur_config import (
  MICRO_ROUGH_TERRAINS_CFG,
  make_microtaur_terrain_scan_cfg,
)
from microtaur_velocity.microtaur_constants import (
  ACTION_JOINT_NAMES,
  CONTROL_SPINE,
  FOOT_GEOM_NAMES,
  FOOT_SITE_NAMES,
  HAS_ACTIVE_SPINE,
  INIT_STATE,
  LEG_JOINT_NAMES,
  MICROTAUR_STAND_A_OFFSETS,
  MICROTAUR_STAND_E_OFFSETS,
  MINITAUR_SWING_SIGNS,
  OBS_JOINT_NAMES,
  POSE_JOINT_NAMES,
  ROOT_BODY,
  SPINE_JOINT_NAMES,
  get_spec,
)

from microtaur_velocity.microtaur_kinematics import (
  FOOT_SPHERE_RADIUS_M,
  FULL_JOINT_NAMES as IK_FULL_JOINT_NAMES,
  HIP_MIDPOINTS_ROOT_M,
  LEG_SIGNS as IK_LEG_SIGNS,
  MicrotaurFiveBarKinematics,
)


# -----------------------------------------------------------------------------
# MJLab v1.2 tuple/list compatibility
# -----------------------------------------------------------------------------

def _patch_scene_entity_cfg_tuple_normalization() -> None:
  """Backport tuple normalization used by newer MJLab releases."""
  normalize = getattr(SceneEntityCfg, "_normalize_to_list", None)
  if normalize is None:
    return

  try:
    probe = SceneEntityCfg("__microtaur_probe__")
    if probe._normalize_to_list(("a", "b")) == ["a", "b"]:
      return
  except Exception:
    pass

  def _normalize_to_list_compat(self, value):
    if value is None:
      return None
    if isinstance(value, (str, int)):
      return [value]
    if isinstance(value, list):
      return value
    if isinstance(value, tuple):
      return list(value)
    try:
      return list(value)
    except TypeError:
      return normalize(self, value)

  SceneEntityCfg._normalize_to_list = _normalize_to_list_compat


_patch_scene_entity_cfg_tuple_normalization()


# -----------------------------------------------------------------------------
# Robot and XL330 actuator model
# -----------------------------------------------------------------------------

# ROBOTIS XL330-M288-T output-shaft data:
# supply voltage: (stall torque [N m], no-load speed [rpm], stall current [A])
_XL330_TABLE = {
  3.7: (0.42, 76.0, 1.11),
  5.0: (0.52, 103.0, 1.47),
  6.0: (0.60, 123.0, 1.74),
}

XL330_SUPPLY_VOLTAGE = float(os.environ.get("MICROTAUR_XL330_VOLTAGE", "5.0"))
if XL330_SUPPLY_VOLTAGE not in _XL330_TABLE:
  raise ValueError(
    "MICROTAUR_XL330_VOLTAGE must be exactly 3.7, 5.0, or 6.0; "
    f"got {XL330_SUPPLY_VOLTAGE}."
  )

(
  XL330_STALL_TORQUE_NM,
  XL330_NO_LOAD_SPEED_RPM,
  XL330_STALL_CURRENT_A,
) = _XL330_TABLE[XL330_SUPPLY_VOLTAGE]

XL330_NO_LOAD_SPEED_RAD_S = XL330_NO_LOAD_SPEED_RPM * 2.0 * math.pi / 60.0

# XL330 limits are based on the regulated servo-bus voltage, not the raw
# 2S LiPo voltage. The default assumes a 5 V servo supply.
XL330_POSITION_KP = float(os.environ.get("MICROTAUR_XL330_KP", "1.0"))
XL330_POSITION_KD = float(os.environ.get("MICROTAUR_XL330_KD", "0.045"))
XL330_ARMATURE = float(os.environ.get("MICROTAUR_ARMATURE", "0.0002"))

# Keep the physical stall torque as the motor saturation limit.
# effort_limit is the lower training-time torque cap used during normal control.
XL330_SAFE_TORQUE_FRACTION = float(
  os.environ.get("MICROTAUR_XL330_SAFE_TORQUE_FRACTION", "0.60")
)
XL330_VELOCITY_LIMIT_FRACTION = float(
  os.environ.get("MICROTAUR_XL330_VELOCITY_LIMIT_FRACTION", "0.80")
)
if not 0.0 < XL330_SAFE_TORQUE_FRACTION <= 1.0:
  raise ValueError("MICROTAUR_XL330_SAFE_TORQUE_FRACTION must be in (0, 1].")
if not 0.0 < XL330_VELOCITY_LIMIT_FRACTION <= 1.0:
  raise ValueError("MICROTAUR_XL330_VELOCITY_LIMIT_FRACTION must be in (0, 1].")

XL330_SAFE_EFFORT_NM = (
  XL330_SAFE_TORQUE_FRACTION * XL330_STALL_TORQUE_NM
)
XL330_VELOCITY_LIMIT_RAD_S = (
  XL330_VELOCITY_LIMIT_FRACTION * XL330_NO_LOAD_SPEED_RAD_S
)
XL330_OUTPUT_FRICTIONLOSS_NM = float(
  os.environ.get("MICROTAUR_XL330_FRICTIONLOSS", "0.010")
)
PASSIVE_JOINT_FRICTIONLOSS_NM = float(
  os.environ.get("MICROTAUR_PASSIVE_FRICTIONLOSS", "0.001")
)
PASSIVE_JOINT_ARMATURE = float(
  os.environ.get("MICROTAUR_PASSIVE_ARMATURE", "0.000001")
)
PASSIVE_JOINT_DAMPING = float(
  os.environ.get("MICROTAUR_PASSIVE_DAMPING", "0.0001")
)

PASSIVE_JOINT_NAMES = tuple(
  name
  for leg_idx in range(1, 5)
  for name in (f"leg{leg_idx}_b_joint", f"leg{leg_idx}_d_joint")
)

# The current policy commands motor positions directly. Keep the physical
# joint range wider than the +/-30 degree policy action so the controller has
# recovery margin without exposing the full CAD range.
MICROTAUR_SWING_SCALE = float(
  os.environ.get("MICROTAUR_SWING_SCALE", "0.45")
)
MICROTAUR_LIFT_SCALE = float(
  os.environ.get("MICROTAUR_LIFT_SCALE", "0.45")
)
ACTUATED_JOINT_LIMIT_MARGIN_RAD = float(
  os.environ.get("MICROTAUR_JOINT_LIMIT_MARGIN", "0.08")
)
MICROTAUR_JOINT_HALF_RANGE_RAD = float(
  os.environ.get("MICROTAUR_JOINT_HALF_RANGE_RAD", "0.75")
)


def _active_joint_ranges() -> dict[str, tuple[float, float]]:
  half_range = MICROTAUR_JOINT_HALF_RANGE_RAD
  ranges: dict[str, tuple[float, float]] = {}
  for leg_idx, (stand_a, stand_e) in enumerate(
    zip(MICROTAUR_STAND_A_OFFSETS, MICROTAUR_STAND_E_OFFSETS),
    start=1,
  ):
    ranges[f"leg{leg_idx}_a_joint_act"] = ( # Center each motor limit on its nominal stand angle.
      float(stand_a) - half_range, 
      float(stand_a) + half_range,
    )
    ranges[f"leg{leg_idx}_e_joint_act"] = (
      float(stand_e) - half_range,
      float(stand_e) + half_range,
    )
  return ranges


def _make_xml_validated_spec() -> mujoco.MjSpec:
  """Load the Microtaur XML and repair RL-critical CAD defaults."""
  spec = get_spec()

  for joint_name, limits in _active_joint_ranges().items():
    joint = spec.joint(joint_name)
    if joint is None:
      raise ValueError(f"Missing actuated Microtaur joint: {joint_name}")
    joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    joint.range[:] = np.asarray(limits, dtype=np.float64)
    joint.armature = XL330_ARMATURE
    joint.frictionloss = XL330_OUTPUT_FRICTIONLOSS_NM

  for joint_name in PASSIVE_JOINT_NAMES:
    joint = spec.joint(joint_name)
    if joint is None:
      raise ValueError(f"Missing passive Microtaur joint: {joint_name}")
    joint.armature = PASSIVE_JOINT_ARMATURE
    joint.frictionloss = PASSIVE_JOINT_FRICTIONLOSS_NM
    joint.damping[:] = 0.0
    joint.damping[0] = PASSIVE_JOINT_DAMPING

  for site_name in FOOT_SITE_NAMES:
    if spec.site(site_name) is None:
      raise ValueError(f"Missing Microtaur foot site: {site_name}")

  return spec

#
def _make_microtaur_robot_cfg() -> EntityCfg:
  motor_cfg = DcMotorActuatorCfg(
    target_names_expr=LEG_JOINT_NAMES,
    stiffness=XL330_POSITION_KP,
    damping=XL330_POSITION_KD,
    effort_limit=XL330_SAFE_EFFORT_NM,
    saturation_effort=XL330_STALL_TORQUE_NM,
    velocity_limit=XL330_VELOCITY_LIMIT_RAD_S,
    armature=XL330_ARMATURE,
    frictionloss=XL330_OUTPUT_FRICTIONLOSS_NM,
  )

  return EntityCfg(
    init_state=copy.deepcopy(INIT_STATE),
    collisions=(),
    spec_fn=_make_xml_validated_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=(motor_cfg,),
      soft_joint_pos_limit_factor=0.95,
    ),
  )


# -----------------------------------------------------------------------------
# Forward/yaw command curriculum
# -----------------------------------------------------------------------------

MICROTAUR_LIN_VEL_Y_RANGE = (0.0, 0.0)

# Play-time command source switch.
# False (default): keep normal random/curriculum velocity commands.
# True: in play mode disable random resampling and let an external controller
# write [vx, 0, yaw] into the existing "twist" command before policy inference.
# Training always keeps the normal random/curriculum commands.
USE_JOYSTICK_COMMANDS = os.environ.get(
  "MICROTAUR_USE_JOYSTICK_COMMANDS",
  "0",
).lower() not in {"0", "false", "no"}

# Forward-only is the default. Enable mirroring explicitly when training
# both forward and reverse commands.
MIRROR_FORWARD_VELOCITY = os.environ.get(
  "MICROTAUR_MIRROR_FORWARD_VELOCITY", "0"
).lower() not in {"0", "false", "no"}

# MJLab's environment step counter restarts when a checkpoint is resumed.
# This offset lets a resumed policy continue from the intended command stage.
# Use 0 for a new run; 90,000 selects the final stage used for fine-tuning.
CURRICULUM_START_STEP = int(
  os.environ.get("MICROTAUR_CURRICULUM_START_STEP", "0")
)
if CURRICULUM_START_STEP < 0:
  raise ValueError("MICROTAUR_CURRICULUM_START_STEP must be non-negative.")


def _command_x_range(stage_range: tuple[float, float]) -> tuple[float, float]:
  """Mirror a positive speed curriculum around zero without changing stages."""
  low, high = (float(stage_range[0]), float(stage_range[1]))
  if not MIRROR_FORWARD_VELOCITY:
    return (low, high)
  max_abs = max(abs(low), abs(high))
  return (-max_abs, max_abs)


# Joint angles are dimensionless, so action angles are not scaled by robot
# size. Linear command and reset distances below are chosen in Microtaur-scale
# SI units. The stage table stores positive forward speeds; optional mirroring
# is handled by _command_x_range().
COMMAND_STAGES = (
  {
    "step": 0,
    "lin_vel_x": (0.10, 0.18),
    "ang_vel_z": (0.00, 0.00),
    "standing": 0.00,
  },
  {
    "step": 25_000,
    "lin_vel_x": (0.08, 0.20),
    "ang_vel_z": (-0.10, 0.10), #was -0.10 to 0.10
    "standing": 0.05,
  },
  {
    "step": 50_000,
    "lin_vel_x": (0.08, 0.20),
    "ang_vel_z": (-0.150, 0.150), # was -0.20 to 0.20
    "standing": 0.05,
  },
  {
    "step": 70_000,
    "lin_vel_x": (0.08, 0.20),
    "ang_vel_z": (-0.20, 0.20), #was -0.25 to 0.25
    "standing": 0.10,
  },
  {
    "step": 90_000,
    "lin_vel_x": (0.08, 0.20),
    "ang_vel_z": (-0.250, 0.250), #was -0.3 to 0.3
    "standing": 0.10,
  },
)

PLAY_COMMAND_STAGE = int(
  os.environ.get("MICROTAUR_PLAY_COMMAND_STAGE", str(len(COMMAND_STAGES) - 1))
)


def _set_runtime_command_stage(env, command_name: str, stage: dict) -> None:
  term = env.command_manager.get_term(command_name)
  if term is None:
    raise KeyError(f"Command term '{command_name}' was not found.")
  cfg = term.cfg
  if not isinstance(cfg, UniformVelocityCommandCfg):
    raise TypeError(f"'{command_name}' must be UniformVelocityCommandCfg.")
  cfg.ranges.lin_vel_x = tuple(stage["lin_vel_x"])
  #cfg.ranges.lin_vel_x = _command_x_range(tuple(stage["lin_vel_x"]))
  cfg.ranges.lin_vel_y = MICROTAUR_LIN_VEL_Y_RANGE
  cfg.ranges.ang_vel_z = tuple(stage["ang_vel_z"])
  cfg.rel_standing_envs = float(stage["standing"])


def _enforce_mirrored_speed_band(env, command_name: str, stage: dict) -> None:
  """Keep non-standing x commands outside the curriculum's zero-speed gap.

  UniformVelocityCommand samples one continuous interval. Mirroring (0.08,
  0.20) as (-0.20, 0.20) would otherwise introduce unintended commands with
  magnitudes below 0.08 m/s. This idempotent clamp preserves exact zero for
  designated standing environments and maps only nonzero values in the gap to
  the stage's minimum commanded speed.
  """
  if not MIRROR_FORWARD_VELOCITY:
    return
  command = env.command_manager.get_command(command_name)
  if command is None:
    return
  min_abs = min(abs(float(stage["lin_vel_x"][0])), abs(float(stage["lin_vel_x"][1])))
  if min_abs <= 0.0:
    return
  x = command[:, 0]
  nonzero = torch.abs(x) > 1e-6
  in_gap = nonzero & (torch.abs(x) < min_abs)
  command[in_gap, 0] = torch.sign(x[in_gap]) * min_abs


def microtaur_command_curriculum(
  env,
  env_ids: torch.Tensor | slice,
  command_name: str,
  stages: tuple[dict, ...],
) -> dict[str, torch.Tensor]:
  del env_ids

  step = (
    CURRICULUM_START_STEP
    + int(getattr(env, "common_step_counter", 0))
  )
  stage_idx = 0
  for idx, stage in enumerate(stages):
    if step >= int(stage["step"]):
      stage_idx = idx
    else:
      break

  stage = stages[stage_idx]
  _set_runtime_command_stage(env, command_name, stage)
  _enforce_mirrored_speed_band(env, command_name, stage)

  device = env.episode_length_buf.device
  as_tensor = lambda value: torch.tensor(float(value), device=device)
  return {
    "stage": as_tensor(stage_idx),
    "lin_vel_x_min": as_tensor(_command_x_range(stage["lin_vel_x"])[0]),
    "lin_vel_x_max": as_tensor(_command_x_range(stage["lin_vel_x"])[1]),
    "ang_vel_z_min": as_tensor(stage["ang_vel_z"][0]),
    "ang_vel_z_max": as_tensor(stage["ang_vel_z"][1]),
    "standing_fraction": as_tensor(stage["standing"]),
  }


def _configure_commands(cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
  cmd = cfg.commands["twist"]
  if not isinstance(cmd, UniformVelocityCommandCfg):
    raise TypeError("Expected UniformVelocityCommandCfg for command 'twist'.")

  stage_idx = PLAY_COMMAND_STAGE if play else 0
  stage_idx = max(0, min(stage_idx, len(COMMAND_STAGES) - 1))
  stage = COMMAND_STAGES[stage_idx]

  cmd.viz.z_offset = 0.12
  cmd.rel_heading_envs = 0.0
  if hasattr(cmd, "rel_forward_envs"):
    cmd.rel_forward_envs = 0.0 #fahhed it up before
  cmd.heading_command = False
  if hasattr(cmd, "init_velocity_prob"):
    cmd.init_velocity_prob = 0.0
  cmd.ranges.heading = None

  cfg.curriculum.pop("command_vel", None)

  # Joystick mode is play-only. Training remains random/curriculum-driven.
  if play and USE_JOYSTICK_COMMANDS:
    cmd.resampling_time_range = (1.0e9, 1.0e9)
    cmd.rel_standing_envs = 0.0
    cmd.ranges.lin_vel_x = (0.0, 0.0)
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    cmd.ranges.ang_vel_z = (0.0, 0.0)
    cfg.curriculum.pop("microtaur_command_ranges", None)
    return

  # Normal random-command behavior.
  cmd.resampling_time_range = (6.0, 10.0)
  cmd.rel_standing_envs = float(stage["standing"])
  cmd.ranges.lin_vel_x = tuple(stage["lin_vel_x"])
  # cmd.ranges.lin_vel_x = _command_x_range(tuple(stage["lin_vel_x"]))
  cmd.ranges.lin_vel_y = MICROTAUR_LIN_VEL_Y_RANGE
  cmd.ranges.ang_vel_z = tuple(stage["ang_vel_z"])

  runtime_stages = (stage,) if play else COMMAND_STAGES
  cfg.curriculum["microtaur_command_ranges"] = CurriculumTermCfg(
    func=microtaur_command_curriculum,
    params={"command_name": "twist", "stages": runtime_stages},
  )


def set_joystick_twist_command(
  env,
  forward_velocity: float,
  yaw_velocity: float,
  command_name: str = "twist",
) -> None:
  """Write [vx, 0, yaw] into MJLab's live twist command tensor."""
  if not USE_JOYSTICK_COMMANDS:
    raise RuntimeError(
      "Joystick injection is disabled. Set "
      "MICROTAUR_USE_JOYSTICK_COMMANDS=1 before constructing the play env."
    )

  base_env = getattr(env, "unwrapped", env)
  term = base_env.command_manager.get_term(command_name)
  command = base_env.command_manager.get_command(command_name)

  if command.ndim != 2 or command.shape[1] < 3:
    raise ValueError(
      f"Expected '{command_name}' command shape [N, >=3], got {tuple(command.shape)}."
    )

  final_stage = COMMAND_STAGES[-1]
  vx_min = float(final_stage["lin_vel_x"][0])
  vx_max = float(final_stage["lin_vel_x"][1])
  wz_min = float(final_stage["ang_vel_z"][0])
  wz_max = float(final_stage["ang_vel_z"][1])

  vx_requested = float(forward_velocity)
  vx = 0.0 if vx_requested <= 1.0e-6 else max(vx_min, min(vx_requested, vx_max))
  wz = max(wz_min, min(float(yaw_velocity), wz_max))

  command[:, 0] = vx
  command[:, 1] = 0.0
  command[:, 2] = wz

  # Prevent command resampling even after manual resets.
  if hasattr(term, "time_left"):
    term.time_left.fill_(1.0e9)


# -----------------------------------------------------------------------------
# Closed-chain-consistent reset initialization
# -----------------------------------------------------------------------------

_MICROTAUR_KINEMATICS = MicrotaurFiveBarKinematics()

# Build reset targets around the foot position produced by the policy's stand
# offsets so a zero action does not introduce a pose jump after reset.
_IK_NOMINAL_FOOT_XZ = np.stack(
  [
    _MICROTAUR_KINEMATICS.forward_numpy(
      MICROTAUR_STAND_A_OFFSETS[leg],
      MICROTAUR_STAND_E_OFFSETS[leg],
      leg_index=leg + 1,
    ).foot
    for leg in range(4)
  ],
  axis=0,
)
_IK_NOMINAL_FULL_Q = np.stack(
  [
    _MICROTAUR_KINEMATICS.forward_numpy(
      MICROTAUR_STAND_A_OFFSETS[leg],
      MICROTAUR_STAND_E_OFFSETS[leg],
      leg_index=leg + 1,
    ).full
    for leg in range(4)
  ],
  axis=0,
)


# Derive nominal root height from the Microtaur geometry so the lowest foot
# collision sphere starts at the terrain surface.
MICROTAUR_NOMINAL_ROOT_HEIGHT_M = float(
  FOOT_SPHERE_RADIUS_M
  - np.min(HIP_MIDPOINTS_ROOT_M[:, 2] + _IK_NOMINAL_FOOT_XZ[:, 1])
)
MICROTAUR_MIN_ROOT_HEIGHT_M = float(
  os.environ.get(
    "MICROTAUR_MIN_ROOT_HEIGHT_M",
    str(MICROTAUR_NOMINAL_ROOT_HEIGHT_M - 0.018),
  )
)

IK_RESET_ROOT_X_RANGE_M = (-0.10, 0.10)
IK_RESET_ROOT_Y_RANGE_M = (-0.10, 0.10)
IK_RESET_ROOT_YAW_RANGE_RAD = (-math.pi, math.pi)

# Reset randomization stays close to the nominal workspace. A shared z shift
# changes body height; zero-mean per-leg jitter adds mild recovery states.
IK_RESET_FOOT_X_OFFSET_RANGE_M = (-0.006, 0.006)
IK_RESET_SHARED_Z_OFFSET_RANGE_M = (-0.004, 0.004)
IK_RESET_PER_LEG_Z_JITTER_RANGE_M = (-0.001, 0.001)

# Foot sites are located at the centers of the 6.2 mm collision spheres.
IK_RESET_LOWEST_FOOT_CENTER_Z_RANGE_M = (
  FOOT_SPHERE_RADIUS_M,
  FOOT_SPHERE_RADIUS_M + 0.002,
)


def _env_id_tensor(env, env_ids: torch.Tensor | slice) -> torch.Tensor:
  if isinstance(env_ids, torch.Tensor):
    return env_ids.to(device=env.device, dtype=torch.long)
  return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]


def _sample_uniform_tensor(
  count: int,
  shape_tail: tuple[int, ...],
  value_range: tuple[float, float],
  device,
  dtype=torch.float32,
) -> torch.Tensor:
  low, high = (float(value_range[0]), float(value_range[1]))
  return (
    low
    + (high - low)
    * torch.rand((count, *shape_tail), device=device, dtype=dtype)
  )


def _ordered_joint_ids(
  robot,
  names: tuple[str, ...],
  device,
) -> torch.Tensor:
  ids, resolved = robot.find_joints(
    list(names),
    preserve_order=True,
  )
  if tuple(resolved) != tuple(names):
    raise ValueError(
      "Microtaur IK reset joint ordering mismatch. "
      f"Expected {names}, got {tuple(resolved)}."
    )
  return torch.tensor(ids, device=device, dtype=torch.long)


def microtaur_ik_consistent_reset(
  env,
  env_ids: torch.Tensor | slice,
  *,
  randomize: bool,
  root_x_range: tuple[float, float],
  root_y_range: tuple[float, float],
  root_yaw_range: tuple[float, float],
  foot_x_offset_range: tuple[float, float],
  shared_z_offset_range: tuple[float, float],
  per_leg_z_jitter_range: tuple[float, float],
  lowest_foot_center_z_range: tuple[float, float],
) -> None:
  """Reset the robot from reachable foot targets and a consistent joint state.

  The planar five-bar has an analytical IK solution, so each reset can solve
  active and passive joints together before writing them to simulation. This
  avoids the large constraint impulse caused by resetting only the motors.
  """
  robot = env.scene["robot"]
  env_ids = _env_id_tensor(env, env_ids)
  count = int(env_ids.numel())
  if count == 0:
    return

  device = env.device
  dtype = robot.data.default_joint_pos.dtype

  nominal_targets = torch.tensor(
    _IK_NOMINAL_FOOT_XZ,
    device=device,
    dtype=dtype,
  )[None, :, :].repeat(count, 1, 1)

  stand_a = torch.tensor(
    MICROTAUR_STAND_A_OFFSETS,
    device=device,
    dtype=dtype,
  )
  stand_e = torch.tensor(
    MICROTAUR_STAND_E_OFFSETS,
    device=device,
    dtype=dtype,
  )
  reference_motor = torch.stack((stand_a, stand_e), dim=-1)
  reference_motor = reference_motor[None, :, :].repeat(count, 1, 1)

  if randomize:
    foot_targets = nominal_targets.clone()
    foot_targets[:, :, 0] += _sample_uniform_tensor(
      count,
      (4,),
      foot_x_offset_range,
      device,
      dtype,
    )

    shared_z = _sample_uniform_tensor(
      count,
      (1,),
      shared_z_offset_range,
      device,
      dtype,
    )
    per_leg_z = _sample_uniform_tensor(
      count,
      (4,),
      per_leg_z_jitter_range,
      device,
      dtype,
    )
    # Keep per-leg z jitter zero-mean so shared_z remains the only net body-height
    # offset.
    per_leg_z -= torch.mean(per_leg_z, dim=1, keepdim=True)
    foot_targets[:, :, 1] += shared_z + per_leg_z
  else:
    foot_targets = nominal_targets

  motor_ids = _ordered_joint_ids(
    robot,
    tuple(LEG_JOINT_NAMES),
    device,
  )
  motor_limits = robot.data.joint_pos_limits[env_ids][:, motor_ids, :]
  motor_limits = motor_limits.reshape(count, 4, 2, 2)

  full_q, valid_per_leg = _MICROTAUR_KINEMATICS.solve_all_legs_torch(
    foot_targets,
    reference_motor,
    motor_limits=motor_limits,
  )
  full_q = full_q.reshape(count, 4, 4)

  # Fall back to the nominal closed-chain state if a sampled target lands on a
  # numerical or joint-limit edge case.
  fallback_q = torch.tensor(
    _IK_NOMINAL_FULL_Q,
    device=device,
    dtype=dtype,
  )[None, :, :].repeat(count, 1, 1)
  full_q = torch.where(
    valid_per_leg[:, :, None],
    full_q,
    fallback_q,
  )
  foot_targets = torch.where(
    valid_per_leg[:, :, None],
    foot_targets,
    nominal_targets,
  )

  full_q_flat = full_q.reshape(count, 16)
  full_joint_ids = _ordered_joint_ids(
    robot,
    tuple(IK_FULL_JOINT_NAMES),
    device,
  )

  # Set root height from the lowest sampled foot center. Hip z is 9.5 mm for all
  # four legs in the current geometry.
  hip_z = torch.tensor(
    HIP_MIDPOINTS_ROOT_M[:, 2],
    device=device,
    dtype=dtype,
  )[None, :]
  foot_z_root = hip_z + foot_targets[:, :, 1]

  if randomize:
    lowest_center_z = _sample_uniform_tensor(
      count,
      (),
      lowest_foot_center_z_range,
      device,
      dtype,
    )
  else:
    lowest_center_z = torch.full(
      (count,),
      float(lowest_foot_center_z_range[0]),
      device=device,
      dtype=dtype,
    )

  root_height = (
    lowest_center_z
    - torch.amin(foot_z_root, dim=1)
  )

  origins = env.scene.env_origins[env_ids].to(dtype=dtype)
  root_state = robot.data.default_root_state[env_ids].clone()
  root_state[:, 0:3] = origins
  root_state[:, 2] += root_height
  root_state[:, 7:13] = 0.0

  if randomize:
    root_state[:, 0] += _sample_uniform_tensor(
      count, (), root_x_range, device, dtype
    )
    root_state[:, 1] += _sample_uniform_tensor(
      count, (), root_y_range, device, dtype
    )
    yaw = _sample_uniform_tensor(
      count, (), root_yaw_range, device, dtype
    )
  else:
    yaw = torch.zeros(count, device=device, dtype=dtype)

  half_yaw = 0.5 * yaw
  root_state[:, 3] = torch.cos(half_yaw)
  root_state[:, 4] = 0.0
  root_state[:, 5] = 0.0
  root_state[:, 6] = torch.sin(half_yaw)

  joint_velocity = torch.zeros_like(full_q_flat)

  robot.write_root_state_to_sim(root_state, env_ids=env_ids)
  robot.write_joint_state_to_sim(
    full_q_flat,
    joint_velocity,
    joint_ids=full_joint_ids,
    env_ids=env_ids,
  )

  # Seed the motor targets from the reset state so the first delayed command
  # does not pull the mechanism away from the solved pose.
  motor_q = full_q[:, :, (0, 2)].reshape(count, 8)
  # set_joint_position_target uses advanced indexing. Expand env and joint IDs
  # to [N, 1] and [1, 8] so the destination broadcasts to [N, 8].
  robot.set_joint_position_target(
    motor_q,
    joint_ids=motor_ids.unsqueeze(0),
    env_ids=env_ids.unsqueeze(1),
  )

  env._microtaur_last_ik_reset_targets_xz = foot_targets
  env._microtaur_last_ik_reset_valid = valid_per_leg



# -----------------------------------------------------------------------------
# Direct motor actions and predictive five-bar filtering
# -----------------------------------------------------------------------------

# Action order must match LEG_JOINT_NAMES: (a, e) for legs 1 through 4.
# Microtaur has no lateral/abduction action.

MOTION_COMMAND_THRESHOLD = 0.025

# Olympus uses a larger action range, but Microtaur's smaller five-bar
# workspace becomes invalid sooner. +/-30 degrees keeps the sampled action
# region reachable while leaving room for the coupled safety filter.
OLYMPUS_WALK_ACTION_SCALE_RAD = math.radians(
  float(os.environ.get("MICROTAUR_OLYMPUS_ACTION_SCALE_DEG", "30.0"))
)

# Match the Olympus predictive-filter horizon: 5 policy steps at 60 Hz.
OLYMPUS_FILTER_HORIZON_S = float(
  os.environ.get("MICROTAUR_OLYMPUS_FILTER_HORIZON_S", str(5.0 / 60.0))
)
OLYMPUS_FILTER_VELOCITY_FLOOR_RAD_S = math.radians(
  float(os.environ.get("MICROTAUR_OLYMPUS_FILTER_VEL_DEG_S", "60.0"))
)
OLYMPUS_FILTER_JOINT_MARGIN_RAD = math.radians(
  float(os.environ.get("MICROTAUR_OLYMPUS_JOINT_MARGIN_DEG", "8.0"))
)
OLYMPUS_FILTER_COUPLED_MARGIN_RAD = math.radians(
  float(os.environ.get("MICROTAUR_OLYMPUS_COUPLED_MARGIN_DEG", "10.0"))
)

# Microtaur paired-motor coordinates:
#   da = sign * (common - diff)
#   de = sign * (common + diff)
#
# Hence:
#   common = 0.5 * sign * (da + de)
#   diff   = 0.5 * sign * (de - da)
#
# Olympus's paired-motor sum constraint maps to common. Microtaur also bounds
# diff and |common| + |diff| to keep both coupled motor directions inside the
# usable five-bar workspace.
OLYMPUS_COMMON_LIMIT_RAD = float(
  os.environ.get("MICROTAUR_OLYMPUS_COMMON_LIMIT_RAD", "0.52")
)
OLYMPUS_DIFF_LIMIT_RAD = float(
  os.environ.get("MICROTAUR_OLYMPUS_DIFF_LIMIT_RAD", "0.52")
)
OLYMPUS_DIAMOND_LIMIT_RAD = float(
  os.environ.get("MICROTAUR_OLYMPUS_DIAMOND_LIMIT_RAD", "0.70")
)

OLYMPUS_MIN_ACTION_DELAY_STEPS = int(
  os.environ.get("MICROTAUR_OLYMPUS_MIN_ACTION_DELAY", "1")
)
OLYMPUS_MAX_ACTION_DELAY_STEPS = int(
  os.environ.get("MICROTAUR_OLYMPUS_MAX_ACTION_DELAY", "2")
)
if OLYMPUS_MIN_ACTION_DELAY_STEPS < 0:
  raise ValueError("Minimum action delay must be non-negative.")
if OLYMPUS_MAX_ACTION_DELAY_STEPS < OLYMPUS_MIN_ACTION_DELAY_STEPS:
  raise ValueError("Maximum action delay must be >= minimum action delay.")


# Normalize reward terms before weighting so coefficients remain comparable
# across quantities with different physical units.
MICROTAUR_REWARD_SCALES = {
  "track_forward_velocity": 3.00,
  "forward_progress": 0.20, #was 0.75
  "track_yaw_velocity": 1.20,
  "zero_command_yaw_rate": -0.30,
  "body_height": 1.00,
  "upright": 0.75,
  "spine_position": -0.15,
  "spine_velocity": -0.01,
  "feet_air_time": 0.20, #was 0.15
  "moving_contact_pattern": 0.20, # Previous tuning value: 0.10.
  "standing_pose": 0.50,
  "lateral_velocity": -0.15,
  "vertical_velocity": -0.10,
  "roll_pitch_rate": -0.10,
  "normalized_torque": -0.03, # Previous tuning value: -0.03.
  "normalized_acceleration": -0.01, # Previous tuning value: -0.01.
  "action_rate": -0.04,
  "undesired_contacts": -2.00,
  "filter_correction": -0.10,
  "joint_limit_proximity": -0.05,
}

# Error scales are defined in Microtaur-relevant physical units.
FORWARD_TRACKING_SIGMA_M_S = float(
  os.environ.get("MICROTAUR_FORWARD_TRACKING_SIGMA", "0.04") #was 0.06
)
YAW_TRACKING_SIGMA_RAD_S = float(
  os.environ.get("MICROTAUR_YAW_TRACKING_SIGMA", "0.15")
)
HEIGHT_TRACKING_SIGMA_M = float(
  os.environ.get("MICROTAUR_HEIGHT_TRACKING_SIGMA", "0.012")
)
# Whole-robot uprightness is evaluated from the midpoint front/rear chassis
# orientation as an actual tilt angle. At 15 deg, the reward is 0.5.
UPRIGHT_TILT_SCALE_RAD = math.radians(
  float(os.environ.get("MICROTAUR_UPRIGHT_TILT_SCALE_DEG", "15.0"))
)

# Passive-spine neutral-position regularization. Motion within +/-3 deg of
# flat is free; beyond that, deviation is penalized on a 10 deg scale.
SPINE_NEUTRAL_DEADBAND_RAD = math.radians(
  float(os.environ.get("MICROTAUR_SPINE_NEUTRAL_DEADBAND_DEG", "10.0"))
)
SPINE_POSITION_SCALE_RAD = math.radians(
  float(os.environ.get("MICROTAUR_SPINE_POSITION_SCALE_DEG", "10.0"))
)

# Passive-spine angular velocity scale. qdot=1 rad/s produces a raw penalty of 1.
# The reward weight controls how expensive that motion is.
SPINE_VELOCITY_SCALE_RAD_S = float(
  os.environ.get("MICROTAUR_SPINE_VELOCITY_SCALE_RAD_S", "1.0")
)

LATERAL_VELOCITY_SCALE_M_S = 0.08
VERTICAL_VELOCITY_SCALE_M_S = 0.20
ROLL_PITCH_RATE_SCALE_RAD_S = 1.50
ZERO_COMMAND_YAW_RATE_SCALE_RAD_S = float(
  os.environ.get("MICROTAUR_ZERO_COMMAND_YAW_RATE_SCALE", "0.15")
)
JOINT_ACCELERATION_SCALE_RAD_S2 = 150.0
AIR_TIME_TARGET_S = 0.10
AIR_TIME_SIGMA_S = 0.06


def _safe_log(env, name: str, value: torch.Tensor) -> None:
  extras = getattr(env, "extras", None)
  if isinstance(extras, dict):
    log = extras.get("log")
    if isinstance(log, dict):
      log[name] = value


@configclass
class MicrotaurOlympusWalkActionCfg(JointPositionActionCfg):
  """Direct motor-position action with a predictive five-bar safety filter."""

  class_type: type = MISSING

  action_scale_rad: float = OLYMPUS_WALK_ACTION_SCALE_RAD
  filter_horizon_s: float = OLYMPUS_FILTER_HORIZON_S
  velocity_floor_rad_s: float = OLYMPUS_FILTER_VELOCITY_FLOOR_RAD_S
  joint_margin_rad: float = OLYMPUS_FILTER_JOINT_MARGIN_RAD
  coupled_margin_rad: float = OLYMPUS_FILTER_COUPLED_MARGIN_RAD
  common_limit_rad: float = OLYMPUS_COMMON_LIMIT_RAD
  diff_limit_rad: float = OLYMPUS_DIFF_LIMIT_RAD
  diamond_limit_rad: float = OLYMPUS_DIAMOND_LIMIT_RAD
  min_action_delay_steps: int = OLYMPUS_MIN_ACTION_DELAY_STEPS
  max_action_delay_steps: int = OLYMPUS_MAX_ACTION_DELAY_STEPS

  stand_a_offsets: tuple[float, float, float, float] = (
    0.45,
    -0.45,
    -0.45,
    0.45,
  )
  stand_e_offsets: tuple[float, float, float, float] = (
    -0.45,
    0.45,
    0.45,
    -0.45,
  )
  leg_signs: tuple[float, float, float, float] = (+1.0, -1.0, -1.0, +1.0)
  joint_half_range_rad: float = 0.98

  def __init__(
    self,
    asset_cfg=None,
    scale=None,
    offset=None,
    preserve_order: bool = True,
    use_default_offset: bool = False,
    action_scale_rad: float = OLYMPUS_WALK_ACTION_SCALE_RAD,
    filter_horizon_s: float = OLYMPUS_FILTER_HORIZON_S,
    velocity_floor_rad_s: float = OLYMPUS_FILTER_VELOCITY_FLOOR_RAD_S,
    joint_margin_rad: float = OLYMPUS_FILTER_JOINT_MARGIN_RAD,
    coupled_margin_rad: float = OLYMPUS_FILTER_COUPLED_MARGIN_RAD,
    common_limit_rad: float = OLYMPUS_COMMON_LIMIT_RAD,
    diff_limit_rad: float = OLYMPUS_DIFF_LIMIT_RAD,
    diamond_limit_rad: float = OLYMPUS_DIAMOND_LIMIT_RAD,
    min_action_delay_steps: int = OLYMPUS_MIN_ACTION_DELAY_STEPS,
    max_action_delay_steps: int = OLYMPUS_MAX_ACTION_DELAY_STEPS,
    stand_a_offsets: tuple[float, float, float, float] = (
      0.45,
      -0.45,
      -0.45,
      0.45,
    ),
    stand_e_offsets: tuple[float, float, float, float] = (
      -0.45,
      0.45,
      0.45,
      -0.45,
    ),
    leg_signs: tuple[float, float, float, float] = (
      +1.0,
      -1.0,
      -1.0,
      +1.0,
    ),
    joint_half_range_rad: float = 0.98,
    **kwargs,
  ):
    try:
      super().__init__()
    except TypeError:
      pass

    self.class_type = MicrotaurOlympusWalkAction
    self.entity_name = "robot"
    self.actuator_names = None
    self.preserve_order = preserve_order
    self.use_default_offset = use_default_offset

    if asset_cfg is not None:
      self.asset_cfg = asset_cfg
      if hasattr(asset_cfg, "name"):
        self.entity_name = asset_cfg.name
      elif hasattr(asset_cfg, "entity_name"):
        self.entity_name = asset_cfg.entity_name

      if hasattr(asset_cfg, "joint_names"):
        self.actuator_names = asset_cfg.joint_names
      elif hasattr(asset_cfg, "actuator_names"):
        self.actuator_names = asset_cfg.actuator_names

    if self.actuator_names is None:
      self.actuator_names = tuple(LEG_JOINT_NAMES)

    # This subclass computes absolute motor targets; keep the base transform neutral.
    self.scale = 1.0 if scale is None else scale
    self.offset = 0.0 if offset is None else offset

    self.action_scale_rad = float(action_scale_rad)
    self.filter_horizon_s = float(filter_horizon_s)
    self.velocity_floor_rad_s = float(velocity_floor_rad_s)
    self.joint_margin_rad = float(joint_margin_rad)
    self.coupled_margin_rad = float(coupled_margin_rad)
    self.common_limit_rad = float(common_limit_rad)
    self.diff_limit_rad = float(diff_limit_rad)
    self.diamond_limit_rad = float(diamond_limit_rad)
    self.min_action_delay_steps = int(min_action_delay_steps)
    self.max_action_delay_steps = int(max_action_delay_steps)
    self.stand_a_offsets = tuple(float(x) for x in stand_a_offsets)
    self.stand_e_offsets = tuple(float(x) for x in stand_e_offsets)
    self.leg_signs = tuple(float(x) for x in leg_signs)
    self.joint_half_range_rad = float(joint_half_range_rad)

    for key, value in kwargs.items():
      setattr(self, key, value)

  def build(self, env):
    return MicrotaurOlympusWalkAction(self, env)


class MicrotaurOlympusWalkAction(JointPositionAction):
  """Eight direct planar motor targets with predictive closed-chain filtering."""

  cfg: MicrotaurOlympusWalkActionCfg

  def __init__(self, cfg: MicrotaurOlympusWalkActionCfg, env):
    super().__init__(cfg, env)
    self._walk_env = env
    device = env.device

    # A wrong resolved joint order would send commands to the wrong leg. Check it
    # once at construction instead of relying on preserve_order implicitly.
    resolved_names = tuple(getattr(self, "_target_names", ()))
    expected_names = tuple(LEG_JOINT_NAMES)
    if resolved_names and resolved_names != expected_names:
      raise ValueError(
        "Resolved motor order does not match Microtaur's required "
        "(a, e) per-leg order. "
        f"Expected {expected_names}, got {resolved_names}."
      )

    stand = torch.empty(8, device=device)
    stand[0::2] = torch.tensor(cfg.stand_a_offsets, device=device)
    stand[1::2] = torch.tensor(cfg.stand_e_offsets, device=device)
    self._stand = stand
    self._leg_signs = torch.tensor(cfg.leg_signs, device=device)

    self._joint_lower = stand - float(cfg.joint_half_range_rad)
    self._joint_upper = stand + float(cfg.joint_half_range_rad)

    self._requested_targets = stand[None, :].repeat(env.num_envs, 1)
    self._safe_targets = self._requested_targets.clone()
    self._applied_targets = self._requested_targets.clone()
    self._filter_correction = torch.zeros_like(self._requested_targets)
    self._filter_blend = torch.zeros_like(self._requested_targets)

    delay_size = max(1, int(cfg.max_action_delay_steps) + 1)
    self._delay_buffer = stand[None, None, :].repeat(
      env.num_envs,
      delay_size,
      1,
    )
    self._delay_steps = torch.full(
      (env.num_envs,),
      int(cfg.min_action_delay_steps),
      device=device,
      dtype=torch.long,
    )
    self._delay_write_index = 0

  @property
  def requested_targets(self) -> torch.Tensor:
    return self._requested_targets

  @property
  def safe_targets(self) -> torch.Tensor:
    return self._safe_targets

  @property
  def applied_targets(self) -> torch.Tensor:
    return self._applied_targets

  @property
  def filter_correction(self) -> torch.Tensor:
    return self._filter_correction

  @property
  def filter_blend(self) -> torch.Tensor:
    return self._filter_blend

  def _reset_delay_state(
    self,
    reset: torch.Tensor,
    current_position: torch.Tensor,
  ) -> None:
    if not torch.any(reset):
      return

    count = int(torch.sum(reset).item())
    self._delay_steps[reset] = torch.randint(
      low=int(self.cfg.min_action_delay_steps),
      high=int(self.cfg.max_action_delay_steps) + 1,
      size=(count,),
      device=self._delay_steps.device,
    )

    # Reset poses are not always the nominal stand pose. Seed every delay slot
    # from the actual reset position to avoid a stale stand command on step 1.
    reset_position = current_position[reset]
    self._delay_buffer[reset] = reset_position[:, None, :].expand(
      -1,
      self._delay_buffer.shape[1],
      -1,
    )
    self._requested_targets[reset] = reset_position
    self._safe_targets[reset] = reset_position
    self._applied_targets[reset] = reset_position
    self._filter_correction[reset] = 0.0
    self._filter_blend[reset] = 0.0

  def _predictive_axis_filter(
    self,
    position: torch.Tensor,
    velocity: torch.Tensor,
    requested: torch.Tensor,
    lower: torch.Tensor | float,
    upper: torch.Tensor | float,
    margin: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Olympus-style time-to-limit blending for one set of coordinates."""
    lower_t = torch.as_tensor(
      lower,
      device=position.device,
      dtype=position.dtype,
    )
    upper_t = torch.as_tensor(
      upper,
      device=position.device,
      dtype=position.dtype,
    )

    margin_lower = position - lower_t
    margin_upper = upper_t - position
    outside = (margin_lower < 0.0) | (margin_upper < 0.0)

    velocity_floor = float(self.cfg.velocity_floor_rad_s)
    safe_velocity = torch.where(
      torch.abs(velocity) < velocity_floor,
      torch.sign(velocity) * velocity_floor,
      velocity,
    )

    inf = torch.full_like(position, float("inf"))
    time_upper = torch.where(
      (safe_velocity > 0.0) & (margin_upper > 0.0),
      margin_upper / safe_velocity.clamp_min(1e-6),
      inf,
    )
    time_lower = torch.where(
      (safe_velocity < 0.0) & (margin_lower > 0.0),
      margin_lower / (-safe_velocity).clamp_min(1e-6),
      inf,
    )
    time_to_limit = torch.minimum(time_upper, time_lower)

    horizon = max(float(self.cfg.filter_horizon_s), 1e-6)
    blend = torch.clamp(1.0 - time_to_limit / horizon, 0.0, 1.0)
    blend = torch.where(outside, torch.ones_like(blend), blend)

    near_upper = margin_upper < float(margin)
    near_lower = margin_lower < float(margin)
    command_out_upper = requested > position
    command_out_lower = requested < position

    # Near a limit, block commands that move farther out but leave inward recovery
    # commands untouched.
    blend = torch.where(
      near_upper & command_out_upper,
      torch.ones_like(blend),
      blend,
    )
    blend = torch.where(
      near_lower & command_out_lower,
      torch.ones_like(blend),
      blend,
    )
    blend = torch.where(
      near_upper & (requested < position),
      torch.zeros_like(blend),
      blend,
    )
    blend = torch.where(
      near_lower & (requested > position),
      torch.zeros_like(blend),
      blend,
    )

    clamped = torch.minimum(torch.maximum(requested, lower_t), upper_t)
    filtered = (1.0 - blend) * requested + blend * clamped
    return filtered, blend

  def _filter_targets(
    self,
    position: torch.Tensor,
    velocity: torch.Tensor,
    requested: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    # Apply individual motor limits first.
    individual, individual_blend = self._predictive_axis_filter(
      position=position,
      velocity=velocity,
      requested=requested,
      lower=self._joint_lower,
      upper=self._joint_upper,
      margin=float(self.cfg.joint_margin_rad),
    )

    q = position.reshape(-1, 4, 2)
    qd = velocity.reshape(-1, 4, 2)
    qr = individual.reshape(-1, 4, 2)
    stand = self._stand.reshape(1, 4, 2)
    signs = self._leg_signs.reshape(1, 4)

    da = q[:, :, 0] - stand[:, :, 0]
    de = q[:, :, 1] - stand[:, :, 1]
    da_dot = qd[:, :, 0]
    de_dot = qd[:, :, 1]

    req_da = qr[:, :, 0] - stand[:, :, 0]
    req_de = qr[:, :, 1] - stand[:, :, 1]

    common = 0.5 * signs * (da + de)
    diff = 0.5 * signs * (de - da)
    common_dot = 0.5 * signs * (da_dot + de_dot)
    diff_dot = 0.5 * signs * (de_dot - da_dot)

    req_common = 0.5 * signs * (req_da + req_de)
    req_diff = 0.5 * signs * (req_de - req_da)

    # Then enforce the coupled common-mode limit.
    safe_common, common_blend = self._predictive_axis_filter(
      position=common,
      velocity=common_dot,
      requested=req_common,
      lower=-float(self.cfg.common_limit_rad),
      upper=float(self.cfg.common_limit_rad),
      margin=float(self.cfg.coupled_margin_rad),
    )

    # Enforce the coupled differential-mode limit.
    safe_diff, diff_blend = self._predictive_axis_filter(
      position=diff,
      velocity=diff_dot,
      requested=req_diff,
      lower=-float(self.cfg.diff_limit_rad),
      upper=float(self.cfg.diff_limit_rad),
      margin=float(self.cfg.coupled_margin_rad),
    )

    # Finally project onto |common| + |diff| <= limit, which bounds both motor
    # offsets because max(|common-diff|, |common+diff|) is the same L1 radius.
    radius = torch.abs(common) + torch.abs(diff)
    requested_radius = torch.abs(safe_common) + torch.abs(safe_diff)
    radial_velocity = (
      torch.sign(common) * common_dot
      + torch.sign(diff) * diff_dot
    )
    distance = float(self.cfg.diamond_limit_rad) - radius

    inf = torch.full_like(radius, float("inf"))
    time_to_diamond = torch.where(
      (radial_velocity > 0.0) & (distance > 0.0),
      distance / radial_velocity.clamp_min(1e-6),
      inf,
    )
    horizon = max(float(self.cfg.filter_horizon_s), 1e-6)
    diamond_blend = torch.clamp(
      1.0 - time_to_diamond / horizon,
      0.0,
      1.0,
    )
    diamond_blend = torch.where(
      radius > float(self.cfg.diamond_limit_rad),
      torch.ones_like(diamond_blend),
      diamond_blend,
    )

    near_diamond = distance < float(self.cfg.coupled_margin_rad)
    command_outward = requested_radius > radius
    diamond_blend = torch.where(
      near_diamond & command_outward,
      torch.ones_like(diamond_blend),
      diamond_blend,
    )
    diamond_blend = torch.where(
      near_diamond & (~command_outward),
      torch.zeros_like(diamond_blend),
      diamond_blend,
    )

    projection = torch.clamp(
      float(self.cfg.diamond_limit_rad)
      / (requested_radius + 1e-6),
      max=1.0,
    )
    projected_common = safe_common * projection
    projected_diff = safe_diff * projection

    safe_common = (
      (1.0 - diamond_blend) * safe_common
      + diamond_blend * projected_common
    )
    safe_diff = (
      (1.0 - diamond_blend) * safe_diff
      + diamond_blend * projected_diff
    )

    safe_da = signs * (safe_common - safe_diff)
    safe_de = signs * (safe_common + safe_diff)

    safe_pairs = torch.empty_like(qr)
    safe_pairs[:, :, 0] = stand[:, :, 0] + safe_da
    safe_pairs[:, :, 1] = stand[:, :, 1] + safe_de
    safe = safe_pairs.reshape(-1, 8)

    # Finish with an exact per-joint clamp for numerical safety.
    safe = torch.minimum(
      torch.maximum(safe, self._joint_lower),
      self._joint_upper,
    )

    pair_blend = torch.maximum(
      torch.maximum(common_blend, diff_blend),
      diamond_blend,
    )
    pair_blend = pair_blend[:, :, None].expand(-1, -1, 2).reshape(-1, 8)
    blend = torch.maximum(individual_blend, pair_blend)
    return safe, blend

  def process_actions(self, actions: torch.Tensor):
    if actions.shape[-1] != 8:
      raise ValueError(
        "MicrotaurOlympusWalkAction expects exactly eight planar motor "
        f"actions; got {actions.shape[-1]}."
      )

    actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
    actions = torch.clamp(actions, -1.0, 1.0)

    reset = self._walk_env.episode_length_buf <= 1

    position = self._entity.data.joint_pos[:, self._target_ids]
    velocity = self._entity.data.joint_vel[:, self._target_ids]
    self._reset_delay_state(reset, position)

    requested = (
      self._stand[None, :]
      + float(self.cfg.action_scale_rad) * actions
    )
    safe, blend = self._filter_targets(position, velocity, requested)

    self._requested_targets = requested
    self._safe_targets = safe
    self._filter_correction = requested - safe
    self._filter_blend = blend

    # Apply the configured one-to-two policy-step command delay.
    self._delay_buffer[:, self._delay_write_index, :] = safe
    read_index = torch.remainder(
      self._delay_write_index - self._delay_steps,
      self._delay_buffer.shape[1],
    )
    env_index = torch.arange(
      self._delay_buffer.shape[0],
      device=actions.device,
    )
    applied = self._delay_buffer[env_index, read_index]
    self._delay_write_index = (
      self._delay_write_index + 1
    ) % self._delay_buffer.shape[1]
    self._applied_targets = applied

    # Keep raw policy actions for observations and action-rate regularization.
    # The parent action class applies the processed motor targets.
    self._raw_actions[:] = actions
    self._processed_actions[:] = applied


MicrotaurOlympusWalkActionCfg.class_type = MicrotaurOlympusWalkAction


def _get_walk_action_term(
  env,
  action_name: str,
) -> MicrotaurOlympusWalkAction:
  term = env.action_manager.get_term(action_name)
  if not isinstance(term, MicrotaurOlympusWalkAction):
    raise TypeError(
      f"Action term '{action_name}' must be MicrotaurOlympusWalkAction; "
      f"got {type(term).__name__}."
    )
  return term


def _contact_bool(env, sensor_name: str) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  if sensor.data.found is None:
    raise RuntimeError(f"Contact sensor '{sensor_name}' has no found field.")
  contact = sensor.data.found > 0
  if contact.ndim == 1:
    contact = contact[:, None]
  elif contact.ndim > 2:
    contact = contact.reshape(contact.shape[0], -1)
  return contact


def _ensure_microtaur_reward_state(
  env,
  num_feet: int,
  action_dim: int,
) -> None:
  if hasattr(env, "_microtaur_scaled_reward_state"):
    return

  device = env.device
  env._microtaur_scaled_reward_state = {
    "previous_action": torch.zeros(
      env.num_envs,
      action_dim,
      device=device,
    ),
    "previous_contact": torch.zeros(
      env.num_envs,
      num_feet,
      device=device,
      dtype=torch.bool,
    ),
    "air_time": torch.zeros(
      env.num_envs,
      num_feet,
      device=device,
    ),
  }


def _microtaur_root_height(env, robot) -> torch.Tensor:
  root_position = getattr(robot.data, "root_link_pos_w", None)
  if root_position is None:
    root_position = getattr(robot.data, "root_pos_w", None)
  if root_position is None:
    raise AttributeError(
      "Robot data exposes neither root_link_pos_w nor root_pos_w."
    )
  return root_position[:, 2] - env.scene.env_origins[:, 2]


PASSIVE_SPINE_JOINT_NAME = "spine_joint_passive"

# Current passive-roll XML:
# child neutral quat ~= -90 deg about Y, local joint axis = +Z,
# so the physical hinge axis in the front/root chassis frame is -X.
PASSIVE_SPINE_AXIS_FRONT = (-1.0, 0.0, 0.0)


def _rotate_vector_about_axis(
  vector: torch.Tensor,
  axis: torch.Tensor,
  angle: torch.Tensor,
) -> torch.Tensor:
  """Rotate batched 3D vectors with Rodrigues' formula."""
  axis = axis.to(device=vector.device, dtype=vector.dtype)
  axis = axis / torch.clamp(torch.linalg.vector_norm(axis), min=1e-6)
  axis = axis.unsqueeze(0).expand_as(vector)

  c = torch.cos(angle).unsqueeze(-1)
  s = torch.sin(angle).unsqueeze(-1)

  return (
    vector * c
    + torch.cross(axis, vector, dim=1) * s
    + axis
      * torch.sum(axis * vector, dim=1, keepdim=True)
      * (1.0 - c)
  )


def _microtaur_passive_spine_indices(env) -> tuple[int | None, int | None]:
  """Return (spine_joint_local_id, rear_body_local_id).

  The rear body is resolved from the body attached to spine_joint_passive,
  so no CAD rear-body name is hard-coded.
  """
  cached = getattr(env, "_microtaur_passive_spine_indices", None)
  if cached is not None:
    return cached

  robot = env.scene["robot"]

  # Rigid Microtaur: no passive spine.
  if PASSIVE_SPINE_JOINT_NAME not in tuple(robot.joint_names):
    result = (None, None)
    env._microtaur_passive_spine_indices = result
    return result

  joint_ids, resolved = robot.find_joints(
    [PASSIVE_SPINE_JOINT_NAME],
    preserve_order=True,
  )
  if len(joint_ids) != 1 or tuple(resolved) != (PASSIVE_SPINE_JOINT_NAME,):
    raise RuntimeError(
      "Expected exactly one passive spine joint named "
      f"'{PASSIVE_SPINE_JOINT_NAME}', got {tuple(resolved)}."
    )

  spine_joint_local_id = int(joint_ids[0])
  spine_joint_global_id = int(
    robot.indexing.joint_ids[spine_joint_local_id].item()
  )

  # MuJoCo jnt_bodyid is the body containing this hinge: the rear chassis half.
  rear_body_global_id = int(
    robot.data.model.jnt_bodyid[spine_joint_global_id].item()
  )

  body_global_ids = [
    int(x) for x in robot.indexing.body_ids.detach().cpu().tolist()
  ]
  if rear_body_global_id not in body_global_ids:
    raise RuntimeError(
      "Body attached to spine_joint_passive is not part of the robot entity."
    )

  rear_body_local_id = body_global_ids.index(rear_body_global_id)

  result = (spine_joint_local_id, rear_body_local_id)
  env._microtaur_passive_spine_indices = result
  return result


def microtaur_projected_gravity(env) -> torch.Tensor:
  """Projected gravity of the physical midpoint orientation of both chassis halves.

  The front and rear MuJoCo body frames are not aligned at q=0, so their
  body-frame gravity vectors must not be averaged directly.

  For this single-axis passive hinge:
    rear physical orientation = front orientation rotated by q
    midpoint orientation      = front orientation rotated by q/2

  robot.data.projected_gravity_b is gravity expressed in the front/root frame.
  Rotating it by -q/2 about the physical spine axis therefore expresses gravity
  in the midpoint chassis frame.

  This gives:
    q = 0 -> exactly the normal root projected gravity
    equal/opposite front-rear tilt -> midpoint near upright
    common whole-body tilt -> preserved
  """
  robot = env.scene["robot"]
  spine_joint_local_id, _ = _microtaur_passive_spine_indices(env)

  front_gravity_b = robot.data.projected_gravity_b

  if spine_joint_local_id is None:
    return front_gravity_b

  spine_q = robot.data.joint_pos[:, spine_joint_local_id]
  spine_axis_front = torch.tensor(
    PASSIVE_SPINE_AXIS_FRONT,
    device=front_gravity_b.device,
    dtype=front_gravity_b.dtype,
  )

  return _rotate_vector_about_axis(
    front_gravity_b,
    spine_axis_front,
    -0.5 * spine_q,
  )


def microtaur_spine_joint_position(env) -> torch.Tensor:
  """Passive spine q [rad]; flat front/rear alignment is q=0."""
  robot = env.scene["robot"]
  spine_joint_local_id, _ = _microtaur_passive_spine_indices(env)

  if spine_joint_local_id is None:
    return torch.zeros(
      (env.num_envs, 1),
      device=robot.data.joint_pos.device,
      dtype=robot.data.joint_pos.dtype,
    )

  return robot.data.joint_pos[
    :,
    spine_joint_local_id : spine_joint_local_id + 1,
  ]


def microtaur_spine_joint_velocity(env) -> torch.Tensor:
  """Passive spine qdot [rad/s]."""
  robot = env.scene["robot"]
  spine_joint_local_id, _ = _microtaur_passive_spine_indices(env)

  if spine_joint_local_id is None:
    return torch.zeros(
      (env.num_envs, 1),
      device=robot.data.joint_vel.device,
      dtype=robot.data.joint_vel.dtype,
    )

  return robot.data.joint_vel[
    :,
    spine_joint_local_id : spine_joint_local_id + 1,
  ]


def microtaur_scaled_walk_reward(
  env,
  command_name: str,
  foot_sensor_name: str,
  illegal_sensor_name: str,
  action_name: str = "joint_pos",
) -> torch.Tensor:
  """Compute the Microtaur locomotion reward.

  Velocity tracking, body height, and uprightness provide the main positive
  signal. Motion, effort, contact, filtering, and joint-limit terms regularize
  how the robot achieves that command. The total reward is intentionally left
  unclamped so failure penalties remain visible to PPO.
  """
  robot = env.scene["robot"]
  term = _get_walk_action_term(env, action_name)
  command = env.command_manager.get_command(command_name)
  if command is None:
    raise KeyError(f"Command '{command_name}' was not found.")

  joint_pos = robot.data.joint_pos[:, term._target_ids]
  joint_acc = robot.data.joint_acc[:, term._target_ids]
  actuator_force = robot.data.actuator_force[:, :8]
  action = env.action_manager.action[:, :8]

  foot_contact = _contact_bool(env, foot_sensor_name)
  illegal_contact = _contact_bool(env, illegal_sensor_name)

  _ensure_microtaur_reward_state(
    env,
    num_feet=foot_contact.shape[1],
    action_dim=action.shape[1],
  )
  state = env._microtaur_scaled_reward_state

  reset = env.episode_length_buf <= 1
  if torch.any(reset):
    state["previous_action"][reset] = action[reset]
    state["previous_contact"][reset] = foot_contact[reset]
    state["air_time"][reset] = 0.0

  root_lin_vel_b = robot.data.root_link_lin_vel_b
  root_ang_vel_b = robot.data.root_link_ang_vel_b
  projected_gravity_b = microtaur_projected_gravity(env)
  root_height = _microtaur_root_height(env, robot)

  command_x = command[:, 0]
  command_yaw = command[:, 2]
  actual_x = root_lin_vel_b[:, 0]

  translation_active = torch.abs(command_x) > 0.03
  yaw_active = torch.abs(command_yaw) > 0.03
  standing = (~translation_active) & (~yaw_active)
  moving = translation_active | yaw_active

  # Track commanded x velocity directly; the same expression also supports
  # reverse commands if mirroring is enabled later.
  forward_error = (
    command_x - actual_x
  ) / max(FORWARD_TRACKING_SIGMA_M_S, 1e-6)
  track_forward = torch.exp(-torch.square(forward_error))

  # Reward speed in the commanded direction. sign(command_x) keeps this term
  # valid for optional reverse commands.
  command_abs = torch.clamp(torch.abs(command_x), min=0.05)
  commanded_direction_speed = actual_x * torch.sign(command_x)
  forward_progress = torch.clamp(
    commanded_direction_speed / command_abs,
    min=0.0,
    max=1.0,
  )
  forward_progress = forward_progress * translation_active.float()

  yaw_error = (
    command_yaw - root_ang_vel_b[:, 2]
  ) / max(YAW_TRACKING_SIGMA_RAD_S, 1e-6)

  # Reward yaw-rate tracking only when a turn is actually commanded.
  # Straight commands use the dedicated penalty below, avoiding a positive
  # standing-still bonus for simply having near-zero yaw rate.
  track_yaw = (
    torch.exp(-torch.square(yaw_error))
    * yaw_active.float()
  )

  height_error = (
    root_height - MICROTAUR_NOMINAL_ROOT_HEIGHT_M
  ) / max(HEIGHT_TRACKING_SIGMA_M, 1e-6)
  body_height = torch.exp(-torch.square(height_error))

  # Convert corrected midpoint-chassis projected gravity into whole-robot tilt.
  # Normalize only for this angle calculation.
  gravity_norm = projected_gravity_b / torch.clamp(
    torch.linalg.vector_norm(
      projected_gravity_b,
      dim=1,
      keepdim=True,
    ),
    min=1e-6,
  )
  tilt_angle = torch.atan2(
    torch.linalg.vector_norm(
      gravity_norm[:, :2],
      dim=1,
    ),
    -gravity_norm[:, 2],
  )

  # Cauchy-style upright reward:
  #   0 deg  -> 1.00
  #  15 deg  -> 0.50
  #  30 deg  -> 0.20
  # Unlike exp(-x^2), this retains a useful learning signal at larger tilts.
  upright = 1.0 / (
    1.0
    + torch.square(
      tilt_angle / max(UPRIGHT_TILT_SCALE_RAD, 1e-6)
    )
  )

  # Keep the passive spine near its flat q=0 configuration, while allowing
  # small compliant motion inside a +/-3 deg neutral deadband.
  spine_q = microtaur_spine_joint_position(env).squeeze(-1)
  spine_position_excess = torch.clamp(
    torch.abs(spine_q) - SPINE_NEUTRAL_DEADBAND_RAD,
    min=0.0,
  )
  spine_position = torch.clamp(
    torch.square(
      spine_position_excess / max(SPINE_POSITION_SCALE_RAD, 1e-6)
    ),
    max=4.0,
  )

  # Penalize violent passive-spine motion independently of its position.
  spine_qdot = microtaur_spine_joint_velocity(env).squeeze(-1)
  spine_velocity = torch.clamp(
    torch.square(
      spine_qdot / max(SPINE_VELOCITY_SCALE_RAD_S, 1e-6)
    ),
    max=4.0,
  )

  lateral_velocity = torch.clamp(
    torch.square(
      root_lin_vel_b[:, 1] / LATERAL_VELOCITY_SCALE_M_S
    ),
    max=4.0,
  )
  vertical_velocity = torch.clamp(
    torch.square(
      root_lin_vel_b[:, 2] / VERTICAL_VELOCITY_SCALE_M_S
    ),
    max=4.0,
  )
  roll_pitch_rate = torch.clamp(
    torch.sum(
      torch.square(
        root_ang_vel_b[:, :2] / ROLL_PITCH_RATE_SCALE_RAD_S
      ),
      dim=1,
    ),
    max=4.0,
  )

  # Penalize unintended yaw only when no turn is commanded.
  zero_command_yaw_rate = torch.clamp(
    torch.square(
      root_ang_vel_b[:, 2]
      / max(ZERO_COMMAND_YAW_RATE_SCALE_RAD_S, 1e-6)
    ),
    max=4.0,
  ) * (~yaw_active).float()

  normalized_torque = torch.mean(
    torch.square(
      actuator_force / max(XL330_SAFE_EFFORT_NM, 1e-6)
    ),
    dim=1,
  )
  normalized_torque = torch.clamp(normalized_torque, max=4.0)

  normalized_acceleration = torch.mean(
    torch.square(
      joint_acc / JOINT_ACCELERATION_SCALE_RAD_S2
    ),
    dim=1,
  )
  normalized_acceleration = torch.clamp(
    normalized_acceleration,
    max=4.0,
  )

  action_rate = torch.mean(
    torch.square(action - state["previous_action"]),
    dim=1,
  )

  # Reward touchdown near the chosen Microtaur swing duration; long flight is
  # not rewarded simply for keeping a foot off the ground.
  first_contact = foot_contact & (~state["previous_contact"])
  touchdown_quality = torch.exp(
    -torch.square(
      (state["air_time"] - AIR_TIME_TARGET_S)
      / max(AIR_TIME_SIGMA_S, 1e-6)
    )
  )
  feet_air_time = torch.mean(
    touchdown_quality * first_contact.float(),
    dim=1,
  ) * moving.float()

  dt = float(getattr(env, "step_dt", 0.02))
  state["air_time"] = torch.where(
    foot_contact,
    torch.zeros_like(state["air_time"]),
    state["air_time"] + dt,
  )

  # This term is gait-agnostic: it favors roughly two contacts while moving
  # without prescribing which diagonal or lateral pair should be in stance.
  contact_count = torch.sum(foot_contact.float(), dim=1)
  moving_contact_pattern = torch.exp(
    -torch.square((contact_count - 2.0) / 1.25)
  ) * moving.float()

  motor_error = joint_pos - term._stand[None, :]
  standing_pose = torch.exp(
    -torch.mean(
      torch.square(motor_error / math.radians(12.0)),
      dim=1,
    )
  ) * standing.float()

  undesired_contacts = torch.sum(illegal_contact.float(), dim=1)

  filter_correction = torch.mean(
    torch.square(
      term.filter_correction
      / max(float(term.cfg.action_scale_rad), 1e-6)
    ),
    dim=1,
  )

  normalized_joint_offset = torch.abs(motor_error) / max(
    float(term.cfg.joint_half_range_rad),
    1e-6,
  )
  joint_limit_proximity = torch.mean(
    torch.square(
      torch.clamp(
        (normalized_joint_offset - 0.80) / 0.20,
        min=0.0,
        max=1.0,
      )
    ),
    dim=1,
  )

  raw_terms = {
    "track_forward_velocity": track_forward,
    "forward_progress": forward_progress,
    "track_yaw_velocity": track_yaw,
    "body_height": body_height,
    "upright": upright,
    "spine_position": spine_position,
    "spine_velocity": spine_velocity,
    "feet_air_time": feet_air_time,
    "moving_contact_pattern": moving_contact_pattern,
    "standing_pose": standing_pose,
    "lateral_velocity": lateral_velocity,
    "vertical_velocity": vertical_velocity,
    "roll_pitch_rate": roll_pitch_rate,
    "zero_command_yaw_rate": zero_command_yaw_rate,
    "normalized_torque": normalized_torque,
    "normalized_acceleration": normalized_acceleration,
    "action_rate": action_rate,
    "undesired_contacts": undesired_contacts,
    "filter_correction": filter_correction,
    "joint_limit_proximity": joint_limit_proximity,
  }

  components = {
    name: MICROTAUR_REWARD_SCALES[name] * value
    for name, value in raw_terms.items()
  }
  total = torch.sum(torch.stack(tuple(components.values())), dim=0)

  # Keep training output intentionally compact. The trainer already reports
  # aggregate episode reward/length, termination statistics, and curriculum
  # values, so only log task-level diagnostics that are useful for judging
  # locomotion quality.
  _safe_log(
    env,
    "Metrics/microtaur/forward_velocity_m_s",
    torch.mean(actual_x),
  )
  _safe_log(
    env,
    "Metrics/microtaur/forward_error_abs_m_s",
    torch.mean(torch.abs(command_x - actual_x)),
  )
  _safe_log(
    env,
    "Metrics/microtaur/yaw_rate_abs_rad_s",
    torch.mean(torch.abs(root_ang_vel_b[:, 2])),
  )
  _safe_log(
    env,
    "Metrics/microtaur/yaw_error_abs_rad_s",
    torch.mean(torch.abs(command_yaw - root_ang_vel_b[:, 2])),
  )
  _safe_log(
    env,
    "Metrics/microtaur/tilt_deg",
    torch.mean(torch.rad2deg(tilt_angle)),
  )
  _safe_log(
    env,
    "Metrics/microtaur/spine_angle_deg",
    torch.mean(torch.abs(torch.rad2deg(spine_q))),
  )
  _safe_log(
    env,
    "Metrics/microtaur/contact_count",
    torch.mean(contact_count),
  )

  state["previous_action"] = action.clone()
  state["previous_contact"] = foot_contact.clone()
  return total



# -----------------------------------------------------------------------------
# Reward registration
# -----------------------------------------------------------------------------

def _configure_rewards(cfg: ManagerBasedRlEnvCfg, rough: bool) -> None:
  del rough

  cfg.rewards.clear()
  cfg.rewards["microtaur_scaled_walk"] = RewardTermCfg(
    func=microtaur_scaled_walk_reward,
    weight=1.0,
    params={
      "command_name": "twist",
      "foot_sensor_name": "feet_ground_contact",
      "illegal_sensor_name": "nonfoot_ground_touch",
      "action_name": "joint_pos",
    },
  )


# -----------------------------------------------------------------------------
# Actions, observations, sensors, events
# -----------------------------------------------------------------------------

def _configure_actions(cfg: ManagerBasedRlEnvCfg) -> None:
  half_range = MICROTAUR_JOINT_HALF_RANGE_RAD

  # Eight actions: the two actuated five-bar joints on each leg.
  cfg.actions.clear()
  cfg.actions["joint_pos"] = MicrotaurOlympusWalkActionCfg(
    asset_cfg=SceneEntityCfg(
      "robot",
      joint_names=list(LEG_JOINT_NAMES),
    ),
    scale=1.0,
    offset=0.0,
    preserve_order=True,
    use_default_offset=False,
    action_scale_rad=OLYMPUS_WALK_ACTION_SCALE_RAD,
    filter_horizon_s=OLYMPUS_FILTER_HORIZON_S,
    velocity_floor_rad_s=OLYMPUS_FILTER_VELOCITY_FLOOR_RAD_S,
    joint_margin_rad=OLYMPUS_FILTER_JOINT_MARGIN_RAD,
    coupled_margin_rad=OLYMPUS_FILTER_COUPLED_MARGIN_RAD,
    common_limit_rad=OLYMPUS_COMMON_LIMIT_RAD,
    diff_limit_rad=OLYMPUS_DIFF_LIMIT_RAD,
    diamond_limit_rad=OLYMPUS_DIAMOND_LIMIT_RAD,
    min_action_delay_steps=OLYMPUS_MIN_ACTION_DELAY_STEPS,
    max_action_delay_steps=OLYMPUS_MAX_ACTION_DELAY_STEPS,
    stand_a_offsets=MICROTAUR_STAND_A_OFFSETS,
    stand_e_offsets=MICROTAUR_STAND_E_OFFSETS,
    leg_signs=MINITAUR_SWING_SIGNS,
    joint_half_range_rad=half_range,
  )



def _configure_actor_noise(
  cfg: ManagerBasedRlEnvCfg,
  rough: bool,
) -> None:
  """Set actor observation noise to Microtaur-scale values.

  The base velocity task uses noise sized for much larger robots; applying it
  unchanged would be comparable to, or larger than, Microtaur's commands.
  """
  terms = cfg.observations["actor"].terms

  if "base_lin_vel" in terms:
    terms["base_lin_vel"].noise = Unoise(n_min=-0.025, n_max=0.025)
  if "base_ang_vel" in terms:
    terms["base_ang_vel"].noise = Unoise(n_min=-0.040, n_max=0.040)
  if "projected_gravity" in terms:
    terms["projected_gravity"].noise = Unoise(n_min=-0.015, n_max=0.015)
  if "joint_pos" in terms:
    terms["joint_pos"].noise = Unoise(n_min=-0.005, n_max=0.005)
  if "joint_vel" in terms:
    terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)
  if "spine_joint_pos" in terms:
    terms["spine_joint_pos"].noise = Unoise(n_min=-0.005, n_max=0.005)
  if "spine_joint_vel" in terms:
    terms["spine_joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)
  if rough and "height_scan" in terms:
    terms["height_scan"].noise = Unoise(n_min=-0.004, n_max=0.004)


def _fresh_foot_asset_cfg() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot",
    site_names=list(FOOT_SITE_NAMES),
  )


def _configure_observations(
  cfg: ManagerBasedRlEnvCfg,
  rough: bool,
) -> None:
  # Keep projected_gravity in its existing observation slot but redefine it as
  # gravity in the physical midpoint chassis frame. Append q/qdot.
  for group_name in ("actor", "critic"):
    group = cfg.observations[group_name]

    projected_gravity_term = group.terms.get("projected_gravity")
    if projected_gravity_term is None:
      raise KeyError(
        f"Observation group '{group_name}' has no 'projected_gravity' term."
      )
    projected_gravity_term.func = microtaur_projected_gravity
    projected_gravity_term.params = {}

    group.terms["spine_joint_pos"] = ObservationTermCfg(
      func=microtaur_spine_joint_position,
      params={},
    )
    group.terms["spine_joint_vel"] = ObservationTermCfg(
      func=microtaur_spine_joint_velocity,
      params={},
    )

  _configure_actor_noise(cfg, rough=rough)

  # Keep the same actor latency convention for the new spine encoder signals.
  actor_terms = cfg.observations["actor"].terms
  for term_name in (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "spine_joint_pos",
    "spine_joint_vel",
    "height_scan",
  ):
    term = actor_terms.get(term_name)
    if term is None:
      continue
    if hasattr(term, "delay_min_lag"):
      term.delay_min_lag = 1
      term.delay_max_lag = 2
      if hasattr(term, "delay_per_env"):
        term.delay_per_env = True

  for group_name in ("actor", "critic"):
    group = cfg.observations[group_name]
    group.terms["joint_pos"].params = {
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=list(LEG_JOINT_NAMES),
      )
    }
    group.terms["joint_vel"].params = {
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=list(LEG_JOINT_NAMES),
      )
    }

  # critic_terms = cfg.observations["critic"].terms
  # if "foot_height" in critic_terms:
  #   critic_terms["foot_height"].params["asset_cfg"] = _fresh_foot_asset_cfg()

  if not rough:
    cfg.observations["actor"].terms.pop("height_scan", None)
    cfg.observations["critic"].terms.pop("height_scan", None)


def microtaur_base_too_low(
  env,
  min_height_m: float,
) -> torch.Tensor:
  robot = env.scene["robot"]
  return _microtaur_root_height(env, robot) < float(min_height_m)


def _configure_sensors(
  cfg: ManagerBasedRlEnvCfg,
  play: bool,
  rough: bool,
) -> None:
  """Configure Microtaur sensors without inheriting empty raycast placeholders.

  MJLab's base velocity task intentionally creates robot-agnostic raycast
  placeholders (notably foot_height_scan with frame=()). For Microtaur, rebuild
  the raycast sensors explicitly so no zero-frame sensor can survive into
  RayCastSensor.prepare_rays().
  """

  # Preserve only non-raycast inherited sensors. This removes BOTH the base
  # terrain_scan and the base foot_height_scan placeholders.
  sensors = [
    sensor
    for sensor in (cfg.scene.sensors or ())
    if (
      not isinstance(sensor, RayCastSensorCfg)
      and sensor.name not in {
        "feet_ground_contact",
        "nonfoot_ground_touch",
      }
    )
  ]

  # Rebuild the per-foot terrain-height sensor from scratch.
  foot_height_scan = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=tuple(
      ObjRef(
        type="site",
        name=site_name,
        entity="robot",
      )
      for site_name in FOOT_SITE_NAMES
    ),
    pattern=RingPatternCfg.single_ring(
      radius=0.008,
      num_samples=4,
      include_center=True,
      direction=(0.0, 0.0, -1.0),
    ),
    ray_alignment="world",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
    debug_vis=play,
  )

  # Fail during config construction rather than later at torch.stack([]).
  foot_frames = (
    foot_height_scan.frame
    if isinstance(foot_height_scan.frame, tuple)
    else (foot_height_scan.frame,)
  )
  if len(foot_frames) != 4:
    raise RuntimeError(
      "Microtaur foot_height_scan must have exactly four frames; "
      f"got {foot_height_scan.frame!r}."
    )

  sensors.append(foot_height_scan)

  # Rough terrain additionally needs the root-centered terrain grid used by
  # the inherited height_scan observation. Flat mode deliberately has no
  # terrain_scan because height_scan observations are removed below.
  if rough:
    sensors.append(
      make_microtaur_terrain_scan_cfg(
        root_body_name=ROOT_BODY,
        debug_vis=play,
      )
    )

  sensors.append(
    ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(
        mode="geom",
        pattern=FOOT_GEOM_NAMES,
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      track_air_time=True,
    )
  )

  sensors.append(
    ContactSensorCfg(
      name="nonfoot_ground_touch",
      primary=ContactMatch(
        mode="geom",
        entity="robot",
        pattern=("rl_body_collision", "rl_battery_collision"),
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
    )
  )

  # Final validation: every raycast sensor must resolve from a non-empty frame
  # specification before Scene builds the runtime sensor instances.
  for sensor in sensors:
    if isinstance(sensor, RayCastSensorCfg):
      frames = sensor.frame if isinstance(sensor.frame, tuple) else (sensor.frame,)
      if len(frames) == 0:
        raise RuntimeError(
          f"Raycast sensor '{sensor.name}' has zero configured frames."
        )

  cfg.scene.sensors = tuple(sensors)

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "nonfoot_ground_touch"},
  )
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=microtaur_base_too_low,
    params={"min_height_m": MICROTAUR_MIN_ROOT_HEIGHT_M},
  )


def _configure_events(
  cfg: ManagerBasedRlEnvCfg,
  play: bool,
  rough: bool,
) -> None:
  if "foot_friction" in cfg.events:
    cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg(
      "robot",
      geom_names=list(FOOT_GEOM_NAMES),
    )
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.2)

  if "base_com" in cfg.events:
    cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
      "robot",
      body_names=[ROOT_BODY],
    )
    cfg.events["base_com"].params["ranges"] = {
      0: (-0.003, 0.003),
      1: (-0.003, 0.003),
      2: (-0.004, 0.004),
    }

  # Use one reset event for both the root and all 16 leg joints. Writing the
  # full closed-chain state together avoids conflicting reset events.
  cfg.events.pop("reset_base", None)
  cfg.events.pop("reset_robot_joints", None)
  cfg.events["ik_consistent_reset"] = EventTermCfg(
    func=microtaur_ik_consistent_reset,
    mode="reset",
    params={
      "randomize": not play,
      "root_x_range": IK_RESET_ROOT_X_RANGE_M,
      "root_y_range": IK_RESET_ROOT_Y_RANGE_M,
      "root_yaw_range": IK_RESET_ROOT_YAW_RANGE_RAD,
      "foot_x_offset_range": IK_RESET_FOOT_X_OFFSET_RANGE_M,
      "shared_z_offset_range": IK_RESET_SHARED_Z_OFFSET_RANGE_M,
      "per_leg_z_jitter_range": IK_RESET_PER_LEG_Z_JITTER_RANGE_M,
      "lowest_foot_center_z_range": (
        IK_RESET_LOWEST_FOOT_CENTER_Z_RANGE_M
      ),
    },
  )

  if "push_robot" in cfg.events:
    if play or not rough:
      # Flat-ground training omits periodic pushes; rough training keeps them.
      cfg.events.pop("push_robot", None)
    else:
      cfg.events["push_robot"].interval_range_s = (4.0, 7.0)
      cfg.events["push_robot"].params["velocity_range"] = {
        "x": (-0.06, 0.06),
        "y": (-0.04, 0.04),
        "z": (-0.02, 0.02),
        "roll": (-0.08, 0.08),
        "pitch": (-0.08, 0.08),
        "yaw": (-0.12, 0.12),
      }


# -----------------------------------------------------------------------------
# Environment assembly
# -----------------------------------------------------------------------------

def _make_base_cfg(
  play: bool,
  rough: bool,
) -> ManagerBasedRlEnvCfg:
  cfg = make_velocity_env_cfg()

  # Run the policy at approximately 60 Hz while leaving the base physics
  # timestep unchanged.
  if hasattr(cfg.sim, "dt") and hasattr(cfg, "decimation"):
    cfg.decimation = max(1, int(round((1.0 / 60.0) / float(cfg.sim.dt))))

  cfg.sim.njmax = 1024
  cfg.sim.nconmax = 256
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 256

  cfg.scene.entities = {"robot": _make_microtaur_robot_cfg()}
  cfg.viewer.body_name = ROOT_BODY
  cfg.viewer.distance = 0.7
  cfg.viewer.elevation = -15.0

  _configure_sensors(cfg, play=play, rough=rough)
  _configure_actions(cfg)
  _configure_observations(cfg, rough=rough)
  _configure_commands(cfg, play=play)
  _configure_rewards(cfg, rough=rough)
  _configure_events(cfg, play=play, rough=rough)

  if cfg.scene.terrain is None:
    raise RuntimeError("Base MJLab velocity config did not create terrain.")

  if rough:
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = replace(MICRO_ROUGH_TERRAINS_CFG)
    cfg.scene.terrain.max_init_terrain_level = 1
  else:
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.curriculum.pop("terrain_levels", None)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg


def microtaur_velocity_yaw_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return _make_base_cfg(play=play, rough=False)


def microtaur_velocity_yaw_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return _make_base_cfg(play=play, rough=True)


# Task-name aliases kept for older launch commands.
def microtaur_velocity_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return microtaur_velocity_yaw_flat_env_cfg(play=play)


def microtaur_velocity_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return microtaur_velocity_yaw_rough_env_cfg(play=play)


def microtaur_velocity_gait_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return microtaur_velocity_yaw_flat_env_cfg(play=play)


def microtaur_velocity_gait_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return microtaur_velocity_yaw_rough_env_cfg(play=play)


def microtaur_velocity_trot_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return microtaur_velocity_yaw_flat_env_cfg(play=play)


def microtaur_velocity_trot_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return microtaur_velocity_yaw_rough_env_cfg(play=play)


def microtaur_velocity_twist_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return microtaur_velocity_yaw_flat_env_cfg(play=play)


def microtaur_velocity_straight_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return microtaur_velocity_yaw_flat_env_cfg(play=play)


def microtaur_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return microtaur_velocity_yaw_flat_env_cfg(play=play)


def microtaur_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return microtaur_velocity_yaw_rough_env_cfg(play=play)


def microtaur_velocity_bound_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return microtaur_velocity_yaw_flat_env_cfg(play=play)


def microtaur_velocity_bound_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return microtaur_velocity_yaw_rough_env_cfg(play=play)