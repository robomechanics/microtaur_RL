"""Open-loop MuJoCo replay of the Microtaur CPG action generator.

This script does not load a policy or an MJLab environment. It reproduces the
same command-conditioned CPG equations used by:

  ENV_CFG_REVISION = "2026-07-28-cpg-bank-plus-differential-v1"

It augments the Microtaur XML in memory with:
  - a ground plane
  - bounded position servos for the eight actuated joints

Then it replays:
  - diagonal trot phasing [0, pi, 0, pi]
  - command-based left/right stride differential
  - simultaneous inward bank through side-dependent lift/extension
  - the five-bar-safe |swing| + |lift| projection

Examples
--------
Automatic straight/left/right demo:

  uv run python replay_microtaur_cpg.py --demo

Constant forward command:

  uv run python replay_microtaur_cpg.py --vx 0.12 --yaw 0.0

Forward left turn using differential stride + inward bank:

  uv run python replay_microtaur_cpg.py --vx 0.12 --yaw 0.45

Turn in place:

  uv run python replay_microtaur_cpg.py --vx 0.0 --yaw 0.55

Inspect the leg cycle with the floating base held in place:

  uv run python replay_microtaur_cpg.py --demo --fixed-base

Flip the bank direction when the linkage convention banks outward:

  uv run python replay_microtaur_cpg.py --demo --bank-sign -1
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
  import mujoco
  import mujoco.viewer
except ImportError as exc:
  raise SystemExit(
    "MuJoCo is not installed in the active environment.\n"
    "Run this through the same uv environment that contains MJLab/MuJoCo."
  ) from exc


# -----------------------------------------------------------------------------
# Robot/action constants
# -----------------------------------------------------------------------------

ACTUATED_JOINTS = (
  "leg1_a_joint_act",
  "leg1_e_joint_act",
  "leg2_a_joint_act",
  "leg2_e_joint_act",
  "leg3_a_joint_act",
  "leg3_e_joint_act",
  "leg4_a_joint_act",
  "leg4_e_joint_act",
)

STAND_A = np.asarray((0.45, -0.45, -0.45, 0.45), dtype=np.float64)
STAND_E = np.asarray((-0.45, 0.45, 0.45, -0.45), dtype=np.float64)
LEG_SIGNS = np.asarray((+1.0, -1.0, -1.0, +1.0), dtype=np.float64)

# Working open-loop gait values.
COMMON_SIGN_FLIP = -1.0
COMMON_RAW_AMP = 0.75
DIFF_RAW_AMP = 0.85

SWING_SCALE = 0.45
LIFT_SCALE = 0.45

TRACK_WIDTH_M = 0.09992
MOTION_THRESHOLD = 0.025

BASE_FREQUENCY_HZ = 1.35
SPEED_FREQUENCY_GAIN = 0.80
FREQUENCY_MIN_HZ = 1.20
FREQUENCY_MAX_HZ = 1.75

BANK_RAW_MAX = 0.12
SAFE_MOTOR_OFFSET_RAD = 0.82

SERVO_KP = 1.0
SERVO_KV = 0.045
SERVO_FORCE_LIMIT_NM = 0.52

PHASE_OFFSETS = np.asarray((0.0, math.pi, 0.0, math.pi), dtype=np.float64)
BANK_SIDE_PATTERN = np.asarray((-1.0, +1.0, +1.0, -1.0), dtype=np.float64)


@dataclass(frozen=True)
class Command:
  vx: float
  yaw: float
  label: str


@dataclass(frozen=True)
class CpgOutput:
  motor_targets: np.ndarray
  phase: float
  leg_phase: np.ndarray
  expected_stance: np.ndarray
  frequency_hz: float
  left_speed: float
  right_speed: float
  bank_raw: float
  raw_swing: np.ndarray
  raw_lift: np.ndarray


# -----------------------------------------------------------------------------
# CPG
# -----------------------------------------------------------------------------

def smoothstep(x: np.ndarray) -> np.ndarray:
  x = np.clip(x, 0.0, 1.0)
  return x * x * (3.0 - 2.0 * x)


def cpg_targets(
  phase: float,
  vx_cmd: float,
  yaw_cmd: float,
  bank_sign: float,
  frequency_residual: float = 0.0,
  stride_residual: float = 0.0,
  lift_residual: float = 0.0,
  stance_push_residual: float = 0.0,
  differential_gain_residual: float = 0.0,
  bank_gain_residual: float = 0.0,
  lift_phase_residual: float = 0.0,
  height_residual: float = 0.0,
) -> CpgOutput:
  """Evaluate one CPG sample.

  Residual arguments correspond to the eight RL actions and must be in [-1, 1].
  Zero residuals replay the nominal gait.
  """
  residuals = np.clip(
    np.asarray(
      (
        frequency_residual,
        stride_residual,
        lift_residual,
        stance_push_residual,
        differential_gain_residual,
        bank_gain_residual,
        lift_phase_residual,
        height_residual,
      ),
      dtype=np.float64,
    ),
    -1.0,
    1.0,
  )

  differential_gain = np.clip(
    1.0 + 0.15 * residuals[4],
    0.85,
    1.15,
  )
  bank_gain = np.clip(
    1.0 + 0.20 * residuals[5],
    0.80,
    1.20,
  )

  half_track = 0.5 * TRACK_WIDTH_M
  left_speed = vx_cmd - differential_gain * half_track * yaw_cmd
  right_speed = vx_cmd + differential_gain * half_track * yaw_cmd

  max_side_speed = max(abs(left_speed), abs(right_speed))
  moving = max_side_speed > MOTION_THRESHOLD

  frequency_hz = np.clip(
    BASE_FREQUENCY_HZ
    + SPEED_FREQUENCY_GAIN * max_side_speed
    + 0.15 * residuals[0],
    FREQUENCY_MIN_HZ,
    FREQUENCY_MAX_HZ,
  )

  leg_phase = np.mod(phase + PHASE_OFFSETS, 2.0 * math.pi)
  first_half = leg_phase < math.pi

  u = np.where(
    first_half,
    leg_phase / math.pi,
    (leg_phase - math.pi) / math.pi,
  )
  s = smoothstep(u)

  common_shape = np.where(
    first_half,
    -1.0 + 2.0 * s,
    1.0 - 2.0 * s,
  )

  lift_phase_shift = 0.08 * math.pi * residuals[6]
  lift_phase = np.mod(leg_phase + lift_phase_shift, 2.0 * math.pi)
  lift_first_half = lift_phase < math.pi
  lift_u = np.where(
    lift_first_half,
    lift_phase / math.pi,
    (lift_phase - math.pi) / math.pi,
  )
  diff_shape = np.where(
    lift_first_half,
    -np.sin(math.pi * lift_u),
    np.sin(math.pi * lift_u),
  )

  side_speed = np.asarray((left_speed, right_speed), dtype=np.float64)
  side_amplitude = 0.65 + 0.35 * np.clip(
    np.abs(side_speed) / 0.16,
    0.0,
    1.0,
  )
  side_direction = np.sign(side_speed)
  side_direction[np.abs(side_speed) <= 1e-5] = 1.0

  # left = leg1/leg4, right = leg2/leg3
  leg_amplitude = np.asarray(
    (
      side_amplitude[0],
      side_amplitude[1],
      side_amplitude[1],
      side_amplitude[0],
    ),
    dtype=np.float64,
  )
  leg_direction = np.asarray(
    (
      side_direction[0],
      side_direction[1],
      side_direction[1],
      side_direction[0],
    ),
    dtype=np.float64,
  )

  stride_scale = np.clip(1.0 + 0.15 * residuals[1], 0.85, 1.15)
  lift_scale = np.clip(1.0 + 0.15 * residuals[2], 0.85, 1.15)
  stance_push = np.clip(1.0 + 0.12 * residuals[3], 0.88, 1.12)

  raw_swing = (
    COMMON_SIGN_FLIP
    * COMMON_RAW_AMP
    * common_shape
    * leg_direction
    * leg_amplitude
    * stride_scale
  )

  lift_wave_scale = np.where(first_half, 1.0, stance_push)
  raw_lift = DIFF_RAW_AMP * diff_shape * lift_scale * lift_wave_scale

  # The only turn mode:
  #   differential side speed + simultaneous inward bank.
  yaw_norm = np.clip(yaw_cmd / 0.60, -1.0, 1.0)
  bank_raw = bank_sign * BANK_RAW_MAX * yaw_norm * bank_gain
  raw_lift = raw_lift + bank_raw * BANK_SIDE_PATTERN

  raw_lift = raw_lift + 0.06 * residuals[7]

  if not moving:
    raw_swing[:] = 0.0
    raw_lift[:] = 0.0

  swing = SWING_SCALE * raw_swing
  lift = LIFT_SCALE * raw_lift

  # Closed-chain-safe projection:
  # max(|swing-lift|, |swing+lift|) == |swing| + |lift|.
  combined = np.abs(swing) + np.abs(lift)
  projection = np.minimum(
    SAFE_MOTOR_OFFSET_RAD / (combined + 1e-9),
    1.0,
  )
  swing *= projection
  lift *= projection

  a_targets = STAND_A + LEG_SIGNS * swing - LEG_SIGNS * lift
  e_targets = STAND_E + LEG_SIGNS * swing + LEG_SIGNS * lift

  motor_targets = np.empty(8, dtype=np.float64)
  motor_targets[0::2] = a_targets
  motor_targets[1::2] = e_targets

  expected_stance = np.where(moving, ~first_half, True)

  return CpgOutput(
    motor_targets=motor_targets,
    phase=phase,
    leg_phase=leg_phase,
    expected_stance=expected_stance,
    frequency_hz=float(frequency_hz),
    left_speed=float(left_speed),
    right_speed=float(right_speed),
    bank_raw=float(bank_raw),
    raw_swing=raw_swing,
    raw_lift=raw_lift,
  )


# -----------------------------------------------------------------------------
# XML augmentation
# -----------------------------------------------------------------------------

def _joint_ranges_from_xml(root: ET.Element) -> dict[str, tuple[float, float]]:
  ranges: dict[str, tuple[float, float]] = {}
  for joint in root.findall(".//joint"):
    name = joint.get("name")
    raw_range = joint.get("range")
    if name in ACTUATED_JOINTS and raw_range:
      values = [float(x) for x in raw_range.split()]
      if len(values) == 2:
        ranges[name] = (values[0], values[1])
  return ranges


def make_replay_xml(
  source_xml: Path,
  timestep: float,
  add_ground: bool,
) -> Path:
  """Create a temporary replay XML beside the source so mesh paths still work."""
  tree = ET.parse(source_xml)
  root = tree.getroot()

  option = root.find("option")
  if option is None:
    option = ET.Element("option")
    root.insert(1, option)
  option.set("timestep", str(timestep))
  option.set("integrator", "implicitfast")

  worldbody = root.find("worldbody")
  if worldbody is None:
    raise RuntimeError("The XML has no <worldbody>.")

  if add_ground and worldbody.find("./geom[@name='cpg_replay_ground']") is None:
    ET.SubElement(
      worldbody,
      "geom",
      {
        "name": "cpg_replay_ground",
        "type": "plane",
        "pos": "0 0 0",
        "size": "5 5 0.1",
        "contype": "1",
        "conaffinity": "1",
        "condim": "4",
        "friction": "1.2 0.02 0.001",
        "rgba": "0.20 0.24 0.28 1",
      },
    )

  actuator = root.find("actuator")
  if actuator is None:
    actuator = ET.SubElement(root, "actuator")

  existing_joints = {
    element.get("joint")
    for element in actuator
    if element.get("joint") is not None
  }
  joint_ranges = _joint_ranges_from_xml(root)

  for joint_name in ACTUATED_JOINTS:
    if joint_name in existing_joints:
      continue

    low, high = joint_ranges.get(joint_name, (-1.5, 1.5))
    ET.SubElement(
      actuator,
      "position",
      {
        "name": f"cpg_{joint_name}",
        "joint": joint_name,
        "kp": str(SERVO_KP),
        "kv": str(SERVO_KV),
        "ctrllimited": "true",
        "ctrlrange": f"{low} {high}",
        "forcelimited": "true",
        "forcerange": (
          f"{-SERVO_FORCE_LIMIT_NM} {SERVO_FORCE_LIMIT_NM}"
        ),
      },
    )

  # Save beside the original file so compiler meshdir="assets" remains valid.
  handle = tempfile.NamedTemporaryFile(
    mode="wb",
    suffix="_cpg_replay.xml",
    prefix="microtaur_",
    dir=source_xml.parent,
    delete=False,
  )
  handle.close()
  output_path = Path(handle.name)
  tree.write(output_path, encoding="utf-8", xml_declaration=True)
  return output_path


# -----------------------------------------------------------------------------
# Viewer helpers
# -----------------------------------------------------------------------------

def resolve_default_xml() -> Path:
  explicit = os.environ.get("MICROTAUR_XML")
  if explicit:
    return Path(explicit).expanduser().resolve()

  try:
    from microtaur_velocity.microtaur_constants import MICROTAUR_XML
  except Exception as exc:
    raise SystemExit(
      "Could not resolve the Microtaur XML automatically.\n"
      "Pass it explicitly with --xml C:\\path\\to\\robot_modified.xml"
    ) from exc

  return Path(MICROTAUR_XML).resolve()


def actuator_ids(model: mujoco.MjModel) -> np.ndarray:
  ids = []
  for joint_name in ACTUATED_JOINTS:
    actuator_name = f"cpg_{joint_name}"
    actuator_id = mujoco.mj_name2id(
      model,
      mujoco.mjtObj.mjOBJ_ACTUATOR,
      actuator_name,
    )
    if actuator_id < 0:
      # The source XML may already provide an actuator for this joint.
      joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
      )
      candidates = np.flatnonzero(model.actuator_trnid[:, 0] == joint_id)
      if candidates.size == 0:
        raise RuntimeError(f"No actuator found for {joint_name}.")
      actuator_id = int(candidates[0])
    ids.append(actuator_id)
  return np.asarray(ids, dtype=np.int32)


def find_free_joint(model: mujoco.MjModel) -> int | None:
  for joint_id in range(model.njnt):
    if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
      return joint_id
  return None


def quat_from_roll_pitch_yaw(
  roll: float,
  pitch: float,
  yaw: float,
) -> np.ndarray:
  """Quaternion in MuJoCo's [w, x, y, z] convention."""
  cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
  cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
  cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)

  return np.asarray(
    (
      cr * cp * cy + sr * sp * sy,
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy,
    ),
    dtype=np.float64,
  )


def set_free_joint_pose(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  joint_id: int,
  position: Iterable[float],
  quaternion: Iterable[float],
) -> None:
  qadr = int(model.jnt_qposadr[joint_id])
  dadr = int(model.jnt_dofadr[joint_id])
  data.qpos[qadr : qadr + 3] = np.asarray(tuple(position), dtype=np.float64)
  data.qpos[qadr + 3 : qadr + 7] = np.asarray(
    tuple(quaternion),
    dtype=np.float64,
  )
  data.qvel[dadr : dadr + 6] = 0.0


def demo_command(elapsed: float) -> Command:
  """Repeat a 12-second straight/left/straight/right demonstration."""
  t = elapsed % 12.0
  if t < 3.0:
    return Command(0.12, 0.0, "straight")
  if t < 6.0:
    return Command(0.12, +0.45, "left: differential + inward bank")
  if t < 9.0:
    return Command(0.12, 0.0, "straight")
  return Command(0.12, -0.45, "right: differential + inward bank")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Replay the Microtaur CPG without RL.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    "--xml",
    type=Path,
    default=None,
    help="Path to robot_modified.xml.",
  )
  parser.add_argument(
    "--vx",
    type=float,
    default=0.12,
    help="Constant forward velocity command in m/s.",
  )
  parser.add_argument(
    "--yaw",
    type=float,
    default=0.0,
    help="Constant yaw-rate command in rad/s.",
  )
  parser.add_argument(
    "--demo",
    action="store_true",
    help="Cycle through straight, left, straight, and right commands.",
  )
  parser.add_argument(
    "--duration",
    type=float,
    default=0.0,
    help="Stop after this many seconds; 0 runs until the viewer closes.",
  )
  parser.add_argument(
    "--bank-sign",
    type=float,
    choices=(-1.0, 1.0),
    default=1.0,
    help="Flip when the linkage banks outward instead of inward.",
  )
  parser.add_argument(
    "--initial-z",
    type=float,
    default=0.075,
    help="Initial floating-base height above the plane.",
  )
  parser.add_argument(
    "--timestep",
    type=float,
    default=0.002,
    help="MuJoCo physics timestep.",
  )
  parser.add_argument(
    "--fixed-base",
    action="store_true",
    help="Hold the root pose fixed while the constrained legs cycle.",
  )
  parser.add_argument(
    "--no-ground",
    action="store_true",
    help="Do not insert a ground plane.",
  )
  parser.add_argument(
    "--speed",
    type=float,
    default=1.0,
    help="Replay speed multiplier.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  xml_path = (
    args.xml.expanduser().resolve()
    if args.xml is not None
    else resolve_default_xml()
  )
  if not xml_path.exists():
    raise SystemExit(f"XML not found: {xml_path}")
  if args.timestep <= 0.0:
    raise SystemExit("--timestep must be positive.")
  if args.speed <= 0.0:
    raise SystemExit("--speed must be positive.")

  replay_xml = make_replay_xml(
    source_xml=xml_path,
    timestep=args.timestep,
    add_ground=not args.no_ground,
  )

  try:
    model = mujoco.MjModel.from_xml_path(str(replay_xml))
  finally:
    try:
      replay_xml.unlink()
    except OSError:
      pass

  data = mujoco.MjData(model)
  motor_ids = actuator_ids(model)
  free_joint_id = find_free_joint(model)

  if free_joint_id is not None:
    set_free_joint_pose(
      model,
      data,
      free_joint_id,
      position=(0.0, 0.0, args.initial_z),
      quaternion=(1.0, 0.0, 0.0, 0.0),
    )

  mujoco.mj_forward(model, data)

  phase = 0.0
  last_wall_time = time.perf_counter()
  start_wall_time = last_wall_time
  next_status_time = 0.0
  last_label = ""

  print(f"XML: {xml_path}")
  print(f"Actuators: {list(ACTUATED_JOINTS)}")
  print(
    "Turning mode: command differential stride + simultaneous inward bank"
  )
  print("Close the MuJoCo viewer to stop.")

  with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 0.55
    viewer.cam.elevation = -18.0
    viewer.cam.azimuth = 135.0

    while viewer.is_running():
      now = time.perf_counter()
      wall_dt = now - last_wall_time
      last_wall_time = now
      elapsed = now - start_wall_time

      if args.duration > 0.0 and elapsed >= args.duration:
        break

      command = (
        demo_command(elapsed)
        if args.demo
        else Command(args.vx, args.yaw, "constant command")
      )

      # Advance using simulation time, not rendering frame rate.
      output = cpg_targets(
        phase=phase,
        vx_cmd=command.vx,
        yaw_cmd=command.yaw,
        bank_sign=args.bank_sign,
      )

      moving = max(abs(output.left_speed), abs(output.right_speed)) > (
        MOTION_THRESHOLD
      )
      if moving:
        phase = (
          phase
          + 2.0
          * math.pi
          * output.frequency_hz
          * args.timestep
          * args.speed
        ) % (2.0 * math.pi)

      data.ctrl[motor_ids] = output.motor_targets

      if args.fixed_base and free_joint_id is not None:
        # Hold the floating base while allowing the passive linkage joints and
        # equality constraints to settle dynamically.
        set_free_joint_pose(
          model,
          data,
          free_joint_id,
          position=(0.0, 0.0, args.initial_z),
          quaternion=(1.0, 0.0, 0.0, 0.0),
        )

      mujoco.mj_step(model, data)

      if command.label != last_label:
        print(
          f"\n[{elapsed:5.1f}s] {command.label}: "
          f"vx={command.vx:+.3f}, yaw={command.yaw:+.3f}"
        )
        last_label = command.label

      if elapsed >= next_status_time:
        stance = "".join(
          "S" if value else "-"
          for value in output.expected_stance
        )
        print(
          f"  f={output.frequency_hz:.3f} Hz | "
          f"vL={output.left_speed:+.3f} | "
          f"vR={output.right_speed:+.3f} | "
          f"bank={output.bank_raw:+.3f} | "
          f"stance={stance}"
        )
        next_status_time = elapsed + 1.0

      viewer.sync()

      # Real-time pacing. Multiple physics steps may occur per rendered frame,
      # but this simple pacing is sufficient for visual gait inspection.
      target_wall_dt = args.timestep / args.speed
      spent = time.perf_counter() - now
      if spent < target_wall_dt:
        time.sleep(target_wall_dt - spent)


if __name__ == "__main__":
  main()