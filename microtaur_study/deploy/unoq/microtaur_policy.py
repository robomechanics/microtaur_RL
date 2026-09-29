"""Microtaur v2 student policy: standalone reference implementation (numpy only).

Everything the robot needs between the sensors and the servo targets, bit-for-bit the
same maths as the IsaacLab training pipeline (checked by test_golden.py against a
recording from the simulator):

  sensors -> build_frame() -> ObsHistory.push() -> Policy.__call__() -> ActionPipeline.step() -> targets

Conventions (identical to the simulator):
  * Joint order (canonical): leg1_a, leg1_e, leg2_a, leg2_e, leg3_a, leg3_e, leg4_a, leg4_e.
    leg1 = rear right, leg2 = rear left, leg3 = front left, leg4 = front right.
  * Joint angles: simulator radians (same zero and sign as the MJCF joints); stand pose STAND.
    Use rad_from_ticks() / ticks_from_rad() for the XL330 tick conversion (see README: the
    upstream calibration constants are flagged unverified).
  * Body frame: x forward, y left, z up (the battery / root body).
  * Policy period DT = 0.035 s (28.6 Hz).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------------------
# Interface constants
# -----------------------------------------------------------------------------

DT = 0.035  # s, one policy step (2.5 ms physics x 14)
HISTORY = 15  # frames per observation term
OBS_CLIP = 100.0  # every observation term is clipped to +-OBS_CLIP before the history
ACTION_SCALE_RAD = math.radians(30.0)  # motor offset from STAND at |action| = 1

JOINT_NAMES = tuple(f"leg{i}_{m}_joint_act" for i in range(1, 5) for m in ("a", "e"))
STAND = np.array([0.45, -0.45, -0.45, 0.45, -0.45, 0.45, 0.45, -0.45])  # rad, canonical order
LEG_SIGNS = np.array([1.0, -1.0, -1.0, 1.0])
JOINT_HALF_RANGE_RAD = 0.75

# Observation terms in order: (name, size). Frame = 47 values; the network input is term-major:
# [term0 x 15 frames (oldest first), term1 x 15 frames, ...] = 705 values.
TERMS = (
  ("projected_gravity", 3),  # front IMU: unit gravity direction in the body frame, ~(0, 0, -1) when level
  ("base_ang_vel", 3),  # front IMU: angular velocity, body frame, rad/s
  ("imu2_projected_gravity", 3),  # rear IMU slot (spine robot): ZEROS on the rigid robot
  ("imu2_base_ang_vel", 3),  # rear IMU slot (spine robot): ZEROS on the rigid robot
  ("joint_pos", 8),  # motor angle - STAND, rad
  ("joint_vel", 8),  # motor velocity, rad/s
  ("foot_fk", 8),  # five-bar FK foot (x, z) per leg in the leg plane, m: x1, z1, ..., x4, z4
  ("actions", 8),  # previous RAW network output, NOT clipped (IsaacLab last_action; zeros after reset)
  ("command", 3),  # (vx m/s, vy = 0, yaw rate rad/s), body frame
)
FRAME_DIM = sum(n for _, n in TERMS)  # 47
OBS_DIM = FRAME_DIM * HISTORY  # 705

# Commands seen in training: vx 0.10-0.35 m/s, yaw rate -0.25..0.25 rad/s, vy always 0.
COMMAND_RANGE = {"vx": (0.10, 0.35), "vy": (0.0, 0.0), "wz": (-0.25, 0.25)}

# -----------------------------------------------------------------------------
# Five-bar forward kinematics (port of microtaur_common.kinematics.forward_torch)
# -----------------------------------------------------------------------------

_L0 = 0.0205000  # motor pivot distance
_L1 = 0.0464129  # proximal link
_L2 = 0.0708302  # distal link
_LF = 0.08296474  # distal joint to foot
_FOOT_DELTA = math.radians(4.31513)
_THETA_AB0 = -3.0 * math.pi / 4.0
_THETA_ED0 = -math.pi / 4.0
_AX, _AZ = -0.5 * _L0, 0.0
_EX, _EZ = +0.5 * _L0, 0.0


def foot_fk(q_abs: np.ndarray) -> np.ndarray:
  """Foot (x, z) per leg from the 8 absolute motor angles (canonical order) -> [x1, z1, ..., x4, z4] (m)."""
  q = np.asarray(q_abs, dtype=np.float64)
  qa, qe = q[0::2], q[1::2]
  theta_ab = _THETA_AB0 - LEG_SIGNS * qa
  theta_ed = _THETA_ED0 - LEG_SIGNS * qe
  bx, bz = _AX + _L1 * np.cos(theta_ab), _AZ + _L1 * np.sin(theta_ab)
  dx, dz = _EX + _L1 * np.cos(theta_ed), _EZ + _L1 * np.sin(theta_ed)
  ux, uz = dx - bx, dz - bz
  dist = np.maximum(np.sqrt(ux * ux + uz * uz), 1e-12)
  ex, ez = ux / dist, uz / dist
  h = np.sqrt(np.maximum(_L2 * _L2 - 0.25 * dist * dist, 0.0))
  mx, mz = bx + 0.5 * ux, bz + 0.5 * uz
  c1z, c2z = mz + h * ex, mz - h * ex  # the two closure candidates; keep the lower one
  lower_first = c1z <= c2z
  cx = np.where(lower_first, mx - h * ez, mx + h * ez)
  cz = np.where(lower_first, c1z, c2z)
  theta_df = np.arctan2(cz - dz, cx - dx) + _FOOT_DELTA
  out = np.empty(8)
  out[0::2] = dx + _LF * np.cos(theta_df)
  out[1::2] = dz + _LF * np.sin(theta_df)
  return out


# -----------------------------------------------------------------------------
# Observation frame and history
# -----------------------------------------------------------------------------


def build_frame(gravity_b, ang_vel_b, q_abs, qd, last_action, command) -> dict[str, np.ndarray]:
  """One observation frame (47 values, as a dict per term) from the robot's signals.

  gravity_b   (3,) unit gravity direction in the body frame (from the IMU orientation, or the
              normalised accelerometer when static: gravity_b = -acc / |acc|)
  ang_vel_b   (3,) gyro, body frame, rad/s
  q_abs       (8,) motor angles, simulator radians, canonical order
  qd          (8,) motor velocities, rad/s
  last_action (8,) the previous RAW network output Policy(obs), unclipped (zeros after reset)
  command     (3,) (vx, 0, wz)
  """
  q = np.asarray(q_abs, dtype=np.float64)
  return {
    "projected_gravity": np.asarray(gravity_b, dtype=np.float64),
    "base_ang_vel": np.asarray(ang_vel_b, dtype=np.float64),
    "imu2_projected_gravity": np.zeros(3),
    "imu2_base_ang_vel": np.zeros(3),
    "joint_pos": q - STAND,
    "joint_vel": np.asarray(qd, dtype=np.float64),
    "foot_fk": foot_fk(q),
    "actions": np.asarray(last_action, dtype=np.float64),
    "command": np.asarray(command, dtype=np.float64),
  }


class ObsHistory:
  """Per-term ring buffers exactly as IsaacLab's ObservationManager with history_length=15 and
  flatten_history_dim=True: each term keeps its own 15 frames (oldest first) and the network
  input is the concatenation term by term. The first push after reset() fills all 15 frames."""

  def __init__(self):
    self.reset()

  def reset(self):
    self._buf: dict[str, np.ndarray] | None = None

  def push(self, frame: dict[str, np.ndarray]) -> np.ndarray:
    if self._buf is None:
      self._buf = {n: np.tile(np.clip(frame[n], -OBS_CLIP, OBS_CLIP), (HISTORY, 1)) for n, _ in TERMS}
    else:
      for n, _ in TERMS:
        self._buf[n] = np.vstack((self._buf[n][1:], np.clip(frame[n], -OBS_CLIP, OBS_CLIP)[None]))
    return np.concatenate([self._buf[n].reshape(-1) for n, _ in TERMS]).astype(np.float32)


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------


class Policy:
  """obs (705,) -> action (8,), unclipped. Normalisation (x - mean) / (std + eps) and the ELU MLP
  from policy_weights.npz; or onnxruntime on policy.onnx (same maths, normalisation included)."""

  def __init__(self, weights_npz: str | Path, onnx_path: str | Path | None = None):
    w = np.load(weights_npz)
    self.mean, self.std, self.eps = w["obs_mean"], w["obs_std"], float(w["obs_eps"])
    self.layers = [(w[f"W{i}"], w[f"b{i}"]) for i in range(int(w["n_layers"]))]
    self._ort = None
    if onnx_path is not None:
      import onnxruntime as ort  # optional

      self._ort = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    if self._ort is not None:
      return self._ort.run(None, {"obs": obs.reshape(1, -1).astype(np.float32)})[0][0]
    x = (obs.astype(np.float32) - self.mean) / (self.std + self.eps)
    for i, (W, b) in enumerate(self.layers):
      x = W @ x + b
      if i < len(self.layers) - 1:
        x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))  # ELU(alpha=1)
    return x


# -----------------------------------------------------------------------------
# Action -> servo targets (port of microtaur_common.walk_action.WalkActionCore, one robot,
# no per-episode gain / bias randomisation, no artificial delay)
# -----------------------------------------------------------------------------


class ActionPipeline:
  """clip(action, -1, 1) -> requested = STAND + 30 deg * action -> safety filter (joint limits
  STAND +- 0.75 rad with a time-to-limit blend, per-leg common / diff limits 0.52 rad and
  |common| + |diff| <= 0.70 rad) -> target (rad). The filter needs the measured q and qd."""

  horizon_s = 0.10
  velocity_floor = math.radians(60.0)
  joint_margin = math.radians(8.0)
  coupled_margin = math.radians(10.0)
  common_limit = 0.52
  diff_limit = 0.52
  diamond_limit = 0.70

  def __init__(self):
    self.lower = STAND - JOINT_HALF_RANGE_RAD
    self.upper = STAND + JOINT_HALF_RANGE_RAD

  def _axis(self, pos, vel, req, lo, hi, margin):
    lo = np.broadcast_to(np.asarray(lo, dtype=np.float64), pos.shape)
    hi = np.broadcast_to(np.asarray(hi, dtype=np.float64), pos.shape)
    m_lo, m_hi = pos - lo, hi - pos
    outside = (m_lo < 0.0) | (m_hi < 0.0)
    v = np.where(np.abs(vel) < self.velocity_floor, np.sign(vel) * self.velocity_floor, vel)
    with np.errstate(divide="ignore", invalid="ignore"):
      t_hi = np.where((v > 0.0) & (m_hi > 0.0), m_hi / np.maximum(v, 1e-6), np.inf)
      t_lo = np.where((v < 0.0) & (m_lo > 0.0), m_lo / np.maximum(-v, 1e-6), np.inf)
    blend = np.clip(1.0 - np.minimum(t_hi, t_lo) / self.horizon_s, 0.0, 1.0)
    blend = np.where(outside, 1.0, blend)
    near_hi, near_lo = m_hi < margin, m_lo < margin
    blend = np.where(near_hi & (req > pos), 1.0, blend)
    blend = np.where(near_lo & (req < pos), 1.0, blend)
    blend = np.where(near_hi & (req < pos), 0.0, blend)
    blend = np.where(near_lo & (req > pos), 0.0, blend)
    clamped = np.minimum(np.maximum(req, lo), hi)
    return (1.0 - blend) * req + blend * clamped, blend

  def filter(self, q, qd, requested):
    ind, _ = self._axis(q, qd, requested, self.lower, self.upper, self.joint_margin)
    s = LEG_SIGNS
    st = STAND.reshape(4, 2)
    q2, qd2, qr = q.reshape(4, 2), qd.reshape(4, 2), ind.reshape(4, 2)
    da, de = q2[:, 0] - st[:, 0], q2[:, 1] - st[:, 1]
    rda, rde = qr[:, 0] - st[:, 0], qr[:, 1] - st[:, 1]
    common, diff = 0.5 * s * (da + de), 0.5 * s * (de - da)
    common_dot, diff_dot = 0.5 * s * (qd2[:, 0] + qd2[:, 1]), 0.5 * s * (qd2[:, 1] - qd2[:, 0])
    req_common, req_diff = 0.5 * s * (rda + rde), 0.5 * s * (rde - rda)
    safe_common, _ = self._axis(common, common_dot, req_common, -self.common_limit, self.common_limit, self.coupled_margin)
    safe_diff, _ = self._axis(diff, diff_dot, req_diff, -self.diff_limit, self.diff_limit, self.coupled_margin)
    radius = np.abs(common) + np.abs(diff)
    req_radius = np.abs(safe_common) + np.abs(safe_diff)
    radial_v = np.sign(common) * common_dot + np.sign(diff) * diff_dot
    dist = self.diamond_limit - radius
    with np.errstate(divide="ignore", invalid="ignore"):
      t = np.where((radial_v > 0.0) & (dist > 0.0), dist / np.maximum(radial_v, 1e-6), np.inf)
    dia = np.clip(1.0 - t / self.horizon_s, 0.0, 1.0)
    dia = np.where(radius > self.diamond_limit, 1.0, dia)
    near, outward = dist < self.coupled_margin, req_radius > radius
    dia = np.where(near & outward, 1.0, dia)
    dia = np.where(near & ~outward, 0.0, dia)
    proj = np.minimum(self.diamond_limit / (req_radius + 1e-6), 1.0)
    safe_common = (1.0 - dia) * safe_common + dia * (safe_common * proj)
    safe_diff = (1.0 - dia) * safe_diff + dia * (safe_diff * proj)
    pairs = np.empty((4, 2))
    pairs[:, 0] = st[:, 0] + s * (safe_common - safe_diff)
    pairs[:, 1] = st[:, 1] + s * (safe_common + safe_diff)
    return np.minimum(np.maximum(pairs.reshape(8), self.lower), self.upper)

  def step(self, action, q, qd):
    """-> (clipped action, target motor angles in rad). Feed the RAW (unclipped) network output
    back as last_action, not this clipped one."""
    a = np.clip(np.nan_to_num(np.asarray(action, dtype=np.float64), nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    requested = STAND + ACTION_SCALE_RAD * a
    return a, self.filter(np.asarray(q, dtype=np.float64), np.asarray(qd, dtype=np.float64), requested)


# -----------------------------------------------------------------------------
# XL330 ticks <-> simulator radians (constants from upstream
# sim2real/microtaur_hw_characterization/config/hardware_config.py, CALIBRATION_VERIFIED = False)
# -----------------------------------------------------------------------------

RAD_PER_TICK = 2.0 * math.pi / 4096
JOINT_TO_DXL_ID = (5, 6, 3, 4, 1, 2, 7, 8)  # canonical order
HW_SIGN = np.array([-1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0])
STAND_TICK = np.array([3590, 625, 3538, 532, 3473, 491, 3685, 606])


def rad_from_ticks(ticks) -> np.ndarray:
  """Present Position ticks (canonical joint order) -> simulator radians."""
  return STAND + HW_SIGN * (np.asarray(ticks, dtype=np.float64) - STAND_TICK) * RAD_PER_TICK


def ticks_from_rad(q) -> np.ndarray:
  """Simulator radians (canonical order) -> Goal Position ticks (rounded)."""
  return np.rint(STAND_TICK + HW_SIGN * (np.asarray(q, dtype=np.float64) - STAND) / RAD_PER_TICK).astype(np.int64)


# -----------------------------------------------------------------------------
# One control step
# -----------------------------------------------------------------------------


class Controller:
  """Glue for the robot loop, called once every DT:

    ctrl = Controller("policy_weights.npz"); ctrl.reset()
    target_rad = ctrl.step(gravity_b, ang_vel_b, q_abs, qd, command)
  """

  def __init__(self, weights_npz, onnx_path=None):
    self.policy = Policy(weights_npz, onnx_path)
    self.pipeline = ActionPipeline()
    self.history = ObsHistory()
    self.last_action = np.zeros(8)

  def reset(self):
    self.history.reset()
    self.last_action = np.zeros(8)

  def step(self, gravity_b, ang_vel_b, q_abs, qd, command):
    obs = self.history.push(build_frame(gravity_b, ang_vel_b, q_abs, qd, self.last_action, command))
    raw = self.policy(obs)
    _, target = self.pipeline.step(raw, q_abs, qd)
    self.last_action = raw  # unclipped, as in training
    return target
