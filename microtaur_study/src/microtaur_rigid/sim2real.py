"""Sim2real randomisation profile, one row per stage.

Stage 0 is the measured nominal model, stage 1 adds narrow uncertainty, stage 2
adds deployment-oriented robustness. All values are copied unchanged from the
2026-09-20 upstream config, where they were spread over six functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Sim2RealStage:
  # Actor observation noise (uniform half-widths). BNO055 stationary gyro sigma
  # was measured at ~0.0011-0.0015 rad/s.
  ang_vel_noise: float
  gravity_noise: float
  joint_pos_noise: float
  joint_vel_noise: float
  # Per-episode command-side uncertainty, applied in the action term.
  action_gain_range: tuple[float, float]
  action_bias_rad: float
  # Startup events. None disables the event.
  encoder_bias_rad: float
  foot_friction_range: tuple[float, float] | None
  root_com_xy_m: float | None
  root_com_z_m: float | None
  push: bool


STAGES = {
  0: Sim2RealStage(
    ang_vel_noise=0.003, gravity_noise=0.004, joint_pos_noise=0.002, joint_vel_noise=0.08,
    action_gain_range=(1.0, 1.0), action_bias_rad=0.0,
    encoder_bias_rad=0.0015, foot_friction_range=None,
    root_com_xy_m=None, root_com_z_m=None, push=False,
  ),
  1: Sim2RealStage(
    ang_vel_noise=0.006, gravity_noise=0.008, joint_pos_noise=0.004, joint_vel_noise=0.15,
    action_gain_range=(0.97, 1.03), action_bias_rad=math.radians(0.20),
    encoder_bias_rad=0.0030, foot_friction_range=(0.60, 0.85),
    root_com_xy_m=0.002, root_com_z_m=0.001, push=False,
  ),
  2: Sim2RealStage(
    ang_vel_noise=0.010, gravity_noise=0.015, joint_pos_noise=0.006, joint_vel_noise=0.25,
    action_gain_range=(0.93, 1.07), action_bias_rad=math.radians(0.45),
    encoder_bias_rad=0.0060, foot_friction_range=(0.50, 1.00),
    root_com_xy_m=0.004, root_com_z_m=0.002, push=True,
  ),
}

# Push magnitudes (flat terrain uses 0.65 of these).
PUSH_INTERVAL_S = (6.0, 10.0)
PUSH_VELOCITY_RANGE = {
  "x": 0.06, "y": 0.04, "z": 0.015, "roll": 0.06, "pitch": 0.06, "yaw": 0.10,
}
PUSH_FLAT_SCALE = 0.65
