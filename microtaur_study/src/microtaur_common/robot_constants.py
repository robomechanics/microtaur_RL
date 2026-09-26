"""Rigid Microtaur constants: names, stand pose, XL330 motor and electrical model.

Framework-free; used by both the mjlab and the IsaacLab environment. Model:
assets/rigid_microtaur/robot_modified.xml (upstream 2026-09-20, total 0.540 kg).
"""

from __future__ import annotations

import math
from pathlib import Path

ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "rigid_microtaur"
XML_PATH = ASSET_DIR / "robot_modified.xml"

EXPECTED_TOTAL_MASS_KG = 0.540
ROOT_BODY = "battery"

# Joint order is part of the action/observation interface: (a, e) for legs 1-4.
LEG_JOINT_NAMES = tuple(
  f"leg{i}_{m}_joint_act" for i in range(1, 5) for m in ("a", "e")
)
PASSIVE_JOINT_NAMES = tuple(
  f"leg{i}_{m}_joint" for i in range(1, 5) for m in ("b", "d")
)
FOOT_SITE_NAMES = tuple(f"leg{i}_foot_site" for i in range(1, 5))
FOOT_GEOM_NAMES = tuple(f"leg{i}_foot_collision" for i in range(1, 5))
BODY_COLLISION_GEOM_NAMES = ("rl_body_collision", "rl_battery_collision")

# Nominal stand pose (motor angles, rad) and the per-leg mirror sign.
STAND_A = (0.45, -0.45, -0.45, 0.45)
STAND_E = (-0.45, 0.45, 0.45, -0.45)
LEG_SIGNS = (1.0, -1.0, -1.0, 1.0)
STAND_JOINT_POS = {
  name: (STAND_A if name.endswith("_a_joint_act") else STAND_E)[int(name[3]) - 1]
  for name in LEG_JOINT_NAMES
}
JOINT_HALF_RANGE_RAD = 0.75


# -----------------------------------------------------------------------------
# ROBOTIS XL330-M077-T, output shaft, 5.0 V servo bus (datasheet)
# -----------------------------------------------------------------------------

XL330_SUPPLY_V = 5.0
XL330_STALL_TORQUE_NM = 0.215
XL330_STALL_CURRENT_A = 1.47
XL330_NO_LOAD_SPEED_RAD_S = 383.0 * 2.0 * math.pi / 60.0

KP = 1.0
KD = 0.0  # 2026-09-20 hardware calibration; never validated in isolation.
ARMATURE = 2e-4
EFFORT_LIMIT_NM = 0.60 * XL330_STALL_TORQUE_NM  # training-time torque cap
VELOCITY_LIMIT_RAD_S = 0.80 * XL330_NO_LOAD_SPEED_RAD_S
FRICTIONLOSS_NM = 0.010

# Electrical model for the energy reward, derived from the stall point:
#   R_eff = V / I_stall, k_t = tau_stall / I_stall, copper loss = (R / k_t^2) tau^2.
# Datasheet-derived only: k_t = 0.146 N m/A disagrees by 17% with the back-EMF
# constant V / w_no_load = 0.125 V s/rad, so treat this as an estimate until it
# is checked against the current traces from the hardware characterization.
XL330_R_OHM = XL330_SUPPLY_V / XL330_STALL_CURRENT_A
XL330_KT_NM_PER_A = XL330_STALL_TORQUE_NM / XL330_STALL_CURRENT_A
XL330_COPPER_W_PER_NM2 = XL330_R_OHM / XL330_KT_NM_PER_A**2  # ~159


def motor_joint_ranges() -> dict[str, tuple[float, float]]:
  """Motor limits centred on the stand pose."""
  return {
    name: (stand - JOINT_HALF_RANGE_RAD, stand + JOINT_HALF_RANGE_RAD)
    for name, stand in STAND_JOINT_POS.items()
  }

# Passive (unactuated) five-bar joints. Already set per joint in the XML; listed
# here for simulators that build joints from these numbers (IsaacLab).
PASSIVE_ARMATURE = 1e-6
PASSIVE_FRICTIONLOSS_NM = 0.001
PASSIVE_DAMPING = 1e-4
