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

KP = 1.0  # measured omega_n is within 4% of sqrt(KP/J); do not move it

# 2026-10-28 sweep on the rigid robot, suspended, one joint at a time
# (docs/ACTUATOR_SYSID_2026-10-28.md): omega_n 11.0 Hz, zeta 0.292, a 1.79x
# gain peak at 10 Hz, and a 9.6-11.6 ms transport delay flat across 1-6 Hz.
# zeta = KD/(2 sqrt(KP*J)) with J = 2.284e-4 from mj_fullM gives KD = 0.0092.
#
# !! KD AND THE FIRMWARE'S SERVO D GAIN ARE ONE QUANTITY. !!
# 0.0092 is the value for a robot whose firmware runs Position D Gain = 200.
# The upstream characterization used D = 0, measured 63% overshoot and 144 ms
# settling (zeta 0.146) and fitted KD = 0.0 to it, matching to 1 ms. Both are
# right; they are different controllers. Changing one without the other breaks
# the pairing that sim2real exists to maintain, and the deployment firmware
# (sim2real/async_deployment_v1/.../sketch.ino) still writes D = 0.
# D = 200 is the better-behaved robot -- half the resonant peak, 1.79x against
# 3.46x -- which is why this file carries its value.
KD = 0.0092
ARMATURE = 2e-4  # equals BAM's identified 2.0005e-4 for this motor

# 0.60 of stall = 0.129 N m (871 mA). effort_limit is the one protection setting
# here (velocity_limit is the zero-torque speed, so lowering that weakens the
# motor at every speed instead of capping anything).
# 0.40 (581 mA) was tried on 2026-10-28 (t3) and is too weak: friction 0.028 plus
# the worst static holding demand (~0.030) already take ~68% of 0.086 N m, the
# trained teacher sat at the cap 54-66% of motor-steps, walked 0.14 m/s at any
# command and could not climb the 37 mm step. It is a protection choice, not a
# measured limit: the deployed firmware sets no current limit and hardware drew
# 1404 mA (~0.208 N m, ~97% of stall) transiently while braking. If motor current
# has to be limited, set the servo's Current Limit register and copy that value here.
EFFORT_LIMIT_NM = 0.60 * XL330_STALL_TORQUE_NM

# Unchanged: 0.80 * no-load is 306 rpm, between the measured floor of 248 rpm
# (a full-travel step, which had not saturated) and the voltage-scaled
# datasheet figure near 345 rpm.
VELOCITY_LIMIT_RAD_S = 0.80 * XL330_NO_LOAD_SPEED_RAD_S

# Measured Coulomb equivalent, from the intercept of the loaded holding fit
# err ~= 9.9 + 1.41*tau(mNm): 9.9 ticks of dead zone is 28 mNm, 13% of stall.
# The spread between joints is real and about 2x -- id6 draws twice id1's
# current and lags 1.3-2.4x more below 8 Hz while their gain curves overlap --
# so randomising this per joint over roughly [0.018, 0.050] models device
# variation rather than noise.
FRICTIONLOSS_NM = 0.028

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

# -----------------------------------------------------------------------------
# Deployment loop timing. Measured upstream (aryan-chandra-cmu/Microtaur_RL
# main, sim2real/README.md and env_cfgs.py:43-50), not here; kept because
# nothing in this repo measures it and the policy period depends on it.
# -----------------------------------------------------------------------------
# 35 ms is a ceiling, not a preference: 25-33 Hz was reliable, 35-37 Hz began
# missing deadlines and 40 Hz was not deterministic. Enforce it with an integer
# decimation and raise if the product does not land on it exactly.
HARDWARE_CONTROL_PERIOD_S = 0.035

# Motor sample to action mailbox, i.e. the sensing and compute path. This is a
# DIFFERENT segment from the 9.6-11.6 ms actuation delay measured here (goal
# written to joint moving); the two do not overlap and should not be summed
# into one "delay" without saying which is which.
HARDWARE_SENSOR_TO_ACTION_MEAN_S = 0.0540
HARDWARE_SENSOR_TO_ACTION_MAX_S = 0.0624

# The action delay is one policy step because the lag was measured at 100% one
# frame, not fitted. Any extra-step probability is for scheduler tails only.
HARDWARE_ACTION_DELAY_STEPS = 1

# -----------------------------------------------------------------------------
# Spine actuator, for when a spine variant is added. Measured 2026-10-28 on the
# yaw robot, spine only (XL330-M288, model 1190, 288.4:1, driven at the robot's
# own P gain of 3000 against the legs' 1500): omega_n 3.4 Hz, zeta 0.341, a
# 1.56x peak at 3.0 Hz. It is a different KIND of system from a leg -- its lag
# in milliseconds is not flat with frequency (19 -> 72 -> 48 ms), so it is its
# own second-order response rather than a transport delay, and at 4.431 Hz it
# sits about 109 degrees behind its command where a leg sits 21-30 behind.
#
# !! The simulated spine gets omega_n right by cancellation. !! The yaw model's
# spine joint has J = 2.546e-3 and inherits the legs' KP = 1.0, giving 3.15 Hz
# against 3.4 Hz measured -- but omega_n fixes only k/J. The peak current
# separates them: at 2 Hz and a 120-tick half amplitude the hardware drew
# 1233 mA, which those sim values predict as 180 mA and (k 7.44, J 0.0163)
# predicts as 1338 mA. Both stiffness and inertia are about 6-7x low, so the
# ratio is right and the absolute torque is wrong by the same factor. Fixing
# one alone moves omega_n off the measurement.
SPINE_OMEGA_N_HZ_MEASURED = 3.4
SPINE_ZETA_MEASURED = 0.341
SPINE_KP_FROM_MEASUREMENT = 7.44       # against the legs' 1.0
SPINE_INERTIA_FROM_MEASUREMENT = 0.0163
# Usable linear amplitude is narrow: roughly 90-180 ticks at 2 Hz, with 59 ms
# of stiction lag below it (three times the large-amplitude value, where the
# legs showed 1.8x) and torque saturation above it.

# Passive (unactuated) five-bar joints. Already set per joint in the XML; listed
# here for simulators that build joints from these numbers (IsaacLab).
PASSIVE_ARMATURE = 1e-6
PASSIVE_FRICTIONLOSS_NM = 0.001
PASSIVE_DAMPING = 1e-4

# -----------------------------------------------------------------------------
# Active-yaw variant: assets/active_yaw_microtaur/robot_2imu.xml (the user's 2026-10-29 model with two IMU
# sites; chassis masses NOT yet re-measured -- see the XML header). Legs are identical to the rigid model.
# -----------------------------------------------------------------------------
YAW_ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "active_yaw_microtaur"
YAW_XML_PATH = YAW_ASSET_DIR / "robot_2imu.xml"
YAW_ROOT_BODY = "s_microtaur_bottom_half_motor_mount"  # front half (legs 3, 4), carries IMU 1
YAW_REAR_BODY = "s_microtaur_top_half_motor_mount"  # rear half (legs 1, 2), carries IMU 2
SPINE_JOINT_NAME = "spine_joint_act"
# IMU sites (from the XML): IMU 1 is aligned with the root frame; IMU 2's site is rotated 180 deg about y
# in the rear body (whose own frame is flipped), so both read in the robot frame (x fwd, z up) at stand.
YAW_IMU1_QUAT_IN_ROOT = (1.0, 0.0, 0.0, 0.0)  # wxyz
YAW_IMU2_QUAT_IN_REAR = (0.0, 0.0, 1.0, 0.0)  # wxyz

# Spine actuator, XL330-M288 (288.4:1), from the 2026-10-28 measurement above (SPINE_*_MEASURED):
# omega_n 3.4 Hz, zeta 0.341 at the robot's own P gain 3000. The model's spine inertia (link 2.35e-3 +
# armature 2e-4) is ~6-7x below what the measured current needs, so the armature makes up the measured
# total; KD follows from zeta. isaac_make_usd_yaw.py checks the resulting inertia and natural frequency.
XL330_M288_STALL_TORQUE_NM = 0.52  # datasheet
SPINE_KP = SPINE_KP_FROM_MEASUREMENT  # 7.44 N m / rad
SPINE_LINK_INERTIA = 2.3838e-3  # mj_fullM about the spine axis at the stand pose, without armature (checked by the build)
SPINE_ARMATURE = SPINE_INERTIA_FROM_MEASUREMENT - SPINE_LINK_INERTIA  # 0.01395 -> total 0.0163 kg m^2
SPINE_KD = 2.0 * SPINE_ZETA_MEASURED * math.sqrt(SPINE_KP * SPINE_INERTIA_FROM_MEASUREMENT)  # 0.238
# Measured, not a protection choice: the current clamps near 1300 mA (2026-10-28 amplitude sweep, torque
# saturation at 320 ticks); 1233 mA <-> 0.474 N m implies kt 0.384 N m / A, so the ceiling is ~0.50 N m.
SPINE_EFFORT_LIMIT_NM = 0.50
SPINE_VELOCITY_LIMIT_RAD_S = 0.80 * 103.0 * 2.0 * math.pi / 60.0  # datasheet no-load 103 rpm (TO CONFIRM)
SPINE_FRICTIONLOSS_NM = 0.010  # the XML's value; not measured
