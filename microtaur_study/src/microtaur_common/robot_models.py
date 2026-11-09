"""The robot models the IsaacLab env can train: one entry per MJCF, with the per-model numbers the env needs.

The env cfg field `variant` (saved in every run's params/env.yaml) names the entry. "rigid" and "yaw" are the
models of every run up to 2026-11-04 and keep their numbers exactly; the "*_uno" entries are the 2026-11-04
hardware layout (MKR1000 removed, UNO Q + DYNAMIXEL shield 41 g and one 3 g board per IMU added, battery kept;
built by /home/rml3/Documents/ben/spine/models_uno_imu_2026-11-04/build_uno_imu.py from the current models).

Framework-free (no torch / isaaclab). Numbers taken from the MJCF at the stand pose with the spine at 0
(MuJoCo kinematics); scripts/check_robot_models.py recomputes and checks every one of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

_ASSETS = Path(__file__).resolve().parents[2] / "assets"

# Foot bodies and foot-sphere centres in them, legs 1..4 (RR, RL, FL, FR)
_RIGID_FEET = ("leglink2_v2_with_leg_down", "leglink2_v2_with_leg_down_2", "leglink2_v2_with_leg_down_3",
               "leglink2_v2_with_leg_down_4")
_RIGID_FOOT_OFFSETS = ((-0.017059, -0.081192, -0.0015), (-0.017059, 0.081192, -0.0015), (0.08273, 0.006242, -0.0015),
                       (0.08273, -0.006242, -0.0015))
# Active-yaw MJCF: same foot body names, other lower-link CAD frames, so the sphere sits elsewhere in its body.
_YAW_FOOT_OFFSETS = ((-0.01860744, -0.0808511886, -0.0015), (0.0825952176, 0.00782186238, -0.0015),
                     (-0.01860744, 0.0808511886, -0.0015), (-0.01860744, -0.0808511886, -0.0015))


@dataclass(frozen=True)
class RobotModel:
  name: str
  asset_dir: Path
  xml: str  # file names inside asset_dir
  usd: str
  visual_usd: str
  root_body: str  # articulation root (carries the root IMU: imu site aligned with the root frame)
  total_mass_kg: float  # MJCF total (motor energy is normalised by m g v)
  nominal_root_height_m: float  # root height with the lowest foot sphere on the ground, stand pose, spine 0
  hip_z_root_m: tuple[float, float, float, float]  # leg-plane origin (motor-pivot midpoint) heights, root frame
  foot_body_names: tuple[str, str, str, str]
  foot_offset_in_body_m: tuple
  foot_extension_angle_rad: float  # foot bend off the distal link (kinematics.MicrotaurFiveBarKinematics)
  spine: str | None = None  # None (rigid), "yaw", "roll" or "pitch": what the 9th motor turns
  other_half_body: str | None = None  # spine robots: the second trunk half (second IMU, illegal contact)
  imu2_quat_in_other_half: tuple[float, float, float, float] | None = None  # wxyz, second IMU site in that body

  @property
  def has_spine(self) -> bool:
    return self.spine is not None

  @property
  def spine_mirror_sign(self) -> float:
    """Spine value of the left/right-mirrored robot: a yaw or roll bend flips, a pitch bend does not."""
    return {"yaw": -1.0, "roll": -1.0, "pitch": 1.0}[self.spine]

  @property
  def xml_path(self) -> Path:
    return self.asset_dir / self.xml

  @property
  def usd_path(self) -> Path:
    return self.asset_dir / self.usd

  @property
  def visual_usd_path(self) -> Path:
    return self.asset_dir / self.visual_usd


_RIGID = dict(
  asset_dir=_ASSETS / "rigid_microtaur", root_body="battery", nominal_root_height_m=0.07017281465145506,
  hip_z_root_m=(0.0095, 0.0095, 0.0095, 0.0095), foot_body_names=_RIGID_FEET, foot_offset_in_body_m=_RIGID_FOOT_OFFSETS,
  foot_extension_angle_rad=math.radians(4.31513),
)
# Yaw: front half = root (legs 3, 4, IMU 1); the rear half's motor axes sit 5.2 mm higher.
_YAW = dict(
  asset_dir=_ASSETS / "active_yaw_microtaur", root_body="s_microtaur_bottom_half_motor_mount",
  nominal_root_height_m=0.076492, hip_z_root_m=(0.0095, 0.0095, 0.0043, 0.0043), foot_body_names=_RIGID_FEET,
  foot_offset_in_body_m=_YAW_FOOT_OFFSETS, foot_extension_angle_rad=math.radians(5.40982), spine="yaw",
  other_half_body="s_microtaur_top_half_motor_mount", imu2_quat_in_other_half=(0.0, 0.0, 1.0, 0.0),
)

MODELS: dict[str, RobotModel] = {
  "rigid": RobotModel("rigid", xml="robot_modified.xml", usd="microtaur_rigid.usda",
                      visual_usd="microtaur_rigid_visual.usd", total_mass_kg=0.540, **_RIGID),
  "yaw": RobotModel("yaw", xml="robot_2imu.xml", usd="microtaur_yaw.usda", visual_usd="microtaur_yaw_visual.usd",
                    total_mass_kg=0.723783, **_YAW),
  "rigid_uno": RobotModel("rigid_uno", xml="robot_uno_battery.xml", usd="microtaur_rigid_uno.usda",
                          visual_usd="microtaur_rigid_uno_visual.usd", total_mass_kg=0.55200036, **_RIGID),
  "yaw_uno": RobotModel("yaw_uno", xml="robot_uno_battery.xml", usd="microtaur_yaw_uno.usda",
                        visual_usd="microtaur_yaw_uno_visual.usd", total_mass_kg=0.73878336, **_YAW),
  # Active roll (upstream "active twist"): spine axis = -x of the root; the ROOT is the rear half (legs 1, 2,
  # root IMU = the MJCF's imu2_site), the front half (legs 3, 4) carries the second IMU (the MJCF's imu_site) and
  # sits 11.5 mm higher. Legs as the yaw robot's (same parts, foot angle, offsets), foot bodies in another order.
  # Spine actuator assumed the yaw robot's XL330-M288 (same joint parameters in the MJCF; not measured).
  "roll_uno": RobotModel(
    "roll_uno", asset_dir=_ASSETS / "active_twist_microtaur", xml="robot_uno_battery.xml", usd="microtaur_roll_uno.usda",
    visual_usd="microtaur_roll_uno_visual.usd", root_body="twisting_microtaur_bottom_half_motor_mount",
    total_mass_kg=0.49384336, nominal_root_height_m=0.08079193565950653, hip_z_root_m=(0.0, 0.0, 0.0115, 0.0115),
    foot_body_names=("leglink2_v2_with_leg_down_3", "leglink2_v2_with_leg_down_4", "leglink2_v2_with_leg_down",
                     "leglink2_v2_with_leg_down_2"),
    foot_offset_in_body_m=_YAW_FOOT_OFFSETS, foot_extension_angle_rad=math.radians(5.40982), spine="roll",
    other_half_body="twisting_microtaur_top_half", imu2_quat_in_other_half=(0.707107, 0.0, 0.707107, 0.0)),
  # Active pitch, upstream c6d3dcd (2026-11-08 "fixed pitch motor attachment points": legs now assembled as the rigid
  # robot's, same foot geometry / offsets / foot angle), built by models_uno_imu_2026-11-09/build_uno_imu.py (UNO Q at
  # the yaw pose, IMU boards, battery kept, root frame moved to the hip centre, the legs 1 / 2 b-link inertials
  # copied from the same-side front leg, bodies / meshes renamed: USD prim names cannot start with a digit). ROOT = front half (legs 3, 4, root IMU = MJCF imu_site); the rear half
  # (legs 1, 2, UNO Q) carries the second IMU (MJCF imu2_site, aligned with the root at spine 0); spine axis +y of
  # the root (pitch), all four hips 11.5 mm above the root origin. Spine actuator assumed the yaw's XL330-M288.
  "pitch_uno": RobotModel(
    "pitch_uno", asset_dir=_ASSETS / "active_pitch_microtaur", xml="robot_uno_battery.xml", usd="microtaur_pitch_uno.usda",
    visual_usd="microtaur_pitch_uno_visual.usd", root_body="pitch_microtaur_top_half_1", total_mass_kg=0.50927592,
    nominal_root_height_m=0.06817292823489918, hip_z_root_m=(0.0115, 0.0115, 0.0115, 0.0115),
    foot_body_names=("leg1_d_link", "leg2_d_link", "leg3_d_link", "leg4_d_link"),
    foot_offset_in_body_m=((-0.0170594, -0.0811919, -0.0015), (-0.0170594, 0.0811919, -0.0015),
                           (0.0827296, 0.00624248, -0.0015), (0.0827296, -0.00624248, -0.0015)),
    foot_extension_angle_rad=math.radians(4.31513), spine="pitch",
    other_half_body="pitch_microtaur_rear_half", imu2_quat_in_other_half=(0.0, 0.0, 0.70710678, 0.70710678)),
}


def robot_model(variant: str) -> RobotModel:
  try:
    return MODELS[variant]
  except KeyError:
    raise ValueError(f"unknown robot variant {variant!r} (known: {', '.join(MODELS)})") from None
