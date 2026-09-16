"""Shared robot constants and XML loading helpers for Microtaur.

The module defines joint/site ordering, nominal stand angles, actuator defaults,
and compatibility constants used by the older common/differential action
interface. For that interface, each leg maps ``common`` and ``diff`` into its
paired ``a``/``e`` motors using the per-leg sign convention
``(+1, -1, -1, +1)``.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]

MORPHOLOGY_ROOT = Path(
  os.environ.get("MICROTAUR_MORPHOLOGY_ROOT", _PROJECT_ROOT / "microtaur_xmls")
)
MICROTAUR_VARIANT = os.environ.get("MICROTAUR_VARIANT", "rigid_microtaur")
#MICROTAUR_VARIANT = os.environ.get("MICROTAUR_VARIANT", "active_twist_microtaur")

_DEFAULT_XML = MORPHOLOGY_ROOT / MICROTAUR_VARIANT / "robot_modified.xml"
if not _DEFAULT_XML.exists():
  _DEFAULT_XML = MORPHOLOGY_ROOT / MICROTAUR_VARIANT / "scene.xml"

MICROTAUR_XML = Path(os.environ.get("MICROTAUR_XML", _DEFAULT_XML))
assert MICROTAUR_XML.exists(), f"Missing Microtaur XML: {MICROTAUR_XML}"


def _read_model_name(xml_path: Path) -> str:
  root = ET.parse(xml_path).getroot()
  return root.attrib.get("model", "")


MODEL_NAME = _read_model_name(MICROTAUR_XML)

CONTROL_SPINE = os.environ.get("MICROTAUR_CONTROL_SPINE", "0").lower() not in {
  "0",
  "false",
  "no",
}

# Joint order is part of the action interface: (a, e) for legs 1 through 4.
LEG_JOINT_NAMES = (
  "leg1_a_joint_act",
  "leg1_e_joint_act",
  "leg2_a_joint_act",
  "leg2_e_joint_act",
  "leg3_a_joint_act",
  "leg3_e_joint_act",
  "leg4_a_joint_act",
  "leg4_e_joint_act",
)

SPINE_JOINT_NAMES = ("spine_joint_act",)

HAS_ACTIVE_SPINE = MODEL_NAME == "active_twist_microtaur"

if HAS_ACTIVE_SPINE:
  ROOT_BODY = "twisting_microtaur_bottom_half_motor_mount"
  BODY_ANG_VEL_BODY = ROOT_BODY
  ACTION_JOINT_NAMES = LEG_JOINT_NAMES + (SPINE_JOINT_NAMES if CONTROL_SPINE else ())
  OBS_JOINT_NAMES = LEG_JOINT_NAMES + SPINE_JOINT_NAMES
  POSE_JOINT_NAMES = LEG_JOINT_NAMES + SPINE_JOINT_NAMES
else:
  ROOT_BODY = "battery"
  BODY_ANG_VEL_BODY = "battery"
  ACTION_JOINT_NAMES = LEG_JOINT_NAMES
  OBS_JOINT_NAMES = LEG_JOINT_NAMES
  POSE_JOINT_NAMES = LEG_JOINT_NAMES


FOOT_SITE_NAMES = (
  "leg1_foot_site",
  "leg2_foot_site",
  "leg3_foot_site",
  "leg4_foot_site",
)

FOOT_BODY_NAMES = (
  "leglink2_v2_with_leg_down",
  "leglink2_v2_with_leg_down_2",
  "leglink2_v2_with_leg_down_3",
  "leglink2_v2_with_leg_down_4",
)

FOOT_GEOM_NAMES = (
  "leg1_foot_collision",
  "leg2_foot_collision",
  "leg3_foot_collision",
  "leg4_foot_collision",
)


# ---------------------------------------------------------------------------
# Stand pose
# ---------------------------------------------------------------------------

# Nominal stand angles for the rigid Microtaur model.
MICROTAUR_STAND_A_OFFSETS = tuple(
  float(x)
  for x in os.environ.get(
    "MICROTAUR_STAND_A_OFFSETS",
    "0.45,-0.45,-0.45,0.45",
  ).split(",")
)

MICROTAUR_STAND_E_OFFSETS = tuple(
  float(x)
  for x in os.environ.get(
    "MICROTAUR_STAND_E_OFFSETS",
    "-0.45,0.45,0.45,-0.45",
  ).split(",")
)


# ---------------------------------------------------------------------------
# Common/differential action signs
# ---------------------------------------------------------------------------

# The actuator diagnostic showed that the mirrored legs need opposite signs in
# common mode: (+1, -1, -1, +1). Keep the historical
# MINITAUR_SWING_SIGNS name because existing configs still import it.
MINITAUR_SWING_SIGNS = tuple(
  float(x)
  for x in os.environ.get(
    "MICROTAUR_SWING_SIGNS",
    "1,-1,-1,1",
  ).split(",")
)

# Legacy extension signs retained for older scripts; the current paired action
# transform does not consume them directly.
MINITAUR_EXTENSION_SIGNS = tuple(
  float(x)
  for x in os.environ.get(
    "MICROTAUR_EXTENSION_SIGNS",
    "-1,1,1,-1",
  ).split(",")
)


# ---------------------------------------------------------------------------
# Legacy per-joint action-scale map
# ---------------------------------------------------------------------------

MICROTAUR_ACTION_SCALE: dict[str, float] = {
  "leg1_a_joint_act": 1.0,
  "leg1_e_joint_act": 1.0,
  "leg2_a_joint_act": 1.0,
  "leg2_e_joint_act": 1.0,
  "leg3_a_joint_act": 1.0,
  "leg3_e_joint_act": 1.0,
  "leg4_a_joint_act": 1.0,
  "leg4_e_joint_act": 1.0,
}

if HAS_ACTIVE_SPINE and CONTROL_SPINE:
  MICROTAUR_ACTION_SCALE["spine_joint_act"] = 1.0


MICROTAUR_STAND_JOINT_POS = {
  "leg1_a_joint_act": MICROTAUR_STAND_A_OFFSETS[0],
  "leg1_e_joint_act": MICROTAUR_STAND_E_OFFSETS[0],
  "leg2_a_joint_act": MICROTAUR_STAND_A_OFFSETS[1],
  "leg2_e_joint_act": MICROTAUR_STAND_E_OFFSETS[1],
  "leg3_a_joint_act": MICROTAUR_STAND_A_OFFSETS[2],
  "leg3_e_joint_act": MICROTAUR_STAND_E_OFFSETS[2],
  "leg4_a_joint_act": MICROTAUR_STAND_A_OFFSETS[3],
  "leg4_e_joint_act": MICROTAUR_STAND_E_OFFSETS[3],
}

if HAS_ACTIVE_SPINE:
  MICROTAUR_STAND_JOINT_POS["spine_joint_act"] = 0.0


# ---------------------------------------------------------------------------
# Actuator settings
# ---------------------------------------------------------------------------

# Default actuator values for get_microtaur_robot_cfg(). Environment-specific
# configs may replace these with a more detailed motor model.
EFFORT_LIMIT = float(os.environ.get("MICROTAUR_EFFORT_LIMIT", "0.25"))
STIFFNESS = float(os.environ.get("MICROTAUR_STIFFNESS", "1.0"))
DAMPING = float(os.environ.get("MICROTAUR_DAMPING", "0.045"))
ARMATURE = float(os.environ.get("MICROTAUR_ARMATURE", "0.0002"))

MICROTAUR_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=ACTION_JOINT_NAMES,
  stiffness=STIFFNESS,
  damping=DAMPING,
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE,
)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, float(os.environ.get("MICROTAUR_INIT_Z", "0.10"))),
  joint_pos={
    joint_name: MICROTAUR_STAND_JOINT_POS.get(joint_name, 0.0)
    for joint_name in OBS_JOINT_NAMES
  },
  joint_vel={".*": 0.0},
)

MICROTAUR_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(MICROTAUR_ACTUATOR_CFG,),
  soft_joint_pos_limit_factor=0.95,
)


# ---------------------------------------------------------------------------
# Legacy common/differential action scales
# ---------------------------------------------------------------------------

# Paired-action convention:
#   raw[0::2] -> common/sweep coordinate
#   raw[1::2] -> differential/shape coordinate
# Keep both scales conservative because common mode has the larger mechanical
# effect on the five-bar leg.
MINITAUR_PAIR_SWING_SCALE = float(
  os.environ.get("MICROTAUR_PAIR_SWING_SCALE", "0.45")
)

MINITAUR_PAIR_EXTENSION_SCALE = float(
  os.environ.get("MICROTAUR_PAIR_EXTENSION_SCALE", "0.45")
)

MINITAUR_PAIR_SWING_OFFSET = float(
  os.environ.get("MICROTAUR_PAIR_SWING_OFFSET", "0.0")
)

MINITAUR_PAIR_EXTENSION_OFFSET = float(
  os.environ.get("MICROTAUR_PAIR_EXTENSION_OFFSET", "0.0")
)


# ---------------------------------------------------------------------------
# MuJoCo model and asset loading
# ---------------------------------------------------------------------------

def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  search_roots: list[Path] = []

  if meshdir:
    search_roots.append(MICROTAUR_XML.parent / meshdir)

  search_roots.append(MICROTAUR_XML.parent)

  for root in search_roots:
    if not root.exists():
      continue

    for path in root.rglob("*"):
      if not path.is_file():
        continue

      rel_key = path.relative_to(MICROTAUR_XML.parent).as_posix()
      data = path.read_bytes()

      assets[rel_key] = data
      assets[path.name] = data

  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(MICROTAUR_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


def get_microtaur_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(),
    spec_fn=get_spec,
    articulation=MICROTAUR_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_microtaur_robot_cfg())
  viewer.launch(robot.spec.compile())