"""Shared configuration for Microtaur test and terrain scripts.

This module keeps the open-loop gait reference parameters and terrain presets
used by diagnostic, viewer, and gait-test scripts in one place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg


# ---------------------------------------------------------------------
# Open-loop gait reference parameters
# ---------------------------------------------------------------------

GAIT_TROT = "trot"
GAIT_BOUND = "bound"

# Reference frequency used by the validated trot test.
GAIT_FREQ_HZ = 1.45

# Raw common/differential amplitudes used by the reference gait.
COMMON_RAW_AMP = 0.75 # Common-mode sweep amplitude shared across legs.
DIFF_RAW_AMP = 0.85
DIFF_RAW_BIAS = 0.00

# Sign convention used by the validated trot test.
COMMON_SIGN_FLIP = -1.0

# Per-leg common-mode signs measured in the actuator diagnostic.
COMMON_SIGN = np.array([+1.0, -1.0, -1.0, +1.0], dtype=np.float64)

RAMP_TIME_S = 2.0
REALTIME = True


# ---------------------------------------------------------------------
# Microtaur terrain scan sensor configuration
# ---------------------------------------------------------------------

# Use a local height map sized for Microtaur rather than the much larger
# default scan inherited from the base velocity task.
MICROTAUR_TERRAIN_SCAN_SIZE = (0.40, 0.30)
MICROTAUR_TERRAIN_SCAN_RESOLUTION = 0.05
MICROTAUR_TERRAIN_SCAN_MAX_DISTANCE = 0.30


def make_microtaur_terrain_scan_cfg(
  root_body_name: str,
  debug_vis: bool = True,
) -> RayCastSensorCfg:
  """Build the terrain height scanner used by velocity observations.

  The sensor name must remain ``terrain_scan`` because the inherited
  ``height_scan`` observation looks it up by name. Rays point downward and
  rotate with robot yaw only, so body roll and pitch do not tilt the scan grid.
  """
  return RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name=root_body_name, entity="robot"),
    pattern=GridPatternCfg(
      size=MICROTAUR_TERRAIN_SCAN_SIZE,
      resolution=MICROTAUR_TERRAIN_SCAN_RESOLUTION,
      direction=(0.0, 0.0, -1.0), # Cast rays downward in the world-gravity direction.
    ),
    ray_alignment="yaw", # Follow heading (z-axis rotation) without following roll or pitch.
    max_distance=MICROTAUR_TERRAIN_SCAN_MAX_DISTANCE,
    exclude_parent_body=True,
    include_geom_groups=(0,), # Query terrain geoms only; robot geoms are excluded.
    debug_vis=debug_vis,
  )


@dataclass(frozen=True)
class MicrotaurGaitCfg:
  gait_type: str = GAIT_TROT # Default reference gait.
  freq_hz: float = GAIT_FREQ_HZ
  common_raw_amp: float = COMMON_RAW_AMP
  diff_raw_amp: float = DIFF_RAW_AMP
  diff_raw_bias: float = DIFF_RAW_BIAS
  common_sign_flip: float = COMMON_SIGN_FLIP
  ramp_time_s: float = RAMP_TIME_S
  realtime: bool = REALTIME

# Reference gait generator retained for diagnostics; it is not used by the RL environment.
# def get_phase_offsets(gait_type: str) -> np.ndarray:
#   """Return per-leg phase offsets.

#   Leg order:
#     leg1, leg2, leg3, leg4

#   Trot:
#     leg1 + leg3 together
#     leg2 + leg4 together
#     [0, pi, 0, pi]

#   Bound:
#     leg1 + leg2 together
#     leg3 + leg4 together
#     [0, 0, pi, pi]
#   """
#   if gait_type == GAIT_TROT:
#     return np.array(
#       [
#         0.0,
#         math.pi,
#         0.0,
#         math.pi,
#       ],
#       dtype=np.float64,
#     )

#   if gait_type == GAIT_BOUND:
#     return np.array(
#       [
#         0.0,
#         0.0,
#         math.pi,
#         math.pi,
#       ],
#       dtype=np.float64,
#     )

#   raise ValueError(f"Unknown gait_type: {gait_type}")


# def smoothstep(x: float) -> float:
#   x = float(np.clip(x, 0.0, 1.0)) # clip x between 0 and 1
#   return x * x * (3.0 - 2.0 * x) # 3x^2 - 2x^3 : because x is between 0 and 1 x^2 > x^3 so this term will vary between 0 and 1


# def make_reference_raw_action(
#   t: float,
#   gait_cfg: MicrotaurGaitCfg = MicrotaurGaitCfg(),
# ) -> np.ndarray:
#   """Return raw 8D common/diff action.

#   Raw action order:
#     [
#       leg1_common, leg1_diff,
#       leg2_common, leg2_diff,
#       leg3_common, leg3_diff,
#       leg4_common, leg4_diff,
#     ]

#   Matches the corrected trot_test convention:

#     swing:
#       common_shape = -1 + 2*smoothstep(u)
#       diff_shape   = -sin(pi*u)

#     stance:
#       common_shape = 1 - 2*smoothstep(u)
#       diff_shape   = +sin(pi*u)
#   """
#   phase_offsets = get_phase_offsets(gait_cfg.gait_type) #trot is [0,pi,0,pi] and bound is [0,0,pi,pi] 
#   base_phase = 2.0 * math.pi * gait_cfg.freq_hz * t # 2*pi*w*t - this is an angle in the cpg pahse diagram

#   raw = np.zeros(8, dtype=np.float64)

#   for leg_idx in range(4):
#     phi = (base_phase + phase_offsets[leg_idx]) % (2.0 * math.pi)

#     if phi < math.pi:
#       u = phi / math.pi
#       s = smoothstep(u)

#       common_shape = -1.0 + 2.0 * s
#       diff_shape = -math.sin(math.pi * u)
#     else:
#       u = (phi - math.pi) / math.pi
#       s = smoothstep(u)

#       common_shape = 1.0 - 2.0 * s
#       diff_shape = math.sin(math.pi * u)

#     common_raw = (
#       gait_cfg.common_sign_flip
#       * gait_cfg.common_raw_amp
#       * common_shape
#     )
#     diff_raw = gait_cfg.diff_raw_bias + gait_cfg.diff_raw_amp * diff_shape

#     raw[2 * leg_idx] = float(np.clip(common_raw, -1.0, 1.0))
#     raw[2 * leg_idx + 1] = float(np.clip(diff_raw, -1.0, 1.0))

#   return raw


# ---------------------------------------------------------------------
# Microtaur-safe terrain configuration
# ---------------------------------------------------------------------

# The base MJLab rough-terrain preset is too large for this robot. Start with
# millimeter-scale obstacles and increase difficulty only after flat locomotion
# is stable.
MICRO_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(4.0, 4.0),
  border_width=8.0,
  num_rows=4,
  num_cols=8,
  difficulty_range=(0.0, 1.0),
  sub_terrains={
    # Keep a large flat fraction so early rough-terrain fine-tuning remains recoverable.
    "flat": terrain_gen.BoxFlatTerrainCfg(
      proportion=0.40,
    ),

    # Small stair obstacles.
    "tiny_pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.20,
      step_height_range=(0.005, 0.01),
      step_width=0.20,
      platform_width=1.0,
      border_width=0.25,
    ),

    "tiny_pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.10,
      step_height_range=(0.00, 0.015),
      step_width=0.20,
      platform_width=1.0,
      border_width=0.25,
    ),

    # Millimeter-scale random heightfield.
    "micro_random_rough": terrain_gen.HfRandomUniformTerrainCfg(
      proportion=0.20,
      noise_range=(-0.008, 0.008),
      noise_step=0.004,
      horizontal_scale=0.15,
      vertical_scale=0.001,
      border_width=0.25,
    ),

    # Low-amplitude wave terrain.
    "micro_wave": terrain_gen.HfWaveTerrainCfg(
      proportion=0.10,
      amplitude_range=(0.0, 0.009),
      num_waves=5,
      horizontal_scale=0.15,
      vertical_scale=0.001,
      border_width=0.25,
    ),
  },
  add_lights=True,
)


# Medium preset for the next rough-terrain fine-tuning stage.
MICRO_ROUGH_TERRAINS_CFG_MEDIUM = TerrainGeneratorCfg(
  size=(4.0, 4.0),
  border_width=8.0,
  num_rows=4,
  num_cols=8,
  difficulty_range=(0.0, 1.0),
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(
      proportion=0.25,
    ),

    "small_pyramid_stairs": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.25,
      step_height_range=(0.01, 0.025),
      step_width=0.20,
      platform_width=1.0,
      border_width=0.25,
    ),

    "small_pyramid_stairs_inv": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.10,
      step_height_range=(0.01, 0.025),
      step_width=0.20,
      platform_width=1.0,
      border_width=0.25,
    ),

    "small_random_rough": terrain_gen.HfRandomUniformTerrainCfg(
      proportion=0.25,
      noise_range=(-0.02, 0.02),
      noise_step=0.004,
      horizontal_scale=0.15,
      vertical_scale=0.001,
      border_width=0.25,
    ),

    "small_wave": terrain_gen.HfWaveTerrainCfg(
      proportion=0.15,
      amplitude_range=(0.0, 0.025),
      num_waves=3,
      horizontal_scale=0.15,
      vertical_scale=0.001,
      border_width=0.25,
    ),
  },
  add_lights=True,
)