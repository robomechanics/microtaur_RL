"""Heightfields for the teacher terrains (pure numpy; no simulator).

  A  flat.
  B  one FIXED map of flat-topped square cells whose heights are Gaussian,
     clipped to the ceiling. Variety comes from spawning at random positions
     and headings on it, not from regenerating the map.
  C  low flat 0.5 m -> left half raised for 4 m (right half stays low) ->
     low flat 0.5 m, along +x. Straight-line commands only.

Difficulty d in [0, 1] scales the heights of the same pattern linearly, so a
terrain level is "the same map, taller".

Ceiling: 50% of the leg's vertical travel in the stance pose, +10.8 mm up and
-7.6 mm down (see UP_MAX_M / DOWN_MAX_M). Numbers come from the 2026-09 rigid RL
gait (0.465 kg model; the leg kinematics are the same), so recheck them once
the new policy's stance is known.

Axes: x (axis 0) is the direction of travel, y (axis 1) lateral (+y = left),
heights in metres. B is zero-mean; C starts and ends on z = 0.
"""

from __future__ import annotations

import numpy as np

# Heightfield sample spacing. A heightfield interpolates linearly between
# samples, so every cell edge is a ramp one sample wide. Adjacent cells differ
# by only 6.3 mm at the median, so at 10 mm spacing the median edge was a 32 deg
# slope (71% of edges gentler than 45 deg). At 2.5 mm the median edge is 68 deg
# and the ramp is narrower than the 6.2 mm foot radius.
DX_M = 0.0025
CELL_M = 0.070
# Ceiling: 50% of the leg's vertical travel in the stance pose, not the stand
# pose. The recorded rigid gait keeps the feet behind the hip; the tightest
# point of stance is liftoff (common +0.38), where retract is 21.6 mm and
# extend 15.2 mm. (At the stand pose they are 26.2 / 32.9 mm.)
UP_MAX_M = 0.0108  # 50% of 21.6 mm retract at liftoff
DOWN_MAX_M = 0.0076  # 50% of 15.2 mm extend at liftoff
SIGMA_M = DOWN_MAX_M / 2.0  # clip at 2 sigma on the tighter (down) side


def _zero_mean(z: np.ndarray) -> np.ndarray:
  return (z - z.mean()).astype(np.float32)


def flat(nx: int, ny: int) -> np.ndarray:
  return np.zeros((nx, ny), dtype=np.float32)


def gaussian_blocks(nx: int, ny: int, *, difficulty: float, seed: int = 0, dx: float = DX_M,
                    cell_m: float = CELL_M, sigma_m: float = SIGMA_M,
                    up_max_m: float = UP_MAX_M, down_max_m: float = DOWN_MAX_M) -> np.ndarray:
  """Terrain B. The cell pattern depends only on `seed`; `difficulty` scales it."""
  c = max(int(round(cell_m / dx)), 1)
  gx, gy = -(-nx // c), -(-ny // c)
  h = np.random.default_rng(seed).normal(0.0, sigma_m, size=(gx, gy))
  h = np.clip(h, -down_max_m, up_max_m)
  z = np.kron(h, np.ones((c, c)))[:nx, :ny]
  return _zero_mean(float(difficulty) * z)


def flat_step_flat(nx: int, ny: int, *, difficulty: float, dx: float = DX_M,
                   flat_before_m: float = 0.5, step_len_m: float = 4.0,
                   delta_m: float = UP_MAX_M + DOWN_MAX_M) -> np.ndarray:
  """Terrain C. Low flat approach, left half raised for most of the course, low flat exit.

  Everything is at z = 0 except the left half (y > 0) of the step section,
  which is raised by difficulty * delta_m. At 100% delta = 18.4 mm: with the
  body lowered, the left legs retract ~10.8 mm and the right legs extend
  ~7.6 mm, i.e. 50% of each at the tightest point of stance. The step section is most of the
  course (4 m of 5 m), because the sustained offset is what C is for.

  Not zero-meaned: approach and exit must stay on the base level, and a course
  that starts and ends at the same height has no net grade.
  """
  z = np.zeros((nx, ny), dtype=np.float32)
  i0 = int(round(flat_before_m / dx))
  i1 = min(nx, i0 + int(round(step_len_m / dx)))
  y = np.arange(ny) - (ny - 1) / 2.0
  z[i0:i1, y > 0] = float(difficulty) * delta_m
  return z
