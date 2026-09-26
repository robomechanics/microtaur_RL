"""Heightfields for the teacher terrains (pure numpy; no simulator).

  A  flat.
  B  one FIXED map of flat-topped square cells whose heights are Gaussian,
     clipped to the ceiling. Variety comes from spawning at random positions
     and headings on it, not from regenerating the map.
  C  flat -> sustained lateral step (left high, right low) -> flat, along +x.
     Straight-line commands only.

Difficulty d in [0, 1] scales the heights of the same pattern linearly, so a
terrain level is "the same map, taller".

Ceiling: 50% of the leg's vertical travel in each direction, +13.1 mm up and
-16.4 mm down, measured at the stand pose (common = 0). ⚠ The recorded rigid
gait keeps the feet behind the hip (common +0.26 at touchdown, +0.38 at
liftoff), where the travel is smaller -- 50% there is +10.8 / -7.6 mm -- so the
top levels are harder than their stand-pose percentage suggests.

Axes: x (axis 0) is the direction of travel, y (axis 1) lateral, heights in
metres, every field zero-mean.
"""

from __future__ import annotations

import numpy as np

DX_M = 0.010  # heightfield sample spacing: 7 samples per 70 mm cell -> flat tops
CELL_M = 0.070
UP_MAX_M = 0.0131  # 50% of retract (26.22 mm) at the stand pose
DOWN_MAX_M = 0.0164  # 50% of extend (32.86 mm) at the stand pose
SIGMA_M = UP_MAX_M / 2.0  # clip at 2 sigma on the tighter (up) side


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
                   flat_before_m: float = 1.0, step_len_m: float = 2.0,
                   up_m: float = UP_MAX_M, down_m: float = DOWN_MAX_M) -> np.ndarray:
  """Terrain C. Flat approach, then left half +up / right half -down, then flat.

  Seam along the centreline y = 0. Not zero-meaned: the two halves already
  cancel to within (up - down) * step fraction, and removing a mean would lift
  the flat approach off z = 0.
  """
  z = np.zeros((nx, ny), dtype=np.float32)
  i0 = int(round(flat_before_m / dx))
  i1 = min(nx, i0 + int(round(step_len_m / dx)))
  y = np.arange(ny) - (ny - 1) / 2.0
  z[i0:i1, y > 0] = float(difficulty) * up_m
  z[i0:i1, y < 0] = -float(difficulty) * down_m
  return z
