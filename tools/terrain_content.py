"""Measure the geometric content of a heightfield, per spine axis.

Pure numpy. This is the substrate for the study's central analysis: instead of
building one terrain per spine axis and reporting the diagonal, generate random
terrain, **measure** what each map actually demands, and fit

    delta_performance(spined - rigid) ~ b_roll*twist + b_pitch*curvature
                                        + b_path*path + eps

so that axis specificity is a fitted coefficient rather than a property of a
course designed to produce it.

The three regressors are all "differential rotation per body length", because
that is what a trunk joint about a given axis actually buys. The reasoning, for
the roll case, is worth stating because it is the thing a hand-designed probe
gets wrong:

  A sustained left-high / right-low surface is a *static* roll demand. A rigid
  trunk satisfies it by rolling as a whole, or by extending the legs on one
  side further than the other. A roll spine only pays when the front and rear
  pairs need **different** roll angles at the same instant. So the quantity to
  measure is not the lateral height difference, it is how much that difference
  *changes* between the front and rear axles.

The same argument applies to pitch: a constant grade needs no differential
pitch, a *change* of grade does. Hence both regressors are second differences.

Sampling model: the robot at (x, y) stands with feet at
(x +/- L/2, y +/- W/2) using the measured footprint, L = 0.170 m,
W = 0.105 m, and travels along +x.
"""

from __future__ import annotations

import numpy as np

from terrain_geometry import (
  BODY_LENGTH_M,
  BODY_WIDTH_M,
  LEG_VERTICAL_TRAVEL_M,
  _footprint_range,
)

REGRESSORS = ("twist", "curvature", "path")


def content(z: np.ndarray, dx: float) -> dict[str, float]:
  """Return RMS twist / curvature / path demand for one heightfield.

  All three are in radians of differential rotation per body length, except
  ``path``, which is a dimensionless proxy (see below).
  """
  nx, ny = z.shape
  il = max(int(round((BODY_LENGTH_M / 2.0) / dx)), 1)
  iw = max(int(round((BODY_WIDTH_M / 2.0) / dx)), 1)
  n_x = nx - 2 * il
  n_y = ny - 2 * iw
  if n_x <= 0 or n_y <= 0:
    raise ValueError(
      f"grid {z.shape} at dx={dx} is smaller than the robot footprint "
      f"({BODY_LENGTH_M} x {BODY_WIDTH_M} m)"
    )

  def at(dxi: int, dyi: int) -> np.ndarray:
    return z[il + dxi : il + dxi + n_x, iw + dyi : iw + dyi + n_y]

  z_fl, z_fr = at(+il, +iw), at(+il, -iw)
  z_ml, z_mr = at(0, +iw), at(0, -iw)
  z_rl, z_rr = at(-il, +iw), at(-il, -iw)

  # --- roll: lateral slope at the front axle minus at the rear axle ---------
  roll_front = (z_fl - z_fr) / BODY_WIDTH_M
  roll_rear = (z_rl - z_rr) / BODY_WIDTH_M
  twist = roll_front - roll_rear

  # --- pitch: fore-aft slope of the front half minus of the rear half ------
  front = 0.5 * (z_fl + z_fr)
  mid = 0.5 * (z_ml + z_mr)
  rear = 0.5 * (z_rl + z_rr)
  half = BODY_LENGTH_M / 2.0
  curvature = (front - mid) / half - (mid - rear) / half

  # --- path: lateral gradient of traversal difficulty ----------------------
  # A proxy, and deliberately a weak one. A heightfield on open ground creates
  # almost no *path* curvature demand: nothing forces the robot to turn, so it
  # can go straight. Real yaw demand needs walls, pillars or gaps, which are not
  # heightfield features. This regressor is included so the fit has a column for
  # it and so its near-zero value is visible rather than assumed; do not read a
  # large b_path off open terrain.
  cost = _footprint_range(z, dx) / LEG_VERTICAL_TRAVEL_M
  dcost_dy = np.gradient(cost, dx, axis=1)
  path = BODY_LENGTH_M * dcost_dy

  rms = lambda a: float(np.sqrt((np.asarray(a, dtype=np.float64) ** 2).mean()))
  return {"twist": rms(twist), "curvature": rms(curvature), "path": rms(path)}


def identifiability(
  rows: list[dict[str, float]], keys: tuple[str, ...] = REGRESSORS
) -> dict:
  """Check whether the regressors vary independently across a set of maps.

  **This is a gate, not a diagnostic.** If twist and curvature content are
  nearly collinear across the evaluation set, the per-axis coefficients are not
  identifiable and the regression will report confident nonsense. It fails
  silently: a set of isotropic, same-scale random maps looks varied and has
  almost perfectly correlated regressors.

  Returns the correlation matrix, the largest absolute off-diagonal
  correlation, and a pass/fail against a 0.8 threshold.
  """
  if len(rows) < 3:
    raise ValueError("need at least 3 maps to estimate a correlation matrix")
  m = np.array([[r[k] for k in keys] for r in rows], dtype=np.float64)
  sd = m.std(axis=0)
  if np.any(sd < 1e-12):
    dead = [k for k, s in zip(keys, sd) if s < 1e-12]
    return {
      "corr": None,
      "max_abs_offdiag": float("nan"),
      "ok": False,
      "reason": f"no variation across maps in: {dead}",
    }
  corr = np.corrcoef(m, rowvar=False)
  off = corr - np.eye(len(keys))
  worst = float(np.abs(off).max())
  return {
    "keys": keys,
    "corr": corr,
    "max_abs_offdiag": worst,
    "ok": bool(worst < 0.8),
    "reason": "ok" if worst < 0.8 else f"max |off-diagonal| = {worst:.3f} >= 0.8",
  }


if __name__ == "__main__":
  import terrain_geometry as tg

  NX, NY, DX = 192, 96, 0.01

  print("=== single families, difficulty 1.0 ===")
  print(f"{'family':38s} {'twist':>10s} {'curvature':>10s} {'path':>10s}")
  cases = [
    ("flat", tg.flat(NX, NY)),
    (
      "band_limited orient=0 (across path)",
      tg.band_limited_random(
        NX, NY, DX, difficulty=1.0, wavelength_m=0.34, orientation_deg=0.0,
        rng=np.random.default_rng(0),
      ),
    ),
    (
      "band_limited orient=90 (along path)",
      tg.band_limited_random(
        NX, NY, DX, difficulty=1.0, wavelength_m=0.34, orientation_deg=90.0,
        rng=np.random.default_rng(0),
      ),
    ),
    ("alternating_lateral_steps", tg.alternating_lateral_steps(NX, NY, DX, difficulty=1.0)),
  ]
  for name, z in cases:
    c = content(z, DX)
    print(f"{name:38s} {c['twist']:10.4f} {c['curvature']:10.4f} {c['path']:10.4f}")

  print()
  print("=== identifiability of a 24-map evaluation set ===")
  maps = tg.make_map_set(24, NX, NY, DX, difficulty=0.7, seed=7)
  rows = [content(m["z"], DX) for m in maps]

  def report(keys):
    ident = identifiability(rows, keys)
    if ident["corr"] is None:
      print(f"  {ident['reason']}")
      return ident
    print(f"  {'':12s}" + "".join(f"{k:>12s}" for k in keys))
    for i, k in enumerate(keys):
      print(f"  {k:12s}" + "".join(f"{ident['corr'][i, j]:12.3f}" for j in range(len(keys))))
    print(f"  max |off-diagonal| = {ident['max_abs_offdiag']:.3f}   "
          f"{'PASS' if ident['ok'] else 'FAIL'}")
    return ident

  print("\nall three regressors:")
  report(REGRESSORS)
  print("\ntwist + curvature only (the pair the study can actually fit):")
  report(("twist", "curvature"))
  print("""
Reading: `path` comes out strongly anti-correlated with `curvature`, because on
an open heightfield both are essentially functions of ridge orientation -- when
the ridges run along the path the lateral cost gradient is large and the fore-aft
curvature is small, and vice versa. So **a heightfield cannot supply an
independent path/yaw regressor**, and the regression should be fitted on
(twist, curvature) only.

Getting a real yaw regressor needs features a heightfield does not have -- walls,
pillars, gaps -- which is a separate terrain family and also needs a different
command protocol, since the robot has to be commanded along a turning
trajectory rather than straight ahead.""")
