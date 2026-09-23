"""Heightfield generators for the Microtaur spine study.

Pure numpy. No simulator, no framework. A generator here returns a float32
heightfield in metres, sampled on a regular grid with spacing `dx`, with the
robot's direction of travel along **+x** (axis 0) and lateral offset along
**y** (axis 1).

Wrapping these for a simulator is a separate, small job: mjlab wants an
`HfRandomUniformTerrainCfg`-style callable, IsaacLab wants a function returning
a trimesh registered through a `SubTerrainBaseCfg`. Keeping the geometry here
means that wrapper is the only framework-specific part, and that the terrain a
paper figure refers to is reproducible from a seed without either simulator
installed.

Two design constraints come from the study and are enforced, not optional:

1. **Zero mean.** A net grade adds potential-energy change to the numerator of
   cost of transport, so CoT would stop measuring dissipation. It also shifts
   the nominal posture that the `body_height`, `standing_pose` and `upright`
   reward terms are written against. Every generator here returns a field whose
   mean is subtracted off.

2. **Difficulty is one scalar.** Each generator takes a single `difficulty` in
   [0, 1] which interpolates its amplitude, so the terrain-level curriculum has
   something monotone to walk along and so results can be plotted against it.

Robot scale, for choosing the numbers (measured, see the 2026-09-23 memo):
fore-aft foot spacing 0.170 m, lateral 0.105 m, nominal root height 0.07017 m,
and the run terminates if the root drops below 0.05217 m. Features therefore
belong in the millimetre-to-centimetre range; a 15 mm step is already 21% of
standing height.
"""

from __future__ import annotations

import numpy as np

# Measured robot geometry (metres).
BODY_LENGTH_M = 0.170
BODY_WIDTH_M = 0.105
NOMINAL_ROOT_HEIGHT_M = 0.07017281465145506
MIN_ROOT_HEIGHT_M = NOMINAL_ROOT_HEIGHT_M - 0.018

# Usable single-joint travel is +/- 0.75 rad about the stand pose, which at this
# link geometry is worth roughly 25 mm of foot height travel per leg. Terrain
# amplitude is expressed as a fraction of that so the difficulty scale means
# something mechanical rather than just "millimetres".
LEG_VERTICAL_TRAVEL_M = 0.025


def _zero_mean(z: np.ndarray) -> np.ndarray:
  return (z - float(z.mean())).astype(np.float32)


def _footprint_range(z: np.ndarray, dx: float) -> np.ndarray:
  """Height range within a body-footprint window, at every position.

  This is the mechanically meaningful amplitude: it is the height difference the
  four legs have to absorb while standing on that patch, which is what
  ``LEG_VERTICAL_TRAVEL_M`` bounds. Global RMS is the wrong scale to normalise a
  random field by — a band-limited Gaussian field has peak-to-peak around 7x its
  RMS, so normalising RMS to the leg travel produces features several times the
  robot's standing height.

  Separable min/max filters, so memory stays O(nx * ny * window) rather than
  O(nx * ny * window^2).
  """
  from numpy.lib.stride_tricks import sliding_window_view

  n0 = int(np.clip(round(BODY_LENGTH_M / dx), 2, z.shape[0]))
  n1 = int(np.clip(round(BODY_WIDTH_M / dx), 2, z.shape[1]))

  a = sliding_window_view(z, n0, axis=0)
  a_max, a_min = a.max(axis=-1), a.min(axis=-1)
  b_max = sliding_window_view(a_max, n1, axis=1).max(axis=-1)
  b_min = sliding_window_view(a_min, n1, axis=1).min(axis=-1)
  return b_max - b_min


def flat(nx: int, ny: int) -> np.ndarray:
  """The reference terrain. Present so that every family has the same signature."""
  return np.zeros((nx, ny), dtype=np.float32)


def band_limited_random(
  nx: int,
  ny: int,
  dx: float,
  *,
  difficulty: float,
  wavelength_m: float,
  orientation_deg: float,
  bandwidth: float = 0.5,
  anisotropy: float = 0.25,
  amplitude_frac: float = 1.0,
  rng: np.random.Generator | None = None,
) -> np.ndarray:
  """Anisotropic band-limited random field — the general-capability terrain.

  White noise is filtered in the frequency domain by a Gaussian bump centred on
  wavenumber ``1 / wavelength_m`` and oriented at ``orientation_deg`` from the
  direction of travel, then rescaled to the target RMS.

  The two knobs exist for a specific reason. ``orientation_deg`` controls
  whether the ridges run across the path or along it, and therefore whether the
  field demands differential *pitch* (ridges across travel, 0 deg) or
  differential *roll* (ridges along travel, 90 deg); ``wavelength_m`` controls
  the spatial scale relative to the body. Randomising both across a map set is
  what **decorrelates** the geometric regressors so that
  ``terrain_content.py``'s per-axis coefficients are identifiable. A set of
  isotropic same-scale maps looks varied and is useless for that: twist and
  curvature content come out almost perfectly collinear.

  Args:
    dx: grid spacing, metres.
    difficulty: 0..1, scales RMS amplitude linearly from 0.
    wavelength_m: centre wavelength of the pass band.
    orientation_deg: ridge normal, measured from the direction of travel.
      Measured twist / curvature ratio at wavelength 0.34 m, from
      ``terrain_content.content`` (see that module's __main__):

        0 deg  twist 0.033  curvature 0.148   ratio 0.22   (pure pitch demand)
       45 deg  twist 0.117  curvature 0.060   ratio 1.96
       60 deg  twist 0.117  curvature 0.028   ratio 4.16
       90 deg  twist 0.069  curvature 0.008   ratio 8.87

      Note that **twist peaks at oblique orientations, around 45-60 deg, not at
      90 deg.** Ridges running exactly along the path give a lateral profile
      that is identical at every x, so the front and rear axles see the same
      roll demand and the differential is zero -- that is the "sustained
      left-high / right-low" case, which a rigid trunk handles by rolling as a
      whole. It is nonzero here only because the finite tangential bandwidth
      lets the ridges meander. The terrain that demands a roll spine is
      *oblique*, not lateral.
    bandwidth: relative width of the Gaussian pass band.
    amplitude_frac: at difficulty 1.0, the 95th-percentile height range inside
      a body footprint, as a fraction of ``LEG_VERTICAL_TRAVEL_M``. So
      ``difficulty=1.0, amplitude_frac=1.0`` means "on the hardest 5% of the
      map, the legs are at the edge of their travel".
  """
  rng = np.random.default_rng() if rng is None else rng
  noise = rng.standard_normal((nx, ny))

  fx = np.fft.fftfreq(nx, d=dx)[:, None]
  fy = np.fft.fftfreq(ny, d=dx)[None, :]
  theta = np.deg2rad(orientation_deg)
  # Decompose into the ridge normal (the surface varies along this) and the
  # ridge tangent (the surface should be coherent along this).
  f_normal = fx * np.cos(theta) + fy * np.sin(theta)
  f_tangent = -fx * np.sin(theta) + fy * np.cos(theta)

  f0 = 1.0 / float(wavelength_m)
  sigma = max(bandwidth * f0, 1e-9)
  # Band pass along the normal selects the ridge spacing. Low pass along the
  # tangent is what actually makes the field anisotropic: without it the filter
  # constrains only one projection and admits arbitrary perpendicular content,
  # so orientation has almost no effect on the measured twist/curvature split.
  sigma_t = max(float(anisotropy) * f0, 1e-9)
  gain = np.exp(-0.5 * ((np.abs(f_normal) - f0) / sigma) ** 2) * np.exp(
    -0.5 * (f_tangent / sigma_t) ** 2
  )

  field = np.real(np.fft.ifft2(np.fft.fft2(noise) * gain))

  # Normalise on the height range inside a body footprint, not on global RMS.
  ref = float(np.percentile(_footprint_range(field, dx), 95))
  if ref < 1e-12:
    return flat(nx, ny)
  target = float(difficulty) * float(amplitude_frac) * LEG_VERTICAL_TRAVEL_M
  return _zero_mean(field * (target / ref))


def alternating_lateral_steps(
  nx: int,
  ny: int,
  dx: float,
  *,
  difficulty: float,
  wavelength_m: float = 2 * BODY_LENGTH_M,
  approach_m: float = BODY_LENGTH_M,
  amplitude_frac: float = 0.6,
) -> np.ndarray:
  """Alternating lateral steps — the transition-dense terrain.

  One side of the path is raised and the other lowered, and the sign flips every
  ``wavelength_m`` along the direction of travel. Flat for ``approach_m`` at
  each end.

  **Why alternating rather than one long step.** A sustained left-high /
  right-low surface is a *static* roll demand: a rigid trunk satisfies it by
  rolling as a whole, or by extending the legs on one side further than the
  other. A roll spine only pays when the front and rear pairs need *different*
  roll angles at the same instant — which happens exactly at the entry and exit
  of a step, over roughly one body length, while the front feet are up and the
  rear feet are not. The sustained middle of a long step contributes nothing.
  So the useful quantity is transitions per metre, and this generator maximises
  it at a chosen wavelength instead of maximising step length.

  Travelling this field in the opposite direction reverses which side is high,
  which is why the study commands both directions.
  """
  z = np.zeros((nx, ny), dtype=np.float64)
  height = float(difficulty) * float(amplitude_frac) * LEG_VERTICAL_TRAVEL_M

  x = np.arange(nx) * dx
  y = np.arange(ny) * dx
  y = y - y.mean()

  # Square wave along travel; +1 / -1 with period wavelength_m.
  phase = np.floor(x / float(wavelength_m)) % 2
  sign_x = np.where(phase == 0, 1.0, -1.0)
  # Which lateral half we are on.
  sign_y = np.sign(y)
  sign_y[sign_y == 0] = 1.0

  z = 0.5 * height * sign_x[:, None] * sign_y[None, :]

  # Flat approach and exit so every episode starts and finishes on known ground.
  n_appr = int(round(float(approach_m) / dx))
  if n_appr > 0:
    z[:n_appr, :] = 0.0
    z[-n_appr:, :] = 0.0

  return _zero_mean(z)


def make_map_set(
  n_maps: int,
  nx: int,
  ny: int,
  dx: float,
  *,
  difficulty: float,
  seed: int = 0,
  wavelength_range_m: tuple[float, float] = (0.5 * BODY_LENGTH_M, 4.0 * BODY_LENGTH_M),
) -> list[dict]:
  """Generate a held-out evaluation set with decorrelated geometric content.

  Orientation is swept deterministically over [0, 180) and wavelength sampled
  log-uniformly, which is the cheapest way to make the three regressors in
  ``terrain_content.py`` vary independently across the set. Returns dicts so the
  generating parameters travel with the heightfield — a map in a figure should
  be reproducible from its row.
  """
  rng = np.random.default_rng(seed)
  lo, hi = wavelength_range_m
  out: list[dict] = []
  for i in range(n_maps):
    orientation = (180.0 * i) / n_maps
    wavelength = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    out.append(
      {
        "index": i,
        "family": "band_limited_random",
        "orientation_deg": orientation,
        "wavelength_m": wavelength,
        "difficulty": float(difficulty),
        "seed": int(rng.integers(0, 2**31 - 1)),
        "z": band_limited_random(
          nx,
          ny,
          dx,
          difficulty=difficulty,
          wavelength_m=wavelength,
          orientation_deg=orientation,
          rng=rng,
        ),
      }
    )
  return out


if __name__ == "__main__":
  NX, NY, DX = 128, 96, 0.01  # 1.28 m x 0.96 m at 10 mm

  print("flat:", flat(NX, NY).shape, "mean", flat(NX, NY).mean())

  for orient in (0.0, 45.0, 90.0):
    z = band_limited_random(
      NX, NY, DX, difficulty=1.0, wavelength_m=0.34, orientation_deg=orient,
      rng=np.random.default_rng(0),
    )
    fr = _footprint_range(z, DX)
    print(
      f"band_limited orient={orient:5.1f}  mean={z.mean():+.2e}  "
      f"rms={np.sqrt((z**2).mean()) * 1e3:5.2f} mm  "
      f"p2p={(z.max() - z.min()) * 1e3:6.2f} mm  "
      f"footprint_range p50/p95={np.percentile(fr, 50) * 1e3:5.2f}/"
      f"{np.percentile(fr, 95) * 1e3:5.2f} mm"
    )

  for d in (0.25, 0.5, 1.0):
    z = alternating_lateral_steps(NX, NY, DX, difficulty=d)
    print(
      f"alt_steps    diff={d:4.2f}       mean={z.mean():+.2e}  "
      f"step={(z.max() - z.min()) * 1e3:5.2f} mm"
    )

  maps = make_map_set(5, NX, NY, DX, difficulty=0.6, seed=1)
  print(f"map set: {len(maps)} maps, "
        f"wavelengths {[round(m['wavelength_m'], 3) for m in maps]}")
