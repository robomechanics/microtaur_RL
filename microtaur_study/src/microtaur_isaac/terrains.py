"""Teacher terrains A / B / C as IsaacLab sub-terrains (trimesh, no heightfields).

Import only after the Kit app is launched (isaaclab.terrains imports omni).
All numbers come from microtaur_common.terrain_maps; this module only turns
them into triangle meshes.

  A_flat    flat plane at z = 0.
  B_blocks  one FIXED grid of flat-topped 70 mm cells (same seed on every
            tile), Gaussian heights clipped to [-DOWN_MAX_M, +UP_MAX_M],
            zero-mean per tile, scaled linearly by the level's difficulty.
            Each cell is a top quad; vertical walls are added only where
            neighbouring heights differ, so walls are exactly vertical and a
            tile is ~27k triangles (a 2.5 mm heightfield would be ~5M).
  C_step    along +x: low flat 0.5 m -> 4.0 m with the left half (y > 0)
            raised by d * (UP_MAX_M + DOWN_MAX_M) -> low flat 0.5 m. The
            course is centred on the tile; the rest of the tile is low flat.

Every tile has a skirt: vertical walls from its edge heights down to
SKIRT_BASE_Z_M, so there is never an open gap between tiles of different
heights (the IsaacLab border boxes also have vertical inner walls).

IsaacLab layout (TerrainGenerator, curriculum=True)
  rows    = levels, laid out along world x: tile (row, col) is centred at
            ((row + 0.5) * size_x, (col + 0.5) * size_y) minus half the grid.
            Row r gets difficulty (r + U(0,1)) / num_rows; the sub-terrain
            functions here snap that to level r -> d = r / (NUM_LEVELS - 1),
            i.e. 0, 0.25, 0.5, 0.75, 1.0 exactly (requires num_rows ==
            NUM_LEVELS and difficulty_range == (0, 1)).
  columns = terrain type, along world y, assigned by proportion (not random):
            A 0.2 / B 0.5 / C 0.3 over 10 columns -> COLUMN_TERRAIN_TYPES.
  A robot walking +x leaves its tile into the NEXT LEVEL of the same type
  (same column); from row NUM_LEVELS - 1 it walks onto the flat border.
  Walking +/-y it enters a tile of the same level, possibly another type.

Env origins
  TerrainImporter.env_origins[i] = terrain_origins[level, column] = the tile
  centre at z = 0 for all three types (B is zero-mean, so its centre cell can
  be up to d * UP_MAX_M above the origin: spawn with >= B_MAX_HEIGHT_M extra
  clearance or read the height scan). TerrainImporter.terrain_types[i] is the
  column index (fixed per env; levels change via update_env_origins); use
  env_terrain_type_ids() / COLUMN_TERRAIN_TYPES to get the type.
  C: spawn at env_origin + (C_SPAWN_OFFSET_XY, 0), yaw C_SPAWN_YAW_RAD (+x).
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from microtaur_common.terrain_maps import CELL_M, DOWN_MAX_M, SIGMA_M, UP_MAX_M, gaussian_blocks

from . import TERRAIN_TYPES

# ---------------------------------------------------------------- constants
NUM_LEVELS = 5  # d = 0, 0.25, 0.5, 0.75, 1.0
TILE_CELLS = (90, 50)  # integer number of 70 mm cells per tile
TILE_SIZE_M = (TILE_CELLS[0] * CELL_M, TILE_CELLS[1] * CELL_M)  # (6.30, 3.50) m
BORDER_WIDTH_M = 3.0  # flat z = 0 around the grid; > 7 m walk (0.35 m/s x 20 s) from the last C spawn
SKIRT_BASE_Z_M = -0.05  # tile skirts go down to here (well below -DOWN_MAX_M)
B_PATTERN_SEED = 0  # same seed as viz_terrains
B_MAX_HEIGHT_M = UP_MAX_M  # upper bound of any B cell top (the mean shift makes it slightly lower)

C_FLAT_BEFORE_M = 0.5
C_STEP_LEN_M = 4.0
C_FLAT_AFTER_M = 0.5
C_COURSE_LEN_M = C_FLAT_BEFORE_M + C_STEP_LEN_M + C_FLAT_AFTER_M  # 5.0 m, centred on the tile
C_DELTA_M = UP_MAX_M + DOWN_MAX_M  # 18.4 mm at d = 1
C_SPAWN_ALONG_M = 0.3  # spawn this far into the approach flat (as in viz_terrains); front feet ~0.115 m before the step
# Spawn point relative to the tile centre (= env origin), tile frame (+x forward, +y left).
C_SPAWN_OFFSET_XY = (-0.5 * C_COURSE_LEN_M + C_SPAWN_ALONG_M, 0.0)  # (-2.2, 0.0) m, straddling the y = 0 step edge
C_SPAWN_YAW_RAD = 0.0  # heading +x; straight-line commands only
# Raised section's x start / end relative to the tile centre.
C_STEP_X_RANGE = (-0.5 * C_COURSE_LEN_M + C_FLAT_BEFORE_M, -0.5 * C_COURSE_LEN_M + C_FLAT_BEFORE_M + C_STEP_LEN_M)


def level_difficulty(difficulty: float, num_levels: int | None) -> float:
  """Snap IsaacLab's per-tile difficulty (row + U(0,1)) / num_rows to level row -> row / (num_levels - 1)."""
  if not num_levels or num_levels < 2:
    return float(difficulty)
  level = min(max(int(math.floor(difficulty * num_levels + 1e-9)), 0), num_levels - 1)
  return level / (num_levels - 1)


# ---------------------------------------------------------------- quad primitives
# Each returns corners (N, 4, 3), counter-clockwise seen from the side the face
# points to; _quads_to_mesh splits each quad into triangles (0,1,2), (0,2,3).

def _flat(*arrays):
  return [a.ravel() for a in np.broadcast_arrays(*(np.asarray(a) for a in arrays))]


def _tops(x0, x1, y0, y1, z) -> np.ndarray:
  x0, x1, y0, y1, z = _flat(x0, x1, y0, y1, z)
  return np.stack([np.stack([x0, y0, z], -1), np.stack([x1, y0, z], -1),
                   np.stack([x1, y1, z], -1), np.stack([x0, y1, z], -1)], axis=1).astype(np.float64)  # normal +z


def _walls_x(x, y0, y1, za, zb, face_pos) -> np.ndarray:
  """Walls in the plane x = const spanning za..zb; normal +x where face_pos else -x."""
  x, y0, y1, za, zb, face_pos = _flat(x, y0, y1, za, zb, face_pos)
  lo, hi = np.minimum(za, zb), np.maximum(za, zb)
  q = np.stack([np.stack([x, y0, lo], -1), np.stack([x, y1, lo], -1),
                np.stack([x, y1, hi], -1), np.stack([x, y0, hi], -1)], axis=1).astype(np.float64)  # normal +x
  return np.where(face_pos.astype(bool)[:, None, None], q, q[:, ::-1])


def _walls_y(y, x0, x1, za, zb, face_pos) -> np.ndarray:
  """Walls in the plane y = const spanning za..zb; normal +y where face_pos else -y."""
  y, x0, x1, za, zb, face_pos = _flat(y, x0, x1, za, zb, face_pos)
  lo, hi = np.minimum(za, zb), np.maximum(za, zb)
  q = np.stack([np.stack([x0, y, lo], -1), np.stack([x1, y, lo], -1),
                np.stack([x1, y, hi], -1), np.stack([x0, y, hi], -1)], axis=1).astype(np.float64)  # normal -y
  return np.where(face_pos.astype(bool)[:, None, None], q[:, ::-1], q)


def _quads_to_mesh(quads: list[np.ndarray]) -> trimesh.Trimesh:
  q = np.concatenate([a for a in quads if len(a)], axis=0)
  base = 4 * np.arange(len(q))[:, None]
  faces = np.concatenate([base + [0, 1, 2], base + [0, 2, 3]], axis=0)
  return trimesh.Trimesh(vertices=q.reshape(-1, 3), faces=faces, process=True)


def _height_grid_mesh(h: np.ndarray, cell: tuple[float, float], base_z: float, eps: float = 1e-7) -> trimesh.Trimesh:
  """Flat-topped cells h[i, j] on [i*cx, (i+1)*cx] x [j*cy, (j+1)*cy]; vertical walls; skirt down to base_z."""
  gx, gy = h.shape
  xs = np.arange(gx + 1) * cell[0]
  ys = np.arange(gy + 1) * cell[1]
  ii, jj = np.meshgrid(np.arange(gx), np.arange(gy), indexing="ij")
  quads = [_tops(xs[ii], xs[ii + 1], ys[jj], ys[jj + 1], h)]
  hp = np.pad(h.astype(np.float64), 1, constant_values=base_z)  # outside the tile = base -> skirt
  # walls x = xs[p] between padded cells p (behind) and p + 1 (ahead); padded rows q = 1..gy
  p, q = np.meshgrid(np.arange(gx + 1), np.arange(1, gy + 1), indexing="ij")
  za, zb = hp[p, q], hp[p + 1, q]
  m = np.abs(za - zb) > eps
  quads.append(_walls_x(xs[p[m]], ys[q[m] - 1], ys[q[m]], za[m], zb[m], za[m] > zb[m]))
  # walls y = ys[q] between padded cells q (right) and q + 1 (left); padded columns p = 1..gx
  p, q = np.meshgrid(np.arange(1, gx + 1), np.arange(gy + 1), indexing="ij")
  za, zb = hp[p, q], hp[p, q + 1]
  m = np.abs(za - zb) > eps
  quads.append(_walls_y(ys[q[m]], xs[p[m] - 1], xs[p[m]], za[m], zb[m], za[m] > zb[m]))
  return _quads_to_mesh(quads)


# ---------------------------------------------------------------- terrain functions
def flat_terrain(difficulty: float, cfg: MicrotaurFlatTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
  """A: plane at z = 0 plus skirt (10 triangles)."""
  mesh = _height_grid_mesh(np.zeros((1, 1)), tuple(cfg.size), cfg.skirt_base_z)
  return [mesh], np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])


def block_cell_heights(size: tuple[float, float], difficulty: float, cell_m: float = CELL_M,
                       seed: int = B_PATTERN_SEED) -> np.ndarray:
  """B cell heights (gx, gy) for a tile of `size` at difficulty d (already snapped). Pure numpy."""
  g = [size[k] / cell_m for k in range(2)]
  gx, gy = (int(round(v)) for v in g)
  if abs(g[0] - gx) > 1e-6 or abs(g[1] - gy) > 1e-6:
    raise ValueError(f"B_blocks tile size {size} must be an integer number of {cell_m} m cells")
  # dx = cell_m -> one sample per cell: exactly terrain_maps' clipped, zero-mean, d-scaled cells.
  return gaussian_blocks(gx, gy, difficulty=difficulty, seed=seed, dx=cell_m, cell_m=cell_m).astype(np.float64)


def blocks_terrain(difficulty: float, cfg: MicrotaurBlocksTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
  """B: fixed Gaussian cell pattern, scaled by the level's difficulty."""
  d = level_difficulty(difficulty, cfg.num_levels)
  h = cfg.height_scale * block_cell_heights(cfg.size, d, cfg.cell_m, cfg.pattern_seed)
  mesh = _height_grid_mesh(h, (cfg.cell_m, cfg.cell_m), cfg.skirt_base_z)
  return [mesh], np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])


def step_terrain(difficulty: float, cfg: MicrotaurStepTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
  """C: low flat -> left half (y > 0) raised -> low flat along +x, course centred on the tile."""
  d = level_difficulty(difficulty, cfg.num_levels)
  lx, ly = cfg.size
  course = cfg.flat_before_m + cfg.step_len_m + cfg.flat_after_m
  if course > lx + 1e-9:
    raise ValueError(f"C_step course {course} m does not fit tile length {lx} m")
  if 0.5 * ly < cfg.min_half_width_m:
    raise ValueError(f"C_step half width {0.5 * ly} m < {cfg.min_half_width_m} m")
  x0 = 0.5 * (lx - course) + cfg.flat_before_m
  x1 = x0 + cfg.step_len_m
  ym = 0.5 * ly
  z = d * cfg.delta_m
  b = cfg.skirt_base_z
  quads = [
    # right half full length, left before, left after (z = 0), raised left section (z)
    _tops([0.0, 0.0, x1, x0], [lx, x0, lx, x1], [0.0, ym, ym, ym], [ym, ly, ly, ly], [0.0, 0.0, 0.0, z]),
    # skirt: x = 0 and x = lx, y = 0 (all z = 0), y = ly (three segments)
    _walls_x([0.0, lx], 0.0, ly, 0.0, b, [False, True]),
    _walls_y(0.0, 0.0, lx, 0.0, b, False),
    _walls_y(ly, [0.0, x0, x1], [x0, x1, lx], [0.0, z, 0.0], b, True),
  ]
  if z > 1e-7:  # step faces: entry (faces -x), exit (faces +x), side (faces -y, toward the low right half)
    quads += [_walls_x([x0, x1], ym, ly, 0.0, z, [False, True]), _walls_y(ym, x0, x1, 0.0, z, False)]
  return [_quads_to_mesh(quads)], np.array([0.5 * lx, 0.5 * ly, 0.0])


# ---------------------------------------------------------------- cfgs
@configclass
class MicrotaurFlatTerrainCfg(SubTerrainBaseCfg):
  """A_flat."""
  function = flat_terrain
  skirt_base_z: float = SKIRT_BASE_Z_M


@configclass
class MicrotaurBlocksTerrainCfg(SubTerrainBaseCfg):
  """B_blocks. (Not named `seed`: TerrainGenerator overwrites cfg.seed with its own seed.)"""
  function = blocks_terrain
  cell_m: float = CELL_M
  pattern_seed: int = B_PATTERN_SEED
  height_scale: float = 1.0  # multiplies the whole map (1 = clip at 50% of the stance budget, 2 = 100%)
  num_levels: int | None = NUM_LEVELS  # None: use IsaacLab's continuous difficulty
  skirt_base_z: float = SKIRT_BASE_Z_M


@configclass
class MicrotaurStepTerrainCfg(SubTerrainBaseCfg):
  """C_step."""
  function = step_terrain
  flat_before_m: float = C_FLAT_BEFORE_M
  step_len_m: float = C_STEP_LEN_M
  flat_after_m: float = C_FLAT_AFTER_M
  delta_m: float = C_DELTA_M
  min_half_width_m: float = 1.0
  num_levels: int | None = NUM_LEVELS
  skirt_base_z: float = SKIRT_BASE_Z_M


MICROTAUR_TERRAINS_CFG = TerrainGeneratorCfg(
  seed=0,
  curriculum=True,
  size=TILE_SIZE_M,
  border_width=BORDER_WIDTH_M,
  border_height=1.0,
  num_rows=NUM_LEVELS,
  num_cols=10,
  difficulty_range=(0.0, 1.0),
  use_cache=False,
  sub_terrains={
    "A_flat": MicrotaurFlatTerrainCfg(proportion=0.2),
    "B_blocks": MicrotaurBlocksTerrainCfg(proportion=0.5),
    "C_step": MicrotaurStepTerrainCfg(proportion=0.3),
  },
)
"""5 levels (rows, along x) x 10 columns (along y): columns 0-1 A, 2-6 B, 7-9 C."""


def scaled_terrains_cfg(scale: float) -> TerrainGeneratorCfg:
  """MICROTAUR_TERRAINS_CFG with B heights and the C step multiplied by `scale`
  (2.0: B clipped at -15.2 / +21.6 mm with sigma 7.6 mm, C step 36.8 mm = 100% of the
  stance-phase budget at the top level). Same layout, seed and B pattern."""
  cfg = MICROTAUR_TERRAINS_CFG.copy()
  subs = dict(cfg.sub_terrains)
  subs["B_blocks"] = subs["B_blocks"].replace(height_scale=scale)
  subs["C_step"] = subs["C_step"].replace(delta_m=C_DELTA_M * scale)
  cfg.sub_terrains = subs
  return cfg


# ---------------------------------------------------------------- terrain type lookup
def column_terrain_types(cfg: TerrainGeneratorCfg = MICROTAUR_TERRAINS_CFG) -> tuple[str, ...]:
  """Sub-terrain name per column, same rule as TerrainGenerator._generate_curriculum_terrains."""
  names = list(cfg.sub_terrains.keys())
  prop = np.array([s.proportion for s in cfg.sub_terrains.values()], dtype=np.float64)
  cum = np.cumsum(prop / prop.sum())
  return tuple(names[int(np.min(np.where(c / cfg.num_cols + 0.001 < cum)[0]))] for c in range(cfg.num_cols))


COLUMN_TERRAIN_TYPES = column_terrain_types()
"""('A_flat',) * 2 + ('B_blocks',) * 5 + ('C_step',) * 3, indexed by TerrainImporter.terrain_types."""
COLUMN_TERRAIN_TYPE_IDS = tuple(TERRAIN_TYPES.index(n) for n in COLUMN_TERRAIN_TYPES)
"""Index into TERRAIN_TYPES per column: (0, 0, 1, 1, 1, 1, 1, 2, 2, 2)."""


def env_terrain_type_ids(terrain, cfg: TerrainGeneratorCfg | None = None):
  """Per-env index into TERRAIN_TYPES (0 A, 1 B, 2 C) from a TerrainImporter (torch long, env device)."""
  import torch

  ids = COLUMN_TERRAIN_TYPE_IDS if cfg is None else tuple(TERRAIN_TYPES.index(n) for n in column_terrain_types(cfg))
  lut = torch.tensor(ids, dtype=torch.long, device=terrain.terrain_types.device)
  return lut[terrain.terrain_types]


__all__ = [
  "B_MAX_HEIGHT_M", "BORDER_WIDTH_M", "CELL_M", "C_DELTA_M", "C_SPAWN_OFFSET_XY", "C_SPAWN_YAW_RAD",
  "C_STEP_X_RANGE", "COLUMN_TERRAIN_TYPES", "COLUMN_TERRAIN_TYPE_IDS", "DOWN_MAX_M", "MICROTAUR_TERRAINS_CFG",
  "MicrotaurBlocksTerrainCfg", "MicrotaurFlatTerrainCfg", "MicrotaurStepTerrainCfg", "NUM_LEVELS", "SIGMA_M",
  "SKIRT_BASE_Z_M", "TILE_SIZE_M", "UP_MAX_M", "block_cell_heights", "column_terrain_types",
  "env_terrain_type_ids", "level_difficulty", "scaled_terrains_cfg",
]
