"""Test terrains for Microtaur rollouts: flat, curb, steps, weave.

Rebuilt for this repo from the geometry documented in export_print_terrains.py,
steps_onfield.py / resummarize.py and weave_drive.py (the original
src/microtaur_velocity/microtaur_terrains.py was not shipped with the package):

  course   4.0 m (travel, +x) x 1.2 m, one patch, 1.0 m flat border. The border
           reproduces the logged out_of_terrain_bounds reset at |y| = 1.30 m.
  spawn    course-local (0.175, 0.60), heading +x.
  curb     kerb edge on the centreline (y = 0.60): +y feet on the kerb top, -y
           feet on the road. Spawn is on the kerb-top height so no foot starts
           inside the kerb; the -y feet then drop onto the road.
  steps    80 mm tiles from x = 0.35 m, heights drawn from 5 even levels up to
           max_height. Tiles span origin-relative x 0.175-3.775, |y| <= 0.60.
  weave    flat ground plus visual-only poles (no collision, excluded from the
           height scan), first pole 0.40 m ahead of spawn.
  flat     the task's own plane.

apply_terrain() swaps the terrain after the task cfg is built, so observations,
actions and sensors are unchanged. Because the robot can end up below its spawn
height (curb), the base_too_low threshold is lowered by surface_drop_m().
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)
from mjlab.terrains.utils import make_plane

COURSE_SIZE_M = (4.0, 1.2)
BORDER_WIDTH_M = 1.0
SPAWN_XY_M = (0.175, 0.60)
TERRAIN_SEED = 0  # one fixed step field for every variant, speed and seed

ROAD_COLOR = (0.45, 0.45, 0.45, 1.0)
FEATURE_COLOR = (0.75, 0.55, 0.30, 1.0)
POLE_COLOR = (0.85, 0.15, 0.15, 1.0)


def _box(body, center, half_size, color):
  geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=half_size, pos=center)
  return TerrainGeometry(geom=geom, color=color)


def _road(body):
  return TerrainGeometry(geom=make_plane(body, COURSE_SIZE_M, 0.0, center_zero=False)[0],
                         color=ROAD_COLOR)


@dataclass(kw_only=True)
class CurbTerrainCfg(SubTerrainCfg):
  curb_height: float = 0.015

  def function(self, difficulty, spec, rng):
    del difficulty, rng
    body = spec.body("terrain")
    length, width = self.size
    y_edge = SPAWN_XY_M[1]
    kerb = _box(body,
                (length / 2, (y_edge + width) / 2, self.curb_height / 2),
                (length / 2, (width - y_edge) / 2, self.curb_height / 2),
                FEATURE_COLOR)
    origin = np.array([SPAWN_XY_M[0], SPAWN_XY_M[1], self.curb_height])
    return TerrainOutput(origin=origin, geometries=[_road(body), kerb])


@dataclass(kw_only=True)
class RandomStepsTerrainCfg(SubTerrainCfg):
  max_height: float = 0.010
  tile_size: float = 0.08
  num_levels: int = 5
  flat_start: float = 0.35

  def function(self, difficulty, spec, rng):
    del difficulty
    body = spec.body("terrain")
    length, width = self.size
    levels = np.linspace(0.0, self.max_height, self.num_levels)
    geometries = [_road(body)]
    nx = int((length - self.flat_start) // self.tile_size)
    ny = int(width // self.tile_size)
    for i in range(nx):
      for j in range(ny):
        h = float(rng.choice(levels))
        if h <= 0.0:
          continue
        center = (self.flat_start + (i + 0.5) * self.tile_size,
                  (j + 0.5) * self.tile_size, h / 2)
        shade = 0.35 + 0.5 * h / max(self.max_height, 1e-9)
        geometries.append(_box(body, center, (self.tile_size / 2, self.tile_size / 2, h / 2),
                               (shade, shade * 0.8, 0.3, 1.0)))
    origin = np.array([SPAWN_XY_M[0], SPAWN_XY_M[1], 0.0])
    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class WeaveTerrainCfg(SubTerrainCfg):
  spacing: float = 0.80
  num_poles: int = 5
  lead_in: float = 0.40
  pole_radius: float = 0.004
  pole_height: float = 0.15

  def function(self, difficulty, spec, rng):
    del difficulty, rng
    body = spec.body("terrain")
    geometries = [_road(body)]
    for k in range(self.num_poles):
      x = SPAWN_XY_M[0] + self.lead_in + k * self.spacing
      pole = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(self.pole_radius, self.pole_height / 2, 0.0),
        pos=(x, SPAWN_XY_M[1], self.pole_height / 2),
        contype=0, conaffinity=0,
        group=2,  # height scan only sees group 0
      )
      pole.rgba[:] = POLE_COLOR
      geometries.append(TerrainGeometry(geom=pole, color=POLE_COLOR))
    origin = np.array([SPAWN_XY_M[0], SPAWN_XY_M[1], 0.0])
    return TerrainOutput(origin=origin, geometries=geometries)


_SUB_TERRAINS = {
  "curb": CurbTerrainCfg,
  "steps": RandomStepsTerrainCfg,
  "weave": WeaveTerrainCfg,
}


def surface_drop_m(terrain: str, **kw) -> float:
  """How far below its spawn height the robot can legitimately end up, metres."""
  if terrain == "curb":
    return float(kw.get("curb_height", CurbTerrainCfg.curb_height))
  return 0.0


def apply_terrain(env_cfg, terrain: str, **kw) -> None:
  """Replace the task terrain with a test course.

  The checkpoints are rough-task policies, whose default terrain is the training
  generator, so 'flat' must explicitly switch to a plane.
  """
  if terrain == "flat":
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    for name in ("terrain_levels", "terrain_level_tracking"):
      env_cfg.curriculum.pop(name, None)
    return
  if terrain not in _SUB_TERRAINS:
    raise ValueError(f"unknown terrain {terrain!r}; choices: flat, {', '.join(_SUB_TERRAINS)}")

  sub_terrain = _SUB_TERRAINS[terrain](size=COURSE_SIZE_M, **kw)
  terrain_cfg = env_cfg.scene.terrain
  terrain_cfg.terrain_type = "generator"
  terrain_cfg.terrain_generator = TerrainGeneratorCfg(
    seed=TERRAIN_SEED,
    size=COURSE_SIZE_M,
    border_width=BORDER_WIDTH_M,
    num_rows=1,
    num_cols=1,
    color_scheme="height",  # use each geometry's own colour
    sub_terrains={terrain: sub_terrain},
    add_lights=True,
  )
  terrain_cfg.max_init_terrain_level = 0

  drop = surface_drop_m(terrain, **kw)
  base_too_low = env_cfg.terminations.get("base_too_low")
  if drop > 0.0 and base_too_low is not None:
    base_too_low.params["min_height_m"] = float(base_too_low.params["min_height_m"]) - drop
