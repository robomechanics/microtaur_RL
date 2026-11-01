"""1:1 real-world copies of the Teacher-Cur terrains B (random blocks) and C (side step), levels 1-6, for robot tests.

  python microtaur_study/cad/real_terrain/export_real_terrain.py [--out DIR]   (writes next to this script by default)

Every piece is a 2.0 m x 1.0 m window of a simulated tile, as one closed solid with a flat bottom at z = 0, written
as STL and 3MF in millimetres. Piece frame: x 0..2000 mm = the robot's walking direction, y 0..1000 mm with +y = the
robot's left, z up from the bottom.

  B_lv<l>  the B tile from 0.25 m behind its centre (the spawn point) to 1.75 m ahead, +-0.5 m sideways. The cell
           pattern is the same on every level; heights scale with l / 6 (lv6: -15.2 / +21.6 mm about the simulated
           ground before the tile's zero-mean shift). Cells are 70 mm; the window cuts the edge cells.
  C_lv<l>  the middle 2 m of the 4 m raised section, centred on the step edge: right half (y < 500 mm) at the
           simulated ground, left half raised by l / 6 x 36.8 mm. No guard rails.

The simulated ground (z = 0 in the sim) sits `base` above the bottom: for B the depth of the piece's lowest cell
plus MIN_THICKNESS_M (the thinnest cell is MIN_THICKNESS_M thick), for C MIN_THICKNESS_M. Also written:
B_lv<l>_cells.csv (cell edges and heights), preview.png and pieces.json (numbers per piece).

The numbers come from the training code, read-only: B heights from microtaur_common.terrain_maps.gaussian_blocks
called as microtaur_isaac.terrains.block_cell_heights calls it, and the terrain constants parsed from terrains.py
(that module imports Isaac Sim, so it is not imported here). No simulator, no GPU.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh

STUDY = Path(__file__).resolve().parents[2]  # microtaur_study/
sys.path.insert(0, str(STUDY / "src"))
from microtaur_common.terrain_maps import CELL_M, DOWN_MAX_M, UP_MAX_M, gaussian_blocks  # noqa: E402

LEVELS = range(1, 7)
PIECE_M = (2.0, 1.0)  # length along the walking direction, width
B_WINDOW_X_M = (-0.25, 1.75)  # along x, relative to the B tile centre (= spawn point)
MIN_THICKNESS_M = 0.003
EPS_M = 1e-9


def terrain_constants(path: Path) -> dict:
  """Module-level `NAME = <literal>` constants of a Python source file."""
  out = {}
  for node in ast.parse(path.read_text()).body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
      try:
        out[node.targets[0].id] = ast.literal_eval(node.value)
      except (ValueError, TypeError, SyntaxError):
        pass  # computed from other names
  return out


K = terrain_constants(STUDY / "src" / "microtaur_isaac" / "terrains.py")
GX, GY = K["TILE_CELLS"]
C_DELTA_M = (UP_MAX_M + DOWN_MAX_M) * K["CUR_TERRAIN_SCALE"]  # terrains.py: C_DELTA_M x CUR_TERRAIN_SCALE


def difficulty(level: int) -> float:
  return level / (K["CUR_NUM_LEVELS"] - 1)  # terrains.level_difficulty: level r -> r / (num_levels - 1)


def b_tile_heights(level: int) -> np.ndarray:
  """Simulated B cell heights (GX, GY) in metres about the ground; cell (i, j) spans [i, i+1] x [j, j+1] * CELL_M."""
  h = gaussian_blocks(GX, GY, difficulty=difficulty(level), seed=K["B_PATTERN_SEED"], dx=CELL_M, cell_m=CELL_M)
  return K["CUR_TERRAIN_SCALE"] * h.astype(np.float64)


def cut(n_cells: int, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
  """Cell edges of the window [lo, hi] (tile metres) shifted to start at 0, and the tile cell index of each piece."""
  full = np.arange(n_cells + 1) * CELL_M
  if lo < -EPS_M or hi > full[-1] + EPS_M:
    raise ValueError(f"window [{lo}, {hi}] m leaves the tile [0, {full[-1]}] m")
  edges = np.concatenate([[lo], full[(full > lo + EPS_M) & (full < hi - EPS_M)], [hi]])
  return edges - lo, np.floor(0.5 * (edges[:-1] + edges[1:]) / CELL_M).astype(int)


def solid_from_cells(ex: np.ndarray, ey: np.ndarray, top: np.ndarray) -> trimesh.Trimesh:
  """Closed solid of flat-topped cells: cell (i, j) spans [ex[i], ex[i+1]] x [ey[j], ey[j+1]] x [0, top[i, j]].

  Vertical walls where neighbouring tops differ (outside = 0). Each wall's vertical edges are split at every top
  height that meets at that grid vertex, so the mesh has no T-junctions. Where four cells meet diagonally (two high,
  two low), the two high cells touch along a vertical edge only; split_pinched_edges gives each high cell its own
  copy of that edge, so every edge has exactly two faces (as 3MF requires)."""
  nx, ny = top.shape
  pad = np.zeros((nx + 2, ny + 2))
  pad[1:-1, 1:-1] = top
  # tops meeting at grid vertex (i, j) = padded cells (i..i+1, j..j+1)
  z_at = [[np.unique(pad[i:i + 2, j:j + 2]) for j in range(ny + 1)] for i in range(nx + 1)]
  verts, index, faces, owner = [], {}, [], []  # owner: the cell whose boundary the face is (padded index)

  def v(i: int, j: int, z: float) -> int:
    key = (i, j, round(float(z), 12))
    if key not in index:
      index[key] = len(verts)
      verts.append((ex[i], ey[j], z))
    return index[key]

  def tri(a: int, b: int, c: int, normal: tuple[float, float, float], cell: tuple[int, int]):
    p = np.asarray([verts[a], verts[b], verts[c]])
    faces.append((a, b, c) if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), normal) > 0 else (a, c, b))
    owner.append(cell[0] * (ny + 2) + cell[1])

  def wall(v1: tuple[int, int], v2: tuple[int, int], lo: float, hi: float, normal: tuple[float, float, float],
           cell: tuple[int, int]):
    side = [[z for z in z_at[i][j] if lo - EPS_M <= z <= hi + EPS_M] for i, j in (v1, v2)]
    a = [v(*v1, z) for z in side[0]]
    b = [v(*v2, z) for z in side[1]]
    p = q = 0
    while p < len(a) - 1 or q < len(b) - 1:  # zip the two vertical edges together from the bottom up
      if q == len(b) - 1 or (p < len(a) - 1 and side[0][p + 1] <= side[1][q + 1]):
        tri(a[p], b[q], a[p + 1], normal, cell)
        p += 1
      else:
        tri(a[p], b[q], b[q + 1], normal, cell)
        q += 1

  for i in range(nx):
    for j in range(ny):
      for z, n in ((top[i, j], (0, 0, 1)), (0.0, (0, 0, -1))):
        c = [v(i, j, z), v(i + 1, j, z), v(i + 1, j + 1, z), v(i, j + 1, z)]
        tri(c[0], c[1], c[2], n, (i + 1, j + 1))
        tri(c[0], c[2], c[3], n, (i + 1, j + 1))
  for i in range(nx + 1):  # walls in the planes x = ex[i]
    for j in range(ny):
      ha, hb = pad[i, j + 1], pad[i + 1, j + 1]
      if abs(ha - hb) > EPS_M:
        wall((i, j), (i, j + 1), min(ha, hb), max(ha, hb), (1 if ha > hb else -1, 0, 0),
             (i, j + 1) if ha > hb else (i + 1, j + 1))
  for i in range(nx):  # walls in the planes y = ey[j]
    for j in range(ny + 1):
      ha, hb = pad[i + 1, j], pad[i + 1, j + 1]
      if abs(ha - hb) > EPS_M:
        wall((i, j), (i + 1, j), min(ha, hb), max(ha, hb), (0, 1 if ha > hb else -1, 0),
             (i + 1, j) if ha > hb else (i + 1, j + 1))
  return trimesh.Trimesh(*split_pinched_edges(np.asarray(verts), np.asarray(faces), np.asarray(owner)), process=False)


def split_pinched_edges(verts: np.ndarray, faces: np.ndarray, owner: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """At every vertex of an edge shared by more than two faces, give each fan of faces its own copy of the vertex.
  Faces around the vertex join a fan through an edge with exactly two faces, or through a shared edge with more
  than two faces when they bound the same cell (owner), so the faces of a pinched edge pair up by cell at both of
  its ends."""
  edge_faces = defaultdict(list)
  for f, (a, b, c) in enumerate(faces):
    for u, w in ((a, b), (b, c), (c, a)):
      edge_faces[(min(u, w), max(u, w))].append(f)
  pinched = sorted({x for e, fs in edge_faces.items() if len(fs) > 2 for x in e})
  out_v, out_f = list(verts), faces.copy()
  for x in pinched:
    around = list(np.nonzero((faces == x).any(axis=1))[0])
    root = {f: f for f in around}

    def find(f: int) -> int:
      while root[f] != f:
        f = root[f]
      return f

    for f in around:
      for y in faces[f]:
        fs = edge_faces[(min(x, y), max(x, y))] if y != x else []
        for g in fs:
          if len(fs) == 2 or owner[g] == owner[f]:
            root[find(g)] = find(f)
    fans = defaultdict(list)
    for f in around:
      fans[find(f)].append(f)
    for fan in list(fans.values())[1:]:
      out_v.append(verts[x])
      for f in fan:
        out_f[f][faces[f] == x] = len(out_v) - 1
  return np.asarray(out_v), out_f


def check(mesh: trimesh.Trimesh, ex: np.ndarray, ey: np.ndarray, top: np.ndarray, name: str):
  """Closed, consistently wound, the right size, every upward face at its cell's height, volume = sum(area x top)."""
  area = np.outer(np.diff(ex), np.diff(ey))
  assert mesh.is_watertight and mesh.is_winding_consistent, f"{name}: not a closed solid"
  assert np.allclose(mesh.bounds, [[0, 0, 0], [ex[-1], ey[-1], top.max()]], atol=1e-9), f"{name}: bounds {mesh.bounds}"
  assert abs(mesh.volume - float((area * top).sum())) < 1e-9, f"{name}: volume {mesh.volume} vs {(area * top).sum()}"
  up = mesh.face_normals[:, 2] > 0.999
  c = mesh.triangles_center[up]
  i = np.searchsorted(ex, c[:, 0]) - 1
  j = np.searchsorted(ey, c[:, 1]) - 1
  assert np.allclose(c[:, 2], top[i, j], atol=1e-9), f"{name}: an upward face is off its cell height"
  assert abs(mesh.area_faces[up].sum() - ex[-1] * ey[-1]) < 1e-9, f"{name}: upward area"


def export(mesh: trimesh.Trimesh, out: Path, name: str):
  mm = mesh.copy()
  mm.apply_scale(1000.0)
  mm.export(out / f"{name}.stl")
  (out / f"{name}.3mf").write_bytes(trimesh.Scene({name: mm}).export(file_type="3mf"))  # 3MF unit: millimetre


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
  out = ap.parse_args().out
  out.mkdir(parents=True, exist_ok=True)
  cx, cy = 0.5 * GX * CELL_M, 0.5 * GY * CELL_M  # B tile centre (tile frame)
  bx, ix = cut(GX, cx + B_WINDOW_X_M[0], cx + B_WINDOW_X_M[1])
  by, iy = cut(GY, cy - 0.5 * PIECE_M[1], cy + 0.5 * PIECE_M[1])
  assert np.isclose(bx[-1], PIECE_M[0]) and np.isclose(by[-1], PIECE_M[1])
  pieces, b_sim = [], {}
  for level in LEVELS:
    h = b_tile_heights(level)[np.ix_(ix, iy)]  # about the simulated ground
    base = MIN_THICKNESS_M - h.min()
    b_sim[level] = h
    name = f"B_lv{level}"
    mesh = solid_from_cells(bx, by, base + h)
    check(mesh, bx, by, base + h, name)
    export(mesh, out, name)
    with open(out / f"{name}_cells.csv", "w", newline="") as fh:
      w = csv.writer(fh)
      w.writerow(["x0_mm", "x1_mm", "y0_mm", "y1_mm", "top_above_bottom_mm", "height_about_sim_ground_mm"])
      for a in range(len(ix)):
        for b in range(len(iy)):
          w.writerow([f"{1e3 * bx[a]:.1f}", f"{1e3 * bx[a + 1]:.1f}", f"{1e3 * by[b]:.1f}", f"{1e3 * by[b + 1]:.1f}",
                      f"{1e3 * (base + h[a, b]):.2f}", f"{1e3 * h[a, b]:+.2f}"])
    pieces.append({"piece": name, "sim_ground_above_bottom_mm": 1e3 * base, "top_min_mm": 1e3 * (base + h.min()),
                   "top_max_mm": 1e3 * (base + h.max()), "height_about_sim_ground_mm": [1e3 * h.min(), 1e3 * h.max()],
                   "cells": [len(ix), len(iy)], "triangles": len(mesh.faces)})
  cex, cey = np.array([0.0, PIECE_M[0]]), np.array([0.0, 0.5 * PIECE_M[1], PIECE_M[1]])
  for level in LEVELS:
    step = difficulty(level) * C_DELTA_M
    top = MIN_THICKNESS_M + np.array([[0.0, step]])  # right (low) half, left (raised) half
    name = f"C_lv{level}"
    mesh = solid_from_cells(cex, cey, top)
    check(mesh, cex, cey, top, name)
    export(mesh, out, name)
    pieces.append({"piece": name, "sim_ground_above_bottom_mm": 1e3 * MIN_THICKNESS_M, "step_mm": 1e3 * step,
                   "top_min_mm": 1e3 * top.min(), "top_max_mm": 1e3 * top.max(), "triangles": len(mesh.faces)})
  summary = {"tile_cells": [GX, GY], "cell_mm": 1e3 * CELL_M, "b_window_x_from_tile_centre_m": B_WINDOW_X_M,
             "pieces": pieces}
  (out / "pieces.json").write_text(json.dumps(summary, indent=1))
  preview(out, bx, by, b_sim)
  for p in pieces:
    print(p["piece"], {k: (round(v, 2) if isinstance(v, float) else v) for k, v in p.items() if k != "piece"})


def preview(out: Path, bx: np.ndarray, by: np.ndarray, b_sim: dict):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

  cmap = LinearSegmentedColormap.from_list("blue_gray_red", ["#1c5cab", "#f0efec", "#e34948"])  # below / at / above
  lim = max(abs(h).max() for h in b_sim.values()) * 1e3
  norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
  fig, axs = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
  for ax, (level, h) in zip(axs.ravel(), b_sim.items()):
    im = ax.pcolormesh(1e3 * bx, 1e3 * by, 1e3 * h.T, cmap=cmap, norm=norm)
    ax.set_aspect("equal")
    ax.set_title(f"B_lv{level}: {1e3 * h.min():+.1f} .. {1e3 * h.max():+.1f} mm about the simulated ground",
                 fontsize=10)
    ax.set_xlabel("x mm (walking direction)")
    ax.set_ylabel("y mm (+ = left)")
  fig.colorbar(im, ax=axs, shrink=0.6, label="cell height about the simulated ground (mm)")
  fig.suptitle("B pieces, top view (2000 x 1000 mm). C pieces: two flat halves, left half raised by "
               + ", ".join(f"{1e3 * difficulty(lv) * C_DELTA_M:.1f}" for lv in LEVELS) + " mm (lv1-6)")
  fig.savefig(out / "preview.png", dpi=90)


if __name__ == "__main__":
  main()
