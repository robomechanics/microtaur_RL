"""Build the IsaacLab teacher terrains (microtaur_isaac.terrains) and measure them.

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_terrain_check.py --out figures/isaac_terrain_check [--device cpu]

Launches Kit headless, runs TerrainGenerator on MICROTAUR_TERRAINS_CFG and
measures every tile from the generated mesh (warp ray casts, not from the
height arrays): triangle count, top-surface height range, face orientation,
B clip fractions / max adjacent-cell step / wall sharpness, C step height,
tile-edge continuity. Then imports it with TerrainImporter (env origins,
terrain type per env) and drops foot-sized spheres on B and C to check PhysX
collision against the cooked mesh. Writes <out>/terrain_check.log and
<out>/terrain_topview.png (matplotlib top view of ray-cast heights; no Kit
viewport render).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="figures/isaac_terrain_check")
ap.add_argument("--num_envs", type=int, default=200)
ap.add_argument("--no_physics", action="store_true")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app

out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)
LOG = open(out / "terrain_check.log", "w")


def log(*a):
  s = " ".join(str(x) for x in a)
  LOG.write(s + "\n")
  LOG.flush()


try:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import numpy as np
  import torch

  sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
  import isaacsim.core.utils.prims as prim_utils

  import isaaclab.sim as sim_utils
  from isaaclab.assets import RigidObject, RigidObjectCfg
  from isaaclab.terrains import TerrainGenerator, TerrainImporter, TerrainImporterCfg
  from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

  from microtaur_common.terrain_maps import CELL_M, DOWN_MAX_M, UP_MAX_M
  from microtaur_isaac import TERRAIN_TYPES
  from microtaur_isaac import terrains as T

  dev = args.device if args.device else "cuda:0"
  log(f"device {dev}; tile {T.TILE_SIZE_M} m = {T.TILE_CELLS} cells; border {T.BORDER_WIDTH_M} m")
  log(f"UP_MAX {UP_MAX_M * 1e3:.1f} mm, DOWN_MAX {DOWN_MAX_M * 1e3:.1f} mm, SIGMA {T.SIGMA_M * 1e3:.2f} mm, "
      f"C delta {T.C_DELTA_M * 1e3:.1f} mm")
  log(f"C_SPAWN_OFFSET_XY {T.C_SPAWN_OFFSET_XY}, C_STEP_X_RANGE {T.C_STEP_X_RANGE}, yaw {T.C_SPAWN_YAW_RAD}")
  log(f"COLUMN_TERRAIN_TYPES {T.COLUMN_TERRAIN_TYPES}")
  log(f"COLUMN_TERRAIN_TYPE_IDS {T.COLUMN_TERRAIN_TYPE_IDS}")

  cfg = T.MICROTAUR_TERRAINS_CFG.copy()
  t0 = time.time()
  gen = TerrainGenerator(cfg, device=dev)
  log(f"generator built in {time.time() - t0:.2f} s")
  R, C = cfg.num_rows, cfg.num_cols
  LX, LY = cfg.size
  mesh = gen.terrain_mesh
  log(f"TOTAL generator mesh: {len(mesh.faces)} triangles, {len(mesh.vertices)} vertices "
      f"(border part: {len(gen.terrain_meshes[-1].faces)} tris)")
  wpm = convert_to_warp_mesh(np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int32), device=dev)
  origins = gen.terrain_origins  # (R, C, 3), world frame (grid centred)

  def height(x, y):
    """Top-surface height at world (x, y) from a downward ray cast (numpy in, numpy out)."""
    x, y = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64))
    shp = x.shape
    st = torch.tensor(np.stack([x.ravel(), y.ravel(), np.full(x.size, 1.0)], -1), dtype=torch.float32, device=dev)
    dr = torch.zeros_like(st)
    dr[:, 2] = -1.0
    hit = raycast_mesh(st, dr, wpm)[0]
    return hit[:, 2].cpu().numpy().astype(np.float64).reshape(shp)

  def tile_mesh(r, c):
    return gen.terrain_meshes[c * R + r]  # generator loops col-major (for col: for row)

  def diff_of(r):
    return T.level_difficulty((r + 0.5) / R, T.NUM_LEVELS)

  log(f"row -> difficulty: {[diff_of(r) for r in range(R)]}  (IsaacLab raw: (row + U(0,1)) / {R})")

  # ---------------------------------------------------------- per tile
  bad_orient = 0
  total_tile_tris = 0
  rows_out = []
  gx, gy = T.TILE_CELLS
  ic = (np.arange(gx) + 0.5) * CELL_M - LX / 2
  jc = (np.arange(gy) + 0.5) * CELL_M - LY / 2
  b_cells = {}
  for c in range(C):
    name = T.COLUMN_TERRAIN_TYPES[c]
    for r in range(R):
      d = diff_of(r)
      m = tile_mesh(r, c)
      total_tile_tris += len(m.faces)
      n = m.face_normals
      top = n[:, 2] > 0.5
      vert = np.abs(n[:, 2]) < 1e-9
      other = ~(top | vert)
      zt = m.triangles_center[top, 2]
      # orientation: every vertical face must look at air (surface lower 1 mm in front of its centroid)
      # (interior walls: 1 mm in front of the centroid the surface is lower; perimeter skirts: face outward)
      cen = m.triangles_center[vert]
      loc = cen[:, :2] - np.array([(r + 0.5) * LX, (c + 0.5) * LY])  # tile-centre frame
      rim = (np.abs(np.abs(loc[:, 0]) - LX / 2) < 1e-6) | (np.abs(np.abs(loc[:, 1]) - LY / 2) < 1e-6)
      bo = int(np.sum(np.sum(n[vert][rim, :2] * loc[rim], axis=1) <= 0))
      if (~rim).any():
        p = loc[~rim] + 1e-3 * n[vert][~rim, :2] + origins[r, c, :2]
        bo += int(np.sum(height(p[:, 0], p[:, 1]) > cen[~rim, 2] + 1e-6))
      bad_orient += bo + int(other.sum()) + int(np.sum(n[:, 2] < -0.5))
      ox, oy = origins[r, c, 0], origins[r, c, 1]
      line = (f"{name:8s} col {c} row {r} d={d:.2f}: {len(m.faces):6d} tris, top z [{zt.min() * 1e3:+7.3f}, "
              f"{zt.max() * 1e3:+7.3f}] mm, vertical {int(vert.sum())}, non-axis {int(other.sum())}, "
              f"down-facing {int(np.sum(n[:, 2] < -0.5))}, walls facing into terrain {bo}; origin "
              f"({ox:+.3f}, {oy:+.3f}, {origins[r, c, 2]:+.3f})")
      if name == "B_blocks":
        X, Y = np.meshgrid(ox + ic, oy + jc, indexing="ij")
        h = height(X, Y)
        exp = T.block_cell_heights(cfg.size, d)
        b_cells[(r, c)] = h
        err = np.abs(h - exp).max()
        if d > 0:  # clip counts on the float64 cell array the mesh was built from (mesh == it to < 0.5 um, above)
          hi, lo = exp.max(), exp.min()
          f_up = np.mean(np.abs(exp - hi) < 1e-12)
          f_dn = np.mean(np.abs(exp - lo) < 1e-12)
        else:
          f_up = f_dn = 0.0
        stepx = np.abs(np.diff(h, axis=0))
        stepy = np.abs(np.diff(h, axis=1))
        smax = max(stepx.max(), stepy.max())
        smed = np.median(np.concatenate([stepx.ravel(), stepy.ravel()]))
        # wall sharpness: 0.25 mm either side of every x cell boundary must read the two cell heights
        xb = ox - LX / 2 + np.arange(1, gx) * CELL_M
        XB, YB = np.meshgrid(xb, oy + jc, indexing="ij")
        hm, hp = height(XB - 2.5e-4, YB), height(XB + 2.5e-4, YB)
        sharp = max(np.abs(hm - h[:-1]).max(), np.abs(hp - h[1:]).max())
        line += (f"\n           cells: mean {h.mean() * 1e6:+.2f} um, std {h.std() * 1e3:.3f} mm, |mesh - expected| "
                 f"max {err * 1e6:.2f} um, at +clip {100 * f_up:.2f}%, at -clip {100 * f_dn:.2f}%, max adjacent "
                 f"step {smax * 1e3:.3f} mm (median {smed * 1e3:.3f}), wall sharpness err @0.25mm "
                 f"{sharp * 1e6:.2f} um")
      elif name == "C_step":
        sx0, sx1 = T.C_STEP_X_RANGE
        xs_mid = ox + np.linspace(sx0 + 0.05, sx1 - 0.05, 50)
        hl = height(xs_mid, oy + 0.5)
        hr = height(xs_mid, oy - 0.5)
        xa = ox + np.linspace(-LX / 2 + 0.01, sx0 - 0.01, 50)
        xe = ox + np.linspace(sx1 + 0.01, LX / 2 - 0.01, 50)
        h_app = np.concatenate([height(xa, oy + 0.5), height(xa, oy - 0.5), height(xe, oy + 0.5), height(xe, oy - 0.5)])
        edge = height(ox + sx0 + np.array([-5e-4, 5e-4]), oy + 0.5)
        side = height(ox + 1.0, oy + np.array([-5e-4, 5e-4]))
        spawn = height(ox + T.C_SPAWN_OFFSET_XY[0] + np.array([-0.1, 0.1, 0.1, -0.1]),
                       oy + T.C_SPAWN_OFFSET_XY[1] + np.array([0.06, 0.06, -0.06, -0.06]))
        line += (f"\n           step: left {hl.mean() * 1e3:.3f} mm (range {np.ptp(hl) * 1e6:.2f} um), right "
                 f"{hr.mean() * 1e3:.3f} mm -> measured step {(hl.mean() - hr.mean()) * 1e3:.3f} mm (target "
                 f"{d * T.C_DELTA_M * 1e3:.3f}); approach/exit |z| max {np.abs(h_app).max() * 1e6:.2f} um; entry "
                 f"edge +-0.5mm {edge[0] * 1e3:.3f}->{edge[1] * 1e3:.3f} mm; side edge +-0.5mm "
                 f"{side[0] * 1e3:.3f}->{side[1] * 1e3:.3f} mm; spawn footprint z {np.abs(spawn).max() * 1e6:.2f} um")
      rows_out.append(line)
  for line in rows_out:
    log(line)
  log(f"sum of tile triangles {total_tile_tris}; faces with wrong orientation / non-axis normals: {bad_orient}")

  # B: same pattern at each level (up to scale), identical across B columns at a level
  bcols = [c for c in range(C) if T.COLUMN_TERRAIN_TYPES[c] == "B_blocks"]
  same_col = max(np.abs(b_cells[(r, c)] - b_cells[(r, bcols[0])]).max() for r in range(R) for c in bcols)
  ref = b_cells[(R - 1, bcols[0])]
  scale_err = max(np.abs(b_cells[(r, bcols[0])] - diff_of(r) * ref).max() for r in range(R))
  log(f"B identical across columns at a level: max diff {same_col * 1e6:.3f} um; level r == d_r * level_max: "
      f"max diff {scale_err * 1e6:.3f} um")

  # ---------------------------------------------------------- tile-edge continuity
  yq = (np.arange(gy) + 0.5) * CELL_M - LY / 2  # sample at cell centres (away from cell boundaries)
  xq = (np.arange(gx) + 0.5) * CELL_M - LX / 2
  eps = 1e-3
  log("tile edges along x (row r -> r+1, same column): max |z(+1mm) - z(-1mm)| in mm")
  for c in range(C):
    v = []
    for r in range(R - 1):
      xb = origins[r, c, 0] + LX / 2
      y = origins[r, c, 1] + yq
      v.append(np.abs(height(xb + eps, y) - height(xb - eps, y)).max() * 1e3)
    xb0 = origins[0, c, 0] - LX / 2
    xb1 = origins[R - 1, c, 0] + LX / 2
    y = origins[0, c, 1] + yq
    bd = max(np.abs(height(xb0 - eps, y) - height(xb0 + eps, y)).max(),
             np.abs(height(xb1 + eps, origins[R - 1, c, 1] + yq) - height(xb1 - eps, origins[R - 1, c, 1] + yq)).max())
    log(f"  col {c} {T.COLUMN_TERRAIN_TYPES[c]:8s}: " + " ".join(f"{a:6.3f}" for a in v) + f" | vs border {bd * 1e3:6.3f}")
  log("tile edges along y (col c -> c+1, same row): max |z(+1mm) - z(-1mm)| in mm")
  for c in range(C - 1):
    v = []
    for r in range(R):
      yb = origins[r, c, 1] + LY / 2
      x = origins[r, c, 0] + xq
      v.append(np.abs(height(x, yb + eps) - height(x, yb - eps)).max() * 1e3)
    log(f"  {T.COLUMN_TERRAIN_TYPES[c]:8s}|{T.COLUMN_TERRAIN_TYPES[c + 1]:8s} ({c}|{c + 1}): "
        + " ".join(f"{a:6.3f}" for a in v))
  # holes: any ray over the whole grid + border that misses?
  gxs = np.arange(-R * LX / 2 - T.BORDER_WIDTH_M + 0.005, R * LX / 2 + T.BORDER_WIDTH_M, 0.01)
  gys = np.arange(-C * LY / 2 - T.BORDER_WIDTH_M + 0.005, C * LY / 2 + T.BORDER_WIDTH_M, 0.01)
  GX, GY = np.meshgrid(gxs, gys, indexing="ij")
  HM = height(GX, GY)
  log(f"top view {HM.shape} rays at 10 mm: misses {int(np.sum(~np.isfinite(HM)))}, z range "
      f"[{np.nanmin(HM) * 1e3:+.3f}, {np.nanmax(HM) * 1e3:+.3f}] mm")

  # ---------------------------------------------------------- image
  fig, axs = plt.subplots(1, 3, figsize=(22, 8), gridspec_kw={"width_ratios": [2.2, 1, 1]})
  vmax = 1e3 * T.C_DELTA_M
  im = axs[0].imshow(1e3 * HM.T, origin="lower", extent=[gxs[0], gxs[-1], gys[0], gys[-1]], cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax, interpolation="nearest")
  for r in range(R):
    for c in range(C):
      o = origins[r, c]
      axs[0].text(o[0], o[1], f"{T.COLUMN_TERRAIN_TYPES[c][0]} d{diff_of(r):.2f}", ha="center", va="center", fontsize=7)
      if T.COLUMN_TERRAIN_TYPES[c] == "C_step":
        axs[0].plot(o[0] + T.C_SPAWN_OFFSET_XY[0], o[1] + T.C_SPAWN_OFFSET_XY[1], "k>", ms=5)
  axs[0].set_title("whole generator (rows = levels along x, columns = type along y); > = C spawn, heading +x")
  axs[0].set_xlabel("world x [m]")
  axs[0].set_ylabel("world y [m]")
  for ax, (r, c), ttl in ((axs[1], (R - 1, bcols[0]), "B_blocks d=1.0"),
                          (axs[2], (R - 1, [k for k in range(C) if T.COLUMN_TERRAIN_TYPES[k] == "C_step"][0]),
                           "C_step d=1.0")):
    o = origins[r, c]
    sel_x = np.abs(gxs - o[0]) <= LX / 2
    sel_y = np.abs(gys - o[1]) <= LY / 2
    ax.imshow(1e3 * HM[np.ix_(sel_x, sel_y)].T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
              extent=[-LX / 2, LX / 2, -LY / 2, LY / 2], interpolation="nearest")
    ax.set_title(f"{ttl} (tile frame, +y left)")
    ax.set_xlabel("x [m]")
  axs[2].plot(*T.C_SPAWN_OFFSET_XY, "k>", ms=8)
  fig.colorbar(im, ax=axs.tolist(), label="height [mm]", shrink=0.8)
  fig.savefig(out / "terrain_topview.png", dpi=110)
  log(f"saved {out / 'terrain_topview.png'}")

  # ---------------------------------------------------------- TerrainImporter + physics
  sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=dev))
  imp = TerrainImporter(TerrainImporterCfg(prim_path="/World/ground", terrain_type="generator",
                                           terrain_generator=T.MICROTAUR_TERRAINS_CFG.copy(), num_envs=args.num_envs,
                                           max_init_terrain_level=None))
  tt = imp.terrain_types
  ids = T.env_terrain_type_ids(imp)
  log(f"TerrainImporter: terrain_origins {tuple(imp.terrain_origins.shape)}, env_origins "
      f"{tuple(imp.env_origins.shape)}, max_terrain_level {imp.max_terrain_level}")
  log(f"  envs per column {torch.bincount(tt, minlength=C).tolist()}")
  log(f"  envs per type {dict(zip(TERRAIN_TYPES, torch.bincount(ids, minlength=3).tolist()))}")
  chk = (imp.env_origins - imp.terrain_origins[imp.terrain_levels, tt]).abs().max().item()
  log(f"  env_origins == terrain_origins[level, type column]: max diff {chk:.2e}; levels hist "
      f"{torch.bincount(imp.terrain_levels, minlength=R).tolist()}")
  tog = imp.terrain_origins.cpu().numpy()
  log(f"  importer origins == generator origins: max diff {np.abs(tog - origins).max():.2e}")

  if not args.no_physics:
    # foot-sized spheres (r = 6.2 mm) on the d = 1 B tile (cell centres) and the d = 1 C tile (both halves)
    rad = 0.0062
    rb, cb = R - 1, bcols[0]
    cc = [k for k in range(C) if T.COLUMN_TERRAIN_TYPES[k] == "C_step"][0]
    rng = np.random.default_rng(1)
    pts, expz = [], []
    hB = b_cells[(rb, cb)]
    for _ in range(48):
      i, j = rng.integers(0, gx), rng.integers(0, gy)
      pts.append((origins[rb, cb, 0] + ic[i], origins[rb, cb, 1] + jc[j]))
      expz.append(hB[i, j])
    for k in range(16):
      x = origins[R - 1, cc, 0] + T.C_STEP_X_RANGE[0] + 0.2 + 0.2 * (k // 2)
      y = origins[R - 1, cc, 1] + (0.3 if k % 2 == 0 else -0.3)
      pts.append((x, y))
      expz.append(T.C_DELTA_M if k % 2 == 0 else 0.0)
    for k, (x, y) in enumerate(pts):
      prim_utils.create_prim(f"/World/Ball{k:03d}", "Xform", translation=(x, y, expz[k] + rad + 0.01))
    balls = RigidObject(RigidObjectCfg(
      prim_path="/World/Ball.*/Sphere",
      spawn=sim_utils.SphereCfg(radius=rad, rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                                mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
                                collision_props=sim_utils.CollisionPropertiesCfg()),
      init_state=RigidObjectCfg.InitialStateCfg()))
    sim.reset()
    z_first = None
    for k in range(300):
      sim.step(render=False)
      balls.update(sim.get_physics_dt())
      if k == 0:
        z_first = balls.data.root_pos_w[:, 2].cpu().numpy().copy()
    log(f"PhysX spheres: height above rest after 1 step {1e3 * (z_first - rad - np.array(expz)).mean():.3f} mm")
    pz = balls.data.root_pos_w[:, 2].cpu().numpy()
    pxy = balls.data.root_pos_w[:, :2].cpu().numpy()
    drift = np.linalg.norm(pxy - np.array(pts), axis=1)
    e = pz - rad - np.array(expz)
    vz = balls.data.root_lin_vel_w[:, 2].abs().max().item()
    log(f"PhysX spheres dropped from 10 mm, after 1.5 s (|vz| max {vz:.2e} m/s): B (48) rest height - cell top: "
        f"mean {e[:48].mean() * 1e6:+.2f} um, max |.| {np.abs(e[:48]).max() * 1e6:.2f} um, xy drift max "
        f"{drift[:48].max() * 1e6:.2f} um")
    log(f"                           C (16) rest height - surface: mean {e[48:].mean() * 1e6:+.2f} um, "
        f"max |.| {np.abs(e[48:]).max() * 1e6:.2f} um, xy drift max {drift[48:].max() * 1e6:.2f} um; "
        f"left z {np.mean(pz[48::2] - rad) * 1e3:.3f} mm, right z {np.mean(pz[49::2] - rad) * 1e3:.3f} mm")
  log("DONE")
except Exception:
  import traceback

  log(traceback.format_exc())
finally:
  LOG.flush()
  LOG.close()
  sys.stdout.flush()
  os._exit(0)
