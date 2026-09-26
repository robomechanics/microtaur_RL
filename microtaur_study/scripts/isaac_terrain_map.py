"""Top-down height map of a Microtaur terrain grid, ray-cast from the generated mesh (warp).

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_terrain_map.py --out <png> [--cur] [--scale 2.0] [--res 0.01]

Left: the whole grid (rows = levels along x, columns = A / B / C along y), run-out
rows labelled. Right: C lane cross-sections at mid-step for every level, and the
height profile along a B tile's centre line at the top level and its run-out row.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--cur", action="store_true", help="Teacher-Cur terrain (7 levels, run-out row, railed C)")
ap.add_argument("--scale", type=float, default=2.0)
ap.add_argument("--res", type=float, default=0.01)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.terrains import TerrainGenerator  # noqa: E402
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh  # noqa: E402

from microtaur_isaac import terrains as TR  # noqa: E402

cfg = TR.curriculum_terrains_cfg(args.scale) if args.cur else TR.scaled_terrains_cfg(args.scale)
gen = TerrainGenerator(cfg, device="cpu")
mesh = gen.terrain_mesh
dev = "cuda"
wm = convert_to_warp_mesh(np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32), device=dev)
lx, ly = cfg.size
R, Cn = cfg.num_rows, cfg.num_cols
x0, y0 = -0.5 * R * lx, -0.5 * Cn * ly  # grid is centred on the world origin


def cast(xs, ys):
  X, Y = np.meshgrid(xs, ys, indexing="ij")
  starts = torch.tensor(np.stack([X.ravel(), Y.ravel(), np.full(X.size, 1.0)], 1), dtype=torch.float32, device=dev)
  dirs = torch.zeros_like(starts); dirs[:, 2] = -1.0
  hits = raycast_mesh(starts[None], dirs[None], wm, max_dist=3.0)[0][0]
  return hits[:, 2].reshape(X.shape).cpu().numpy()


r = args.res
xs = np.arange(x0 + 0.5 * r, x0 + R * lx, r)
ys = np.arange(y0 + 0.5 * r, y0 + Cn * ly, r)
H = cast(xs, ys)
n_lv = int(cfg.sub_terrains["B_blocks"].num_levels or R)
types = TR.column_terrain_types(cfg)

fig = plt.figure(figsize=(16, 10))
ax = fig.add_axes([0.04, 0.06, 0.50, 0.88])
im = ax.imshow(H.T * 1e3, origin="lower", extent=[x0, x0 + R * lx, y0, y0 + Cn * ly], cmap="terrain",
               vmin=-20, vmax=max(40.0, float(np.nanpercentile(H, 99.9)) * 1e3), aspect="equal", interpolation="nearest")
fig.colorbar(im, ax=ax, fraction=0.03, label="height (mm)")
for i in range(R):
  lab = f"lvl {i}" if i < n_lv else f"run-out\n(= lvl {n_lv - 1})"
  ax.text(x0 + (i + 0.5) * lx, y0 + Cn * ly + 0.3, lab, ha="center", va="bottom", fontsize=8)
for j, t in enumerate(types):
  ax.text(x0 - 0.3, y0 + (j + 0.5) * ly, t.split("_")[0], ha="right", va="center", fontsize=9)
ax.set_xlabel("x (m) -> walking direction on C"); ax.set_ylabel("y (m)")
ax.set_title(f"{'Teacher-Cur' if args.cur else 'terrain'} x{args.scale}: {n_lv} levels + {R - n_lv} run-out row(s)", pad=30)

# C lane cross-sections (first C column) at mid-step, per level
cc = types.index("C_step")
yc = y0 + (cc + 0.5) * ly
yy = np.arange(yc - 0.25, yc + 0.25, 0.002)
xmid = 0.5 * (TR.C_STEP_X_RANGE[0] + TR.C_STEP_X_RANGE[1])
ax2 = fig.add_axes([0.62, 0.56, 0.35, 0.36])
for i in range(R):
  h = cast(np.array([x0 + (i + 0.5) * lx + xmid]), yy)[0]
  ax2.plot((yy - yc) * 1e3, h * 1e3, label=f"lvl {i}" if i < n_lv else "run-out", lw=1.2)
for s in (-1, 1):
  ax2.axvline(s * 50, color="k", ls=":", lw=0.8)
  ax2.axvline(s * 75, color="r", ls=":", lw=0.8)
ax2.set_xlabel("y from step edge (mm); dotted: stance feet +/-50, red: body box +/-75")
ax2.set_ylabel("z (mm)"); ax2.set_title("C lane cross-section at mid-step"); ax2.legend(fontsize=7, ncol=2)
ax2.set_ylim(-5, 130)

# B profile along the tile centre line through the top level and the run-out row
bc = types.index("B_blocks") + 2
yb = y0 + (bc + 0.5) * ly
xa = np.arange(x0 + (n_lv - 2) * lx, x0 + R * lx + 1.5, 0.005)
hb = cast(xa, np.array([yb]))[:, 0]
ax3 = fig.add_axes([0.62, 0.08, 0.35, 0.36])
ax3.plot(xa, hb * 1e3, lw=0.8)
for i in range(n_lv - 2, R + 1):
  ax3.axvline(x0 + i * lx, color="k", ls="--", lw=0.8)
ax3.set_xlabel("x (m): levels %d, %d, run-out, border" % (n_lv - 2, n_lv - 1)); ax3.set_ylabel("z (mm)")
ax3.set_title(f"B centre line (column {bc})")
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=110)
print("MAP_OK", args.out, flush=True)
import os  # noqa: E402

os._exit(0)
