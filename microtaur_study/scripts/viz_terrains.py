"""Render the three teacher terrains (A flat, B Gaussian blocks, C flat-step-flat).

  DISPLAY=:1 MUJOCO_GL=glfw python scripts/viz_terrains.py --out figures/

Writes
  terrains_overview.png  top views, cross-sections, B height histogram, levels
  render_B.png, render_C.png  the robot settled on each terrain (MuJoCo)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from microtaur_rigid import terrain_maps as T  # noqa: E402
from microtaur_rigid.kinematics import MicrotaurFiveBarKinematics  # noqa: E402
from microtaur_rigid.robot import LEG_JOINT_NAMES, STAND_A, STAND_E, XML_PATH  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="figures")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

BODY = (0.170, 0.105)  # fore-aft x lateral foot spacing, m
NX_B, NY_B = 1600, 1600  # 4 x 4 m at 2.5 mm
NX_C, NY_C = 2000, 800  # 5 x 2 m at 2.5 mm
B = T.gaussian_blocks(NX_B, NY_B, difficulty=1.0, seed=0)
C = T.flat_step_flat(NX_C, NY_C, difficulty=1.0)
A = T.flat(NX_B, NY_B)


def extent(nx, ny):
  return [0, nx * T.DX_M, -ny * T.DX_M / 2, ny * T.DX_M / 2]


def footprint(ax, x, y):
  ax.add_patch(plt.Rectangle((x - BODY[0] / 2, y - BODY[1] / 2), *BODY, fill=False, ec="k", lw=1.5))


# ---------------------------------------------------------------- overview
fig, axs = plt.subplots(3, 3, figsize=(16, 13))
vmax = 1e3 * max(T.UP_MAX_M, T.DOWN_MAX_M)
for ax, (name, z, nx, ny) in zip(axs[0], (("A flat", A, NX_B, NY_B), ("B Gaussian blocks (100%)", B, NX_B, NY_B),
                                          ("C low flat - left raised - low flat (100%)", C, NX_C, NY_C))):
  im = ax.imshow(1e3 * z.T, origin="lower", extent=extent(nx, ny), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
  footprint(ax, 0.3 if name.startswith("C") else 2.0, 0.0)
  ax.set_title(f"{name}\n(box = robot foot spacing 170 x 105 mm)")
  ax.set_xlabel("x, direction of travel [m]")
  ax.set_ylabel("y, lateral [m] (+y = left)")
fig.colorbar(im, ax=axs[0].tolist(), label="height [mm]", shrink=0.8)

# Zoomed B with cell grid
ax = axs[1][0]
zoom = B[600:1000, 600:1000]
ax.imshow(1e3 * zoom.T, origin="lower", extent=[1.5, 2.5, -0.5, 0.5], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
footprint(ax, 2.0, 0.0)
ax.set_title("B zoom 1 x 1 m: 70 mm flat-topped cells")
ax.set_xlabel("x [m]")

# Cross-sections
ax = axs[1][1]
x = np.arange(NX_B) * T.DX_M
ax.plot(x, 1e3 * B[:, NY_B // 2], lw=1)
ax.set_xlim(1.5, 2.5)
ax.axhline(1e3 * T.UP_MAX_M, ls="--", c="r", lw=0.8, label="+13.1 mm (up ceiling)")
ax.axhline(-1e3 * T.DOWN_MAX_M, ls="--", c="b", lw=0.8, label="-16.4 mm (down ceiling)")
ax.set_title("B cross-section along x (1 m): flat tops, vertical edges")
ax.set_xlabel("x [m]")
ax.set_ylabel("height [mm]")
ax.legend(fontsize=8)

ax = axs[1][2]
xc = np.arange(NX_C) * T.DX_M
ax.plot(xc, 1e3 * C[:, NY_C * 3 // 4], label="left half (y > 0)")
ax.plot(xc, 1e3 * C[:, NY_C // 4], label="right half (y < 0)")
ax.set_title("C along x: flat 0.5 m | step 4 m | flat 0.5 m")
ax.set_xlabel("x [m]")
ax.set_ylabel("height [mm]")
ax.legend(fontsize=8)

# B height distribution
ax = axs[2][0]
c = int(round(T.CELL_M / T.DX_M))
cells = 1e3 * B[::c, ::c].ravel()
ax.hist(cells, bins=60, color="0.5")
ax.axvline(1e3 * T.UP_MAX_M, ls="--", c="r")
ax.axvline(-1e3 * T.DOWN_MAX_M, ls="--", c="b")
clipped_up = np.mean(cells >= 1e3 * T.UP_MAX_M - 1e-6 + 1e3 * (B.mean()))
ax.set_title(f"B cell heights: Gaussian sigma = {1e3 * T.SIGMA_M:.2f} mm, clipped\n"
             f"{len(cells)} cells, std {cells.std():.2f} mm, "
             f"{100 * np.mean(cells >= cells.max() - 1e-3):.1f}% at the up clip")
ax.set_xlabel("cell height [mm]")

# Difficulty levels
ax = axs[2][1]
for d in (0.25, 0.5, 0.75, 1.0):
  ax.plot(x, 1e3 * T.gaussian_blocks(NX_B, NY_B, difficulty=d, seed=0)[:, NY_B // 2], lw=1, label=f"{int(100 * d)}%")
ax.set_xlim(1.5, 2.5)
ax.set_title("B levels: same map, heights scaled")
ax.set_xlabel("x [m]")
ax.legend(fontsize=8)

ax = axs[2][2]
for d in (0.25, 0.5, 0.75, 1.0):
  Cd = T.flat_step_flat(NX_C, NY_C, difficulty=d)
  ax.plot(np.arange(NY_C) * T.DX_M - NY_C * T.DX_M / 2, 1e3 * Cd[NX_C // 2], label=f"{int(100 * d)}%")
ax.set_title("C across y at mid-step: left raised, right stays low")
ax.set_xlabel("y [m]")
ax.legend(fontsize=8)

fig.savefig(out / "terrains_overview.png", dpi=110, bbox_inches="tight")
print("wrote", out / "terrains_overview.png")


# ---------------------------------------------------------------- MuJoCo renders
KIN = MicrotaurFiveBarKinematics()
FULL_Q = np.concatenate([KIN.forward_numpy(STAND_A[i], STAND_E[i], leg_index=i + 1).full for i in range(4)])
FULL_NAMES = [f"leg{i}_{j}" for i in range(1, 5) for j in ("a_joint_act", "b_joint", "e_joint_act", "d_joint")]


def render(z: np.ndarray, spawn_xy: tuple[float, float], name: str, cam: dict):
  spec = mujoco.MjSpec.from_file(str(XML_PATH))
  nx, ny = z.shape
  zmin, zmax = float(z.min()), float(z.max())
  span = max(zmax - zmin, 1e-4)
  hf = spec.add_hfield(name="terrain", nrow=ny, ncol=nx,
                       size=[nx * T.DX_M / 2, ny * T.DX_M / 2, span, 0.01])
  hf.userdata = ((z.T - zmin) / span).ravel().tolist()  # MuJoCo rows are y, columns x
  spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="terrain",
                          pos=[nx * T.DX_M / 2, 0, zmin], rgba=[0.75, 0.72, 0.65, 1])
  spec.worldbody.add_light(pos=[spawn_xy[0], spawn_xy[1], 1.5], dir=[0, 0, -1], castshadow=0)
  spec.visual.global_.offwidth, spec.visual.global_.offheight = 1080, 720
  m = spec.compile()
  m.opt.timestep = 0.005
  d = mujoco.MjData(m)
  ix, iy = int(spawn_xy[0] / T.DX_M), int(spawn_xy[1] / T.DX_M + ny / 2)
  hx, hy = int(0.09 / T.DX_M), int(0.06 / T.DX_M)
  ground = float(z[ix - hx: ix + hx, iy - hy: iy + hy].max())
  d.qpos[0:7] = (spawn_xy[0], spawn_xy[1], ground + 0.075, 1, 0, 0, 0)
  qadr = [m.joint(n).qposadr[0] for n in FULL_NAMES]
  d.qpos[qadr] = FULL_Q
  motors = [m.joint(n) for n in LEG_JOINT_NAMES]
  target = np.array([FULL_Q[FULL_NAMES.index(n)] for n in LEG_JOINT_NAMES])
  for _ in range(200):  # settle 1 s under a stand PD (kp 1.0, torque cap 0.129)
    q = np.array([d.qpos[j.qposadr[0]] for j in motors])
    qd = np.array([d.qvel[j.dofadr[0]] for j in motors])
    d.qfrc_applied[:] = 0
    d.qfrc_applied[[j.dofadr[0] for j in motors]] = np.clip(1.0 * (target - q) - 0.02 * qd, -0.129, 0.129)
    mujoco.mj_step(m, d)
  r = mujoco.Renderer(m, 720, 1080)
  c = mujoco.MjvCamera()
  c.lookat[:] = d.qpos[0:3]
  c.distance, c.azimuth, c.elevation = cam["distance"], cam["azimuth"], cam["elevation"]
  opt = mujoco.MjvOption()
  r.update_scene(d, camera=c, scene_option=opt)
  plt.imsave(out / f"render_{name}.png", r.render())
  feet = {f"leg{i}": d.site_xpos[m.site(f"leg{i}_foot_site").id] for i in range(1, 5)}
  desc = "  ".join(f"{k}(x {p[0]:.2f}, y {p[1]:+.3f}) z {1e3 * (p[2] - 0.0062):+5.1f}mm" for k, p in feet.items())
  print("wrote", out / f"render_{name}.png", f"root z {d.qpos[2]:.4f} m | foot-bottom heights: {desc}")


render(B[400:1200, 400:1200], (1.0, 0.0), "B", {"distance": 0.75, "azimuth": 140, "elevation": -28})
# C at three points along the course: still on the approach, front feet just
# onto the step, fully on the step. Left (+y) half is high, right half low.
for label, x in (("C_approach", 0.35), ("C_entry", 0.55), ("C_mid", 2.5)):
  render(C, (x, 0.0), label, {"distance": 0.55, "azimuth": 180, "elevation": -12})
