"""Render an IsaacLab trajectory (scripts/isaac_eval_gait.py traj_<v>.npz) with MuJoCo.

  DISPLAY=:1 MUJOCO_GL=glfw python scripts/isaac_render_traj.py --traj <dir>/traj_0.20.npz [--seconds 4]

The rigid Microtaur MJCF has the same joint names as the USD, so the recorded
root pose and 16 joint angles are written to qpos by name and mj_forward places
every body (kinematic replay, no MuJoCo dynamics). A floor plane and a light are
added for the picture. Writes <traj>.mp4 (real time) and <traj>_frames.png
(24 side-view frames over ~1.4 s).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from microtaur_common.robot_constants import ROOT_BODY, XML_PATH  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True)
ap.add_argument("--seconds", type=float, default=4.0)
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=400)
args = ap.parse_args()

tr = np.load(args.traj)
dt = float(tr["dt"])
names = [str(n) for n in tr["joint_names"]]

spec = mujoco.MjSpec.from_file(str(XML_PATH))
floor = spec.worldbody.add_geom()
floor.type = mujoco.mjtGeom.mjGEOM_PLANE
floor.size = [5.0, 5.0, 0.01]
floor.rgba = [0.82, 0.84, 0.86, 1.0]
light = spec.worldbody.add_light()
light.pos = [0.0, -1.0, 2.0]
light.dir = [0.0, 0.5, -1.0]
spec.visual.global_.offwidth = max(args.width, 640)
spec.visual.global_.offheight = max(args.height, 480)
m = spec.compile()
d = mujoco.MjData(m)
free_adr = int(m.jnt_qposadr[int(m.body_jntadr[m.body(ROOT_BODY).id])])  # the root's free joint
adr = [int(m.jnt_qposadr[m.joint(n).id]) for n in names]

renderer = mujoco.Renderer(m, args.height, args.width)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.elevation, cam.azimuth = 0.45, -10.0, 90.0

T = min(len(tr["root_pos"]), int(args.seconds / dt))
frames = []
for t in range(T):
  d.qpos[free_adr:free_adr + 3] = tr["root_pos"][t]
  d.qpos[free_adr + 3:free_adr + 7] = tr["root_quat_wxyz"][t]
  d.qpos[adr] = tr["joint_pos"][t]
  mujoco.mj_forward(m, d)
  cam.lookat[:] = [d.qpos[free_adr], d.qpos[free_adr + 1], 0.05]
  renderer.update_scene(d, camera=cam)
  frames.append(renderer.render().copy())

stem = Path(args.traj).with_suffix("")
imageio.mimsave(f"{stem}.mp4", frames, fps=round(1.0 / dt))
strip = frames[: min(len(frames), 48): 2]
cols = 6
rows = math.ceil(len(strip) / cols)
fig, axs = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.2))
for i, ax in enumerate(np.ravel(axs)):
  ax.axis("off")
  if i < len(strip):
    ax.imshow(strip[i]); ax.set_title(f"t = {i * 2 * dt:.2f} s", fontsize=8)
yaw = float(tr["cmd_yaw_rad_s"]) if "cmd_yaw_rad_s" in tr else 0.0
fig.suptitle(f"IsaacLab rollout replayed in MuJoCo, cmd {float(tr['cmd_m_s']):.2f} m/s, yaw {yaw:+.2f} rad/s")
fig.savefig(f"{stem}_frames.png", dpi=90, bbox_inches="tight")
print("wrote", f"{stem}.mp4", f"{stem}_frames.png")
