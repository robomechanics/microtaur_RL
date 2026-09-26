"""Is the IsaacLab robot posed like the MJCF? Body poses vs MuJoCo FK, and side-view pictures.

  OMNI_KIT_ACCEPT_EULA=YES DISPLAY=:1 MUJOCO_GL=glfw python tests/check_isaac_pose.py --out <dir>

One Play env (visual USD). At reset (nominal stand) and after 30 zero-action
steps: every body's world pose in PhysX is compared with MuJoCo forward
kinematics of the MJCF at the same root pose and 16 joint angles, and both are
rendered from the same side view (isaac_<tag>.png, mujoco_<tag>.png).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
LOG = open(out / "pose.log", "w")


def log(*a):
  LOG.write(" ".join(map(str, a)) + "\n"); LOG.flush()


import imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from microtaur_common.robot_constants import ROOT_BODY, XML_PATH  # noqa: E402
from microtaur_isaac.env_cfg import MicrotaurFlatPlayEnvCfg  # noqa: E402

cfg = MicrotaurFlatPlayEnvCfg()
cfg.scene.num_envs = 1
cfg.viewer.eye = (0.0, -0.45, 0.05)
cfg.viewer.lookat = (0.0, 0.0, 0.0)
cfg.viewer.resolution = (960, 600)
env = ManagerBasedRLEnv(cfg, render_mode="rgb_array")
robot = env.scene["robot"]

spec = mujoco.MjSpec.from_file(str(XML_PATH))
floor = spec.worldbody.add_geom()
floor.type = mujoco.mjtGeom.mjGEOM_PLANE
floor.size = [5.0, 5.0, 0.01]
floor.rgba = [0.3, 0.3, 0.3, 1.0]
light = spec.worldbody.add_light()
light.pos = [0.0, -1.0, 2.0]
light.dir = [0.0, 0.5, -1.0]
spec.visual.global_.offwidth, spec.visual.global_.offheight = 960, 600
m = spec.compile()
d = mujoco.MjData(m)
free_adr = int(m.jnt_qposadr[int(m.body_jntadr[m.body(ROOT_BODY).id])])
renderer = mujoco.Renderer(m, 600, 960)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance, cam.elevation, cam.azimuth = 0.45, -6.4, 90.0


def compare(tag: str):
  origin = env.scene.env_origins[0]
  root = robot.data.root_link_pose_w[0].cpu().numpy()
  root[:3] -= origin.cpu().numpy()
  q = robot.data.joint_pos[0].cpu().numpy()
  d.qpos[free_adr:free_adr + 3] = root[:3]
  d.qpos[free_adr + 3:free_adr + 7] = root[3:7]
  for name, v in zip(robot.joint_names, q):
    d.qpos[m.jnt_qposadr[m.joint(name).id]] = v
  mujoco.mj_kinematics(m, d)
  poses = robot.data.body_link_pose_w[0].cpu().numpy()
  worst_p, worst_r = 0.0, 0.0
  for i, name in enumerate(robot.body_names):
    b = m.body(name).id
    dp = np.linalg.norm(poses[i, :3] - origin.cpu().numpy() - d.xpos[b])
    dq = 2 * np.degrees(np.arccos(min(1.0, abs(float(np.dot(poses[i, 3:7], d.xquat[b]))))))
    worst_p, worst_r = max(worst_p, dp), max(worst_r, dq)
  log(f"[{tag}] root z {1e3 * root[2]:.2f} mm; joints " + ", ".join(f"{n} {v:+.3f}" for n, v in zip(robot.joint_names, q)))
  log(f"[{tag}] PhysX body poses vs MuJoCo FK at the same q: max position diff {1e3 * worst_p:.4f} mm, max rotation diff {worst_r:.4f} deg")
  for _ in range(3):
    env.sim.render()
  imageio.imwrite(out / f"isaac_{tag}.png", env.render())
  cam.lookat[:] = [d.qpos[free_adr], d.qpos[free_adr + 1], d.qpos[free_adr + 2]]
  renderer.update_scene(d, camera=cam)
  imageio.imwrite(out / f"mujoco_{tag}.png", renderer.render())


env.reset()
compare("reset")
zero = torch.zeros(1, 8, device=env.device)
for _ in range(30):
  env.step(zero)
compare("settled")
LOG.close()
os._exit(0)
