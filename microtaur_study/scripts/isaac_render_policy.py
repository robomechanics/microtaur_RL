"""Render a policy (teacher or student) walking the Teacher-Cur terrain, with a steady follow camera.

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_render_policy.py --checkpoint <model.pt> --env 9 --out C_lvl6.mp4 \
      [--steps 420] [--speed 0.20] [--eye -0.45 -0.18 0.32] [--lookat 0.25 0 0] [--still-eye ...] [--no-rails]

10 envs, env i on terrain column i (0-1 A, 2-6 B, 7-9 C) at the level in LEVELS (B env 6 = run-out row).
The camera follows the robot in xy with a ~0.2 s low-pass and a fixed height (terrain + nominal root
height), so the trot's vertical bob shows against a still background. Prints VIDEO_OK, resets, final pose.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--env", type=int, required=True)
ap.add_argument("--steps", type=int, default=420)
ap.add_argument("--speed", type=float, default=0.20)
ap.add_argument("--eye", type=float, nargs=3, default=(0.30, -0.45, 0.18))
ap.add_argument("--lookat", type=float, nargs=3, default=(0.0, 0.0, 0.04))
ap.add_argument("--no-rails", action="store_true")
ap.add_argument("--still-eye", type=float, nargs=3, default=None, help="extra still frame, camera relative to the robot")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from microtaur_isaac.env_cfg import MicrotaurTeacherCurPlayEnvCfg  # noqa: E402
from microtaur_isaac.policy_loader import load_policy, prepare_env_cfg, variant_kwargs  # noqa: E402

LEVELS = [0, 3, 0, 2, 4, 6, 7, 0, 3, 6]  # per env / column
cfg = MicrotaurTeacherCurPlayEnvCfg(**variant_kwargs(args.checkpoint))
cfg.scene.num_envs = 10
if args.no_rails:
  g = cfg.scene.terrain.terrain_generator
  subs = dict(g.sub_terrains)
  subs["C_step"] = subs["C_step"].replace(lane_half_width_m=None)
  g.sub_terrains = subs
cfg.curriculum.command_ranges = None
cfg.commands.twist.resampling_time_range = (1e9, 1e9)
cfg.viewer.env_index = args.env
cfg.viewer.resolution = (960, 540)
cfg.scene.sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=900.0, color=(0.9, 0.9, 0.95)))
prepare_env_cfg(cfg, args.checkpoint)
env = ManagerBasedRLEnv(cfg, render_mode="rgb_array")
t = env.scene.terrain
t.terrain_levels[:] = torch.tensor(LEVELS, device=env.device)
t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]
w = RslRlVecEnvWrapper(env)
policy, policy_reset = load_policy(w, args.checkpoint)
cmd = env.command_manager.get_term("twist")


def pin():
  cmd.vel_command_b[:, 0] = args.speed
  cmd.vel_command_b[:, 1:] = 0.0


pin()
obs = w.get_observations()
cc = env.viewport_camera_controller
cc.update_view_to_world()
eye_off, look_off = np.array(args.eye), np.array(args.lookat)
robot = env.scene["robot"]
tgt = robot.data.root_link_pos_w[args.env].cpu().numpy().copy()
tgt[2] = float(t.env_origins[args.env, 2]) + 0.07


def follow():
  p = robot.data.root_link_pos_w[args.env].cpu().numpy()
  if np.linalg.norm(p[:2] - tgt[:2]) > 0.5:  # after a reset
    tgt[:2] = p[:2]
  tgt[:2] += 0.15 * (p[:2] - tgt[:2])
  cc.update_view_location(eye=tgt + eye_off, lookat=tgt + look_off)


follow()
frames, resets = [], {}
with torch.inference_mode():
  for k in range(args.steps):
    pin()
    obs, _, dones, _ = w.step(policy(obs))
    policy_reset(dones)
    if bool(dones[args.env]):
      tm = env.termination_manager
      c = next((n for n in tm.active_terms if bool(tm.get_term(n)[args.env])), "?")
      resets[c] = resets.get(c, 0) + 1
    if k == 5 and args.still_eye is not None:
      cc.update_view_location(eye=tgt + np.array(args.still_eye), lookat=tgt + np.array((0.5, 0.0, 0.0)))
      for _ in range(2):
        env.sim.render()
      imageio.imwrite(args.out.replace(".mp4", "_still.png"), env.render())
    follow()
    frames.append(env.render())
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
imageio.mimsave(args.out, frames, fps=round(1.0 / env.step_dt))
r = robot.data.root_link_pos_w[args.env] - t.env_origins[args.env]
print("VIDEO_OK", args.out, "resets", resets, "final root rel origin", [round(float(v), 3) for v in r], flush=True)
sys.stdout.flush()
os._exit(0)
