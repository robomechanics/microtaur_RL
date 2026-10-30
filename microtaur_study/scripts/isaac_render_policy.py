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
ap.add_argument("--tour", action="store_true",
                help="turn tour instead of straight --speed: TOUR segments (vx, yaw rate, s); frames carry the command, the "
                     "measured yaw rate and (yaw robot) the spine angle")
ap.add_argument("--schedule", default=None,
                help='custom command segments "vx,wz,seconds;vx,wz,seconds;..." (implies the --tour overlay), e.g. a circle '
                     '"0.10,0.25,30"')
ap.add_argument("--fixed-cam", action="store_true", help="camera stays where it starts (eye / lookat relative to the spawn)")
ap.add_argument("--trace", action="store_true",
                help="also save <out>.npz per policy step: time, command, yaw rate, root xy / yaw, spine angle, foot contacts")
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
# straight, left, straight, right, straight; +-0.25 rad/s is the trained yaw-rate range (86 deg per 6 s turn)
TOUR = [(0.20, 0.0, 3.0), (0.15, 0.25, 6.0), (0.20, 0.0, 2.0), (0.15, -0.25, 6.0), (0.20, 0.0, 3.0)]
if args.schedule:
  TOUR = [tuple(float(v) for v in seg.split(",")) for seg in args.schedule.split(";")]
  args.tour = True
cfg = MicrotaurTeacherCurPlayEnvCfg(**variant_kwargs(args.checkpoint))
cfg.scene.num_envs = 10
cfg.sim.device = args.device  # --device cpu: physics on the CPU when the shared GPU is full (rendering stays on the GPU)
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


def command(k):
  if not args.tour:
    return args.speed, 0.0
  t_s = k * env.step_dt
  for vx, wz, dur in TOUR:
    if t_s < dur:
      return vx, wz
    t_s -= dur
  return TOUR[-1][0], TOUR[-1][1]


def pin(k=0):
  vx, wz = command(k)
  cmd.vel_command_b[:, 0] = vx
  cmd.vel_command_b[:, 1] = 0.0
  cmd.vel_command_b[:, 2] = wz


if args.tour:
  args.steps = round(sum(d for _, _, d in TOUR) / env.step_dt)


pin()
obs = w.get_observations()
cc = env.viewport_camera_controller
cc.update_view_to_world()
eye_off, look_off = np.array(args.eye), np.array(args.lookat)
robot = env.scene["robot"]
tgt = robot.data.root_link_pos_w[args.env].cpu().numpy().copy()
tgt[2] = float(t.env_origins[args.env, 2]) + 0.07


def follow():
  if args.fixed_cam:
    cc.update_view_location(eye=tgt + eye_off, lookat=tgt + look_off)
    return
  p = robot.data.root_link_pos_w[args.env].cpu().numpy()
  if np.linalg.norm(p[:2] - tgt[:2]) > 0.5:  # after a reset
    tgt[:2] = p[:2]
  tgt[:2] += 0.15 * (p[:2] - tgt[:2])
  cc.update_view_location(eye=tgt + eye_off, lookat=tgt + look_off)


follow()
frames, resets = [], {}
spine = robot.joint_names.index("spine_joint_act") if "spine_joint_act" in robot.joint_names else None
trace = {k: [] for k in ("t", "cmd_vx", "cmd_wz", "wz", "vx", "x", "y", "yaw", "spine", "contact")}
if args.trace:
  from microtaur_isaac.mdp.contact import foot_contact_timers
  from microtaur_isaac.mdp.observations import feet_sensor_cfg
  fsc = feet_sensor_cfg()
  fsc.resolve(env.scene)
with torch.inference_mode():
  for k in range(args.steps):
    pin(k)
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
    if args.trace:
      d, e = robot.data, args.env
      q = d.root_link_quat_w[e]
      for key, v in (("t", k * env.step_dt), ("cmd_vx", command(k)[0]), ("cmd_wz", command(k)[1]),
                     ("wz", float(d.root_link_ang_vel_b[e, 2])), ("vx", float(d.root_link_lin_vel_b[e, 0])),
                     ("x", float(d.root_link_pos_w[e, 0] - t.env_origins[e, 0])), ("y", float(d.root_link_pos_w[e, 1] - t.env_origins[e, 1])),
                     ("yaw", float(torch.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))),
                     ("spine", float(d.joint_pos[e, spine]) if spine is not None else 0.0),
                     ("contact", foot_contact_timers(env, fsc)[0][e].cpu().numpy().astype(float))):
        trace[key].append(v)
    img = env.render()
    if args.tour:
      from PIL import Image, ImageDraw, ImageFont
      vx, wz = command(k)
      txt = f"t {k * env.step_dt:4.1f} s  cmd vx {vx:.2f} m/s  yaw {wz:+.2f} rad/s | yaw rate {float(robot.data.root_link_ang_vel_b[args.env, 2]):+.2f}"
      if spine is not None:
        txt += f"  spine {np.degrees(float(robot.data.joint_pos[args.env, spine])):+5.1f} deg"
      im = Image.fromarray(img)
      font, dr = ImageFont.load_default(size=20), ImageDraw.Draw(im)
      x0, y0, x1, y1 = dr.textbbox((12, 12), txt, font=font)
      dr.rectangle((x0 - 6, y0 - 4, x1 + 6, y1 + 4), fill=(0, 0, 0))  # readable on the light floor
      dr.text((12, 12), txt, fill=(255, 255, 255), font=font)
      img = np.asarray(im)
    frames.append(img)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
imageio.mimsave(args.out, frames, fps=round(1.0 / env.step_dt))
if args.trace:
  np.savez(args.out.replace(".mp4", ".npz"), **{k: np.asarray(v) for k, v in trace.items()})
r = robot.data.root_link_pos_w[args.env] - t.env_origins[args.env]
print("VIDEO_OK", args.out, "resets", resets, "final root rel origin", [round(float(v), 3) for v in r], flush=True)
sys.stdout.flush()
os._exit(0)
