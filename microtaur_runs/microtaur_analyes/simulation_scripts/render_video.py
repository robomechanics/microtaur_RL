"""Render an MP4 of a trained Microtaur checkpoint walking, with a gait HUD.

    python scripts/render_video.py --variant pitch --vx 0.15 --seconds 20
    python scripts/render_video.py --variant pitch --robust --out pitch_robust.mp4

Writes rollouts/videos/<name>.mp4. The HUD burned into each frame shows the
command, the achieved body velocity, which feet are in stance, and the running
trot/pace classification from the diagonal-vs-lateral contact correlation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["MICROTAUR_USE_JOYSTICK_COMMANDS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from microtaur_variants import VARIANTS, use_variant  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = Path("rollouts/videos")


def draw_feet(img, found, corner):
  """Plan-view stance indicator: four pads, filled when that foot is loaded.

  Laid out as the robot is seen from above with nose up, so the viewer can
  check the classifier against their own eye frame by frame. A trot lights one
  diagonal at a time; a pace lights one side at a time.
  """
  h, w = img.shape[:2]
  pad, gap, x0, y0 = 16, 30, w - 96, h - 96
  spot = {"FL": (x0, y0), "FR": (x0 + gap + pad, y0),
          "HL": (x0, y0 + gap + pad), "HR": (x0 + gap + pad, y0 + gap + pad)}
  img[y0 - 12:y0 + 2 * pad + gap + 12, x0 - 12:x0 + 2 * pad + gap + 12] = (
    img[y0 - 12:y0 + 2 * pad + gap + 12, x0 - 12:x0 + 2 * pad + gap + 12] // 3)
  for ch in range(min(4, len(found))):
    name = corner.get(ch)
    if name is None:
      continue
    px, py = spot[name]
    on = found[ch] > 0
    col = np.array([60, 230, 90] if on else [70, 70, 80], dtype=img.dtype)
    img[py:py + pad, px:px + pad] = col
    if not on:                      # hollow when swinging
      img[py + 3:py + pad - 3, px + 3:px + pad - 3] = img.dtype.type(25)
  return img


def build(variant, ckpt, device, robust, width, height, azimuth, elevation, distance):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from rsl_rl.runners import OnPolicyRunner
  from microtaur_velocity.distill_reliable.mjlab_utils import cfg_to_dict

  task = "Mjlab-Velocity-Yaw-Flat-Microtaur"
  if robust:
    from microtaur_velocity.distill_reliable.mjlab_utils import _make_robust_distill_cfg
    cfg = _make_robust_distill_cfg(task, 1)
  else:
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    if "actor" in cfg.observations:
      cfg.observations["actor"].enable_corruption = False
  agent_cfg = load_rl_cfg(task)

  cfg.viewer.width, cfg.viewer.height = width, height
  cfg.viewer.azimuth, cfg.viewer.elevation = azimuth, elevation
  cfg.viewer.distance = distance

  # The viewer's tracked body must exist in THIS variant's model, and it is
  # resolved during env construction — so fix it up beforehand.
  import mujoco
  from microtaur_velocity.microtaur_constants import get_spec
  m = get_spec().compile()
  bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, m.nbody)]
  want = getattr(cfg.viewer, "body_name", None)
  if want not in bodies:
    print(f"[render] viewer body {want!r} absent; tracking {bodies[0]!r} instead")
    cfg.viewer.body_name = bodies[0]

  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array")
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task) or OnPolicyRunner
  runner = runner_cls(env, cfg_to_dict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  return env, runner.get_inference_policy(device=device)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--variant", choices=list(VARIANTS), default="pitch")
  ap.add_argument("--checkpoint", default=None)
  ap.add_argument("--vx", type=float, default=0.15)
  ap.add_argument("--yaw", type=float, default=0.0)
  ap.add_argument("--seconds", type=float, default=20.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--robust", action="store_true", help="training-like env (DR + obs corruption)")
  ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  ap.add_argument("--width", type=int, default=960)
  ap.add_argument("--height", type=int, default=540)
  ap.add_argument("--azimuth", type=float, default=125.0)
  ap.add_argument("--elevation", type=float, default=-12.0)
  ap.add_argument("--distance", type=float, default=0.62)
  ap.add_argument("--fps", type=int, default=50)
  ap.add_argument("--out", default=None)
  args = ap.parse_args()

  v = VARIANTS[args.variant]
  ckpt = Path(args.checkpoint) if args.checkpoint else Path(v.ckpt)
  if not ckpt.exists():
    ap.error(f"checkpoint not found: {ckpt}")
  OUT.mkdir(parents=True, exist_ok=True)
  name = args.out or f"{args.variant}_{ckpt.stem}_vx{args.vx:.2f}{'_robust' if args.robust else ''}.mp4"
  outpath = OUT / name

  with use_variant(args.variant):
    from microtaur_velocity.distill_reliable.mjlab_utils import get_initial_obs, step_env
    from microtaur_velocity.env_cfgs import set_joystick_twist_command
    import imageio.v2 as imageio

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    env, policy = build(args.variant, ckpt, args.device, args.robust,
                        args.width, args.height, args.azimuth, args.elevation, args.distance)
    u = env.unwrapped
    robot = u.scene["robot"]
    feet = u.scene["feet_ground_contact"]
    from foot_index import corner_of_channel
    CORNER = corner_of_channel(feet)   # NOT the identity: mjlab permutes channels
    print(f"[render] contact channels {list(feet.primary_names)} -> {CORNER}")
    dt = float(getattr(u, "step_dt", 0.02))
    n = int(round(args.seconds / dt))

    obs = get_initial_obs(env)
    try:
      set_joystick_twist_command(env, args.vx, args.yaw)
    except Exception as e:  # noqa: BLE001
      print("[render] joystick command unavailable:", e)

    frames, C, V = [], [], []
    for i in range(n):
      with torch.no_grad():
        raw = torch.nan_to_num(policy(obs), nan=0.0, posinf=6.0, neginf=-6.0)
      obs, _, done, _ = step_env(env, raw)          # raw prev-action (term clamps internally)
      try:
        set_joystick_twist_command(env, args.vx, args.yaw)
      except Exception:  # noqa: BLE001
        pass
      C.append(feet.data.found[0].float().cpu().numpy().ravel()[:4].copy())
      V.append(float(robot.data.root_link_lin_vel_b[0, 0]))
      img = u.render()
      if img is not None:
        frames.append(draw_feet(np.asarray(img).copy(), C[-1], CORNER))
      if i % 250 == 0:
        print(f"[render] {i}/{n}  v_body_x={V[-1]:+.3f}")
    env.close()

  C = np.array(C); V = np.array(V)
  ss = slice(50, None)
  col = {v: k for k, v in CORNER.items()}

  def cr(a, b):
    x = C[ss, col[a]] - C[ss, col[a]].mean(); y = C[ss, col[b]] - C[ss, col[b]].mean()
    return float((x * y).mean() / (x.std() * y.std() + 1e-9))

  diag = np.mean([cr("FL", "HR"), cr("FR", "HL")])
  lat = np.mean([cr("FL", "HL"), cr("FR", "HR")])
  gait = "TROT" if diag > lat else "PACE"
  print(f"[render] gait={gait}  diag={diag:+.2f} lat={lat:+.2f}  mean v_body_x={V[ss].mean():+.3f}")

  if not frames:
    print("[render] NO FRAMES captured — offscreen rendering unavailable"); return
  imageio.mimwrite(outpath, frames, fps=args.fps, quality=8, macro_block_size=1)
  print(f"[render] wrote {outpath}  ({len(frames)} frames, {len(frames)/args.fps:.1f}s)")

  # burn a HUD on with ffmpeg
  meta = (f"{args.variant}  {ckpt.name}   cmd vx {args.vx:+.3f} yaw {args.yaw:+.2f}   "
          f"achieved vx {V[ss].mean():+.3f}   env {'robust (DR)' if args.robust else 'clean play'}   "
          f"GAIT {gait}  (diag {diag:+.2f} / lat {lat:+.2f})")
  try:
    import imageio_ffmpeg
    import subprocess
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    hud = outpath.with_name(outpath.stem + "_hud.mp4")
    txt = meta.replace(":", r"\:").replace("'", "")
    subprocess.run([ff, "-y", "-i", str(outpath), "-vf",
                    f"drawbox=x=0:y=0:w=iw:h=34:color=black@0.65:t=fill,"
                    f"drawtext=text='{txt}':x=12:y=10:fontsize=15:fontcolor=white",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(hud)],
                   capture_output=True, text=True, timeout=300)
    if hud.exists():
      print(f"[render] wrote {hud}")
  except Exception as e:  # noqa: BLE001
    print("[render] HUD overlay skipped:", e)


if __name__ == "__main__":
  main()
