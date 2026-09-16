"""Spawn a Microtaur on a task terrain and render a still (or a short clip).

    python scripts/render_terrain.py --terrain curb
    python scripts/render_terrain.py --terrain all --variant rigid
    python scripts/render_terrain.py --terrain weave --seconds 12   # writes an mp4 too

The robot is driven by its trained checkpoint so it stands and walks naturally
rather than collapsing; the still is taken after --settle seconds.
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

OUT = Path("rollouts/terrains")

# per-terrain camera: (azimuth, elevation, distance)
CAMS = {
  # Looks across the lip from the road side, so the riser is in profile and
  # the two feet on the kerb read at a different height from the two below.
  "curb":  (100.0, -10.0, 0.45),
  "steps": (140.0, -18.0, 0.60),
  "weave": (135.0, -26.0, 0.95),
  "flat":  (140.0, -14.0, 0.60),
}


def build(ckpt, device, terrain, width, height, cam, terrain_kw, clip_actions):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from rsl_rl.runners import OnPolicyRunner
  from microtaur_velocity.distill_reliable.mjlab_utils import cfg_to_dict
  from microtaur_terrains import apply_terrain
  from microtaur_variants import TASK_ID

  task = TASK_ID
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = 1
  if "actor" in cfg.observations:
    cfg.observations["actor"].enable_corruption = False
  for sensor in cfg.scene.sensors or ():
    if hasattr(sensor, "debug_vis"):
      sensor.debug_vis = False  # hide the height-scan ray markers
  agent_cfg = load_rl_cfg(task)
  agent_cfg.clip_actions = clip_actions  # must match how the checkpoint was trained

  # Terrain is swapped after the task cfg is built, so observations, actions and
  # sensors are identical to the task the checkpoints were trained on.
  apply_terrain(cfg, terrain, **terrain_kw)

  az, el, dist = cam
  cfg.viewer.width, cfg.viewer.height = width, height
  cfg.viewer.azimuth, cfg.viewer.elevation, cfg.viewer.distance = az, el, dist

  import mujoco
  from microtaur_velocity.microtaur_constants import get_spec
  m = get_spec().compile()
  bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, m.nbody)]
  if getattr(cfg.viewer, "body_name", None) not in bodies:
    cfg.viewer.body_name = bodies[0]

  env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array")
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task) or OnPolicyRunner
  runner = runner_cls(env, cfg_to_dict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  return env, runner.get_inference_policy(device=device)


def run_one(variant, terrain, args):
  from microtaur_velocity.distill_reliable.mjlab_utils import get_initial_obs, step_env
  from microtaur_velocity.env_cfgs import set_joystick_twist_command
  import imageio.v2 as imageio

  ckpt = Path(args.checkpoint) if args.checkpoint else Path(VARIANTS[variant].ckpt)
  cam = (args.azimuth if args.azimuth is not None else CAMS[terrain][0],
         args.elevation if args.elevation is not None else CAMS[terrain][1],
         args.distance if args.distance is not None else CAMS[terrain][2])
  kw = {}
  if terrain == "curb" and args.curb_height is not None:
    kw["curb_height"] = args.curb_height
  if terrain == "steps":
    if args.step_height is not None:
      kw["max_height"] = args.step_height
    if args.tile_size is not None:
      kw["tile_size"] = args.tile_size
  if terrain == "weave":
    if args.pole_spacing is not None:
      kw["spacing"] = args.pole_spacing
    if args.num_poles is not None:
      kw["num_poles"] = args.num_poles

  torch.manual_seed(args.seed); np.random.seed(args.seed)
  env, policy = build(ckpt, args.device, terrain, args.width, args.height, cam, kw,
                      VARIANTS[variant].clip_actions)
  u = env.unwrapped
  # Seed exactly as rollout_log.py does (manual_seed -> build -> env.seed), so
  # render seed k reproduces logged run k. Without this the two paths drew
  # different domain randomisation and a render could land in the opposite mode
  # of a bimodal cell from the logged run it was meant to illustrate.
  try:
    u.seed(args.seed)
  except Exception:  # noqa: BLE001
    pass
  robot = u.scene["robot"]
  dt = float(getattr(u, "step_dt", 0.02))
  mjm = u.sim.mj_model
  print(f"[terrain] {terrain}: model has {mjm.ngeom} geoms, "
        f"{mjm.nbody} bodies; camera az/el/d = {cam}")

  # Same steering the rollouts use: the policy has no lateral-position feedback,
  # so the curb needs a lane keeper to stay on the lip and the (physics-less)
  # weave poles need a path follower to produce any slalom at all.
  driver = None
  if args.lane_keep or args.weave_drive:
    from weave_drive import WeaveDriver, first_pole_x, yaw_from_quat_wxyz
    o = u.scene.env_origins[0].detach().cpu().numpy()
    if args.weave_drive:
      sp = float(kw.get("spacing", 0.10))
      driver = WeaveDriver(x0=first_pole_x(u, sp), spacing=sp, y_center=float(o[1]))
    else:
      driver = WeaveDriver(x0=float(o[0]), spacing=1.0, amplitude=0.0,
                           y_center=float(o[1]))

  def cmd_yaw():
    if driver is None:
      return args.yaw
    p = robot.data.root_link_pos_w[0].cpu().numpy()
    q = robot.data.root_link_quat_w[0].cpu().numpy()
    v = float(robot.data.root_link_lin_vel_b[0, 0])
    return driver.yaw_command(float(p[0]), float(p[1]), yaw_from_quat_wxyz(q),
                              speed=max(v, 0.02))

  obs = get_initial_obs(env)
  set_joystick_twist_command(env, args.vx, cmd_yaw())

  n_settle = int(round(args.settle / dt))
  n_total = int(round(max(args.settle, args.seconds) / dt))
  still, Y, TGT, V, X = None, [], [], [], []
  n_frames, last_img = 0, None
  origin_w = u.scene.env_origins[0].detach().cpu().numpy()
  OUT.mkdir(parents=True, exist_ok=True)
  tag = args.tag or f"{variant}_{terrain}"
  mp4 = OUT / f"{tag}.mp4"
  # Stream frames to the encoder; buffering 800 HD frames costs ~2 GB per process.
  writer = (imageio.get_writer(mp4, fps=50, quality=8, macro_block_size=1)
            if args.seconds > 0 else None)

  # Optional first-order low-pass on the 8 leg actions. Leg targets are
  # stand + scale * clamp(action, -1, 1), so this filters the leg joint targets
  # (before the action term's safety limiter). The spine action is untouched.
  alpha = None
  if args.leg_lowpass_hz:
    tau = 1.0 / (2.0 * np.pi * args.leg_lowpass_hz)
    alpha = dt / (dt + tau)
    names = list(getattr(u.action_manager, "active_terms", []))
    print(f"[terrain] leg low-pass {args.leg_lowpass_hz:g} Hz (alpha {alpha:.3f} at "
          f"{1/dt:.0f} Hz); action terms {names}")
    if names and names[0] != "joint_pos":
      raise RuntimeError(f"expected leg term first in action vector, got {names}")
  leg_state = None
  for i in range(n_total):
    with torch.no_grad():
      raw = torch.nan_to_num(policy(obs), nan=0.0, posinf=6.0, neginf=-6.0)
      if alpha is not None:
        raw = raw.clamp(-1.0, 1.0)
        if leg_state is None:
          leg_state = torch.zeros_like(raw[:, :8])  # stand pose
        leg_state = leg_state + alpha * (raw[:, :8] - leg_state)
        raw = torch.cat([leg_state, raw[:, 8:]], dim=1)
    obs, _, _done, _ = step_env(env, raw)
    if alpha is not None and bool(_done.any()):
      leg_state = None  # robot respawned at stand
    set_joystick_twist_command(env, args.vx, cmd_yaw())
    p = robot.data.root_link_pos_w[0].cpu().numpy()
    Y.append(float(p[1]) - float(origin_w[1]))
    X.append(float(p[0]) - float(origin_w[0]))
    TGT.append(driver.target_y(float(p[0])) - float(origin_w[1]) if driver else 0.0)
    V.append(float(robot.data.root_link_lin_vel_b[0, 0]))
    img = u.render()
    if img is not None:
      if i + 1 == n_settle:
        still = np.asarray(img).copy()
      last_img = img
      if writer is not None:
        writer.append_data(np.asarray(img))
        n_frames += 1
  if writer is not None:
    writer.close()
  z = float(robot.data.root_link_pos_w[0, 2])
  vx = float(robot.data.root_link_lin_vel_b[0, 0])
  env.close()

  Y, TGT, V, X = np.array(Y), np.array(TGT), np.array(V), np.array(X)
  # a jump of >5 cm in one 20 ms step is a reset (robot teleported to spawn)
  jumps = np.where(np.hypot(np.diff(X), np.diff(Y)) > 0.05)[0]
  n_reset = int(len(jumps))
  ss = slice(min(100, len(Y) // 4), None)
  note = f"achieved vx {V[ss].mean():+.3f}"
  if terrain == "steps":
    # tile region in the origin-relative frame (see scripts/steps_onfield.py)
    on = (X >= 0.175) & (X <= 3.775) & (np.abs(Y) <= 0.60)
    note += (f"   progress {X.max():.2f} m   on tiles {on[ss].mean()*100:.0f}% of run   max |y| {np.abs(Y).max():.2f} m"
             f"   resets {n_reset}")
  if terrain == "curb":
    frac = float((np.abs(Y[ss]) < 0.04).mean())
    note += f"   on-lip {frac*100:.0f}% of run   lateral drift {Y[-1]-Y[0]:+.3f} m"
  elif terrain == "weave" and driver is not None:
    a_r = Y[ss].std() * np.sqrt(2); a_t = TGT[ss].std() * np.sqrt(2)
    note += f"   weave amplitude {a_r*1000:.0f}mm vs target {a_t*1000:.0f}mm"
    note += f"  (ratio {a_r/max(a_t,1e-9):.2f})"

  if alpha is not None:
    note += f"   leg low-pass {args.leg_lowpass_hz:g} Hz"
  if still is None and last_img is not None:
    still = np.asarray(last_img)
  if still is None:
    print(f"[terrain] {terrain}: NO FRAMES captured"); return None
  png = OUT / f"{tag}.png"
  imageio.imwrite(png, still)
  print(f"[terrain] {terrain}: wrote {png}  (root z={z:.3f} m, v_body_x={vx:+.3f})")
  if args.seconds > 0 and n_frames:
    print(f"[terrain] {terrain}: wrote {mp4}  ({n_frames} frames)")
    label = args.label or terrain
    steer = ("lane-keep" if args.lane_keep else
             "weave-drive" if args.weave_drive else "no steering")
    meta = (f"{label}   {variant} model_4499   cmd vx {args.vx:+.2f}   "
            f"steering {steer}   {note}")
    try:
      import imageio_ffmpeg, subprocess
      ff = imageio_ffmpeg.get_ffmpeg_exe()
      hud = mp4.with_name(mp4.stem + "_hud.mp4")
      txt = meta.replace(":", r"\:").replace("'", "").replace("%", r"\%")
      subprocess.run([ff, "-y", "-i", str(mp4), "-vf",
                      "drawbox=x=0:y=0:w=iw:h=34:color=black@0.65:t=fill,"
                      f"drawtext=text='{txt}':x=12:y=10:fontsize=15:fontcolor=white",
                      "-c:v", "libx264", "-pix_fmt", "yuv420p", str(hud)],
                     capture_output=True, text=True, timeout=600)
      if hud.exists():
        print(f"[terrain] {terrain}: wrote {hud}")
    except Exception as e:  # noqa: BLE001
      print("[terrain] HUD overlay skipped:", e)
  print(f"[terrain] {terrain}: {note}")
  return png


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--terrain", default="all",
                  help="curb | steps | weave | flat | all")
  ap.add_argument("--variant", choices=list(VARIANTS), default="rigid")
  ap.add_argument("--checkpoint", default=None)
  ap.add_argument("--vx", type=float, default=0.14)
  ap.add_argument("--yaw", type=float, default=0.0)
  ap.add_argument("--settle", type=float, default=2.5,
                  help="seconds to walk before the still is grabbed")
  ap.add_argument("--seconds", type=float, default=0.0,
                  help=">0 also writes an mp4 of this length")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  ap.add_argument("--width", type=int, default=1100)
  ap.add_argument("--height", type=int, default=680)
  ap.add_argument("--azimuth", type=float, default=None)
  ap.add_argument("--elevation", type=float, default=None)
  ap.add_argument("--distance", type=float, default=None)
  ap.add_argument("--curb-height", type=float, default=None)
  ap.add_argument("--step-height", type=float, default=None)
  ap.add_argument("--tile-size", type=float, default=None)
  ap.add_argument("--pole-spacing", type=float, default=None)
  ap.add_argument("--num-poles", type=int, default=None)
  ap.add_argument("--lane-keep", action="store_true",
                  help="hold the spawn line (on curb: stay straddling the lip)")
  ap.add_argument("--weave-drive", action="store_true",
                  help="steer the slalom through the poles")
  ap.add_argument("--label", default=None, help="text for the HUD banner")
  ap.add_argument("--tag", default=None, help="output filename stem override")
  ap.add_argument("--leg-lowpass-hz", type=float, default=None,
                  help="first-order low-pass cutoff on the 8 leg actions")
  args = ap.parse_args()

  names = ["curb", "steps", "weave"] if args.terrain == "all" else [args.terrain]
  with use_variant(args.variant):
    for t in names:
      run_one(args.variant, t, args)


if __name__ == "__main__":
  main()
