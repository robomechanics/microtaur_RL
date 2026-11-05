"""Tight-turn S tours with the spine free or locked: does the policy use its spine to turn? (rigid robot = baseline)

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_spine_turn.py --checkpoint <model.pt> --label y2 --out <dir>

All envs stand on one flat tile (A_flat, level 0; envs do not collide with each other). Env groups = command set x
spine mode. Every set is an S: 2 s straight at the set's vx, then four 6 s arcs at +wz, -wz, +wz, -wz.
  in_train  vx 0.10  wz 0.25  commanded radius 0.40 m: the tightest turn of the training commands
  1L_fast   vx 0.10  wz 0.50  radius 0.20 m, about one body length (hip pairs 0.167 m apart); wz twice the trained max
  1L_slow   vx 0.05  wz 0.25  radius 0.20 m; vx half the trained minimum
  spin      vx 0.00  wz 0.50  turn on the spot
Yaw robot: in half the envs of every set the spine is locked at 0 (the policy's spine action is replaced by 0, so the
servo holds 0). Per set and mode, over the last 4 s of each arc, sign-corrected so + = toward the commanded turn,
over the envs that have not reset: yaw rate (ratio to the command), forward speed, path radius = speed / yaw rate,
spine angle mean and std (+ = front half rotated toward the turn, i.e. bent into it), and the spine's strongest
oscillation above 1 Hz (stride frequency). The legs are planar (no abduction), so each half can only push along its
own heading, as on an articulated vehicle (front / rear axle = hip-pair midpoints, a_f / a_r from the spine axis):
without foot slip a bend phi turns the robot on the radius R with phi = atan(a_f / R) + atan(a_r / R), about
R = a / tan(phi / 2). Printed: the no-slip bend for the measured radius and the no-slip radius for the measured bend.
Writes spine_turn_<label>.txt / .json, spine_turn_<label>.npz (traces of the first env of every group) and
spine_turn_<label>.png (paths and spine angle of those envs).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--envs-per-group", type=int, default=12)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
from microtaur_common.gpu import pick_device  # noqa: E402

args.device = pick_device(args.device)
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from microtaur_common.robot_constants import SPINE_JOINT_NAME  # noqa: E402
from microtaur_common.robot_models import robot_model  # noqa: E402
from microtaur_isaac.env_cfg import MicrotaurTeacherCurPlayEnvCfg  # noqa: E402
from microtaur_isaac.mdp.contact import foot_contact_timers  # noqa: E402
from microtaur_isaac.mdp.observations import feet_sensor_cfg  # noqa: E402
from microtaur_isaac.policy_loader import load_policy, prepare_env_cfg, variant_kwargs  # noqa: E402

SETS = [("in_train", 0.10, 0.25), ("1L_fast", 0.10, 0.50), ("1L_slow", 0.05, 0.25), ("spin", 0.00, 0.50)]
STRAIGHT_S, ARC_S, WINDOW_S = 2.0, 6.0, 4.0
ARC_SIGNS = (1, -1, 1, -1)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
G = args.envs_per_group
cfg = MicrotaurTeacherCurPlayEnvCfg(**variant_kwargs(args.checkpoint))
cfg.scene.num_envs = len(SETS) * 2 * G  # env = (set * 2 + locked) * G + i
cfg.sim.device = args.device
cfg.curriculum.command_ranges = None
cfg.commands.twist.resampling_time_range = (1.0e9, 1.0e9)
prepare_env_cfg(cfg, args.checkpoint)
env = ManagerBasedRLEnv(cfg)
t = env.scene.terrain
t.terrain_levels[:] = 0
t.terrain_types[:] = 0
t.env_origins[:] = t.terrain_origins[0, 0]
w = RslRlVecEnvWrapper(env)  # resets: robots spawn at the new origins
policy, policy_reset = load_policy(w, args.checkpoint)
robot, cmd = env.scene["robot"], env.command_manager.get_term("twist")
fc = feet_sensor_cfg(cfg.variant)
fc.resolve(env.scene)
yaw = SPINE_JOINT_NAME in robot.joint_names  # any spine robot (the spine-locked groups)
yaw_bend = robot_model(cfg.variant).spine == "yaw"
sid = robot.joint_names.index(SPINE_JOINT_NAME) if yaw else None
# hip-pair midpoints along the body from the spine axis (robot_2imu.xml: motor axes at x -94.3 / -73.8 and 72.3 / 92.8 mm)
A_REAR, A_FRONT = 0.08405, 0.08255

n = env.num_envs
set_of = torch.arange(n, device=env.device) // (2 * G)
locked = ((torch.arange(n, device=env.device) // G) % 2 == 1) & yaw
vx_set = torch.tensor([s[1] for s in SETS], device=env.device)[set_of]
wz_set = torch.tensor([s[2] for s in SETS], device=env.device)[set_of]
dt = env.step_dt
n_straight, n_arc = round(STRAIGHT_S / dt), round(ARC_S / dt)
T = n_straight + len(ARC_SIGNS) * n_arc
arc_sign = np.zeros(T)
for i, s in enumerate(ARC_SIGNS):
  arc_sign[n_straight + i * n_arc:n_straight + (i + 1) * n_arc] = s

rec = {k: [] for k in ("wz", "speed", "spine", "x", "y", "done", "contact")}
with torch.inference_mode():
  obs = w.get_observations()
  for k in range(T):
    cmd.vel_command_b[:, 0], cmd.vel_command_b[:, 1], cmd.vel_command_b[:, 2] = vx_set, 0.0, wz_set * arc_sign[k]
    act = policy(obs)
    if yaw:
      act[locked, -1] = 0.0  # spine target 0 (the 9th action is the spine)
    obs, _, dones, _ = w.step(act)
    policy_reset(dones)
    d = robot.data
    # .copy(): with CPU physics .cpu() is a no-op and numpy would alias the reused PhysX buffer
    rec["wz"].append(d.root_link_ang_vel_b[:, 2].cpu().numpy().copy())
    rec["speed"].append(torch.linalg.norm(d.root_link_lin_vel_w[:, :2], dim=1).cpu().numpy().copy())
    rec["spine"].append(d.joint_pos[:, sid].cpu().numpy().copy() if yaw else np.zeros(n))
    rec["x"].append((d.root_link_pos_w[:, 0] - t.env_origins[:, 0]).cpu().numpy().copy())
    rec["y"].append((d.root_link_pos_w[:, 1] - t.env_origins[:, 1]).cpu().numpy().copy())
    rec["done"].append(dones.cpu().numpy().copy())
    rec["contact"].append(foot_contact_timers(env, fc)[0].cpu().numpy().copy())
R = {k: np.stack(v) for k, v in rec.items()}  # [T, N(, 4)]
alive_env = R["done"].sum(axis=0) == 0
set_np, locked_np = set_of.cpu().numpy(), locked.cpu().numpy()


def osc(sp_deg: np.ndarray) -> tuple[float, float]:
  """Strongest spectral line above 1 Hz of one window (amplitude deg, frequency Hz)."""
  spec = np.abs(np.fft.rfft(sp_deg - sp_deg.mean())) * 2 / sp_deg.size
  f = np.fft.rfftfreq(sp_deg.size, dt)
  k = np.argmax(np.where(f > 1.0, spec, 0))
  return float(spec[k]), float(f[k])


nw = round(WINDOW_S / dt)
res = {"label": args.label, "checkpoint": args.checkpoint, "variant": cfg.variant, "rows": []}
modes = (False, True) if yaw else (False,)
lines = [f"{args.label}: {args.checkpoint}",
         f"{res['variant']} robot, {G} envs per group (rigid: {2 * G}), S tour = {STRAIGHT_S:.0f} s straight + 4 arcs of {ARC_S:.0f} s "
         f"(+ - + -), stats over the last {WINDOW_S:.0f} s of every arc, sign-corrected (+ = toward the commanded turn)",
         "set      cmd vx  wz   R_cmd | spine  | envs ok | yaw rate (ratio) | speed | radius m | spine deg mean +- std | osc deg @ Hz | no-slip: bend for R deg, R for bend m"]
for si, (name, vx, wz) in enumerate(SETS):
  for lock in modes:
    envs = np.nonzero((set_np == si) & (locked_np == lock if yaw else True) & alive_env)[0]
    n_grp = int(((set_np == si) & (locked_np == lock if yaw else True)).sum())
    wz_m, sp_m, sp_s, spd, oa, of = [], [], [], [], [], []
    for i in range(len(ARC_SIGNS)):
      idx = np.arange(n_straight + (i + 1) * n_arc - nw, n_straight + (i + 1) * n_arc)
      s = ARC_SIGNS[i]
      if envs.size == 0:
        continue
      wz_m.append(s * R["wz"][idx][:, envs].mean())
      spd.append(R["speed"][idx][:, envs].mean())
      sp = np.degrees(R["spine"][idx][:, envs]) * s
      sp_m.append(sp.mean())
      sp_s.append(sp.std(axis=0).mean())  # within-window std per env, averaged
      o = [osc(sp[:, j]) for j in range(envs.size)] if yaw else [(0.0, 0.0)]
      oa.append(np.mean([a for a, _ in o]))
      of.append(np.median([f for _, f in o]))
    if not wz_m:
      lines.append(f"{name:8s} all envs reset")
      continue
    yr, sv = float(np.mean(wz_m)), float(np.mean(spd))
    radius = sv / yr if yr > 1e-3 else float("inf")
    bend = math.degrees(math.atan(A_FRONT / radius) + math.atan(A_REAR / radius)) if math.isfinite(radius) else 0.0
    sp_bar = float(np.mean(sp_m))
    # the no-slip radius of a bend: planar yaw bend only (a roll / pitch bend does not steer the halves' headings)
    r_bend = ((A_FRONT + A_REAR) / 2 / math.tan(math.radians(sp_bar) / 2) if yaw_bend and sp_bar > 0.05
              else float("inf") if yaw_bend else float("nan"))
    row = {"set": name, "cmd_vx": vx, "cmd_wz": wz, "spine_locked": bool(lock), "envs": n_grp, "envs_ok": int(envs.size),
           "yaw_rate": yr, "ratio": yr / wz, "speed": sv, "radius_m": radius, "spine_deg_mean": float(np.mean(sp_m)),
           "spine_deg_std": float(np.mean(sp_s)), "osc_deg": float(np.mean(oa)), "osc_hz": float(np.median(of)),
           "noslip_bend_for_radius_deg": bend, "noslip_radius_for_bend_m": r_bend}
    res["rows"].append(row)
    rc = f"{vx / wz:4.2f}" if vx else "0.00"
    mode = ("locked" if lock else "free  ") if yaw else "rigid "
    lines.append(f"{name:8s} {vx:.2f}  {wz:.2f}  {rc}  | {mode} | {envs.size:2d}/{n_grp:2d}   | {yr:+.3f} ({yr / wz:4.2f})    | {sv:.3f} | "
                 f"{radius:6.2f}   | " + (f"{row['spine_deg_mean']:+5.2f} +- {row['spine_deg_std']:4.2f}       | "
                                         f"{row['osc_deg']:4.2f} @ {row['osc_hz']:3.1f}   | {bend:5.2f}  {r_bend:6.2f}" if yaw else
                                         f"n/a                   | n/a          | {bend:5.2f}"))
(out / f"spine_turn_{args.label}.txt").write_text("\n".join(lines) + "\n")
(out / f"spine_turn_{args.label}.json").write_text(json.dumps(res, indent=1))
first = [int(np.nonzero((set_np == si) & (locked_np == lock))[0][0]) for si in range(len(SETS)) for lock in modes]
np.savez(out / f"spine_turn_{args.label}.npz", t=np.arange(T) * dt, arc_sign=arc_sign, envs=np.array(first),
         sets=np.array([s[0] for s in SETS]), cmd_vx=np.array([s[1] for s in SETS]), cmd_wz=np.array([s[2] for s in SETS]),
         locked=locked_np[first], **{k: R[k][:, first] for k in ("wz", "speed", "spine", "x", "y", "done", "contact")})

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, axs = plt.subplots(2, len(SETS), figsize=(5 * len(SETS), 9), squeeze=False)
tt = np.arange(T) * dt
for si, (name, vx, wz) in enumerate(SETS):
  for lock in modes:
    e = int(np.nonzero((set_np == si) & (locked_np == lock))[0][0])
    lab = ("spine locked" if lock else "spine free") if yaw else "rigid"
    axs[0, si].plot(R["x"][:, e], R["y"][:, e], lw=1.2, label=lab, color="C1" if lock else "C0")
    axs[1, si].plot(tt, np.degrees(R["spine"][:, e]), lw=0.8, label=lab, color="C1" if lock else "C0")
  axs[0, si].plot(0, 0, "go")
  axs[0, si].set_aspect("equal", adjustable="datalim")
  axs[0, si].set_title(f"{name}: cmd vx {vx:.2f} m/s, wz +-{wz:.2f} rad/s (R_cmd {vx / wz:.2f} m)")
  axs[0, si].set_xlabel("x m")
  axs[0, si].legend(fontsize=8)
  ax2 = axs[1, si].twinx()
  ax2.plot(tt, arc_sign * wz, "k--", lw=0.8)
  ax2.set_ylabel("yaw command rad/s (dashed)")
  axs[1, si].axhline(0, color="k", lw=0.4)
  axs[1, si].set_title("spine angle deg (" + {"yaw": "+ = bent to the left", "roll": "+ = front half rolled left side down"}.get(robot_model(cfg.variant).spine, "joint angle") + ")")
  axs[1, si].set_xlabel("t s")
axs[0, 0].set_ylabel("y m")
fig.suptitle(f"{args.label}: S tours, first env of every group (start = green dot)")
fig.tight_layout()
fig.savefig(out / f"spine_turn_{args.label}.png", dpi=100)
print("\n".join(lines), flush=True)
sys.stdout.flush()
os._exit(0)
