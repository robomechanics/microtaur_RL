"""Scripted IK trot as a reference ("what should a good gait score?").

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_reference_gait.py --out <dir> \
      [--speed 0.20] [--periods 0.4 0.28 0.14] [--lift-mm 10] [--num-envs 32] [--seconds 8]

For each trot period T: every foot follows a leg-plane trajectory around the stand
foot position -- stance (first half of the cycle): on the ground, sweeping back from
+L/2 to -L/2 with L = speed * T / 2 (the body-relative stroke at the commanded speed);
swing: lifted by lift-mm (sin profile) while returning forward (cosine blend).
Diagonal pairs RR+FL and RL+FR half a cycle apart. The five-bar IK turns the foot
targets into motor angles, which go through the SAME action pipeline as the policy
(action = (q - stand) / action_scale, clipped, safety filter, 1-step delay), open loop.
The flat env uses the reward set of the terrain run c1 (numbers below) with the
command pinned to (speed, 0, 0).

Reports per period: every reward term's mean weighted rate (what an episode sum per
second would be), their total, achieved speed, heading drift, body height and tilt,
falls; plus each term's analytic ceiling. Writes reference_gait.json and
ref_T<period>.npz trajectories (render with scripts/isaac_render_traj.py).
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
ap.add_argument("--out", required=True)
ap.add_argument("--speed", type=float, default=0.20)
ap.add_argument("--periods", type=float, nargs="+", default=[0.4, 0.28, 0.14])
ap.add_argument("--lift-mm", type=float, default=10.0)
ap.add_argument("--num-envs", type=int, default=32)
ap.add_argument("--seconds", type=float, default=8.0)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
LOG = open(out / "reference_gait.log", "w")


def log(*a):
  LOG.write(" ".join(map(str, a)) + "\n"); LOG.flush()


import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from microtaur_common.kinematics import MicrotaurFiveBarKinematics  # noqa: E402
from microtaur_common.robot_constants import STAND_A, STAND_E  # noqa: E402
from microtaur_isaac.env_cfg import MicrotaurFlatEnvCfg  # noqa: E402
from microtaur_isaac.mdp.contact import foot_contact_timers  # noqa: E402
from microtaur_isaac.mdp.observations import feet_sensor_cfg  # noqa: E402

# c1 reward set (Hydra overrides of the terrain run), applied to the flat env
C1 = {"action_rate": -0.08, "feet_air_time": 1.0, "flat_orientation_l2": -2.5, "heading_tracking": 1.0}
cfg = MicrotaurFlatEnvCfg(play=True)
cfg.scene.num_envs = args.num_envs
cfg.curriculum.command_ranges = None
cfg.commands.twist.resampling_time_range = (1.0e9, 1.0e9)
for k, w in C1.items():
  getattr(cfg.rewards, k).weight = w
cfg.rewards.trot_gait.params["std"] = 0.1
cfg.rewards.trot_gait.params["max_err"] = 0.2
cfg.rewards.track_lin_vel_xy.params["sigma"] = 0.07
cfg.rewards.feet_air_time.params["mode_time_s"] = 0.2
env = ManagerBasedRLEnv(cfg)
robot = env.scene["robot"]
term = env.action_manager.get_term("joint_pos")
cmd = env.command_manager.get_term("twist")
rm = env.reward_manager
dt = env.step_dt
feet = feet_sensor_cfg(); feet.resolve(env.scene)
scale = float(cfg.actions.joint_pos.action_scale_rad)
K = MicrotaurFiveBarKinematics()
stand = np.empty(8); stand[0::2] = STAND_A; stand[1::2] = STAND_E
x0, z0 = (lambda s: (s.foot_x, s.foot_z))(K.forward_numpy(STAND_A[0], STAND_E[0], 1))
PHASE = (0.0, 0.5, 0.0, 0.5)  # legs 1 RR, 2 RL, 3 FL, 4 FR: diagonals in phase


def foot(phi: float, L: float, h: float) -> tuple[float, float]:
  if phi < 0.5:  # stance: sweep back on the ground
    s = phi / 0.5
    return x0 + L / 2 - L * s, z0
  s = (phi - 0.5) / 0.5  # swing: forward and up
  return x0 - L / 2 + L * 0.5 * (1 - math.cos(math.pi * s)), z0 + h * math.sin(math.pi * s)


def targets(t: float, T: float, L: float, h: float) -> np.ndarray:
  q = np.empty(8)
  for k in range(4):
    fx, fz = foot(((t / T) + PHASE[k]) % 1.0, L, h)
    sol = K.inverse_numpy(fx, fz, k + 1, reference_motor=(stand[2 * k], stand[2 * k + 1]))
    if not sol.valid:
      raise RuntimeError(f"IK failed leg {k + 1} at ({fx:.4f}, {fz:.4f})")
    q[2 * k], q[2 * k + 1] = sol.q_a, sol.q_e
  return q


def ceilings() -> dict:
  w = {n: rm.get_term_cfg(n).weight for n in rm.active_terms}
  mode = cfg.rewards.feet_air_time.params["mode_time_s"]
  return {"track_lin_vel_xy": w["track_lin_vel_xy"], "track_ang_vel_z": w["track_ang_vel_z"],
          "trot_gait": w["trot_gait"], "heading_tracking": w["heading_tracking"],
          # Spot air_time: every phase exactly mode_time -> each foot's running time averages mode/2
          "feet_air_time": w["feet_air_time"] * 4 * mode / 2, "flat_orientation_l2": 0.0, "action_rate": 0.0,
          "termination": 0.0}


results = {"speed": args.speed, "lift_mm": args.lift_mm, "c1_reward_overrides": C1, "ceilings": None, "periods": {}}
N = env.num_envs
for T in args.periods:
  L = args.speed * T / 2
  env.reset()
  cmd.vel_command_b[:, 0] = args.speed; cmd.vel_command_b[:, 1:] = 0.0
  sums = {n: 0.0 for n in rm.active_terms}
  rec = {"vx": [], "wz": [], "z": [], "g": [], "root_pos": [], "root_quat_wxyz": [], "joint_pos": [], "contact": [], "air": [], "con": []}
  falls, n_meas = 0, 0
  steps = int(args.seconds / dt)
  for i in range(steps):
    cmd.vel_command_b[:, 0] = args.speed; cmd.vel_command_b[:, 1:] = 0.0
    q = targets(i * dt, T, L, args.lift_mm * 1e-3)
    a = torch.tensor(np.clip((q - stand) / scale, -1.0, 1.0), dtype=torch.float32, device=env.device).expand(N, 8)
    _, _, term_, trunc, _ = env.step(a)
    if i < int(2.0 / dt):
      continue
    n_meas += 1
    falls += int(term_.sum())
    for j, n in enumerate(rm.active_terms):
      sums[n] += float(rm._step_reward[:, j].mean())
    d = robot.data
    rec["vx"].append(float(d.root_link_lin_vel_b[:, 0].mean())); rec["wz"].append(float(d.root_link_ang_vel_b[:, 2].mean()))
    rec["z"].append(float((d.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).mean()))
    rec["g"].append(d.projected_gravity_b[0].cpu().numpy())
    rec["root_pos"].append((d.root_link_pos_w[0] - env.scene.env_origins[0]).cpu().numpy())
    rec["root_quat_wxyz"].append(d.root_link_quat_w[0].cpu().numpy()); rec["joint_pos"].append(d.joint_pos[0].cpu().numpy())
    c, air, con = foot_contact_timers(env, feet)
    rec["contact"].append(c[0].cpu().numpy()); rec["air"].append(air.cpu().numpy()); rec["con"].append(con.cpu().numpy())
  rates = {n: v / n_meas for n, v in sums.items()}
  g = np.array(rec["g"]); tilt = np.degrees(np.arcsin(np.clip(np.linalg.norm(g[:, :2], axis=1), 0, 1)))
  qmax = np.abs((np.array([targets(k * dt, T, L, args.lift_mm * 1e-3) for k in range(int(T / dt) + 1)]) - stand)).max()
  r = {"period_s": T, "stride_hz": 1 / T, "stroke_mm": 1e3 * L, "max_motor_offset_deg": math.degrees(qmax),
       "action_clipped": bool(qmax > scale), "speed_m_s": float(np.mean(rec["vx"])), "yaw_rate_rad_s": float(np.mean(rec["wz"])),
       "body_height_mm": 1e3 * float(np.mean(rec["z"])), "tilt_deg_mean": float(tilt.mean()),
       "terminations_per_env_s": falls / (N * n_meas * dt), "reward_rates": rates, "reward_total": float(sum(rates.values()))}
  C = np.array(rec["contact"]); A = np.array(rec["air"]); Cn = np.array(rec["con"])  # [T, 4] env 0 / [T, N, 4]
  e2, std = 0.2 ** 2, 0.1
  k = lambda x, y: np.exp(-(np.minimum(x ** 2, e2) + np.minimum(y ** 2, e2)) / std)  # noqa: E731
  rr, rl, fl, fr = 0, 1, 2, 3
  kern = {"sync RR-FL": k(A[..., rr] - A[..., fl], Cn[..., rr] - Cn[..., fl]), "sync RL-FR": k(A[..., rl] - A[..., fr], Cn[..., rl] - Cn[..., fr]),
          "anti RR-RL": k(A[..., rr] - Cn[..., rl], Cn[..., rr] - A[..., rl]), "anti FL-FR": k(A[..., fl] - Cn[..., fr], Cn[..., fl] - A[..., fr]),
          "anti RR-FR": k(A[..., rr] - Cn[..., fr], Cn[..., rr] - A[..., fr]), "anti RL-FL": k(A[..., rl] - Cn[..., fl], Cn[..., rl] - A[..., fl])}
  r["duty_env0"] = [round(float(x), 2) for x in C.mean(0)]
  r["contact_env0_first_40_steps"] = ["".join("#" if c else "." for c in C[:40, j]) for j in range(4)]
  r["trot_kernels_mean"] = {n: round(float(v.mean()), 3) for n, v in kern.items()}
  r["phase_len_s_mean"] = round(float(np.maximum(A, Cn).mean()), 3)
  results["periods"][f"{T:.2f}"] = r
  np.savez(out / f"ref_T{T:.2f}.npz", dt=dt, joint_names=np.array(robot.joint_names), cmd_m_s=args.speed, cmd_yaw_rad_s=0.0,
           root_pos=np.array(rec["root_pos"]), root_quat_wxyz=np.array(rec["root_quat_wxyz"]), joint_pos=np.array(rec["joint_pos"]))
results["ceilings"] = ceilings()
(out / "reference_gait.json").write_text(json.dumps(results, indent=1))

names = list(rm.active_terms)
log(f"scripted IK trot, {args.speed} m/s commanded, lift {args.lift_mm} mm, {N} envs, measured {args.seconds - 2:.0f} s\n")
hdr = f"{'':26s}" + "".join(f"{('T=' + p + ' s'):>14s}" for p in results["periods"]) + f"{'ceiling':>12s}"
log(hdr)
for key, fmt in (("stride_hz", "{:.2f}"), ("stroke_mm", "{:.0f}"), ("max_motor_offset_deg", "{:.1f}"), ("speed_m_s", "{:.3f}"),
                 ("yaw_rate_rad_s", "{:+.3f}"), ("body_height_mm", "{:.1f}"), ("tilt_deg_mean", "{:.1f}"), ("terminations_per_env_s", "{:.3f}")):
  log(f"{key:26s}" + "".join(f"{fmt.format(r[key]):>14s}" for r in results["periods"].values()))
log("reward rate (weighted, /s):")
for n in names:
  c = results["ceilings"].get(n)
  log(f"  {n:24s}" + "".join(f"{r['reward_rates'][n]:>14.3f}" for r in results["periods"].values()) + (f"{c:>12.3f}" if c is not None else ""))
log(f"  {'TOTAL':24s}" + "".join(f"{r['reward_total']:>14.3f}" for r in results["periods"].values())
    + f"{sum(v for v in results['ceilings'].values()):>12.3f}")
log("\ncontact of env 0 (first 40 steps, # = contact) and GaitReward kernels (mean):")
for p, r in results["periods"].items():
  log(f"  T={p}: duty {r['duty_env0']}, mean running phase time {r['phase_len_s_mean']} s")
  for j, leg in enumerate(("RR", "RL", "FL", "FR")):
    log(f"     {leg} {r['contact_env0_first_40_steps'][j]}")
  log("     kernels " + ", ".join(f"{n} {v}" for n, v in r["trot_kernels_mean"].items()))
LOG.close()
os._exit(0)
