"""Evaluate a Microtaur IsaacLab policy's gait at fixed commands (flat): straight and turning.

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_eval_gait.py --checkpoint <model_N.pt> --out <dir> \
      [--speeds 0.10 0.15 0.20 0.25 0.30 0.35] [--turns 0.20:-0.25 0.20:0.25 ...] [--envs-per-speed 16]

One group of envs per command, all in one scene; the play config (no noise, no
randomisation, no episode timeout). The command is pinned every step: forward
speed v, no lateral, yaw rate w (0 for --speeds, given for --turns "v:w"). 2 s settle, 6 s measured. Same statistics as
scripts/eval_gait.py (mjlab): tracking, duty factor, stride frequency, phase vs
leg 1 -> trot / pace / bound, swing clearance, body height and roll/pitch,
motor travel, torque saturation, target-vs-actual error, power and CoT.

Writes summary.json, gait_<v>.png (contact diagram, leg-1 motor traces, foot
heights of the first env at each speed) and traj_<v>.npz (root pose and all
joint angles of that env at the policy rate) for scripts/isaac_render_traj.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--speeds", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
ap.add_argument("--turns", nargs="*", default=["0.20:-0.25", "0.20:-0.12", "0.20:0.12", "0.20:0.25"],
                help='turning commands "v:w" (m/s : rad/s)')
ap.add_argument("--envs-per-speed", type=int, default=16)
ap.add_argument("--settle-s", type=float, default=2.0)
ap.add_argument("--measure-s", type=float, default=6.0)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
LOG = open(out / "eval.log", "w")


def log(*a):
  LOG.write(" ".join(map(str, a)) + "\n"); LOG.flush()


import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from microtaur_common.gait_analysis import LEGS, classify, onset_phases  # noqa: E402
from microtaur_common.robot_constants import (  # noqa: E402
  EFFORT_LIMIT_NM, EXPECTED_TOTAL_MASS_KG, JOINT_HALF_RANGE_RAD, LEG_JOINT_NAMES, STAND_A, STAND_E,
  XL330_COPPER_W_PER_NM2,
)
from microtaur_isaac import FOOT_BODY_NAMES, FOOT_OFFSET_IN_BODY_M  # noqa: E402
from microtaur_isaac.agents import MicrotaurPPORunnerCfg  # noqa: E402
from microtaur_isaac.env_cfg import MicrotaurFlatEnvCfg  # noqa: E402
from microtaur_isaac.mdp.contact import foot_contact_timers  # noqa: E402
from microtaur_isaac.mdp.observations import feet_sensor_cfg  # noqa: E402

FOOT_R = 0.0062
CMDS = [(v, 0.0) for v in args.speeds] + [tuple(map(float, t.split(":"))) for t in args.turns]
LABELS = [f"{v:.2f}" if w == 0.0 else f"{v:.2f}_w{w:+.2f}" for v, w in CMDS]
S, K = len(CMDS), args.envs_per_speed
N = S * K

cfg = MicrotaurFlatEnvCfg(play=True)
cfg.scene.num_envs = N
cfg.curriculum.command_ranges = None  # the command is pinned below
cfg.commands.twist.resampling_time_range = (1.0e9, 1.0e9)
env = ManagerBasedRLEnv(cfg)
wrapped = RslRlVecEnvWrapper(env)
agent = MicrotaurPPORunnerCfg()
runner = OnPolicyRunner(wrapped, agent.to_dict(), log_dir=None, device=env.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.device)

robot = env.scene["robot"]
sensor = env.scene["contact_forces"]
term = env.action_manager.get_term("joint_pos")
cmd_term = env.command_manager.get_term("twist")
mids, _ = robot.find_joints(list(LEG_JOINT_NAMES), preserve_order=True)
fids, _ = robot.find_bodies(list(FOOT_BODY_NAMES), preserve_order=True)
feet_cfg = feet_sensor_cfg()
feet_cfg.resolve(env.scene)  # contact = touched in any substep of the policy step (as the reward)
foot_off = torch.tensor(FOOT_OFFSET_IN_BODY_M, device=env.device)
v_cmd = torch.tensor([c[0] for c in CMDS], device=env.device).repeat_interleave(K)
w_cmd = torch.tensor([c[1] for c in CMDS], device=env.device).repeat_interleave(K)
first = [s * K for s in range(S)]  # env recorded for the trajectory / plots


def pin_command():
  c = cmd_term.vel_command_b
  c[:, 0] = v_cmd
  c[:, 1] = 0.0
  c[:, 2] = w_cmd


dt = env.step_dt
n_settle, n_meas = int(args.settle_s / dt), int(args.measure_s / dt)
keys = ("vx", "vy", "wz", "z", "roll", "pitch", "contact", "foot_z", "q", "target", "tau", "mech", "copper", "reset")
rec = {k: [] for k in keys}
traj = {"root_pos": [], "root_quat_wxyz": [], "joint_pos": []}
pin_command()
obs = wrapped.get_observations()
with torch.inference_mode():
  for t in range(n_settle + n_meas):
    pin_command()
    obs, _, dones, _ = wrapped.step(policy(obs))
    if t < n_settle:
      continue
    d = robot.data
    g = d.projected_gravity_b
    tau, qd = d.applied_torque[:, mids], d.joint_vel[:, mids]
    feet = d.body_link_pose_w[:, fids]
    foot_w = feet[..., :3] + quat_apply(feet[..., 3:7].reshape(-1, 4), foot_off.repeat(N, 1)).reshape(N, 4, 3)
    rec["vx"].append(d.root_link_lin_vel_b[:, 0]); rec["vy"].append(d.root_link_lin_vel_b[:, 1])
    rec["wz"].append(d.root_link_ang_vel_b[:, 2])
    rec["z"].append(d.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2])
    rec["roll"].append(torch.atan2(-g[:, 1], -g[:, 2])); rec["pitch"].append(torch.asin(torch.clamp(g[:, 0], -1, 1)))
    rec["contact"].append(foot_contact_timers(env, feet_cfg)[0].clone())
    rec["foot_z"].append(foot_w[..., 2] - env.scene.env_origins[:, None, 2] - FOOT_R)
    rec["q"].append(d.joint_pos[:, mids]); rec["target"].append(term.core.applied.clone())
    rec["tau"].append(tau)
    rec["mech"].append((tau * qd).abs().sum(1)); rec["copper"].append(XL330_COPPER_W_PER_NM2 * (tau**2).sum(1))
    rec["reset"].append(dones.bool())
    traj["root_pos"].append((d.root_link_pos_w[first] - env.scene.env_origins[first]).clone())
    traj["root_quat_wxyz"].append(d.root_link_quat_w[first].clone())
    traj["joint_pos"].append(d.joint_pos[first].clone())

R = {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}  # [T, N, ...]
TR = {k: torch.stack(v).cpu().numpy() for k, v in traj.items()}  # [T, S, ...]
joint_names = list(robot.joint_names)

stand = np.empty(8); stand[0::2] = STAND_A; stand[1::2] = STAND_E
summary = {"checkpoint": args.checkpoint, "policy_dt_s": dt, "commands": {}}
for si, ((v, w), label) in enumerate(zip(CMDS, LABELS)):
  sl = slice(si * K, (si + 1) * K)
  r = {k: x[:, sl] for k, x in R.items()}
  vx = float(r["vx"].mean())
  duty = r["contact"].mean(axis=(0, 1))
  freqs, phase_list = [], []
  for e in range(K):
    f, ph = onset_phases(r["contact"][:, e, :], dt)
    freqs.append(f); phase_list.append(ph)
  phases = [float(np.nanmedian([p[k] for p in phase_list])) for k in range(4)]
  swing = ~r["contact"]
  clearance = [float(np.percentile(r["foot_z"][:, :, k][swing[:, :, k]], 95) * 1e3) if swing[:, :, k].any() else 0.0
               for k in range(4)]
  travel = np.percentile(np.abs(r["q"] - stand), 95, axis=(0, 1)) / JOINT_HALF_RANGE_RAD
  err = r["target"] - r["q"]
  power = r["mech"].mean() + r["copper"].mean()
  s = {
    "cmd_m_s": v, "cmd_yaw_rad_s": w, "speed_m_s": vx, "speed_ratio": vx / v,
    "yaw_rate_rad_s": float(r["wz"].mean()), "yaw_ratio": float(r["wz"].mean() / w) if w != 0.0 else None,
    "speed_env_min_max": [float(r["vx"].mean(0).min()), float(r["vx"].mean(0).max())],
    "lateral_m_s_rms": float(np.sqrt((r["vy"] ** 2).mean())), "yaw_rate_err_rad_s_rms": float(np.sqrt(((r["wz"] - w) ** 2).mean())),
    "resets_during_measure": int(r["reset"].sum()),
    "duty_factor": [round(float(x), 3) for x in duty], "stride_hz": float(np.nanmedian(freqs)),
    "touchdown_phase_vs_leg1": [round(p, 3) for p in phases], "gait": classify(phases),
    "swing_clearance_p95_mm": [round(c, 1) for c in clearance],
    "body_height_mm": [float(1e3 * r["z"].mean()), float(1e3 * r["z"].std())],
    "roll_pitch_std_deg": [float(np.degrees(r["roll"].std())), float(np.degrees(r["pitch"].std()))],
    "motor_travel_p95_frac_of_limit": [round(float(x), 2) for x in travel],
    "tracking_err_rms_deg": float(np.degrees(np.sqrt((err**2).mean()))),
    "tracking_err_p95_deg": float(np.degrees(np.percentile(np.abs(err), 95))),
    "torque_saturated_frac": float(np.mean(np.abs(r["tau"]) >= 0.99 * EFFORT_LIMIT_NM)),
    "torque_rms_nm": float(np.sqrt((r["tau"] ** 2).mean())),
    "power_mech_w": float(r["mech"].mean()), "power_copper_w": float(r["copper"].mean()),
    "electrical_cot": float(power / (EXPECTED_TOTAL_MASS_KG * 9.81 * max(vx, 1e-3))),
  }
  summary["commands"][label] = s
  log(f"\n=== cmd {v:.2f} m/s, yaw {w:+.2f} rad/s")
  for k, val in s.items():
    log(f"  {k:32s} {val}")

  np.savez(out / f"traj_{label}.npz", dt=dt, joint_names=np.array(joint_names), cmd_m_s=v, cmd_yaw_rad_s=w,
           root_pos=TR["root_pos"][:, si], root_quat_wxyz=TR["root_quat_wxyz"][:, si], joint_pos=TR["joint_pos"][:, si])

  T = min(r["contact"].shape[0], int(2.0 / dt))
  tt = np.arange(T) * dt
  fig, axs = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
  for k in range(4):
    axs[0].fill_between(tt, k + 0.1, k + 0.9, where=r["contact"][:T, 0, k], step="post", color="k")
  axs[0].set_yticks([0.5, 1.5, 2.5, 3.5], LEGS)
  axs[0].set_title(f"IsaacLab, cmd {v:.2f} m/s / {w:+.2f} rad/s, achieved {vx:.3f} / {s['yaw_rate_rad_s']:+.2f}: {s['gait']}, stride {s['stride_hz']:.2f} Hz, "
                   f"duty {np.round(duty, 2).tolist()} (black = stance)")
  for j, lab in ((0, "a"), (1, "e")):
    axs[1].plot(tt, np.degrees(r["q"][:T, 0, j]), label=f"leg1 {lab} actual")
    axs[1].plot(tt, np.degrees(r["target"][:T, 0, j]), "--", label=f"leg1 {lab} target")
  axs[1].set_ylabel("motor angle [deg]"); axs[1].legend(fontsize=7, ncol=4)
  for k in range(4):
    axs[2].plot(tt, 1e3 * r["foot_z"][:T, 0, k], label=LEGS[k])
  axs[2].set_ylabel("foot height [mm]"); axs[2].set_xlabel("t [s]"); axs[2].legend(fontsize=7, ncol=4)
  fig.savefig(out / f"gait_{label}.png", dpi=100, bbox_inches="tight"); plt.close(fig)

(out / "summary.json").write_text(json.dumps(summary, indent=1))
log("\nwrote", out)
LOG.close()
os._exit(0)
