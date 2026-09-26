"""Evaluate a trained rigid Microtaur policy's gait at fixed forward commands.

  DISPLAY=:1 MUJOCO_GL=glfw python scripts/eval_gait.py --checkpoint <model.pt> \
      --out <dir> [--speeds 0.10 0.15 0.20 0.25 0.30] [--num-envs 16]

For each commanded speed (straight, no turning, no standing envs, flat, play
config: no noise or randomisation) the policy runs 2 s to settle and 6 s
measured. Reported per speed:

  tracking   achieved forward speed vs command, lateral speed, yaw rate
  gait       duty factor per leg, stride frequency, onset phase of each leg
             relative to leg 1 -> trot (diagonals together) / pace / bound
  feet       swing clearance (peak foot height above ground while unloaded)
  body       height and roll/pitch oscillation
  effort     motor travel used (p95 |q - stand| / 0.75 rad), torque saturation
             (|tau| >= 99% of the 0.129 N m cap), target-vs-actual tracking
             error (compliance of the KP = 1, KD = 0 servo model), power, CoT

Writes summary.json, gait_<v>.png (contact diagram + leg-1 motor traces) and
frames_<v>.png (side-view frames of env 0 over ~1 s).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.viewer.viewer_config import ViewerConfig  # noqa: E402

from microtaur_rigid.env_cfg import make_env_cfg  # noqa: E402
from microtaur_rigid.rewards import XL330_COPPER_W_PER_NM2  # noqa: E402
from microtaur_rigid.rl_cfg import microtaur_velocity_ppo_runner_cfg  # noqa: E402
from microtaur_rigid.robot import (  # noqa: E402
  EFFORT_LIMIT_NM, EXPECTED_TOTAL_MASS_KG, FOOT_SITE_NAMES, JOINT_HALF_RANGE_RAD, LEG_JOINT_NAMES, ROOT_BODY,
)

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--speeds", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.25, 0.30])
ap.add_argument("--num-envs", type=int, default=16)
ap.add_argument("--settle-s", type=float, default=2.0)
ap.add_argument("--measure-s", type=float, default=6.0)
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

FOOT_R = 0.0062
LEGS = ("leg1 RR", "leg2 RL", "leg3 FL", "leg4 FR")
PAIRS = {"trot": ((0, 2), (1, 3)), "pace": ((0, 3), (1, 2)), "bound": ((0, 1), (2, 3))}


def build(v: float):
  cfg = make_env_cfg(play=True)
  cfg.scene.num_envs = args.num_envs
  cmd = cfg.commands["twist"]
  cmd.ranges.lin_vel_x = (v, v)
  cmd.ranges.ang_vel_z = (0.0, 0.0)
  cmd.rel_standing_envs = 0.0
  cfg.curriculum.pop("command_ranges")  # would overwrite the fixed command on reset
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name=ROOT_BODY,
    distance=0.42, elevation=-8.0, azimuth=90.0, width=480, height=320, max_extra_envs=0,
  )
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
  agent = microtaur_velocity_ppo_runner_cfg()
  runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device="cuda:0")
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  return env, wrapped, runner.get_inference_policy(device="cuda:0")


def onset_phases(contact: np.ndarray, dt: float):
  """Stride frequency from leg-1 touchdowns and each leg's touchdown phase vs leg 1."""
  onsets = [np.flatnonzero(~contact[:-1, k] & contact[1:, k]) + 1 for k in range(4)]
  if len(onsets[0]) < 3:
    return float("nan"), [float("nan")] * 4
  period = float(np.median(np.diff(onsets[0])))
  phases = []
  for k in range(4):
    if len(onsets[k]) == 0:
      phases.append(float("nan"))
      continue
    rel = []
    for t in onsets[k]:
      prev = onsets[0][onsets[0] <= t]
      if len(prev):
        rel.append(((t - prev[-1]) / period) % 1.0)
    # circular mean
    a = 2 * np.pi * np.asarray(rel)
    phases.append(float((np.angle(np.mean(np.exp(1j * a))) / (2 * np.pi)) % 1.0) if len(rel) else float("nan"))
  return 1.0 / (period * dt), phases


def classify(phases):
  """Gait whose in-phase pairs have the smallest phase gap (and anti-phase partner ~0.5)."""
  p = np.asarray(phases)
  if np.any(np.isnan(p)):
    return "unclear"
  def gap(a, b):
    d = abs(p[a] - p[b]) % 1.0
    return min(d, 1 - d)
  score = {}
  for name, (x, y) in PAIRS.items():
    score[name] = gap(*x) + gap(*y) + abs(0.5 - gap(x[0], y[0]))
  best = min(score, key=score.get)
  return best if score[best] < 0.25 else f"irregular (closest {best}, score {score[best]:.2f})"


summary = {"checkpoint": args.checkpoint, "speeds": {}}
for v in args.speeds:
  env, wrapped, policy = build(v)
  robot = env.scene["robot"]
  contact_sensor = env.scene["feet_ground_contact"]
  term = env.action_manager.get_term("joint_pos")
  mids, _ = robot.find_joints(list(LEG_JOINT_NAMES), preserve_order=True)
  fids, _ = robot.find_sites(list(FOOT_SITE_NAMES), preserve_order=True)
  dt = env.step_dt
  n_settle, n_meas = int(args.settle_s / dt), int(args.measure_s / dt)
  obs = wrapped.get_observations()
  rec = {k: [] for k in ("vx", "vy", "wz", "z", "roll", "pitch", "contact", "foot_z", "q", "target", "tau", "mech", "copper")}
  frames, resets = [], 0
  with torch.inference_mode():
    for t in range(n_settle + n_meas):
      obs, _, dones, _ = wrapped.step(policy(obs))
      if t < n_settle:
        continue
      resets += int(dones.sum())
      d = robot.data
      g = d.projected_gravity_b
      tau = d.qfrc_actuator[:, mids]
      qd = d.joint_vel[:, mids]
      rec["vx"].append(d.root_link_lin_vel_b[:, 0]); rec["vy"].append(d.root_link_lin_vel_b[:, 1])
      rec["wz"].append(d.root_link_ang_vel_b[:, 2])
      rec["z"].append(d.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2])
      rec["roll"].append(torch.atan2(-g[:, 1], -g[:, 2])); rec["pitch"].append(torch.asin(torch.clamp(g[:, 0], -1, 1)))
      rec["contact"].append(contact_sensor.data.found.reshape(args.num_envs, -1)[:, :4] > 0)
      rec["foot_z"].append(d.site_pos_w[:, fids, 2] - env.scene.env_origins[:, None, 2] - FOOT_R)
      rec["q"].append(term.resolved_to_canonical(d.joint_pos[:, term._target_ids]))
      rec["target"].append(term.applied_targets)
      rec["tau"].append(tau)
      rec["mech"].append((tau * qd).abs().sum(1)); rec["copper"].append(XL330_COPPER_W_PER_NM2 * (tau**2).sum(1))
      if t - n_settle < 48 and (t - n_settle) % 2 == 0:
        frames.append(env.render())
  R = {k: torch.stack(val).cpu().numpy() for k, val in rec.items()}  # [T, N, ...]
  env.close()

  stand = np.empty(8); stand[0::2] = (0.45, -0.45, -0.45, 0.45); stand[1::2] = (-0.45, 0.45, 0.45, -0.45)
  vx = R["vx"].mean()
  per_env_speed = R["vx"].mean(0)
  duty = R["contact"].mean(axis=(0, 1))
  freqs, phase_list = [], []
  for e in range(args.num_envs):
    f, ph = onset_phases(R["contact"][:, e, :], dt)
    freqs.append(f); phase_list.append(ph)
  phases = [float(np.nanmedian([p[k] for p in phase_list])) for k in range(4)]
  swing = ~R["contact"]
  clearance = [float(np.percentile(R["foot_z"][:, :, k][swing[:, :, k]], 95) * 1e3) if swing[:, :, k].any() else 0.0 for k in range(4)]
  travel = np.percentile(np.abs(R["q"] - stand), 95, axis=(0, 1)) / JOINT_HALF_RANGE_RAD
  err = R["target"] - R["q"]
  sat = np.mean(np.abs(R["tau"]) >= 0.99 * EFFORT_LIMIT_NM)
  power = R["mech"].mean() + R["copper"].mean()
  s = {
    "cmd_m_s": v, "speed_m_s": float(vx), "speed_ratio": float(vx / v), "speed_env_min_max": [float(per_env_speed.min()), float(per_env_speed.max())],
    "lateral_m_s_rms": float(np.sqrt((R["vy"] ** 2).mean())), "yaw_rate_rad_s_rms": float(np.sqrt((R["wz"] ** 2).mean())),
    "resets_during_measure": resets,
    "duty_factor": [round(float(x), 3) for x in duty], "stride_hz": float(np.nanmedian(freqs)),
    "touchdown_phase_vs_leg1": [round(p, 3) for p in phases], "gait": classify(phases),
    "swing_clearance_p95_mm": [round(c, 1) for c in clearance],
    "body_height_mm": [float(1e3 * R["z"].mean()), float(1e3 * R["z"].std())],
    "roll_pitch_std_deg": [float(np.degrees(R["roll"].std())), float(np.degrees(R["pitch"].std()))],
    "motor_travel_p95_frac_of_limit": [round(float(x), 2) for x in travel],
    "tracking_err_rms_deg": float(np.degrees(np.sqrt((err**2).mean()))), "tracking_err_p95_deg": float(np.degrees(np.percentile(np.abs(err), 95))),
    "torque_saturated_frac": float(sat), "torque_rms_nm": float(np.sqrt((R["tau"] ** 2).mean())),
    "power_mech_w": float(R["mech"].mean()), "power_copper_w": float(R["copper"].mean()),
    "electrical_cot": float(power / (EXPECTED_TOTAL_MASS_KG * 9.81 * max(vx, 1e-3))),
  }
  summary["speeds"][f"{v:.2f}"] = s
  print(f"\n=== cmd {v:.2f} m/s")
  for k, val in s.items():
    print(f"  {k:32s} {val}")

  # Gait diagram + leg-1 traces for env 0
  T = min(R["contact"].shape[0], int(2.0 / dt))
  tt = np.arange(T) * dt
  fig, axs = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
  for k in range(4):
    c = R["contact"][:T, 0, k]
    axs[0].fill_between(tt, k + 0.1, k + 0.9, where=c, step="post", color="k")
  axs[0].set_yticks([0.5, 1.5, 2.5, 3.5], LEGS)
  axs[0].set_title(f"cmd {v:.2f} m/s, achieved {vx:.3f}: {s['gait']}, stride {s['stride_hz']:.2f} Hz, "
                   f"duty {np.round(duty, 2).tolist()} (black = stance)")
  for j, lab in ((0, "a"), (1, "e")):
    axs[1].plot(tt, np.degrees(R["q"][:T, 0, j]), label=f"leg1 {lab} actual")
    axs[1].plot(tt, np.degrees(R["target"][:T, 0, j]), "--", label=f"leg1 {lab} target")
  axs[1].set_ylabel("motor angle [deg]"); axs[1].legend(fontsize=7, ncol=4)
  for k in range(4):
    axs[2].plot(tt, 1e3 * R["foot_z"][:T, 0, k], label=LEGS[k])
  axs[2].set_ylabel("foot height [mm]"); axs[2].set_xlabel("t [s]"); axs[2].legend(fontsize=7, ncol=4)
  fig.savefig(out / f"gait_{v:.2f}.png", dpi=100, bbox_inches="tight"); plt.close(fig)

  # Frame strip: 24 frames at 25 Hz (~1 s)
  if frames:
    cols = 6
    rows = math.ceil(len(frames) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.2))
    for i, ax in enumerate(np.ravel(axs)):
      ax.axis("off")
      if i < len(frames):
        ax.imshow(frames[i]); ax.set_title(f"t = {i * 2 * dt:.2f} s", fontsize=8)
    fig.suptitle(f"env 0 side view, cmd {v:.2f} m/s")
    fig.savefig(out / f"frames_{v:.2f}.png", dpi=90, bbox_inches="tight"); plt.close(fig)

(out / "summary.json").write_text(json.dumps(summary, indent=1))
print("\nwrote", out)
