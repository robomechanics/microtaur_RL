"""Servo command vs response at the physics rate (2.5 ms) for one robot on flat ground: policy target
(28.6 Hz steps), target after the safety filter / delay, the actuator target, joint angle, torque.
Two passes: the current zero-order hold (target held for the 14 substeps) and, patched in for this
test only, linear interpolation from the previous to the new target over the 14 substeps.

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_servo_curves.py --checkpoint <model.pt> --out <dir>
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
ap.add_argument("--speed", type=float, default=0.20)
ap.add_argument("--interp", action="store_true", help="patch in linear target interpolation over the substeps")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from microtaur_common.kinematics import MicrotaurFiveBarKinematics  # noqa: E402
from microtaur_common.robot_constants import LEG_JOINT_NAMES, STAND_A, STAND_E  # noqa: E402
from microtaur_common.walk_action import ACTION_SCALE_RAD  # noqa: E402
from microtaur_isaac import terrains as TR  # noqa: E402
from microtaur_isaac.agents import MicrotaurTeacherPPORunnerCfg  # noqa: E402
from microtaur_isaac.env_cfg import MicrotaurTeacherCurPlayEnvCfg  # noqa: E402
from microtaur_isaac.mdp.contact import foot_contact_timers  # noqa: E402
from microtaur_isaac.mdp.observations import feet_sensor_cfg  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
SUB_DT = 0.0025


def run(interp: bool):
  N = 20  # 2 per column; all level 0 (flat) -> A envs 0, 1
  cfg = MicrotaurTeacherCurPlayEnvCfg()
  cfg.scene.num_envs = N
  cfg.curriculum.command_ranges = None
  cfg.commands.twist.resampling_time_range = (1.0e9, 1.0e9)
  env = ManagerBasedRLEnv(cfg)
  t = env.scene.terrain
  t.terrain_levels[:] = 0
  t.env_origins[:] = t.terrain_origins[0, t.terrain_types]
  w = RslRlVecEnvWrapper(env)
  runner = OnPolicyRunner(w, MicrotaurTeacherPPORunnerCfg().to_dict(), log_dir=None, device=env.device)
  runner.load(args.checkpoint)
  policy = runner.get_inference_policy(device=env.device)
  robot = env.scene["robot"]
  term = env.action_manager.get_term("joint_pos")
  mids = [robot.joint_names.index(n) for n in LEG_JOINT_NAMES]
  dec = env.cfg.decimation
  state = {"k": 0, "prev": None}
  orig_process, orig_apply = term.process_actions, term.apply_actions

  def process(actions):
    state["prev"] = term._processed.clone()
    orig_process(actions)
    state["k"] = 0

  def apply():
    if interp and state["prev"] is not None:
      a = (state["k"] + 1) / dec
      keep = term._processed.clone()
      term._processed[:] = state["prev"] + a * (keep - state["prev"])
      orig_apply()
      term._processed[:] = keep
    else:
      orig_apply()
    state["k"] += 1

  term.process_actions, term.apply_actions = process, apply
  rec = {k: [] for k in ("raw", "applied", "target", "q", "tau")}
  orig_mgr_apply = env.action_manager.apply_action

  def mgr_apply():
    orig_mgr_apply()
    d = robot.data
    rec["raw"].append(term._raw[0].cpu().numpy().copy())
    rec["applied"].append(term._processed[0].cpu().numpy().copy())
    rec["target"].append(d.joint_pos_target[0, mids].cpu().numpy().copy())
    rec["q"].append(d.joint_pos[0, mids].cpu().numpy().copy())
    rec["tau"].append(d.applied_torque[0, mids].cpu().numpy().copy())

  env.action_manager.apply_action = mgr_apply
  fc = feet_sensor_cfg()
  fc.resolve(env.scene)
  types = TR.env_terrain_type_ids(t).cpu().numpy()
  C, VX, D = [], [], []
  with torch.inference_mode():
    obs = w.get_observations()
    for k in range(200):
      cmd = env.command_manager.get_term("twist")
      cmd.vel_command_b[:, 0] = args.speed
      cmd.vel_command_b[:, 1:] = 0.0
      obs, _, dones, _ = w.step(policy(obs))
      if k >= 60:
        C.append(foot_contact_timers(env, fc)[0].cpu().numpy().copy())
        VX.append(robot.data.root_link_lin_vel_b[:, 0].cpu().numpy())
        D.append(dones.cpu().numpy())
  R = {k: np.stack(v) for k, v in rec.items()}
  C, VX, D = np.stack(C).astype(float), np.stack(VX), np.stack(D)
  ok = ~D.any(0)

  def corr(a, b):
    a = a - a.mean(0); b = b - b.mean(0)
    return float(np.nanmean((a * b).mean(0) / (a.std(0) * b.std(0) + 1e-9)))

  s = slice(60 * dec, None)
  q = R["q"][s]
  acc = np.diff(q, 2, axis=0) / SUB_DT**2
  dtau = np.diff(R["tau"][s], axis=0) / SUB_DT
  m = {"vx_mean": float(VX[:, ok].mean()), "diag": np.mean([corr(C[:, ok, 0], C[:, ok, 2]), corr(C[:, ok, 1], C[:, ok, 3])]),
       "same_end": np.mean([corr(C[:, ok, 0], C[:, ok, 1]), corr(C[:, ok, 2], C[:, ok, 3])]),
       "flight": float((C[:, ok].sum(2) == 0).mean()), "resets": int(D.sum()),
       "joint_acc_rms": float(np.sqrt((acc**2).mean())), "joint_acc_p99": float(np.percentile(np.abs(acc), 99)),
       "torque_rate_rms": float(np.sqrt((dtau**2).mean())), "target_step_rms": float(np.sqrt((np.diff(R["target"][s], axis=0) ** 2).mean()))}
  return R, m, dec


tag = "interp" if args.interp else "hold"
R0, m0, dec = run(args.interp)
np.savez(out / f"servo_{tag}.npz", **R0)
(out / f"servo_{tag}.json").write_text(json.dumps(m0, indent=1))
print(tag, json.dumps(m0), flush=True)
if not all((out / f"servo_{x}.npz").exists() for x in ("hold", "interp")):
  sys.stdout.flush()
  os._exit(0)
data, res = {}, {}
for x, lab in (("hold", "hold (current)"), ("interp", "linear interp")):
  data[lab] = dict(np.load(out / f"servo_{x}.npz"))
  res[lab] = json.loads((out / f"servo_{x}.json").read_text())

# ---- figure: leg 1, both motors, 0.6 s ----
KIN = MicrotaurFiveBarKinematics()
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
t0 = 120 * dec
n = int(0.6 / SUB_DT)
tt = np.arange(n) * SUB_DT * 1e3
fig, ax = plt.subplots(3, 2, figsize=(14, 10), sharex="col")
for col, lab in enumerate(data):
  R = data[lab]
  for row, j in enumerate((0, 1)):
    a = ax[row, col]
    stand = (STAND_A[0], STAND_E[0])[j]  # leg 1: (a, e)
    req = np.degrees(stand + ACTION_SCALE_RAD * np.clip(R["raw"][t0:t0 + n, j], -1, 1))
    a.step(tt, req, where="post", color=C4, lw=1.2, label="policy target = stand + 30 deg x action (28.6 Hz)")
    a.step(tt, np.degrees(R["applied"][t0:t0 + n, j]), where="post", color=C2, lw=1.2, label="after safety filter + 1-step delay")
    a.plot(tt, np.degrees(R["target"][t0:t0 + n, j]), color=C1, lw=2, label="actuator target (per 2.5 ms substep)")
    a.plot(tt, np.degrees(R["q"][t0:t0 + n, j]), color=C3, lw=2, label="joint angle")
    a.set_ylabel(f"{LEG_JOINT_NAMES[j]} (deg)")
    a.grid(alpha=0.25)
    if row == 0:
      a.set_title(f"{lab}: joint acc rms {res[lab]['joint_acc_rms']:.0f} rad/s^2, torque rate rms {res[lab]['torque_rate_rms']:.1f} N m/s")
  a = ax[2, col]
  q = R["q"][t0:t0 + n]
  F = np.array([(lambda f: (f.foot_x, f.foot_z))(KIN.forward_numpy(x[0], x[1], 1)) for x in q]) * 1e3
  a.plot(tt, F[:, 0], color=C1, lw=2, label="foot x in leg plane")
  a.plot(tt, F[:, 1] - F[:, 1].mean(), color=C2, lw=2, label="foot z (mean removed)")
  a.set_ylabel("foot (mm, FK)")
  a.set_xlabel("time (ms)")
  a.grid(alpha=0.25)
for a in ax.flat:
  a.legend(fontsize=7, loc="upper right")
fig.suptitle(f"{Path(args.checkpoint).parent.name}/{Path(args.checkpoint).name}: leg 1 motors, flat, {args.speed} m/s")
fig.tight_layout()
fig.savefig(out / "servo_curves.png", dpi=110)
print("FIG", out / "servo_curves.png", flush=True)
sys.stdout.flush()
os._exit(0)
