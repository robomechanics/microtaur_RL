"""Turning and virtual-leg measurements on flat ground (A_flat, level 0), one checkpoint per run.

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_turn_legs.py --checkpoint <model.pt> --label y1 --out <dir>

Every A_flat env drives the turn tour of isaac_render_policy.py --tour (straight 0.20 m/s 3 s, left 0.15 m/s
+0.25 rad/s 6 s, straight 2 s, right 6 s, straight 3 s). Per segment, over the envs that have not reset: the
measured yaw rate (body z) and forward speed over the segment's last 4 s (turns) or all of it (straight), and
on the yaw robot the spine angle (mean, std). Virtual leg r = |foot - motor-pivot midpoint| (five-bar FK of the
robot's own legs, kinematics.five_bar_for): the stand value, and over the straight segments per leg the stance
mean / min / max and the swing minimum, theta range; root height above the ground.
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
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True)
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

from microtaur_common.kinematics import five_bar_for  # noqa: E402
from microtaur_common.robot_constants import LEG_JOINT_NAMES, SPINE_JOINT_NAME, STAND_A, STAND_E  # noqa: E402
from microtaur_isaac import terrains as TR  # noqa: E402
from microtaur_isaac.env_cfg import MicrotaurTeacherCurPlayEnvCfg  # noqa: E402
from microtaur_isaac.mdp.contact import foot_contact_timers  # noqa: E402
from microtaur_isaac.mdp.observations import feet_sensor_cfg  # noqa: E402
from microtaur_isaac.policy_loader import load_policy, prepare_env_cfg, variant_kwargs  # noqa: E402

TOUR = [("straight", 0.20, 0.0, 3.0), ("left", 0.15, 0.25, 6.0), ("straight", 0.20, 0.0, 2.0),
        ("right", 0.15, -0.25, 6.0), ("straight", 0.20, 0.0, 3.0)]
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
cfg = MicrotaurTeacherCurPlayEnvCfg(**variant_kwargs(args.checkpoint))
cfg.scene.num_envs = 100  # 10 per column; A_flat = columns 0-1
cfg.sim.device = args.device
cfg.curriculum.command_ranges = None
cfg.commands.twist.resampling_time_range = (1.0e9, 1.0e9)
prepare_env_cfg(cfg, args.checkpoint)
env = ManagerBasedRLEnv(cfg)
t = env.scene.terrain
t.terrain_levels[:] = 0
t.env_origins[:] = t.terrain_origins[0, t.terrain_types]
w = RslRlVecEnvWrapper(env)  # resets: robots spawn at the new origins
policy, policy_reset = load_policy(w, args.checkpoint)
robot, cmd = env.scene["robot"], env.command_manager.get_term("twist")
fc = feet_sensor_cfg()
fc.resolve(env.scene)
yaw = SPINE_JOINT_NAME in robot.joint_names
mids = [robot.joint_names.index(n) for n in LEG_JOINT_NAMES]
sid = robot.joint_names.index(SPINE_JOINT_NAME) if yaw else None
A = (TR.env_terrain_type_ids(t) == 0).cpu().numpy()
K = five_bar_for(yaw)

seg_of = []
for i, (_, _, _, dur) in enumerate(TOUR):
  seg_of += [i] * round(dur / env.step_dt)
rec = {k: [] for k in ("wz", "vx", "spine", "r", "th", "c", "z", "done")}
with torch.inference_mode():
  obs = w.get_observations()
  for k, sg in enumerate(seg_of):
    _, vx, wz, _ = TOUR[sg]
    cmd.vel_command_b[:, 0], cmd.vel_command_b[:, 1], cmd.vel_command_b[:, 2] = vx, 0.0, wz
    obs, _, dones, _ = w.step(policy(obs))
    policy_reset(dones)
    d = robot.data
    q = d.joint_pos[:, mids].double()
    fx, fz = K.forward_torch(q[:, 0::2], q[:, 1::2])
    rec["wz"].append(d.root_link_ang_vel_b[:, 2].cpu().numpy())
    rec["vx"].append(d.root_link_lin_vel_b[:, 0].cpu().numpy())
    rec["spine"].append(d.joint_pos[:, sid].cpu().numpy() if yaw else np.zeros(env.num_envs))
    rec["r"].append(torch.hypot(fx, fz).cpu().numpy())
    rec["th"].append(torch.atan2(fx, -fz).cpu().numpy())
    rec["c"].append(foot_contact_timers(env, fc)[0].cpu().numpy())
    rec["z"].append((d.root_link_pos_w[:, 2] - t.env_origins[:, 2]).cpu().numpy())
    rec["done"].append(dones.cpu().numpy())
R = {k: np.stack(v) for k, v in rec.items()}
seg = np.array(seg_of)
alive = (np.cumsum(R["done"], axis=0) == 0) & A[None]  # [T, N]
res = {"label": args.label, "checkpoint": args.checkpoint, "variant": "yaw" if yaw else "rigid", "n_envs": int(A.sum()),
       "falls": int((R["done"][:, A]).any(axis=0).sum()), "segments": []}
lines = [f"{args.label}: {args.checkpoint}", f"{res['variant']} robot, {res['n_envs']} A_flat envs, turn tour; envs that reset: {res['falls']}",
         "segment  | cmd vx  cmd wz | yaw rate rad/s (ratio) | vx m/s | spine deg mean +- std"]
last4 = round(4.0 / env.step_dt)
for i, (name, vx, wz, dur) in enumerate(TOUR):
  idx = np.nonzero(seg == i)[0]
  if name != "straight":
    idx = idx[-last4:]
  m = alive[idx]
  wz_m, vx_m = float(R["wz"][idx][m].mean()), float(R["vx"][idx][m].mean())
  sp = np.degrees(R["spine"][idx][m])
  row = {"segment": name, "cmd_vx": vx, "cmd_wz": wz, "yaw_rate": wz_m, "vx": vx_m, "spine_deg_mean": float(sp.mean()), "spine_deg_std": float(sp.std())}
  res["segments"].append(row)
  ratio = f"({wz_m / wz:4.2f})" if wz else "      "
  lines.append(f"{name:8s} | {vx:.2f}  {wz:+.2f}  | {wz_m:+.3f} {ratio}        | {vx_m:.3f}  | " +
               (f"{row['spine_deg_mean']:+5.2f} +- {row['spine_deg_std']:.2f}" if yaw else "n/a (rigid)"))
# virtual legs while trotting straight
st_idx = np.nonzero(np.isin(seg, [0, 2, 4]))[0]
st_idx = st_idx[st_idx > round(1.0 / env.step_dt)]  # skip the first second after the reset
m = alive[st_idx]
r, th, c = R["r"][st_idx][m] * 1e3, np.degrees(R["th"][st_idx][m]), R["c"][st_idx][m] > 0.5  # [S, 4]
stand = [K.forward_numpy(STAND_A[j], STAND_E[j], leg_index=j + 1) for j in range(4)]
res["stand_r_mm"] = [float(np.hypot(s.foot_x, s.foot_z) * 1e3) for s in stand]
res["root_z_mm"] = float(R["z"][st_idx][m].mean() * 1e3)
lines += ["", f"virtual leg r (mm), straight trotting; stand pose r = {res['stand_r_mm'][0]:.2f} mm; root height {res['root_z_mm']:.1f} mm",
          "leg | stance mean  min   max | swing min (retracted) | theta range deg"]
res["legs"] = []
for j in range(4):
  s, sw = r[:, j][c[:, j]], r[:, j][~c[:, j]]
  stat = lambda v, fn: float(fn(v)) if v.size else None  # noqa: E731  (a leg may never swing / never touch down)
  row = {"leg": j + 1, "stance_mean": stat(s, np.mean), "stance_min": stat(s, np.min), "stance_max": stat(s, np.max),
         "swing_min": stat(sw, np.min), "theta_min": stat(th[:, j], np.min), "theta_max": stat(th[:, j], np.max)}
  res["legs"].append(row)
  f = lambda v, w=6, p=1, sign="": " " * (w - 3) + "n/a" if v is None else f"{v:{sign}{w}.{p}f}"  # noqa: E731
  lines.append(f" {j + 1}  | {f(row['stance_mean'])}  {f(row['stance_min'], 5)} {f(row['stance_max'], 5)} | {f(row['swing_min'])} "
               f"               | {f(row['theta_min'], 5, 1, '+')} .. {f(row['theta_max'], 5, 1, '+')}")
(out / f"turn_legs_{args.label}.txt").write_text("\n".join(lines) + "\n")
(out / f"turn_legs_{args.label}.json").write_text(json.dumps(res, indent=1))
print("\n".join(lines), flush=True)
sys.stdout.flush()
os._exit(0)
