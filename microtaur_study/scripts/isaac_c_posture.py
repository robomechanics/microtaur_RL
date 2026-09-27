"""Posture on the C step lane: does the policy keep the body level by shortening the legs on the
raised side (a stance offset), or lean? Teacher-Cur-Play, C lanes at levels 0 / 3 / 6, straight 0.20 m/s.

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_c_posture.py --checkpoint <model.pt> --out <dir>

Per level, over steps where the robot is on the raised section (x inside the step): body roll
(deg, + = left side up), and the stance leg length r (five-bar FK from the two motor angles,
leg plane) of the left (raised-side) legs minus the right (low-side) legs. A level body on a
step of height h needs r_right - r_left ~= h (36.8 mm at level 6).
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
ap.add_argument("--steps", type=int, default=400)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from microtaur_common.kinematics import MicrotaurFiveBarKinematics  # noqa: E402
from microtaur_common.robot_constants import LEG_JOINT_NAMES  # noqa: E402
from microtaur_isaac import FOOT_BODY_NAMES  # noqa: E402
from microtaur_isaac import terrains as TR  # noqa: E402
from microtaur_isaac.agents import MicrotaurTeacherPPORunnerCfg  # noqa: E402
from microtaur_isaac.env_cfg import MicrotaurTeacherCurPlayEnvCfg  # noqa: E402
from microtaur_isaac.mdp.contact import foot_contact_timers  # noqa: E402
from microtaur_isaac.mdp.observations import feet_sensor_cfg  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
N = 60  # 6 per column -> C columns 7-9 = envs 42..59
cfg = MicrotaurTeacherCurPlayEnvCfg()
cfg.scene.num_envs = N
cfg.curriculum.command_ranges = None
cfg.commands.twist.resampling_time_range = (1.0e9, 1.0e9)
env = ManagerBasedRLEnv(cfg)
t = env.scene.terrain
LV = [0, 3, 6]
t.terrain_levels[:] = torch.tensor([LV[i % 3] for i in range(N)], device=env.device)
t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]
w = RslRlVecEnvWrapper(env)
runner = OnPolicyRunner(w, MicrotaurTeacherPPORunnerCfg().to_dict(), log_dir=None, device=env.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.device)
robot = env.scene["robot"]
cmd = env.command_manager.get_term("twist")
fc = feet_sensor_cfg()
fc.resolve(env.scene)
mids = [robot.joint_names.index(n) for n in LEG_JOINT_NAMES]
fids = [robot.body_names.index(n) for n in FOOT_BODY_NAMES]
isc = (TR.env_terrain_type_ids(t) == 2).cpu().numpy()
lv = t.terrain_levels.cpu().numpy()
rec = {k: [] for k in ("q", "c", "x", "roll", "fy", "done")}
with torch.inference_mode():
  obs = w.get_observations()
  for k in range(args.steps):
    cmd.vel_command_b[:, 0] = args.speed
    cmd.vel_command_b[:, 1:] = 0.0
    obs, _, dones, _ = w.step(policy(obs))
    d = robot.data
    g = d.projected_gravity_b
    rel = d.body_link_pos_w[:, fids] - d.root_link_pos_w[:, None]
    fy = quat_apply_inverse(d.root_link_quat_w[:, None].expand(-1, 4, -1).reshape(-1, 4), rel.reshape(-1, 3)).reshape(N, 4, 3)[..., 1]
    rec["q"].append(d.joint_pos[:, mids].cpu().numpy())
    rec["c"].append(foot_contact_timers(env, fc)[0].cpu().numpy().copy())
    rec["x"].append((d.root_link_pos_w[:, 0] - t.env_origins[:, 0]).cpu().numpy())
    rec["roll"].append(torch.rad2deg(torch.atan2(-g[:, 1], -g[:, 2])).cpu().numpy())  # + = left side up
    rec["fy"].append(fy.cpu().numpy())
    rec["done"].append(dones.cpu().numpy())
R = {k: np.stack(v) for k, v in rec.items()}
KIN = MicrotaurFiveBarKinematics()
x0, x1 = TR.C_STEP_X_RANGE
res = {}
lines = [f"checkpoint {args.checkpoint}", f"C lane, straight {args.speed} m/s, {args.steps} steps; on-step = root x inside [{x0 + 0.15:.2f}, {x1:.2f}] m",
         "lvl | step mm | n samples | roll deg mean (+ left up) | stance r left - right mm | r_left mm | r_right mm | resets"]
for L in LV:
  sel = np.nonzero(isc & (lv == L))[0]
  roll, dr, rl, rr = [], [], [], []
  for e in sel:
    alive = np.cumsum(R["done"][:, e]) == 0
    on = alive & (R["x"][:, e] > x0 + 0.15) & (R["x"][:, e] < x1)
    for s in np.nonzero(on)[0]:
      r = np.array([np.hypot(*(lambda f: (f.foot_x, f.foot_z))(KIN.forward_numpy(R["q"][s, e, 2 * j], R["q"][s, e, 2 * j + 1], j + 1))) for j in range(4)])
      left = R["fy"][s, e] > 0
      st = R["c"][s, e] > 0.5
      if (st & left).any() and (st & ~left).any():
        rl.append(r[st & left].mean()); rr.append(r[st & ~left].mean()); dr.append(rl[-1] - rr[-1])
      roll.append(R["roll"][s, e])
  h = L / (TR.CUR_NUM_LEVELS - 1) * TR.C_DELTA_M * TR.CUR_TERRAIN_SCALE * 1e3
  row = {"level": L, "step_mm": h, "n": len(dr), "roll_deg": float(np.mean(roll)) if roll else None,
         "dr_left_minus_right_mm": float(np.mean(dr) * 1e3) if dr else None,
         "r_left_mm": float(np.mean(rl) * 1e3) if rl else None, "r_right_mm": float(np.mean(rr) * 1e3) if rr else None,
         "resets": int(R["done"][:, sel].sum())}
  res[L] = row
  f = lambda v: "n/a" if v is None else f"{v:+.1f}"  # noqa: E731
  lines.append(f"{L:3d} | {h:7.1f} | {row['n']:9d} | {f(row['roll_deg']):>8s} | {f(row['dr_left_minus_right_mm']):>8s} | "
               f"{f(row['r_left_mm']):>7s} | {f(row['r_right_mm']):>7s} | {row['resets']}")
lines.append("target on a level body: r_left - r_right ~= -step height (legs on the raised left side shorter), roll ~ 0")
(out / "c_posture.txt").write_text("\n".join(lines) + "\n")
(out / "c_posture.json").write_text(json.dumps(res, indent=1))
print("\n".join(lines), flush=True)
sys.stdout.flush()
os._exit(0)
