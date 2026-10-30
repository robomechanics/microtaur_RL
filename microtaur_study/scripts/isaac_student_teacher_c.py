"""Why does a student fail on the C step lane where its teacher succeeds? Teacher-Cur-Play, C lanes at the given
levels, straight 0.20 m/s, raised side on the left.

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_student_teacher_c.py --student <model.pt> --teacher <model.pt> \
      --label y1 --out <dir> [--levels 3 6] [--seconds 20]

Two passes on the same lanes, one driven by each policy; every step both policies see the same state (student
group / teacher group) and both actions are recorded. Regions by root x relative to the raised section
[x0, x1]: approach (x < x0 - 0.10), edge (x0 - 0.10 .. x0 + 0.15), on step (x0 + 0.15 .. x1). Per pass, level
and region: samples, roll (+ = left up), stance r left - right, leg-motor torque saturation, |student - teacher|
action (clipped to +-1) on the raised-side legs (2, 3), the low-side legs (1, 4) and the spine; per pass and
level: furthest x, time to reach the step, terminations by cause.
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
ap.add_argument("--student", required=True)
ap.add_argument("--teacher", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--levels", type=int, nargs="+", default=[3, 6])
ap.add_argument("--seconds", type=float, default=20.0)
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
from microtaur_common.robot_constants import LEG_JOINT_NAMES, SPINE_JOINT_NAME  # noqa: E402
from microtaur_isaac import terrains as TR  # noqa: E402
from microtaur_isaac.env_cfg import MicrotaurTeacherCurPlayEnvCfg  # noqa: E402
from microtaur_isaac.mdp.contact import foot_contact_timers  # noqa: E402
from microtaur_isaac.mdp.observations import feet_sensor_cfg  # noqa: E402
from microtaur_isaac.policy_loader import load_policy, prepare_env_cfg, variant_kwargs  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
N = 60  # 6 per column; C = columns 7-9
cfg = MicrotaurTeacherCurPlayEnvCfg(**variant_kwargs(args.student))
cfg.scene.num_envs = N
cfg.sim.device = args.device
cfg.c_reverse_frac = 0.0
cfg.curriculum.command_ranges = None
cfg.commands.twist.resampling_time_range = (1.0e9, 1.0e9)
prepare_env_cfg(cfg, args.student)  # adds the student group (history) next to the teacher group
env = ManagerBasedRLEnv(cfg)
t = env.scene.terrain
LV = args.levels
t.terrain_levels[:] = torch.tensor([LV[i % len(LV)] for i in range(N)], device=env.device)
t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]
w = RslRlVecEnvWrapper(env)
act_s, reset_s = load_policy(w, args.student)
act_t, reset_t = load_policy(w, args.teacher)
robot, cmd, tm = env.scene["robot"], env.command_manager.get_term("twist"), env.termination_manager
fc = feet_sensor_cfg()
fc.resolve(env.scene)
yaw = SPINE_JOINT_NAME in robot.joint_names
mids = [robot.joint_names.index(n) for n in LEG_JOINT_NAMES]
K = five_bar_for(yaw)
isc = (TR.env_terrain_type_ids(t) == 2).cpu().numpy()
lv = t.terrain_levels.cpu().numpy()
x0, x1 = TR.C_STEP_X_RANGE
steps = round(args.seconds / env.step_dt)
LEFT, RIGHT = [1, 2], [0, 3]  # legs 2 (rear left), 3 (front left) are on the raised side
na = 9 if yaw else 8
res = {"label": args.label, "student": args.student, "teacher": args.teacher, "passes": {}}
lines = [f"{args.label}: student {args.student}", f"  teacher {args.teacher}",
         f"C lanes levels {LV}, straight 0.20 m/s, {args.seconds:.0f} s; raised section x in [{x0:.2f}, {x1:.2f}] m"]
# both passes in inference mode: the second reset touches tensors made by the first
with torch.inference_mode():
  for driver in ("student", "teacher"):
    obs, _ = w.reset()
    reset_s(torch.ones(N, dtype=torch.bool, device=env.device))
    reset_t(torch.ones(N, dtype=torch.bool, device=env.device))
    rec = {k: [] for k in ("x", "roll", "dr", "sat", "da", "done", "cause")}
    for k in range(steps):
      cmd.vel_command_b[:, 0], cmd.vel_command_b[:, 1:] = 0.20, 0.0
      a_s, a_t = act_s(obs), act_t(obs)
      obs, _, dones, _ = w.step(a_s if driver == "student" else a_t)
      reset_s(dones)
      reset_t(dones)
      d = robot.data
      g = d.projected_gravity_b
      q = d.joint_pos[:, mids].double()
      fx, fz = K.forward_torch(q[:, 0::2], q[:, 1::2])
      r = torch.hypot(fx, fz).float()
      c = foot_contact_timers(env, fc)[0].float()
      rl = (r[:, LEFT] * c[:, LEFT]).sum(1) / c[:, LEFT].sum(1).clamp(min=1)
      rr = (r[:, RIGHT] * c[:, RIGHT]).sum(1) / c[:, RIGHT].sum(1).clamp(min=1)
      ok = (c[:, LEFT].sum(1) > 0) & (c[:, RIGHT].sum(1) > 0)
      rec["x"].append((d.root_link_pos_w[:, 0] - t.env_origins[:, 0]).cpu().numpy())
      rec["roll"].append(torch.rad2deg(torch.atan2(-g[:, 1], -g[:, 2])).cpu().numpy())
      rec["dr"].append(torch.where(ok, rl - rr, torch.full_like(rl, float("nan"))).cpu().numpy() * 1e3)
      rec["sat"].append(((d.computed_torque[:, mids] - d.applied_torque[:, mids]).abs() > 1e-5).float().mean(1).cpu().numpy())
      rec["da"].append((a_s.clamp(-1, 1) - a_t.clamp(-1, 1)).abs()[:, :na].cpu().numpy())
      rec["done"].append(dones.cpu().numpy())
      cause = np.full(N, "", dtype=object)
      for n in tm.active_terms:  # get_term is the sticky last reason; read it only for envs done this step
        v = tm.get_term(n).cpu().numpy() & dones.cpu().numpy().astype(bool)
        cause[v] = n
      rec["cause"].append(cause)
    R = {k: np.stack(v) for k, v in rec.items()}
    alive = np.cumsum(R["done"], axis=0) == 0
    P = {}
    lines += ["", f"=== {driver} drives",
              "lvl region   | samples | roll deg | stance r L-R mm | torque sat | |a_s - a_t| raised legs  low legs  spine"]
    for L in LV:
      sel = isc & (lv == L)
      h = L / (TR.CUR_NUM_LEVELS - 1) * TR.C_DELTA_M * TR.CUR_TERRAIN_SCALE * 1e3
      X = R["x"][:, sel]
      regions = {"approach": X < x0 - 0.10, "edge": (X >= x0 - 0.10) & (X < x0 + 0.15), "on step": (X >= x0 + 0.15) & (X < x1)}
      rows = {}
      for name, m in regions.items():
        m = m & alive[:, sel]
        n = int(m.sum())
        da = R["da"][:, sel][m]  # [S, na]
        left_idx = [2, 3, 4, 5]
        low_idx = [0, 1, 6, 7]
        row = {"n": n, "roll": float(np.nanmean(R["roll"][:, sel][m])) if n else None,
               "dr": float(np.nanmean(R["dr"][:, sel][m])) if n else None, "sat": float(R["sat"][:, sel][m].mean()) if n else None,
               "da_raised": float(da[:, left_idx].mean()) if n else None, "da_low": float(da[:, low_idx].mean()) if n else None,
               "da_spine": float(da[:, 8].mean()) if (n and yaw) else None}
        rows[name] = row
        f = lambda v, p=2: "  n/a" if v is None else f"{v:+.{p}f}"  # noqa: E731
        lines.append(f"{L:3d} {name:8s} | {n:7d} | {f(row['roll'], 1):>8s} | {f(row['dr'], 1):>15s} | {f(row['sat']):>10s} | "
                     f"{f(row['da_raised']):>21s} {f(row['da_low']):>9s} {f(row['da_spine']):>6s}")
      reach = [(np.argmax(X[:, e] >= x0 + 0.15) * env.step_dt if (X[:, e] >= x0 + 0.15).any() else None) for e in range(X.shape[1])]
      causes = {}
      for e in np.nonzero(sel)[0]:
        for cs in R["cause"][:, e][R["done"][:, e].astype(bool)]:
          causes[cs] = causes.get(cs, 0) + 1
      summ = {"step_mm": h, "envs": int(sel.sum()), "furthest_x_mean": float(np.nanmax(np.where(alive[:, sel], X, np.nan), axis=0).mean()),
              "reached_step": int(sum(v is not None for v in reach)), "t_reach_s": [None if v is None else round(float(v), 2) for v in reach],
              "terminations": causes}
      P[L] = {"regions": rows, "summary": summ}
      lines.append(f"    lvl {L} ({h:.1f} mm): {summ['reached_step']}/{summ['envs']} reached the step (x > x0 + 0.15), furthest x mean "
                   f"{summ['furthest_x_mean']:+.2f} m, reach times {summ['t_reach_s']}, terminations {causes}")
    res["passes"][driver] = P
(out / f"student_teacher_c_{args.label}.txt").write_text("\n".join(lines) + "\n")
(out / f"student_teacher_c_{args.label}.json").write_text(json.dumps(res, indent=1, default=str))
print("\n".join(lines), flush=True)
sys.stdout.flush()
os._exit(0)
