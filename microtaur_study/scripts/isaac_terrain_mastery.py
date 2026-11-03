"""Terrain mastery per level: success rate of a teacher on one terrain type, every level, straight walking.

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_terrain_mastery.py --checkpoint <model.pt> --label <name> --out <dir> \
      --terrain C [--reverse] [--levels 0,1,2,3,4,5,6] [--envs-per-level 30] [--speed 0.20] [--seconds 40] [--device cpu]

All envs on the chosen terrain type (B or C columns), --envs-per-level per level, one episode of --seconds at a constant
straight command. Envs ended by physics_unstable (a simulator fault) or out_of_terrain_bounds are not counted (as in the
training gate). Success = the level is got through without a fall, however slowly (Ben, 2026-11-03):
  C: the body got 0.5 m past the entry riser (up on the raised half; 0.7 m from the spawn, measured along the lane in
     the world frame) and no termination during the whole run, until the robot leaves its own course (4.7 m; the next
     tile is another level). This is the training gate's C criterion (gate "complete").
  B: progress along the command (as the command term integrates it) reached --b-complete-m (2 m) before any
     termination; whatever happens after that does not count.
Also reported: C "course" = whole raised section walked (4.4 m, body past the exit riser) within --seconds, the speed
over it, fall causes, and the old rule (alive at 20 s and >= 0.8 x the commanded distance by then; a speed criterion).
--reverse (C only): spawn at the far end facing -x, raised side on the robot's RIGHT (right feet climb). A level is
"mastered" here at 100 %. Writes mastery_<label>.txt / .json.
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
ap.add_argument("--terrain", choices=("B", "C"), required=True)
ap.add_argument("--reverse", action="store_true")
ap.add_argument("--levels", default="0,1,2,3,4,5,6")
ap.add_argument("--envs-per-level", type=int, default=30)
ap.add_argument("--speed", type=float, default=0.20)
ap.add_argument("--seconds", type=float, default=40.0)
ap.add_argument("--b-complete-m", type=float, default=2.0)
ap.add_argument("--promote-frac", type=float, default=0.8, help="old rule: fraction of the commanded distance in 20 s")
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

from microtaur_isaac import TERRAIN_TYPES  # noqa: E402
from microtaur_isaac import terrains as TR  # noqa: E402
from microtaur_isaac.env_cfg import MicrotaurTeacherCurPlayEnvCfg  # noqa: E402
from microtaur_isaac.policy_loader import load_policy, prepare_env_cfg, variant_kwargs  # noqa: E402

LV = [int(v) for v in args.levels.split(",")]
K = args.envs_per_level
N = K * len(LV)
TYPE = {"B": "B_blocks", "C": "C_step"}[args.terrain]
cols = [c for c, name in enumerate(TR.COLUMN_TERRAIN_TYPES) if name == TYPE]
cfg = MicrotaurTeacherCurPlayEnvCfg(**variant_kwargs(args.checkpoint))
cfg.scene.num_envs = N
cfg.sim.device = args.device
cfg.c_reverse_frac = 1.0 if args.reverse else 0.0
cfg.curriculum.command_ranges = None
cfg.commands.twist.resampling_time_range = (1.0e9, 1.0e9)
prepare_env_cfg(cfg, args.checkpoint)
env = ManagerBasedRLEnv(cfg)
t = env.scene.terrain
level = torch.tensor([LV[i // K] for i in range(N)], device=env.device)
t.terrain_types[:] = torch.tensor([cols[i % len(cols)] for i in range(N)], device=env.device)
t.terrain_levels[:] = level
t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]
w = RslRlVecEnvWrapper(env)  # resets: spawns at the new origins (C at the course start)
policy, policy_reset = load_policy(w, args.checkpoint)
cmd = env.command_manager.get_term("twist")
assert bool((TR.env_terrain_type_ids(t) == TERRAIN_TYPES.index(TYPE)).all())
steps = round(args.seconds / env.step_dt)
old_steps = round(20.0 / env.step_dt)
is_c = TYPE == "C_step"
climb_m = TR.C_STEP_X_RANGE[0] - TR.C_SPAWN_OFFSET_XY[0] + 0.5  # 0.7 m
course_m = TR.C_STEP_X_RANGE[1] - TR.C_SPAWN_OFFSET_XY[0] + 0.2  # 4.4 m
leave_m = TR.C_COURSE_LEN_M - TR.C_SPAWN_ALONG_M  # 4.7 m
robot = env.scene["robot"]
x0 = robot.data.root_link_pos_w[:, 0].clone()
sgn = -1.0 if args.reverse else 1.0
alive = torch.ones(N, dtype=torch.bool, device=env.device)   # no termination while it counts
alive_all = torch.ones(N, dtype=torch.bool, device=env.device)  # no termination at all (old rule)
stop = torch.zeros(N, dtype=torch.bool, device=env.device)   # stopped counting (B: 2 m reached, C: left its course)
climbed = torch.zeros(N, dtype=torch.bool, device=env.device)
course = torch.zeros(N, dtype=torch.bool, device=env.device)
t_course = torch.full((N,), float("nan"), device=env.device)
progress = torch.zeros(N, device=env.device)
old_ok = torch.zeros(N, dtype=torch.bool, device=env.device)
cause = [""] * N
excl = np.zeros(N, dtype=bool)  # ended by a truncation that is not the robot's failure: not counted
EXCLUDED = ("physics_unstable", "out_of_terrain_bounds")
tm = env.termination_manager
with torch.inference_mode():
  obs = w.get_observations()
  for k in range(steps):
    cmd.vel_command_b[:, 0] = args.speed
    cmd.vel_command_b[:, 1:] = 0.0
    obs, _, dones, _ = w.step(policy(obs))
    policy_reset(dones)
    d = dones.bool() & alive & ~stop
    for i in d.nonzero().flatten().tolist():
      cause[i] = next((n for n in tm.active_terms if bool(tm.get_term(n)[i])), "?")
      excl[i] = cause[i] in EXCLUDED
    alive &= ~d
    alive_all &= ~dones.bool()
    live = alive & ~stop
    # C: along the lane, world frame; B: the command term's progress (zeroed on reset: keep the last live value)
    now = sgn * (robot.data.root_link_pos_w[:, 0] - x0) if is_c else cmd.cmd_progress
    progress = torch.where(live, now, progress)
    if k + 1 == old_steps:
      old_ok = alive_all & (cmd.cmd_progress >= args.promote_frac * args.speed * 20.0)
    if is_c:
      climbed |= live & (progress >= climb_m)
      new = live & ~course & (progress >= course_m)
      t_course = torch.where(new, torch.full_like(t_course, (k + 1) * env.step_dt), t_course)
      course |= new
      stop |= live & (progress >= leave_m)
    else:
      stop |= live & (progress >= args.b_complete_m)
done = (climbed & alive) if is_c else stop
ok = done.cpu().numpy() & ~excl
fell = (~alive).cpu().numpy() & ~excl
crs = course.cpu().numpy()
td = t_course.cpu().numpy()
prog = progress.cpu().numpy()
oldk = old_ok.cpu().numpy()
lv = level.cpu().numpy()
res = {"label": args.label, "checkpoint": args.checkpoint, "terrain": TYPE, "reverse": args.reverse, "speed": args.speed,
       "seconds": args.seconds, "climb_m": climb_m if is_c else None, "course_m": course_m if is_c else None,
       "b_complete_m": None if is_c else args.b_complete_m, "levels": []}
step_mm = lambda L: (L / (TR.CUR_NUM_LEVELS - 1) * TR.C_DELTA_M * TR.CUR_TERRAIN_SCALE * 1e3) if is_c else None  # noqa: E731
what = ("climbed: body 0.5 m past the entry riser, then no fall until it leaves the course" if is_c
        else f"{args.b_complete_m:.1f} m before any fall")
lines = [f"{args.label}: {args.checkpoint}",
         f"{TYPE}{' REVERSED (raised side on the right)' if args.reverse else ''}, straight {args.speed} m/s for "
         f"{args.seconds:.0f} s, {K} envs per level; success = {what}, any speed",
         "level | step mm | success | " + ("course 4.4 m | " if is_c else "") + "falls | progress m (mean, min) | "
         + ("course speed m/s | " if is_c else "") + "old 20 s rule | not counted | fall causes"]
first_unmastered = None
for L in LV:
  m = lv == L
  causes = {}
  for i in np.nonzero(m & fell)[0]:
    causes[cause[i]] = causes.get(cause[i], 0) + 1
  c = m & ~excl  # counted envs
  nc = max(int(c.sum()), 1)
  row = {"level": L, "n": int(m.sum()), "not_counted": int((m & excl).sum()),
         "success": float(ok[c].sum() / nc), "falls": float(fell[c].sum() / nc),
         "course": float(crs[c].sum() / nc) if is_c else None,
         "progress_mean_m": float(prog[m].mean()), "progress_min_m": float(prog[m].min()),
         "course_speed_m_s": float(course_m / np.nanmean(td[m & crs])) if is_c and (m & crs).any() else None,
         "old_rule_success": float(oldk[m].mean()), "fall_causes": causes}
  res["levels"].append(row)
  if first_unmastered is None and row["success"] < 1.0:
    first_unmastered = L
  h = step_mm(L)
  sp = row["course_speed_m_s"]
  lines.append(f"{L:5d} | {('%.1f' % h) if h is not None else '  -':>7s} | {row['success']:7.0%} | "
               + (f"{row['course']:12.0%} | " if is_c else "")
               + f"{row['falls']:5.0%} | {row['progress_mean_m']:.2f}, {row['progress_min_m']:.2f} | "
               + (f"{('%.3f' % sp) if sp is not None else '-':>16s} | " if is_c else "")
               + f"{row['old_rule_success']:13.0%} | {row['not_counted']:11d} | "
               + (", ".join(f"{c} {n}" for c, n in causes.items()) or "-"))
res["first_unmastered"] = first_unmastered
lines.append(f"first level below 100 %: {first_unmastered if first_unmastered is not None else 'none'}")
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / f"mastery_{args.label}.txt").write_text("\n".join(lines) + "\n")
(out / f"mastery_{args.label}.json").write_text(json.dumps(res, indent=1))
print("\n".join(lines), flush=True)
sys.stdout.flush()
os._exit(0)
