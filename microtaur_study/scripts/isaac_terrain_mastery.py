"""Terrain mastery per level: success rate of a teacher on one terrain type, every level, straight walking.

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_terrain_mastery.py --checkpoint <model.pt> --label <name> --out <dir> \
      --terrain C [--reverse] [--levels 0,1,2,3,4,5,6] [--envs-per-level 30] [--speed 0.20] [--seconds 20] [--device cpu]

All envs on the chosen terrain type (B or C columns), --envs-per-level per level, one episode of --seconds at a constant
straight command. Success = the training curriculum's promotion rule: no termination and progress along the command
(body-frame velocity integrated by the command term, as terrain_levels_progress reads it) >= 0.8 x the commanded distance.
--reverse (C only): spawn at the far end facing -x, raised side on the robot's RIGHT (right feet climb). A level is
"mastered" here at 100 %; the first level below that is where a gated curriculum would restart. Writes
mastery_<label>.txt / .json.
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
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--promote-frac", type=float, default=0.8)
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
alive = torch.ones(N, dtype=torch.bool, device=env.device)
progress = torch.zeros(N, device=env.device)
cmd_dist = torch.zeros(N, device=env.device)
with torch.inference_mode():
  obs = w.get_observations()
  for k in range(steps):
    cmd.vel_command_b[:, 0] = args.speed
    cmd.vel_command_b[:, 1:] = 0.0
    obs, _, dones, _ = w.step(policy(obs))
    policy_reset(dones)
    d = dones.bool()
    alive &= ~d
    # the command term integrates progress / commanded distance and zeroes them on reset: keep the alive envs' values
    progress = torch.where(alive, cmd.cmd_progress, progress)
    cmd_dist = torch.where(alive, cmd.cmd_distance, cmd_dist)
ratio = (progress / (args.speed * args.seconds)).cpu().numpy()
ok = (alive.cpu().numpy()) & (ratio >= args.promote_frac)
lv = level.cpu().numpy()
res = {"label": args.label, "checkpoint": args.checkpoint, "terrain": TYPE, "reverse": args.reverse, "speed": args.speed,
       "seconds": args.seconds, "levels": []}
step_mm = lambda L: (L / (TR.CUR_NUM_LEVELS - 1) * TR.C_DELTA_M * TR.CUR_TERRAIN_SCALE * 1e3) if TYPE == "C_step" else None  # noqa: E731
lines = [f"{args.label}: {args.checkpoint}",
         f"{TYPE}{' REVERSED (raised side on the right)' if args.reverse else ''}, straight {args.speed} m/s for {args.seconds:.0f} s, "
         f"{K} envs per level; success = no termination and progress >= {args.promote_frac} x commanded",
         "level | step mm | success | falls | progress m (mean, min)"]
first_unmastered = None
for L in LV:
  m = lv == L
  row = {"level": L, "n": int(m.sum()), "success": float(ok[m].mean()), "falls": float((~alive.cpu().numpy())[m].mean()),
         "progress_mean_m": float(ratio[m].mean() * args.speed * args.seconds),
         "progress_min_m": float(ratio[m].min() * args.speed * args.seconds)}
  res["levels"].append(row)
  if first_unmastered is None and row["success"] < 1.0:
    first_unmastered = L
  h = step_mm(L)
  lines.append(f"{L:5d} | {('%.1f' % h) if h is not None else '  -':>7s} | {row['success']:6.0%} | {row['falls']:5.0%} | "
               f"{row['progress_mean_m']:.2f}, {row['progress_min_m']:.2f}")
res["first_unmastered"] = first_unmastered
lines.append(f"first level below 100 %: {first_unmastered if first_unmastered is not None else 'none'}")
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / f"mastery_{args.label}.txt").write_text("\n".join(lines) + "\n")
(out / f"mastery_{args.label}.json").write_text(json.dumps(res, indent=1))
print("\n".join(lines), flush=True)
sys.stdout.flush()
os._exit(0)
