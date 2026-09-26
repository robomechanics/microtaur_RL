"""Evaluate a Microtaur teacher policy on the A/B/C terrain grid, per terrain type and level.

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_eval_terrain.py --checkpoint <model.pt> --out <dir> \
      [--num-envs 500] [--speed 0.20] [--seconds 20]

Teacher-Play config (no noise / randomisation / timeout, terrain curriculum off,
robots spread uniformly over all levels and the A/B/C columns). Every env walks
straight at --speed (C is straight anyway) for --seconds. Per env: progress along
its spawn heading until its first termination (or the end), whether and why it
terminated, mean forward speed. Writes terrain_eval.json and prints a table of
type x level: n, fall rate, mean progress, fraction reaching half a tile (the
curriculum's promotion distance), mean speed.
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
ap.add_argument("--num-envs", type=int, default=500)
ap.add_argument("--speed", type=float, default=0.20)
ap.add_argument("--seconds", type=float, default=20.0)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
LOG = open(out / "terrain_eval.log", "w")


def log(*a):
  LOG.write(" ".join(map(str, a)) + "\n"); LOG.flush()


import re  # noqa: E402

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from microtaur_isaac import terrains as TR  # noqa: E402
from microtaur_isaac.agents import MicrotaurTeacherPPORunnerCfg  # noqa: E402
from microtaur_isaac.env_cfg import MicrotaurTeacherPlayEnvCfg  # noqa: E402

cfg = MicrotaurTeacherPlayEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.curriculum.command_ranges = None
cfg.commands.twist.resampling_time_range = (1.0e9, 1.0e9)
y = Path(args.checkpoint).parent / "params" / "env.yaml"
m = re.search(r"action_scale_rad: ([0-9.eE+-]+)", y.read_text()) if y.exists() else None
if m:
  cfg.actions.joint_pos.action_scale_rad = float(m.group(1))
env = ManagerBasedRLEnv(cfg)
wrapped = RslRlVecEnvWrapper(env)
runner = OnPolicyRunner(wrapped, MicrotaurTeacherPPORunnerCfg().to_dict(), log_dir=None, device=env.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.device)

robot = env.scene["robot"]
terrain = env.scene.terrain
cmd = env.command_manager.get_term("twist")
types = TR.env_terrain_type_ids(terrain)  # 0 A, 1 B, 2 C
levels = terrain.terrain_levels.clone()
N = env.num_envs


def pin():
  cmd.vel_command_b[:, 0] = args.speed
  cmd.vel_command_b[:, 1:] = 0.0


pin()
obs = wrapped.get_observations()
d = robot.data
start = (d.root_link_pos_w[:, :2]).clone()
q = d.root_link_quat_w
heading = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
fwd = torch.stack((torch.cos(heading), torch.sin(heading)), dim=1)
alive = torch.ones(N, dtype=torch.bool, device=env.device)
progress = torch.zeros(N, device=env.device)
speed_sum = torch.zeros(N, device=env.device)
steps = torch.zeros(N, device=env.device)
cause = ["" for _ in range(N)]
tm = env.termination_manager
T = int(args.seconds / env.step_dt)
with torch.inference_mode():
  for t in range(T):
    pin()
    prev = d.root_link_pos_w[:, :2].clone()
    obs, _, dones, _ = wrapped.step(policy(obs))
    now = torch.where(dones[:, None].bool(), prev, d.root_link_pos_w[:, :2])  # a reset moves the root: keep the last pose
    p = torch.sum((now - start) * fwd, dim=1)
    progress = torch.where(alive, p, progress)
    speed_sum += torch.where(alive, d.root_link_lin_vel_b[:, 0], torch.zeros_like(p))
    steps += alive.float()
    newly = alive & dones.bool()
    for i in torch.nonzero(newly).flatten().tolist():
      cause[i] = next((n for n in tm.active_terms if bool(tm.get_term(n)[i])), "?")
    alive &= ~dones.bool()

half_tile = TR.TILE_SIZE_M[0] / 2
names = ("A_flat", "B_blocks", "C_step")
rows = []
log(f"checkpoint {args.checkpoint}\nstraight {args.speed} m/s for {args.seconds} s; promotion distance (half a tile) {half_tile:.2f} m\n")
log(f"{'terrain':9s} {'lvl':>3s} {'n':>4s} {'fall':>6s} {'progress m':>11s} {'>= half tile':>12s} {'speed m/s':>10s}  causes")
for ty in range(3):
  for lv in range(int(levels.max()) + 1):
    sel = (types == ty) & (levels == lv)
    n = int(sel.sum())
    if n == 0:
      continue
    fell = (~alive & sel)
    pr = progress[sel]
    sp = (speed_sum[sel] / steps[sel].clamp_min(1))
    cs = {}
    for i in torch.nonzero(fell).flatten().tolist():
      cs[cause[i]] = cs.get(cause[i], 0) + 1
    row = {"terrain": names[ty], "level": lv, "n": n, "fall_rate": float(fell.sum()) / n, "progress_m": float(pr.mean()),
           "reach_half_tile": float((pr >= half_tile).float().mean()), "speed_m_s": float(sp.mean()), "causes": cs}
    rows.append(row)
    log(f"{row['terrain']:9s} {lv:3d} {n:4d} {row['fall_rate']:6.2f} {row['progress_m']:11.2f} {row['reach_half_tile']:12.2f} "
        f"{row['speed_m_s']:10.3f}  {cs}")
(out / "terrain_eval.json").write_text(json.dumps({"checkpoint": args.checkpoint, "speed": args.speed, "rows": rows}, indent=1))
LOG.close()
os._exit(0)
