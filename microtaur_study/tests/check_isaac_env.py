"""Smoke test of the assembled IsaacLab tasks (flat, rough, teacher).

  OMNI_KIT_ACCEPT_EULA=YES python tests/check_isaac_env.py [--num_envs 16] [--out log.txt]

For each task: build the env, reset, 100 zero-action steps then 100 random
steps; report observation dims per group, standing height, falls/NaNs, reward
term means, closure gap, and on rough terrain the per-type env counts, the C
spawn pose and the B spawn clearance.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--num_envs", type=int, default=16)
ap.add_argument("--tasks", nargs="+", default=["Flat", "Rough", "Teacher"])
ap.add_argument("--out", default="check_isaac_env.txt")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app
LOG = open(args.out, "w")


def log(*a):
  LOG.write(" ".join(map(str, a)) + "\n"); LOG.flush(); print(*a, flush=True)


import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

from microtaur_isaac import env_cfg as EC  # noqa: E402
from microtaur_isaac.usd import compile_model  # noqa: E402

M = compile_model()
LOOPS = [(M.body(int(M.site_bodyid[int(M.eq_obj1id[i])])).name, M.site_pos[int(M.eq_obj1id[i])],
          M.body(int(M.site_bodyid[int(M.eq_obj2id[i])])).name, M.site_pos[int(M.eq_obj2id[i])]) for i in range(M.neq)]
ok = True


def closure_gap(robot) -> float:
  pose = robot.data.body_link_pose_w
  worst = 0.0
  for b0, p0, b1, p1 in LOOPS:
    i0, i1 = robot.body_names.index(b0), robot.body_names.index(b1)
    t0 = torch.tensor(p0, dtype=torch.float32, device=pose.device).expand(pose.shape[0], 3)
    t1 = torch.tensor(p1, dtype=torch.float32, device=pose.device).expand(pose.shape[0], 3)
    s0 = pose[:, i0, :3] + quat_apply(pose[:, i0, 3:7], t0)
    s1 = pose[:, i1, :3] + quat_apply(pose[:, i1, 3:7], t1)
    worst = max(worst, float(torch.linalg.norm(s0 - s1, dim=1).max()))
  return worst


for task in args.tasks:
  cfg = getattr(EC, f"Microtaur{task}EnvCfg")()
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRLEnv(cfg)
  robot = env.scene["robot"]
  obs, _ = env.reset()
  log(f"\n===== {task}: obs groups " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in obs.items()))
  z0 = (robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2])
  log(f"  after reset: root z above origin mean {1e3 * float(z0.mean()):.2f} mm, closure gap {1e3 * closure_gap(robot):.4f} mm")
  terrain = env.scene.terrain
  if getattr(terrain, "terrain_types", None) is not None:
    from microtaur_isaac import terrains as TR
    types = TR.env_terrain_type_ids(terrain)
    log(f"  env counts A/B/C: {[int((types == k).sum()) for k in range(3)]}, levels {terrain.terrain_levels.tolist()}")
    rel = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    c = types == 2
    if c.any():
      yaw = torch.atan2(2 * (robot.data.root_quat_w[:, 0] * robot.data.root_quat_w[:, 3]), 1 - 2 * robot.data.root_quat_w[:, 3] ** 2)
      log(f"  C spawn xy (should be {TR.C_SPAWN_OFFSET_XY}): {rel[c][:3].tolist()}  yaw {yaw[c][:3].tolist()}  zero-yaw mask {bool(env.microtaur_zero_yaw_mask[c].all())}")
  worst_gap, fell, rew = 0.0, 0, {}
  for t in range(200):
    act = torch.zeros(env.num_envs, 8, device=env.device) if t < 100 else 0.5 * torch.randn(env.num_envs, 8, device=env.device)
    obs, r, term, trunc, extras = env.step(act)
    worst_gap = max(worst_gap, closure_gap(robot))
    fell += int(term.sum())
    for i, n in enumerate(env.reward_manager.active_terms):
      rew[n] = rew.get(n, 0.0) + float(env.reward_manager._step_reward[:, i].mean()) / 200
    if t == 99:
      z = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
      log(f"  after 100 zero-action steps: root z mean {1e3 * float(z.mean()):.2f} mm, terminated so far {fell}")
    if not torch.isfinite(robot.data.root_pos_w).all():
      log("  NaN in root state!"); ok = False; break
  log(f"  200 steps: terminations {fell}, max closure gap {1e3 * worst_gap:.3f} mm")
  log("  mean step reward rate (raw x weight): " + ", ".join(f"{k} {v:+.3f}" for k, v in rew.items()))
  env.close()
log("\nDONE")
LOG.close()
os._exit(0)
