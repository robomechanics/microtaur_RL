"""Equivalence of microtaur_rigid against the upstream 2026-09-20 env.

mujoco-warp is not bit-deterministic on GPU: upstream run twice with the same
seed and actions diverges after ~2 steps (contact onset amplifies 1e-8
differences). So equivalence is checked in two ways instead of by trajectory:

  A. Bit-exact on the deterministic parts:
     - state after reset (IK-consistent reset, same seed);
     - the action term: upstream's and this package's action terms are built on
       the SAME env and fed the same action sequence; requested / safe / applied
       motor targets must match exactly at every step;
     - the safety filter for every sim2real stage, on random inputs.
  B. Statistical: upstream-vs-refactor differences in rollout statistics must be
     no larger than upstream-vs-upstream (run-to-run noise).

Rewards are not compared: D0 replaces the upstream reward on purpose.

Usage (mjlab 1.6 venv):
  python tests/check_equivalence.py --upstream-src /path/to/microtaur_upstream/src
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--upstream-src", required=True)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--stat-envs", type=int, default=512)
ap.add_argument("--stat-steps", type=int, default=500)
args = ap.parse_args()

os.environ["MICROTAUR_RIGID_SIM2REAL_STAGE"] = "0"
sys.path.insert(0, args.upstream_src)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from mjlab.entity.entity import Entity
from mjlab.envs import ManagerBasedRlEnv

_orig = Entity.set_joint_position_target


def _flat_ids(self, position, joint_ids=None, env_ids=None):
  # Upstream pre-expands ids to [N,1]/[1,8]; mjlab >= 1.6 rejects that.
  if isinstance(joint_ids, torch.Tensor) and joint_ids.ndim == 2:
    joint_ids = joint_ids.reshape(-1)
  if isinstance(env_ids, torch.Tensor) and env_ids.ndim == 2:
    env_ids = env_ids.reshape(-1)
  return _orig(self, position, joint_ids=joint_ids, env_ids=env_ids)


Entity.set_joint_position_target = _flat_ids

from microtaur_rigid.env_cfg import make_env_cfg  # noqa: E402
from microtaur_velocity import env_cfgs as upstream  # noqa: E402

DEV = "cuda:0"
ok = True


def build(cfg, n):
  cfg.scene.num_envs = n
  cfg.seed = args.seed
  torch.manual_seed(args.seed)
  return ManagerBasedRlEnv(cfg, device=DEV)


def report(name, diff, tol=0.0):
  global ok
  passed = diff <= tol
  ok &= passed
  print(f"  [{'PASS' if passed else 'FAIL'}] {name:48s} max diff {diff:.3e}")


# --- A1: state after reset ----------------------------------------------------
print("A1. state after reset (16 envs, same seed)")
states = []
for cfg in (upstream.microtaur_velocity_flat_env_cfg(), make_env_cfg()):
  env = build(cfg, 16)
  obs, _ = env.reset()
  r = env.scene["robot"].data
  states.append((obs["actor"].clone(), obs["critic"].clone(), r.joint_pos.clone(), r.root_link_pose_w.clone()))
  env.close()
for k, name in enumerate(("actor obs", "critic obs", "joint_pos", "root pose")):
  report(name, float((states[0][k] - states[1][k]).abs().max()))

# --- A2: action term on one env -------------------------------------------------
print("A2. action term, upstream vs refactor on the same env (16 envs, 300 steps)")
env = build(make_env_cfg(), 16)
env.reset()
mine = env.action_manager.get_term("joint_pos")
theirs = upstream.microtaur_velocity_flat_env_cfg().actions["joint_pos"].build(env)
g = torch.Generator().manual_seed(123)
worst = {"requested": 0.0, "safe": 0.0, "applied": 0.0, "processed": 0.0}
for _ in range(300):
  a = (0.8 * torch.randn(16, 8, generator=g)).to(DEV)
  theirs.process_actions(a)  # same env state that the next env.step sees
  up = (theirs._requested_targets.clone(), theirs._safe_targets.clone(),
        theirs._applied_targets.clone(), theirs._processed_actions.clone())
  env.step(a)
  me = (mine.requested_targets, mine.safe_targets, mine.applied_targets, mine._processed_actions)
  for k, u, m in zip(worst, up, me):
    worst[k] = max(worst[k], float((u - m).abs().max()))
for k, v in worst.items():
  report(f"{k} motor targets", v)

# --- A3: safety filter ---------------------------------------------------------
# The filter does not depend on the sim2real stage (gain/bias are applied to the
# request before it), so it is compared on three independent random batches.
print("A3. safety filter on random states (3 x 4096 samples)")
for seed in (0, 1, 2):
  gs = torch.Generator(device=DEV).manual_seed(seed)
  stand = mine.stand[None]
  pos = stand + 0.36 * torch.randn(4096, 8, device=DEV, generator=gs)
  vel = 3.0 * torch.randn(4096, 8, device=DEV, generator=gs)
  req = stand + 0.6 * torch.randn(4096, 8, device=DEV, generator=gs)
  s1, b1 = theirs._filter_targets(pos, vel, req)
  s2, b2 = mine._filter_targets(pos, vel, req)
  report(f"batch {seed} filter safe targets / blend", max(float((s1 - s2).abs().max()), float((b1 - b2).abs().max())))
env.close()

# --- B: rollout statistics ------------------------------------------------------
print(f"B. rollout statistics ({args.stat_envs} envs, {args.stat_steps} steps, same random actions)")
g = torch.Generator().manual_seed(321)
acts = [(0.5 * torch.randn(args.stat_envs, 8, generator=g)) for _ in range(args.stat_steps)]


def stats(cfg):
  env = build(cfg, args.stat_envs)
  env.reset()
  robot = env.scene["robot"].data
  contact = env.scene["feet_ground_contact"]
  z, tilt, feet, resets = [], [], [], 0
  for a in acts:
    _, _, term, trunc, _ = env.step(a.to(DEV))
    z.append((robot.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).mean())
    tilt.append(torch.acos(torch.clamp(-robot.projected_gravity_b[:, 2], -1, 1)).mean())
    feet.append((contact.data.found > 0).float().sum(-1).mean())
    resets += int((term | trunc).sum())
  env.close()
  return {"root_z_mm": 1e3 * float(torch.stack(z).mean()), "tilt_deg": float(torch.rad2deg(torch.stack(tilt).mean())),
          "feet_in_contact": float(torch.stack(feet).mean()), "resets": resets}


u1 = stats(upstream.microtaur_velocity_flat_env_cfg())
u2 = stats(upstream.microtaur_velocity_flat_env_cfg())
m1 = stats(make_env_cfg())
for k in u1:
  noise, delta = abs(u1[k] - u2[k]), abs(u1[k] - m1[k])
  print(f"  {k:16s} upstream {u1[k]:9.4f} / {u2[k]:9.4f}  refactor {m1[k]:9.4f}   "
        f"|up-up| {noise:.4f}  |up-refactor| {delta:.4f}")
  # Allow 3x the observed run-to-run noise, with a small absolute floor.
  floor = {"root_z_mm": 0.05, "tilt_deg": 0.05, "feet_in_contact": 0.01, "resets": 0.02 * max(u1["resets"], 1)}[k]
  ok &= delta <= max(3 * noise, floor)

print("RESULT:", "EQUIVALENT" if ok else "NOT EQUIVALENT")
sys.exit(0 if ok else 1)
