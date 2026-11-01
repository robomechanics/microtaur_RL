"""Reward of every term while turning: is turning worth it under the training reward? (yaw: forced spine bends)

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_turn_reward_terms.py --checkpoint <model.pt> --label y3 --out <dir> \
      [--vx 0.15 --wz 0.30 --seconds 20] [--device cpu]

All envs on one flat tile, one command for all (default the 0.50 m circle of the extreme test). Groups: the policy as
trained, and on the yaw robot its spine action replaced by a constant (0 = locked, 0.5, 1.0 = the 18.9 deg limit),
which turns the robot more or less with the same legs. Per group, after the first 3 s, over envs that did not reset:
the measured yaw rate and speed, and the mean per-step reward of every term (weight x term, the training scale per
second; IsaacLab multiplies by dt) with the total. The reward is the one the checkpoint was TRAINED with (weights and
numeric params from its params/env.yaml, the energy ramp's final weight, the last reward_schedule stage), not the
play cfg's defaults. A group that turns better and earns more total reward than the
trained policy means the policy sits in a local optimum; a group that turns better but earns less shows which terms
price turning. Writes turn_reward_<label>.txt / .json.
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
ap.add_argument("--vx", type=float, default=0.15)
ap.add_argument("--wz", type=float, default=0.30)
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--envs-per-group", type=int, default=12)
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

from microtaur_common.robot_constants import SPINE_JOINT_NAME  # noqa: E402
from microtaur_isaac.env_cfg import MicrotaurTeacherCurPlayEnvCfg  # noqa: E402
from microtaur_isaac.policy_loader import load_policy, prepare_env_cfg, variant_kwargs  # noqa: E402



def apply_training_rewards(cfg, checkpoint: str) -> list[str]:
  """Reward weights / numeric params of the run (params/env.yaml), with the curricula at their final stage."""
  import yaml

  y = yaml.unsafe_load(open(Path(checkpoint).parent / "params" / "env.yaml"))
  for name, term in (y.get("rewards") or {}).items():
    t = getattr(cfg.rewards, name, None)
    if t is None or term is None:
      continue
    t.weight = float(term["weight"])
    for k, v in (term.get("params") or {}).items():
      if isinstance(v, (int, float)) and not isinstance(v, bool) and k in t.params:
        t.params[k] = v
  cur = y.get("curriculum") or {}
  notes = []
  stages = ((cur.get("energy_weight") or {}).get("params") or {}).get("stages") or []
  if stages:
    cfg.rewards.motor_energy.weight = float(stages[-1]["weight"])
    notes.append(f"motor_energy {cfg.rewards.motor_energy.weight:g} (ramp end)")
  sched = [st for st in ((cur.get("reward_schedule") or {}).get("params") or {}).get("stages") or [] if st != "off"]
  if sched:
    for item in str(sched[-1]).split("|")[1:]:
      key, val = item.split("=", 1)
      term, attr = key.split(".", 1)
      t = getattr(cfg.rewards, term)
      if attr == "weight":
        t.weight = float(val)
      else:
        t.params[attr] = float(val)
    notes.append(f"schedule {sched[-1]}")
  cfg.curriculum.energy_weight = None  # the curricula would reset the weights to their step-0 values
  cfg.curriculum.reward_schedule = None
  return notes


cfg = MicrotaurTeacherCurPlayEnvCfg(**variant_kwargs(args.checkpoint))
yaw = cfg.variant == "yaw"
GROUPS = [("trained", None)] + ([("spine 0 (locked)", 0.0), ("spine 0.5", 0.5), ("spine 1.0 (limit)", 1.0)] if yaw else [])
G = args.envs_per_group
cfg.scene.num_envs = len(GROUPS) * G
cfg.sim.device = args.device
cfg.curriculum.command_ranges = None
cfg.commands.twist.resampling_time_range = (1.0e9, 1.0e9)
prepare_env_cfg(cfg, args.checkpoint)
reward_notes = apply_training_rewards(cfg, args.checkpoint)
env = ManagerBasedRLEnv(cfg)
t = env.scene.terrain
t.terrain_levels[:] = 0
t.terrain_types[:] = 0
t.env_origins[:] = t.terrain_origins[0, 0]
w = RslRlVecEnvWrapper(env)
policy, policy_reset = load_policy(w, args.checkpoint)
robot, cmd, rm = env.scene["robot"], env.command_manager.get_term("twist"), env.reward_manager
group = torch.arange(env.num_envs, device=env.device) // G
spine_val = torch.tensor([np.nan if v is None else v for _, v in GROUPS], device=env.device)[group]
forced = ~torch.isnan(spine_val)
names = list(rm.active_terms)
steps, skip = round(args.seconds / env.step_dt), round(3.0 / env.step_dt)
rew, wz, spd, done = [], [], [], []
with torch.inference_mode():
  obs = w.get_observations()
  for k in range(steps):
    cmd.vel_command_b[:, 0], cmd.vel_command_b[:, 1], cmd.vel_command_b[:, 2] = args.vx, 0.0, args.wz
    act = policy(obs)
    if yaw:
      act[forced, -1] = spine_val[forced]
    obs, _, dones, _ = w.step(act)
    policy_reset(dones)
    # .copy(): with CPU physics .cpu() is a no-op and numpy would alias the reused PhysX buffer
    rew.append(rm._step_reward.clone().cpu().numpy().copy())  # weight x term per env (before x dt)
    wz.append(robot.data.root_link_ang_vel_b[:, 2].cpu().numpy().copy())
    spd.append(torch.linalg.norm(robot.data.root_link_lin_vel_w[:, :2], dim=1).cpu().numpy().copy())
    done.append(dones.cpu().numpy().copy())
R, WZ, SP, D = (np.stack(x) for x in (rew, wz, spd, done))  # [T, N, terms], [T, N]
ok = D.sum(axis=0) == 0
g_np = group.cpu().numpy()
res = {"label": args.label, "checkpoint": args.checkpoint, "cmd": [args.vx, args.wz], "groups": []}
cols = [n for n in names if np.abs(R[skip:, :, names.index(n)]).max() > 0]
lines = [f"{args.label}: {args.checkpoint}",
         f"command vx {args.vx} m/s, yaw {args.wz} rad/s for {args.seconds:.0f} s (stats after 3 s), {G} envs per group; "
         "reward per step = weight x term (training scale, before x dt); training reward: " + "; ".join(reward_notes),
         "group              envs | yaw rate  speed | " + " ".join(f"{n[:14]:>14s}" for n in cols) + " |   total"]
for gi, (gname, _) in enumerate(GROUPS):
  m = (g_np == gi) & ok
  per = {n: float(R[skip:, m, names.index(n)].mean()) for n in cols}
  tot = float(sum(per.values()))
  row = {"group": gname, "envs_ok": int(m.sum()), "yaw_rate": float(WZ[skip:, m].mean()), "speed": float(SP[skip:, m].mean()),
         "terms": per, "total": tot}
  res["groups"].append(row)
  lines.append(f"{gname:18s} {m.sum():4d} | {row['yaw_rate']:+.3f}  {row['speed']:.3f} | "
               + " ".join(f"{per[n]:+14.4f}" for n in cols) + f" | {tot:+.4f}")
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / f"turn_reward_{args.label}.txt").write_text("\n".join(lines) + "\n")
(out / f"turn_reward_{args.label}.json").write_text(json.dumps(res, indent=1))
print("\n".join(lines), flush=True)
sys.stdout.flush()
os._exit(0)
