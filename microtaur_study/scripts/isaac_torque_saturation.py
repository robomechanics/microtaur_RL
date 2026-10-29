"""How often the leg motors hit their torque limit, per terrain, for a policy under given actuator constants.

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_torque_saturation.py --checkpoint <model.pt> --label t3 --out <json> \
      [--effort 0.086] [--friction 0.028] [--kd 0.0092] [--speed 0.20]

Teacher-Cur-Play, 10 envs per column-level pair (A level 0, B level 6, C levels 3 / 4 / 5), straight at --speed.
Per policy step and motor: the DC-motor model clips the PD torque to the torque-speed envelope (effort cap
below the knee); "saturated" = |computed - applied| > 1e-5 N m. Defaults: the current robot_constants;
pass --effort/--friction/--kd to evaluate a policy under the constants it was trained with.
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
ap.add_argument("--effort", type=float, default=None)
ap.add_argument("--friction", type=float, default=None)
ap.add_argument("--kd", type=float, default=None)
ap.add_argument("--speed", type=float, default=0.20)
ap.add_argument("--steps", type=int, default=400)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from microtaur_common.robot_constants import LEG_JOINT_NAMES  # noqa: E402
from microtaur_isaac import terrains as TR  # noqa: E402
from microtaur_isaac.env_cfg import MicrotaurTeacherCurPlayEnvCfg  # noqa: E402
from microtaur_isaac.policy_loader import load_policy, prepare_env_cfg  # noqa: E402

N = 100  # 10 per column; A = cols 0-1, B = 2-6, C = 7-9
cfg = MicrotaurTeacherCurPlayEnvCfg()
cfg.scene.num_envs = N
cfg.curriculum.command_ranges = None
cfg.commands.twist.resampling_time_range = (1e9, 1e9)
m = cfg.scene.robot.actuators["motors"]
if args.effort is not None:
  m.effort_limit = args.effort
if args.friction is not None:
  m.friction = args.friction
  m.dynamic_friction = args.friction
if args.kd is not None:
  m.damping = args.kd
prepare_env_cfg(cfg, args.checkpoint)
env = ManagerBasedRLEnv(cfg)
t = env.scene.terrain
col = torch.arange(N, device=env.device) // 10
LV = {0: 0, 1: 0, 2: 6, 3: 6, 4: 6, 5: 6, 6: 6, 7: 3, 8: 4, 9: 5}
t.terrain_levels[:] = torch.tensor([LV[int(c)] for c in col], device=env.device)
t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]
w = RslRlVecEnvWrapper(env)
act, reset = load_policy(w, args.checkpoint)
robot = env.scene["robot"]
mids = [robot.joint_names.index(n) for n in LEG_JOINT_NAMES]
cmd = env.command_manager.get_term("twist")
eff = float(m.effort_limit)
SAT, TQ, VX, DN = [], [], [], []
with torch.inference_mode():
  obs = w.get_observations()
  for k in range(args.steps):
    cmd.vel_command_b[:, 0] = args.speed
    cmd.vel_command_b[:, 1:] = 0.0
    obs, _, dones, _ = w.step(act(obs))
    reset(dones)
    d = robot.data
    c, a = d.computed_torque[:, mids], d.applied_torque[:, mids]
    if k >= 40:
      SAT.append(((c - a).abs() > 1e-5).float().cpu().numpy())
      TQ.append(a.abs().cpu().numpy())
      VX.append(d.root_link_lin_vel_b[:, 0].cpu().numpy())
      DN.append(dones.cpu().numpy())
SAT, TQ, VX, DN = map(np.stack, (SAT, TQ, VX, DN))
types = TR.env_terrain_type_ids(t).cpu().numpy()
lv = t.terrain_levels.cpu().numpy()
groups = {"A0": (types == 0) & (lv == 0), "B6": (types == 1) & (lv == 6), "C3": (types == 2) & (lv == 3),
          "C4": (types == 2) & (lv == 4), "C5": (types == 2) & (lv == 5)}
res = {"label": args.label, "effort_limit": eff, "friction": float(m.friction), "kd": float(m.damping), "speed_cmd": args.speed, "groups": {}}
lines = [f"{args.label}: effort cap {eff:.4f} N m, friction {float(m.friction):.4f}, kd {float(m.damping):.4f}, cmd {args.speed} m/s",
         "group | saturated motor-steps | p95 |tau| / cap | mean |tau| / cap | vx m/s | resets"]
for g, sel in groups.items():
  if not sel.any():
    continue
  s, tq = SAT[:, sel], TQ[:, sel]
  r = {"sat_frac": float(s.mean()), "p95_tau_over_cap": float(np.percentile(tq, 95) / eff), "mean_tau_over_cap": float(tq.mean() / eff),
       "vx": float(VX[:, sel].mean()), "resets": int(DN[:, sel].sum())}
  res["groups"][g] = r
  lines.append(f"{g:5s} | {r['sat_frac']:21.3f} | {r['p95_tau_over_cap']:16.2f} | {r['mean_tau_over_cap']:17.2f} | {r['vx']:6.3f} | {r['resets']}")
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(res, indent=1))
print("\n".join(lines), flush=True)
sys.stdout.flush()
os._exit(0)
