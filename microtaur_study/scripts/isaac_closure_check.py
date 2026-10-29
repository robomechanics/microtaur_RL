"""Loop-closure gap of the four five-bar legs in the real training env, for a set of motor KD values.

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_closure_check.py --checkpoint <policy.pt> --kd 0.0 0.0092 0.045 --out <json>

Teacher-Cur training cfg (domain randomisation, pushes, all terrain levels spread over the envs),
the motors' damping overridden per run, the given policy trotting at random commands. The gap of
each closing joint is |p0 + R0 l0 - (p1 + R1 l1)| between the two closing frames
(physics:localPos0 / localPos1 of the "loops" joints in the training USD), sampled every policy
step. Pass: p99 < 1 mm and max < 5 mm (robot.py: at KD 0 the p99 passed and the max reached ~5 mm).
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
ap.add_argument("--kd", type=float, nargs="+", default=[0.0, 0.0092, 0.045])
ap.add_argument("--num-envs", type=int, default=256)
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from microtaur_common.robot_constants import EFFORT_LIMIT_NM, FRICTIONLOSS_NM, KD  # noqa: E402
from microtaur_isaac.env_cfg import MicrotaurTeacherCurEnvCfg  # noqa: E402
from microtaur_isaac.policy_loader import load_policy, prepare_env_cfg  # noqa: E402

# body0, body1, localPos0, localPos1 of the closing joints (assets/rigid_microtaur/microtaur_rigid.usda, "loops")
LOOPS = [
  ("leglink2_v2_with_leg_down", "spacer_for_legs", (-0.00930745, -0.070216, -0.003), (-0.0505419, -0.049623, -0.003)),
  ("leglink2_v2_with_leg_down_2", "spacer_for_legs_2", (-0.00930745, 0.070216, -0.003), (-0.00930745, -0.070216, -0.003)),
  ("leglink2_v2_with_leg_down_3", "spacer_for_legs_3", (0.0708302, 0.0, -0.003), (-0.0698359, 0.0118265, -0.003)),
  ("leglink2_v2_with_leg_down_4", "spacer_for_legs_4", (0.0708302, 0.0, -0.003), (-0.0703557, 0.00818491, -0.003)),
]

results = {"constants": {"KD": KD, "EFFORT_LIMIT_NM": EFFORT_LIMIT_NM, "FRICTIONLOSS_NM": FRICTIONLOSS_NM}, "runs": {}}
lines = [f"checkpoint {args.checkpoint}", f"robot_constants: KD {KD}, effort {EFFORT_LIMIT_NM:.4f} N m, friction {FRICTIONLOSS_NM}",
         f"{args.num_envs} envs x {args.steps} policy steps, training cfg (DR + pushes), all levels",
         "kd      | gap p50 mm | p99 mm | max mm | envs > 1 mm | envs > 5 mm | resets | verdict"]
for kd in args.kd:
  cfg = MicrotaurTeacherCurEnvCfg()
  cfg.scene.num_envs = args.num_envs
  cfg.scene.robot.actuators["motors"].damping = kd
  cfg.scene.terrain.max_init_terrain_level = None  # spread over all levels
  prepare_env_cfg(cfg, args.checkpoint)
  env = ManagerBasedRLEnv(cfg)
  w = RslRlVecEnvWrapper(env)
  act, reset = load_policy(w, args.checkpoint)
  robot = env.scene["robot"]
  ids0 = [robot.body_names.index(b0) for b0, _, _, _ in LOOPS]
  ids1 = [robot.body_names.index(b1) for _, b1, _, _ in LOOPS]
  l0 = torch.tensor([l for _, _, l, _ in LOOPS], device=env.device)
  l1 = torch.tensor([l for _, _, _, l in LOOPS], device=env.device)
  N = env.num_envs
  gaps, resets = [], 0
  with torch.inference_mode():
    obs = w.get_observations()
    for k in range(args.steps):
      obs, _, dones, _ = w.step(act(obs))
      reset(dones)
      resets += int(dones.sum())
      d = robot.data
      p0, q0 = d.body_link_pos_w[:, ids0], d.body_link_quat_w[:, ids0]
      p1, q1 = d.body_link_pos_w[:, ids1], d.body_link_quat_w[:, ids1]
      c0 = p0 + quat_apply(q0.reshape(-1, 4), l0.repeat(N, 1)).reshape(N, 4, 3)
      c1 = p1 + quat_apply(q1.reshape(-1, 4), l1.repeat(N, 1)).reshape(N, 4, 3)
      g = torch.linalg.norm(c0 - c1, dim=-1)
      gaps.append(torch.where(dones[:, None].bool(), torch.zeros_like(g), g))  # skip the reset step itself
  G = torch.nan_to_num(torch.stack(gaps), nan=1.0)  # [T, N, 4]
  per_env = G.amax(dim=(0, 2))
  flat = G.flatten()
  r = {"p50_mm": float(flat.median()) * 1e3, "p99_mm": float(torch.quantile(flat[torch.randperm(flat.numel(), device=flat.device)[:200000]], 0.99)) * 1e3,
       "max_mm": float(flat.max()) * 1e3, "envs_gt_1mm": int((per_env > 1e-3).sum()), "envs_gt_5mm": int((per_env > 5e-3).sum()), "resets": resets}
  r["pass"] = r["p99_mm"] < 1.0 and r["max_mm"] < 5.0
  results["runs"][str(kd)] = r
  lines.append(f"{kd:<7g} | {r['p50_mm']:10.3f} | {r['p99_mm']:6.3f} | {r['max_mm']:6.2f} | {r['envs_gt_1mm']:11d} | {r['envs_gt_5mm']:11d} | {resets:6d} | {'PASS' if r['pass'] else 'FAIL'}")
  print(lines[-1], flush=True)
  env.close()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(results, indent=1))
Path(args.out).with_suffix(".txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
sys.stdout.flush()
os._exit(0)
