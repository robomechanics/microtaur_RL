"""Can a planar-legged Microtaur turn by stride length alone? Scripted IK trot, left legs' stroke smaller.

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_diff_stride_turn.py --variant rigid_uno --out <dir> \
      [--speed 0.20] [--periods 0.21 0.28] [--ks 0 0.25 0.5 0.75 1 1.5 2] [--envs-per-k 8] [--seconds 8]

The open-loop trot of scripts/isaac_reference_gait.py (stance sweeps the foot back by the stroke L = speed * T / 2,
swing returns it 10 mm up, diagonals half a cycle apart, five-bar IK, the policy's action pipeline), but with the
stroke of the LEFT legs (RL, FL) L * (1 - k) and of the RIGHT legs (RR, FR) L * (1 + k): k = 0 straight, k = 1 left
legs step in place, k > 1 left legs step backward (spin-like). A spine robot holds its spine at 0 (action 0). Flat
play env: no randomisation, ground friction 1.0. One env group per k, measured after 2 s.

Printed per period and k: yaw rate (rad/s, + = left), forward speed, the yaw rate a skid-free tracked vehicle would
get from the same left / right stroke speeds, (v_R - v_L) / track = 2 k v_stroke / 0.1 m (v_stroke = L / (T / 2)),
the ratio of the two, path radius, tilt, terminations. Writes diff_stride_turn_<variant>.txt / .json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variant", default="rigid_uno")
ap.add_argument("--out", required=True)
ap.add_argument("--speed", type=float, default=0.20)
ap.add_argument("--periods", type=float, nargs="+", default=[0.21, 0.28])
ap.add_argument("--ks", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
ap.add_argument("--lift-mm", type=float, default=10.0)
ap.add_argument("--envs-per-k", type=int, default=8)
ap.add_argument("--seconds", type=float, default=8.0)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
from microtaur_common.gpu import pick_device  # noqa: E402

args.device = pick_device(args.device)
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from microtaur_common.kinematics import five_bar_for  # noqa: E402
from microtaur_common.robot_constants import STAND_A, STAND_E  # noqa: E402
from microtaur_common.robot_models import robot_model  # noqa: E402
from microtaur_isaac.env_cfg import MicrotaurFlatEnvCfg  # noqa: E402

TRACK_M = 0.100  # left / right foot spacing (feet at y +-50 mm in every model)
LEFT = (False, True, True, False)  # legs 1 RR, 2 RL, 3 FL, 4 FR
PHASE = (0.0, 0.5, 0.0, 0.5)  # diagonals RR + FL, RL + FR in phase

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
G = args.envs_per_k
KS = list(args.ks)
cfg = MicrotaurFlatEnvCfg(play=True, variant=args.variant)
cfg.scene.num_envs = G * len(KS)
cfg.curriculum.command_ranges = None
cfg.commands.twist.resampling_time_range = (1.0e9, 1.0e9)
env = ManagerBasedRLEnv(cfg)
robot = env.scene["robot"]
N, dt = env.num_envs, env.step_dt
n_act = env.action_manager.total_action_dim
scale = float(cfg.actions.joint_pos.action_scale_rad)
KIN = five_bar_for(args.variant)
stand = np.empty(8); stand[0::2] = STAND_A; stand[1::2] = STAND_E
x0, z0 = (lambda s: (s.foot_x, s.foot_z))(KIN.forward_numpy(STAND_A[0], STAND_E[0], 1))


def foot(phi: float, L: float, h: float) -> tuple[float, float]:
  if phi < 0.5:  # stance: sweep back on the ground by L (L < 0: forward)
    s = phi / 0.5
    return x0 + L / 2 - L * s, z0
  s = (phi - 0.5) / 0.5  # swing: return, lifted
  return x0 - L / 2 + L * 0.5 * (1 - math.cos(math.pi * s)), z0 + h * math.sin(math.pi * s)


def targets(t: float, T: float, L: float, k: float, h: float) -> np.ndarray:
  q = np.empty(8)
  for leg in range(4):
    fx, fz = foot(((t / T) + PHASE[leg]) % 1.0, L * ((1 - k) if LEFT[leg] else (1 + k)), h)
    sol = KIN.inverse_numpy(fx, fz, leg + 1, reference_motor=(stand[2 * leg], stand[2 * leg + 1]))
    if not sol.valid:
      raise RuntimeError(f"IK failed leg {leg + 1} at ({fx:.4f}, {fz:.4f}), k {k}")
    q[2 * leg], q[2 * leg + 1] = sol.q_a, sol.q_e
  return q


rows, lines = [], [f"{args.variant}: scripted IK trot, stroke left L(1-k) / right L(1+k), {args.speed} m/s nominal, "
                   f"lift {args.lift_mm} mm, {G} envs per k, measured {args.seconds - 2:.0f} s after 2 s; spine "
                   f"{'held at 0' if robot_model(args.variant).has_spine else 'none'}; ground friction 1.0",
                   "T s   k     | yaw rate rad/s | speed m/s | skid-free yaw rate | ratio | radius m | tilt deg | "
                   "terminated envs | max motor offset deg (clipped)"]
for T in args.periods:
  L = args.speed * T / 2
  v_stroke = L / (T / 2)
  env.reset()
  wz, vx, tilt, term = [], [], [], np.zeros(N, bool)
  qmax = {k: 0.0 for k in KS}
  for i in range(int(args.seconds / dt)):
    a = torch.zeros(N, n_act, device=env.device)
    for j, k in enumerate(KS):
      q = targets(i * dt, T, L, k, args.lift_mm * 1e-3)
      qmax[k] = max(qmax[k], float(np.abs(q - stand).max()))
      a[j * G:(j + 1) * G, :8] = torch.tensor(np.clip((q - stand) / scale, -1.0, 1.0), dtype=torch.float32, device=env.device)
    _, _, te, tr, _ = env.step(a)
    if i < int(2.0 / dt):
      continue
    term |= (te | tr).cpu().numpy()
    d = robot.data
    wz.append(d.root_link_ang_vel_b[:, 2].cpu().numpy().copy())
    vx.append(d.root_link_lin_vel_b[:, 0].cpu().numpy().copy())
    g = d.projected_gravity_b.cpu().numpy()
    tilt.append(np.degrees(np.arcsin(np.clip(np.linalg.norm(g[:, :2], axis=1), 0, 1))))
  wz, vx, tilt = np.array(wz), np.array(vx), np.array(tilt)
  for j, k in enumerate(KS):
    e = np.arange(j * G, (j + 1) * G)
    ok = e[~term[e]]
    if ok.size == 0:
      lines.append(f"{T:.2f}  {k:4.2f}  | all {G} envs terminated")
      continue
    w, v = float(wz[:, ok].mean()), float(vx[:, ok].mean())
    ideal = 2 * k * v_stroke / TRACK_M
    r = {"period_s": T, "k": k, "yaw_rate": w, "speed": v, "skid_free_yaw_rate": ideal,
         "ratio": w / ideal if ideal else None, "radius_m": abs(v / w) if abs(w) > 1e-3 else None,
         "tilt_deg": float(tilt[:, ok].mean()), "terminated": int(term[e].sum()), "envs": G,
         "max_motor_offset_deg": math.degrees(qmax[k]), "clipped": qmax[k] > scale}
    rows.append(r)
    lines.append(f"{T:.2f}  {k:4.2f}  | {w:+.3f}         | {v:+.3f}    | {ideal:+.3f}             | "
                 + (f"{r['ratio']:5.2f}" if ideal else "  -  ") + f" | "
                 + (f"{r['radius_m']:7.2f}" if r["radius_m"] else "    inf") + f"  | {r['tilt_deg']:5.1f}    | "
                 f"{r['terminated']}/{G}             | {r['max_motor_offset_deg']:.1f}" + (" (clipped)" if r["clipped"] else ""))
txt = "\n".join(lines) + "\n"
(out / f"diff_stride_turn_{args.variant}.txt").write_text(txt)
(out / f"diff_stride_turn_{args.variant}.json").write_text(json.dumps({"variant": args.variant, "args": vars(args), "rows": rows},
                                                                      indent=1, default=str))
print(txt, flush=True)
os._exit(0)
