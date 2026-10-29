"""Check the five-bar FK / IK maths, and FK against the simulator's actual foot positions.

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_check_fk_ik.py --checkpoint <policy.pt> --out <dir>

1. Maths (numpy): FK -> IK -> FK round trips over the motor range (stand +- 0.75 rad), the torch FK used
   by the observations against the numpy FK, IK of a grid of foot targets, and the virtual leg
   (r = |(x, z)| from the midpoint of the two motor pivots, theta = atan2(x, -z), 0 = straight down,
   + = foot ahead) at the stand pose.
2. Simulator: the policy walks flat ground in Teacher-Cur-Play; every policy step each foot-sphere
   centre is taken from the simulated body pose, expressed in the root frame and compared with
   hip midpoint + (FK x, lateral offset, FK z) from the four simulated motor angles.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--maths-only", action="store_true")
args, _ = ap.parse_known_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from microtaur_common.kinematics import FOOT_LATERAL_OFFSETS_M, HIP_MIDPOINTS_ROOT_M, MicrotaurFiveBarKinematics  # noqa: E402
from microtaur_common.robot_constants import JOINT_HALF_RANGE_RAD, STAND_A, STAND_E  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
K = MicrotaurFiveBarKinematics()
lines, res = [], {}


def say(s):
  lines.append(s)
  print(s, flush=True)


def leg_polar(x, z):
  return np.hypot(x, z), np.arctan2(x, -z)


# ---------------------------------------------------------------- 1. maths
rng = np.random.default_rng(0)
say("== maths (numpy) ==")
for leg in range(1, 5):
  a0, e0 = STAND_A[leg - 1], STAND_E[leg - 1]
  n, ok, q_err, p_err, tor_err = 0, 0, [], [], []
  for _ in range(4000):
    qa = a0 + rng.uniform(-JOINT_HALF_RANGE_RAD, JOINT_HALF_RANGE_RAD)
    qe = e0 + rng.uniform(-JOINT_HALF_RANGE_RAD, JOINT_HALF_RANGE_RAD)
    s = K.forward_numpy(qa, qe, leg)
    if not s.valid:
      continue
    n += 1
    tx, tz = K.forward_torch(torch.tensor([[qa if j == leg - 1 else a0 for j in range(4)]], dtype=torch.float64),
                             torch.tensor([[qe if j == leg - 1 else e0 for j in range(4)]], dtype=torch.float64))
    tor_err.append(max(abs(float(tx[0, leg - 1]) - s.foot_x), abs(float(tz[0, leg - 1]) - s.foot_z)))
    inv = K.inverse_numpy(s.foot_x, s.foot_z, leg, reference_motor=(qa, qe))
    if inv.valid:
      ok += 1
      q_err.append(max(abs(inv.q_a - qa), abs(inv.q_e - qe)))
      f = K.forward_numpy(inv.q_a, inv.q_e, leg)  # independent: FK of the IK's motor angles
      p_err.append(np.hypot(f.foot_x - s.foot_x, f.foot_z - s.foot_z))
  r = {"samples": n, "ik_valid": ok, "q_err_max_rad": float(np.max(q_err)), "foot_err_max_mm": float(np.max(p_err)) * 1e3,
       "torch_vs_numpy_max_mm": float(np.max(tor_err)) * 1e3}
  res[f"leg{leg}_roundtrip"] = r
  say(f"leg {leg}: {n} valid FK samples, IK valid {ok}; IK(FK(q)) max |dq| {r['q_err_max_rad']:.2e} rad, "
      f"FK(IK(p)) max err {r['foot_err_max_mm']:.2e} mm, torch FK vs numpy {r['torch_vs_numpy_max_mm']:.2e} mm")
# IK from the stand reference over a grid of foot targets (what an IK action space would ask)
a0, e0 = STAND_A[0], STAND_E[0]
s0 = K.forward_numpy(a0, e0, 1)
r0, th0 = leg_polar(s0.foot_x, s0.foot_z)
say(f"stand (leg 1): foot (x, z) = ({s0.foot_x * 1e3:.2f}, {s0.foot_z * 1e3:.2f}) mm, virtual leg r = {r0 * 1e3:.2f} mm, "
    f"theta = {np.degrees(th0):+.2f} deg")
reach, worst = 0, 0.0
xs, zs = np.linspace(-0.04, 0.04, 41), np.linspace(-0.110, -0.040, 36)
for x in xs:
  for z in zs:
    inv = K.inverse_numpy(x, z, 1, reference_motor=(a0, e0))
    if inv.valid and abs(inv.q_a - a0) <= JOINT_HALF_RANGE_RAD and abs(inv.q_e - e0) <= JOINT_HALF_RANGE_RAD:
      reach += 1
      f = K.forward_numpy(inv.q_a, inv.q_e, 1)
      worst = max(worst, float(np.hypot(f.foot_x - x, f.foot_z - z)))
say(f"IK grid x -40..40 mm, z -110..-40 mm: {reach}/{len(xs) * len(zs)} targets reachable inside the motor range, "
    f"max FK(IK) residual {worst * 1e3:.2e} mm")
res["stand"] = {"foot_x_mm": s0.foot_x * 1e3, "foot_z_mm": s0.foot_z * 1e3, "r_mm": r0 * 1e3, "theta_deg": float(np.degrees(th0))}

# ---------------------------------------------------------------- 2. simulator
if not args.maths_only:
  from isaaclab.app import AppLauncher

  ap2 = argparse.ArgumentParser()
  AppLauncher.add_app_launcher_args(ap2)
  largs, _ = ap2.parse_known_args()
  largs.headless = True
  from microtaur_common.gpu import pick_device

  largs.device = pick_device(largs.device)
  app = AppLauncher(largs).app
  from isaaclab.envs import ManagerBasedRLEnv
  from isaaclab.utils.math import quat_apply, quat_apply_inverse
  from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

  from microtaur_common.robot_constants import LEG_JOINT_NAMES
  from microtaur_isaac import FOOT_BODY_NAMES, FOOT_OFFSET_IN_BODY_M
  from microtaur_isaac.env_cfg import MicrotaurTeacherCurPlayEnvCfg
  from microtaur_isaac.mdp.contact import foot_contact_timers
  from microtaur_isaac.mdp.observations import feet_sensor_cfg
  from microtaur_isaac.policy_loader import load_policy, prepare_env_cfg

  cfg = MicrotaurTeacherCurPlayEnvCfg()
  cfg.scene.num_envs = 20  # columns 0-1 = A (flat)
  cfg.sim.device = largs.device
  cfg.curriculum.command_ranges = None
  cfg.commands.twist.resampling_time_range = (1e9, 1e9)
  prepare_env_cfg(cfg, args.checkpoint)
  env = ManagerBasedRLEnv(cfg)
  t = env.scene.terrain
  t.terrain_levels[:] = 0
  t.env_origins[:] = t.terrain_origins[0, t.terrain_types]
  w = RslRlVecEnvWrapper(env)
  act, reset = load_policy(w, args.checkpoint)
  robot = env.scene["robot"]
  mids = [robot.joint_names.index(n) for n in LEG_JOINT_NAMES]
  fids = [robot.body_names.index(n) for n in FOOT_BODY_NAMES]
  off = torch.tensor(FOOT_OFFSET_IN_BODY_M, device=env.device, dtype=torch.float32)
  hip = torch.tensor(HIP_MIDPOINTS_ROOT_M, device=env.device, dtype=torch.float32)
  lat = torch.tensor(FOOT_LATERAL_OFFSETS_M, device=env.device, dtype=torch.float32)
  fc = feet_sensor_cfg()
  fc.resolve(env.scene)
  cmd = env.command_manager.get_term("twist")
  N = env.num_envs
  E, C, R, TH = [], [], [], []
  with torch.inference_mode():
    obs = w.get_observations()
    for k in range(args.steps):
      cmd.vel_command_b[:, 0] = 0.20
      cmd.vel_command_b[:, 1:] = 0.0
      obs, _, dones, _ = w.step(act(obs))
      reset(dones)
      if k < 20:
        continue
      d = robot.data
      fp = d.body_link_pos_w[:, fids] + quat_apply(d.body_link_quat_w[:, fids].reshape(-1, 4), off.repeat(N, 1)).reshape(N, 4, 3)
      rel = quat_apply_inverse(d.root_link_quat_w[:, None].expand(-1, 4, -1).reshape(-1, 4),
                               (fp - d.root_link_pos_w[:, None]).reshape(-1, 3)).reshape(N, 4, 3) - hip
      q = d.joint_pos[:, mids].double()
      fx, fz = K.forward_torch(q[:, 0::2], q[:, 1::2])
      pred = torch.stack((fx.float(), lat.expand(N, 4), fz.float()), dim=-1)
      E.append((rel - pred).cpu().numpy())
      C.append(foot_contact_timers(env, fc)[0].cpu().numpy())
      R.append(torch.hypot(fx, fz).cpu().numpy())
      TH.append(torch.atan2(fx, -fz).cpu().numpy())
  E, C, R, TH = map(np.stack, (E, C, R, TH))
  ok = (t.terrain_types < 2).cpu().numpy()  # A columns (flat) only
  e, c = E[:, ok], C[:, ok].astype(bool)
  say("== simulator (t4 trotting on flat, foot-sphere centre vs hip + FK, root frame) ==")
  for name, axis in (("x (fore-aft)", 0), ("y (lateral)", 1), ("z (vertical)", 2)):
    v = np.abs(e[..., axis]) * 1e3
    say(f"{name:13s}: |err| mean {v.mean():.3f} mm, p99 {np.percentile(v, 99):.3f}, max {v.max():.3f}; stance mean {v[c].mean():.3f}, swing mean {v[~c].mean():.3f}")
  dist = np.linalg.norm(e, axis=-1) * 1e3
  res["sim"] = {"err3d_mean_mm": float(dist.mean()), "err3d_p99_mm": float(np.percentile(dist, 99)), "err3d_max_mm": float(dist.max()),
                "r_range_mm": [float(R[:, ok].min() * 1e3), float(R[:, ok].max() * 1e3)],
                "theta_range_deg": [float(np.degrees(TH[:, ok].min())), float(np.degrees(TH[:, ok].max()))]}
  say(f"3-D |err|: mean {res['sim']['err3d_mean_mm']:.3f} mm, p99 {res['sim']['err3d_p99_mm']:.3f}, max {res['sim']['err3d_max_mm']:.3f}; "
      f"walking r {res['sim']['r_range_mm'][0]:.1f}..{res['sim']['r_range_mm'][1]:.1f} mm, theta {res['sim']['theta_range_deg'][0]:+.1f}..{res['sim']['theta_range_deg'][1]:+.1f} deg")
(out / "fk_ik_check.txt").write_text("\n".join(lines) + "\n")
(out / "fk_ik_check.json").write_text(json.dumps(res, indent=1))
sys.stdout.flush()
os._exit(0)
