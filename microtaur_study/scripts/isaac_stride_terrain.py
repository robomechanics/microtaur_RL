"""Stride rate and how the foot clears the ground, per terrain type and level (Teacher-Cur-Play):
touchdowns/s per foot, duty, swing foot lift (p95 over the stance median, world z), five-bar FK leg
retraction in swing (stance r - swing r, mm), stance foot slip (m/s), forward speed.

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_stride_terrain.py --checkpoint <model.pt> --out <file.txt>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--speed", type=float, default=0.20)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from microtaur_common.kinematics import MicrotaurFiveBarKinematics  # noqa: E402
from microtaur_common.robot_constants import LEG_JOINT_NAMES  # noqa: E402
from microtaur_isaac import FOOT_BODY_NAMES, FOOT_OFFSET_IN_BODY_M  # noqa: E402
from microtaur_isaac import terrains as TR  # noqa: E402
from microtaur_isaac.agents import MicrotaurTeacherPPORunnerCfg  # noqa: E402
from microtaur_isaac.env_cfg import MicrotaurTeacherCurPlayEnvCfg  # noqa: E402
from microtaur_isaac.mdp.contact import foot_contact_timers  # noqa: E402
from microtaur_isaac.mdp.observations import feet_sensor_cfg  # noqa: E402

N = 150  # 15 per column
LV = [0, 3, 6]
cfg = MicrotaurTeacherCurPlayEnvCfg()
cfg.scene.num_envs = N
cfg.curriculum.command_ranges = None
cfg.commands.twist.resampling_time_range = (1.0e9, 1.0e9)
env = ManagerBasedRLEnv(cfg)
t = env.scene.terrain
t.terrain_levels[:] = torch.tensor([LV[i % 3] for i in range(N)], device=env.device)
t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]
w = RslRlVecEnvWrapper(env)
runner = OnPolicyRunner(w, MicrotaurTeacherPPORunnerCfg().to_dict(), log_dir=None, device=env.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=env.device)
robot = env.scene["robot"]
cmd = env.command_manager.get_term("twist")
fc = feet_sensor_cfg()
fc.resolve(env.scene)
fids = [robot.body_names.index(n) for n in FOOT_BODY_NAMES]
mids = [robot.joint_names.index(n) for n in LEG_JOINT_NAMES]
off = torch.tensor(FOOT_OFFSET_IN_BODY_M, device=env.device)
dt = env.step_dt


def feet():
  p = robot.data.body_link_pose_w[:, fids]
  o = off.expand(4, 3) if off.dim() == 1 else off
  return p[..., :3] + quat_apply(p[..., 3:7].reshape(-1, 4), o.repeat(N, 1)).reshape(N, 4, 3)


rec = {k: [] for k in ("c", "z", "v", "q", "vx", "done")}
with torch.inference_mode():
  obs = w.get_observations()
  prev = feet()
  for k in range(300):
    cmd.vel_command_b[:, 0] = args.speed
    cmd.vel_command_b[:, 1:] = 0.0
    obs, _, dones, _ = w.step(policy(obs))
    cur = feet()
    v = torch.linalg.norm((cur - prev)[..., :2], dim=-1) / dt
    prev = cur
    if k < 60:
      continue
    rec["c"].append(foot_contact_timers(env, fc)[0].cpu().numpy().copy())
    rec["z"].append(cur[..., 2].cpu().numpy())
    rec["v"].append(v.cpu().numpy())
    rec["q"].append(robot.data.joint_pos[:, mids].cpu().numpy())
    rec["vx"].append(robot.data.root_link_lin_vel_b[:, 0].cpu().numpy())
    rec["done"].append(dones.cpu().numpy())
R = {k: np.stack(v) for k, v in rec.items()}
KIN = MicrotaurFiveBarKinematics()
types = TR.env_terrain_type_ids(t).cpu().numpy()
lv = t.terrain_levels.cpu().numpy()
ok = ~R["done"].any(0)
lines = [f"checkpoint {args.checkpoint}", f"straight {args.speed} m/s, 8.4 s per env after 2 s warm-up, envs without reset",
         "terrain lvl |  n | vx m/s | stride Hz | duty | swing lift p95 mm | FK retraction mm (stance r - swing r) | stance slip m/s"]
for ty, nm in ((0, "A"), (1, "B"), (2, "C")):
  for L in LV:
    sel = np.nonzero(ok & (types == ty) & (lv == L))[0]
    if len(sel) == 0:
      continue
    c = R["c"][:, sel].astype(bool)
    hz = ((~c[:-1]) & c[1:]).sum(0).mean() / ((c.shape[0] - 1) * dt)
    lift, retr, slip = [], [], []
    for jj, e in enumerate(sel[:6]):
      for f in range(4):
        st, sw = c[:, jj, f], ~c[:, jj, f]
        if st.sum() < 5 or sw.sum() < 5:
          continue
        z = R["z"][:, e, f]
        lift.append(np.percentile(z[sw], 95) - np.median(z[st]))
        r = np.array([np.hypot(*(lambda s: (s.foot_x, s.foot_z))(KIN.forward_numpy(a, b, f + 1)))
                      for a, b in zip(R["q"][:, e, 2 * f], R["q"][:, e, 2 * f + 1])])
        retr.append(np.median(r[st]) - np.percentile(r[sw], 5))
        slip.append(R["v"][:, e, f][st].mean())
    lines.append(f"{nm} {L}  | {len(sel):2d} | {R['vx'][:, sel].mean():.3f}  | {hz:9.1f} | {c.mean():.2f} | {np.mean(lift) * 1e3:17.1f} | "
                 f"{np.mean(retr) * 1e3:38.1f} | {np.mean(slip):.3f}")
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text("\n".join(lines) + "\n")
print("\n".join(lines), flush=True)
sys.stdout.flush()
os._exit(0)
