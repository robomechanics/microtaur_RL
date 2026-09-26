"""Check the left/right mirror used for symmetry augmentation (mdp/symmetry.py).

  OMNI_KIT_ACCEPT_EULA=YES python tests/check_isaac_symmetry.py --out <log>

1. Model: is the MJCF left/right symmetric? For every mirror joint pair (leg1<->leg2,
   leg4<->leg3), the child bodies' mass, inertia, and world position / COM at the
   stand pose with y negated; plus the root body's COM y.
2. Mirror function: env 0 walks a few random steps; env 1 is set to the mirrored state
   (root pose / velocity reflected, every joint of the partner leg negated, action-term
   buffers mirrored). Both then step with mirrored random actions; each step compares
   env 1's observations (policy and critic groups) with mirror(env 0's observations).
   A correct mirror + a symmetric model keep the difference at float noise at first;
   contact makes it grow slowly.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--action-std", type=float, default=0.5)
ap.add_argument("--copy", action="store_true", help="control: env 1 = exact copy of env 0, same actions (no mirror)")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app
LOG = open(args.out, "w")


def log(*a):
  LOG.write(" ".join(map(str, a)) + "\n"); LOG.flush()


import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from microtaur_isaac.env_cfg import MicrotaurFlatEnvCfg  # noqa: E402
from microtaur_isaac.mdp import symmetry as SYM  # noqa: E402
if args.copy:
  mirror_joints = lambda x: x.clone()  # noqa: E731
  mirror_group = lambda env, grp, x: x.clone()  # noqa: E731
else:
  mirror_group, mirror_joints = SYM.mirror_group, SYM.mirror_joints
from microtaur_isaac.usd import compile_model  # noqa: E402

PARTNER = {1: 2, 2: 1, 3: 4, 4: 3}

# --- 1. model symmetry ------------------------------------------------------------
m = compile_model()
d = mujoco.MjData(m)
from microtaur_common.kinematics import FULL_JOINT_NAMES  # noqa: E402
from microtaur_common.reset import NOMINAL_FULL_Q  # noqa: E402
d.qpos[3] = 1.0
for n, q in zip(FULL_JOINT_NAMES, NOMINAL_FULL_Q.reshape(16)):
  d.qpos[m.joint(n).qposadr[0]] = q
mujoco.mj_forward(m, d)
worst = {"mass": 0.0, "inertia": 0.0, "pos": 0.0, "com": 0.0}
for leg in (1, 3):
  for j in ("a_joint_act", "e_joint_act", "b_joint", "d_joint"):
    b1 = m.jnt_bodyid[m.joint(f"leg{leg}_{j}").id]
    b2 = m.jnt_bodyid[m.joint(f"leg{PARTNER[leg]}_{j}").id]
    flip = np.array([1.0, -1.0, 1.0])
    worst["mass"] = max(worst["mass"], abs(m.body_mass[b1] - m.body_mass[b2]))
    worst["inertia"] = max(worst["inertia"], float(np.abs(np.sort(m.body_inertia[b1]) - np.sort(m.body_inertia[b2])).max()))
    R0, p0 = d.xmat[m.body("battery").id].reshape(3, 3), d.xpos[m.body("battery").id]
    rel = lambda x: R0.T @ (x - p0)  # noqa: E731  (in the root body frame)
    worst["pos"] = max(worst["pos"], float(np.abs(rel(d.xpos[b1]) - flip * rel(d.xpos[b2])).max()))
    worst["com"] = max(worst["com"], float(np.abs(rel(d.xipos[b1]) - flip * rel(d.xipos[b2])).max()))
root = m.body("battery").id
log(f"[model] mirror pairs: max |mass diff| {worst['mass']:.2e} kg, |principal inertia diff| {worst['inertia']:.2e}, "
    f"|body pos - mirror| {1e3 * worst['pos']:.3f} mm, |body COM - mirror| {1e3 * worst['com']:.3f} mm")
R0, p0 = d.xmat[root].reshape(3, 3), d.xpos[root]
log(f"[model] root body COM in its frame (mm): {np.round(1e3 * m.body_ipos[root], 3).tolist()}; "
    f"whole-robot COM at stand, root frame (mm): {np.round(1e3 * (R0.T @ (d.subtree_com[root] - p0)), 3).tolist()}")

# --- 2. mirror function -------------------------------------------------------------
cfg = MicrotaurFlatEnvCfg(play=True)
cfg.scene.num_envs = 2
cfg.commands.twist.resampling_time_range = (1e9, 1e9)
env = ManagerBasedRLEnv(cfg)
robot = env.scene["robot"]
term = env.action_manager.get_term("joint_pos")
names = robot.joint_names
jperm = [names.index(n.replace(f"leg{int(n[3])}_", f"leg{PARTNER[int(n[3])]}_", 1)) for n in names]
env.reset()
g = torch.Generator(device=env.device).manual_seed(0)
cmd = env.command_manager.get_term("twist")
cmd.vel_command_b[0] = torch.tensor([0.2, 0.0, 0.15], device=env.device)
cmd.vel_command_b[1] = torch.tensor([0.2, 0.0, -0.15], device=env.device)
for _ in range(8):
  a = 0.5 * torch.randn(2, 8, device=env.device, generator=g)
  env.step(a)

# teleport env 1 to the mirror of env 0
dat = robot.data
pose = dat.root_link_pose_w.clone()
o = env.scene.env_origins
p = pose[0, :3] - o[0]
q = pose[0, 3:7]
F = 1.0 if args.copy else -1.0
pose[1, :3] = o[1] + p * torch.tensor([1.0, F, 1.0], device=env.device)
pose[1, 3:7] = q * torch.tensor([1.0, F, 1.0, F], device=env.device)  # mirror: (w, x, y, z) -> (w, -x, y, -z)
vel = dat.root_link_vel_w.clone()
vel[1] = vel[0] * torch.tensor([1.0, F, 1.0, F, 1.0, F], device=env.device)
jp, jv = dat.joint_pos.clone(), dat.joint_vel.clone()
perm = list(range(len(names))) if args.copy else jperm
jp[1], jv[1] = F * jp[0, perm], F * jv[0, perm]
ids = torch.tensor([1], device=env.device)
robot.write_root_link_pose_to_sim(pose[1:], env_ids=ids)
robot.write_root_link_velocity_to_sim(vel[1:], env_ids=ids)
robot.write_joint_state_to_sim(jp[1:], jv[1:], env_ids=ids)
core = term.core
for buf in ("requested", "safe", "applied"):
  x = getattr(core, buf)
  x[1] = mirror_joints(x[0:1])[0]
bias = getattr(env, "microtaur_encoder_bias", None)
if bias is not None:
  bias[1] = mirror_joints(bias[0:1])[0]
core._delay_buffer[1] = mirror_joints(core._delay_buffer[0])
core._gain[1] = core._gain[0] if args.copy else core._gain[0][[2, 3, 0, 1, 6, 7, 4, 5]]
core._bias[1] = core._bias[0] if args.copy else -core._bias[0][[2, 3, 0, 1, 6, 7, 4, 5]]
core._delay_steps[1] = core._delay_steps[0]
term._raw[1] = mirror_joints(term._raw[0:1])[0]
env.action_manager._action[1] = mirror_joints(env.action_manager._action[0:1])[0]
env.action_manager._prev_action[1] = mirror_joints(env.action_manager._prev_action[0:1])[0]

for t in range(args.steps):
  a0 = args.action_std * torch.randn(1, 8, device=env.device, generator=g)
  obs, _, term_, trunc, _ = env.step(torch.cat((a0, mirror_joints(a0)), dim=0))
  if bool((term_ | trunc).any()):
    log(f"  step {t}: an env reset, stopping"); break
  line = f"  step {t:2d} | applied target {float((core.applied[1] - mirror_joints(core.applied[0:1])[0]).abs().max()):.1e}"
  for grp in ("policy", "critic"):
    diff = (obs[grp][1] - mirror_group(env, grp, obs[grp][0:1])[0]).abs()
    om = env.observation_manager
    i, worst_term = 0, ("", 0.0)
    for n, dims in zip(om.active_terms[grp], om.group_obs_term_dim[grp]):
      w = int(np.prod(dims)); e = float(diff[i:i + w].max()); i += w
      if e > worst_term[1]:
        worst_term = (n, e)
    line += f" | {grp} max {float(diff.max()):.2e} (worst {worst_term[0]} {worst_term[1]:.2e})"
  jd = float((obs["policy"][1, 6:14] - mirror_joints(obs["policy"][0:1, 6:14])[0]).abs().max())
  line += f" | joint_pos {jd:.1e}"
  log(line)
LOG.close()
os._exit(0)
