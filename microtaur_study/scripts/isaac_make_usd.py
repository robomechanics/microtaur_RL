"""Build assets/rigid_microtaur/microtaur_rigid.usda from the MJCF and check it.

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_make_usd.py      (conda env spine)

Spawns the USD in IsaacLab and checks masses and inertias against MuJoCo and the
closure gap at the stand pose.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app
LOG = open(Path(__file__).resolve().parents[1] / "assets" / "rigid_microtaur" / "microtaur_rigid.usda.check.txt", "w")


def log(*a):
  LOG.write(" ".join(map(str, a)) + "\n"); LOG.flush(); print(*a, flush=True)


import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

from microtaur_isaac.robot import MICROTAUR_RIGID_CFG, USD_PATH  # noqa: E402
from microtaur_isaac.usd import build_usd, compile_model  # noqa: E402

info = build_usd(str(USD_PATH))
log(f"wrote {USD_PATH}: {len(info['bodies'])} bodies, loops {info['loops']}, MuJoCo total mass {info['total_mass_kg']:.6f} kg")


@configclass
class SceneCfg(InteractiveSceneCfg):
  robot = MICROTAUR_RIGID_CFG


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.0025, device="cuda:0"))
scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=1.0))
sim.reset()
robot: Articulation = scene["robot"]
m = compile_model()
masses = robot.root_physx_view.get_masses()[0].cpu().numpy()
inert = robot.root_physx_view.get_inertias()[0].cpu().numpy().reshape(-1, 3, 3)
worst_m = worst_i = 0.0
for i, name in enumerate(robot.body_names):
  b = m.body(name).id
  worst_m = max(worst_m, abs(masses[i] - m.body_mass[b]))
  worst_i = max(worst_i, float(np.abs(np.sort(np.linalg.eigvalsh(inert[i])) - np.sort(m.body_inertia[b])).max()))
log(f"PhysX total mass {masses.sum():.6f} kg; max |mass diff| {worst_m:.2e} kg; max |principal inertia diff| {worst_i:.2e}")
log(f"joints ({robot.num_joints}): {robot.joint_names}")
# closure gap at t = 0 from the MJCF closing sites
gaps = []
pose = robot.data.body_link_pose_w[0]
for i in range(m.neq):
  s1, s2 = int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
  b1 = robot.body_names.index(m.body(int(m.site_bodyid[s1])).name)
  b2 = robot.body_names.index(m.body(int(m.site_bodyid[s2])).name)
  p1 = pose[b1, :3] + quat_apply(pose[b1, 3:7][None], torch.tensor(m.site_pos[s1], dtype=torch.float32, device=pose.device)[None])[0]
  p2 = pose[b2, :3] + quat_apply(pose[b2, 3:7][None], torch.tensor(m.site_pos[s2], dtype=torch.float32, device=pose.device)[None])[0]
  gaps.append(float(torch.linalg.norm(p1 - p2)))
log(f"closure gap at stand, max over 8 sites: {max(gaps):.2e} m")
LOG.close()
os._exit(0)
