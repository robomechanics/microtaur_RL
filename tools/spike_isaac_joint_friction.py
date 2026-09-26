"""Is PhysX joint friction a constant torque (like MuJoCo frictionloss) in this Isaac Sim?

IsaacLab 2.3 documents `friction` as a unitless coefficient in Isaac Sim 4.5
and as an effort (N m) from Isaac Sim 5.0. This measures it: the rigid
Microtaur USD from spike_isaac_closed_chain.py floating with gravity off, every
joint's damping and stiffness 0, one motor joint (leg1_a) given friction F and a
constant applied torque tau. If friction is a constant torque, the joint stays
put for tau < F and accelerates for tau > F.

  python tools/spike_isaac_joint_friction.py --usd <microtaur_rigid_revolute.usda> [--friction 0.010]
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--friction", type=float, default=0.010)
ap.add_argument("--out", default="")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app

_log = open(args.out or os.devnull, "w")


def log(*a):
  s = " ".join(str(x) for x in a)
  _log.write(s + "\n")
  _log.flush()
  print(s, flush=True)


import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

TAUS = (0.0, 0.005, 0.009, 0.011, 0.015, 0.019, 0.021, 0.030)
JOINT = "leg1_a_joint_act"


@configclass
class SceneCfg(InteractiveSceneCfg):
  robot: ArticulationCfg = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
      usd_path=args.usd,
      rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
      articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=16, solver_velocity_iteration_count=0,
      ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.3)),
    actuators={
      "test": ImplicitActuatorCfg(joint_names_expr=[JOINT], stiffness=0.0, damping=0.0, friction=args.friction),
      "rest": ImplicitActuatorCfg(joint_names_expr=[f"^(?!{JOINT}$).*"], stiffness=0.0, damping=0.0, friction=0.0),
    },
  )


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.0025, device=args.device))
scene = InteractiveScene(SceneCfg(num_envs=len(TAUS), env_spacing=0.5))
sim.reset()
robot: Articulation = scene["robot"]
j = robot.joint_names.index(JOINT)
log(f"Isaac Sim joint friction test: joint {JOINT}, friction param {args.friction}")
log(f"PhysX friction coefficient read back: {robot.root_physx_view.get_dof_friction_coefficients()[0, j].item():.4f}")

effort = torch.zeros(len(TAUS), robot.num_joints, device=robot.device)
effort[:, j] = torch.tensor(TAUS, device=robot.device)
q0 = robot.data.joint_pos[:, j].clone()
for _ in range(200):  # 0.5 s
  robot.set_joint_effort_target(effort)
  scene.write_data_to_sim()
  sim.step(render=False)
  scene.update(0.0025)
dq = (robot.data.joint_pos[:, j] - q0).cpu()
qd = robot.data.joint_vel[:, j].cpu()
for i, tau in enumerate(TAUS):
  moved = abs(float(dq[i])) > 1e-3
  log(f"  applied {tau:.3f} N m -> moved {float(dq[i]):+.4f} rad, velocity {float(qd[i]):+.3f} rad/s  {'MOVES' if moved else 'held'}")
_log.close()
os._exit(0)
