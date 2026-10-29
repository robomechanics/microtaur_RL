"""Build the active-yaw Microtaur USDs (training + visual) from assets/active_yaw_microtaur/robot_2imu.xml
and check them against MuJoCo.

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_make_usd_yaw.py      (conda env spine)

Checks (written to assets/active_yaw_microtaur/microtaur_yaw.usda.check.txt):
  1. per-body mass and principal inertia, PhysX vs MuJoCo; joint list (16 leg + spine); closure gap at stand
  2. spine inertia about its axis (MuJoCo mass matrix at stand) against SPINE_LINK_INERTIA, and the
     resulting natural frequency / damping ratio against the 2026-10-28 measurement
  3. spine actuator inertia, IsaacLab vs MuJoCo in the same scenario: free floating, gravity off, legs held
     at stand, constant 0.05 N m on the spine, PD off -> angular acceleration and effective inertia
  4. standing on flat ground for 3 s, motors holding the stand pose: IsaacLab vs MuJoCo (same PD gains,
     armature, friction, effort caps): root height, pitch, spine angle
  5. IMU 2 frame (rear body x site rotation) vs MuJoCo's imu2_quat sensor at stand
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_mul  # noqa: E402

from microtaur_common.kinematics import FULL_JOINT_NAMES  # noqa: E402
from microtaur_common.reset import NOMINAL_FULL_Q  # noqa: E402
from microtaur_common import robot_constants as RC  # noqa: E402
from microtaur_isaac.robot import MICROTAUR_RIGID_CFG, MICROTAUR_YAW_CFG, YAW_NOMINAL_ROOT_HEIGHT_M, YAW_USD_PATH, YAW_VISUAL_USD_PATH  # noqa: E402
from microtaur_common.reset import NOMINAL_ROOT_HEIGHT_M  # noqa: E402
from microtaur_isaac.usd import build_usd, compile_model  # noqa: E402

LOG = open(RC.YAW_ASSET_DIR / "microtaur_yaw.usda.check.txt", "w")


def log(*a):
  LOG.write(" ".join(map(str, a)) + "\n"); LOG.flush(); print(*a, flush=True)


XML = RC.YAW_XML_PATH
info = build_usd(str(YAW_USD_PATH), xml_path=XML)
log(f"wrote {YAW_USD_PATH}: {len(info['bodies'])} bodies, loops {info['loops']}, MuJoCo total mass {info['total_mass_kg']:.6f} kg")
build_usd(str(YAW_VISUAL_USD_PATH), visuals=True, xml_path=XML)
log(f"wrote {YAW_VISUAL_USD_PATH} ({YAW_VISUAL_USD_PATH.stat().st_size / 1e6:.1f} MB)")
m = compile_model(XML)
STAND = dict(zip(FULL_JOINT_NAMES, map(float, NOMINAL_FULL_Q.reshape(16))))
STAND[RC.SPINE_JOINT_NAME] = 0.0


# ---------------------------------------------------------------- 2. spine inertia (MuJoCo, analytic)
def mj_at_stand(model, z=0.2):
  d = mujoco.MjData(model)
  free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
  a = model.jnt_qposadr[free]
  d.qpos[a:a + 3] = (0, 0, z); d.qpos[a + 3:a + 7] = (1, 0, 0, 0)
  names = {model.joint(j).name for j in range(model.njnt)}
  for n, q in STAND.items():
    if n in names:  # the rigid control model has no spine
      d.qpos[model.joint(n).qposadr[0]] = q
  mujoco.mj_forward(model, d)
  return d


d = mj_at_stand(m)
M = np.zeros((m.nv, m.nv))
try:
  mujoco.mj_fullM(m, d, M)  # MuJoCo >= 3.3 signature
except TypeError:
  mujoco.mj_fullM(m, M, d.qM)
sdof = m.jnt_dofadr[m.joint(RC.SPINE_JOINT_NAME).id]
j_link = M[sdof, sdof] - m.dof_armature[sdof]
j_tot = j_link + RC.SPINE_ARMATURE
log(f"spine inertia about its axis at stand: link {j_link:.4e} (constant SPINE_LINK_INERTIA {RC.SPINE_LINK_INERTIA:.4e}), "
    f"+ armature {RC.SPINE_ARMATURE:.4e} = {j_tot:.4e} kg m^2 (measured {RC.SPINE_INERTIA_FROM_MEASUREMENT})")
wn = math.sqrt(RC.SPINE_KP / j_tot) / (2 * math.pi)
zeta = RC.SPINE_KD / (2 * math.sqrt(RC.SPINE_KP * j_tot))
log(f"-> spine omega_n {wn:.3f} Hz (measured {RC.SPINE_OMEGA_N_HZ_MEASURED}), zeta {zeta:.3f} (measured {RC.SPINE_ZETA_MEASURED})")


# ---------------------------------------------------------------- 1. USD vs MuJoCo
@configclass
class SceneCfg(InteractiveSceneCfg):
  ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(
    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, friction_combine_mode="max")))
  robot = MICROTAUR_YAW_CFG
  robot_vis = MICROTAUR_YAW_CFG.replace(
    prim_path="{ENV_REGEX_NS}/RobotVis", spawn=MICROTAUR_YAW_CFG.spawn.replace(usd_path=str(YAW_VISUAL_USD_PATH)),
    init_state=MICROTAUR_YAW_CFG.init_state.replace(pos=(0.0, 0.5, YAW_NOMINAL_ROOT_HEIGHT_M)),
  )
  # Control: the rigid robot under the same standing test (separates yaw-specific from engine differences)
  rigid = MICROTAUR_RIGID_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotRigid",
                                      init_state=MICROTAUR_RIGID_CFG.init_state.replace(pos=(0.5, 0.0, NOMINAL_ROOT_HEIGHT_M)))
  # Free floating, gravity off, spine PD off (the articulation root cannot be fixed: its root prim is an
  # Xform without a RigidBodyAPI; rewriting the root pose every step moves the whole articulation, so it
  # does not pin anything either). Compared with MuJoCo in the same scenario.
  robot_fixed = MICROTAUR_YAW_CFG.replace(
    prim_path="{ENV_REGEX_NS}/RobotFloat",
    spawn=MICROTAUR_YAW_CFG.spawn.replace(rigid_props=MICROTAUR_YAW_CFG.spawn.rigid_props.replace(disable_gravity=True)),
    init_state=MICROTAUR_YAW_CFG.init_state.replace(pos=(0.0, -0.5, 0.3)),
    actuators={**MICROTAUR_YAW_CFG.actuators, "spine": MICROTAUR_YAW_CFG.actuators["spine"].replace(
      stiffness=0.0, damping=0.0, friction=0.0, dynamic_friction=0.0)},
  )


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.0025, device="cuda:0"))
scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=1.0))
sim.reset()
robot: Articulation = scene["robot"]
masses = robot.root_physx_view.get_masses()[0].cpu().numpy()
inert = robot.root_physx_view.get_inertias()[0].cpu().numpy().reshape(-1, 3, 3)
worst_m = worst_i = 0.0
for i, name in enumerate(robot.body_names):
  b = m.body(name).id
  worst_m = max(worst_m, abs(masses[i] - m.body_mass[b]))
  worst_i = max(worst_i, float(np.abs(np.sort(np.linalg.eigvalsh(inert[i])) - np.sort(m.body_inertia[b])).max()))
log(f"PhysX total mass {masses.sum():.6f} kg; max |mass diff| {worst_m:.2e} kg; max |principal inertia diff| {worst_i:.2e}")
log(f"joints ({robot.num_joints}): {robot.joint_names}")
assert RC.SPINE_JOINT_NAME in robot.joint_names and robot.num_joints == 17, "expected 16 leg joints + the spine"
log(f"root body: {robot.body_names[0]} (expected {RC.YAW_ROOT_BODY})")
vis: Articulation = scene["robot_vis"]
assert vis.body_names == robot.body_names and vis.joint_names == robot.joint_names
dm = float((vis.root_physx_view.get_masses() - robot.root_physx_view.get_masses()).abs().max())
log(f"visual USD vs training USD: max |mass diff| {dm:.1e} kg")
pose = robot.data.body_link_pose_w[0]
gaps = []
for i in range(m.neq):
  s1, s2 = int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
  b1 = robot.body_names.index(m.body(int(m.site_bodyid[s1])).name)
  b2 = robot.body_names.index(m.body(int(m.site_bodyid[s2])).name)
  p1 = pose[b1, :3] + quat_apply(pose[b1, 3:7][None], torch.tensor(m.site_pos[s1], dtype=torch.float32, device=pose.device)[None])[0]
  p2 = pose[b2, :3] + quat_apply(pose[b2, 3:7][None], torch.tensor(m.site_pos[s2], dtype=torch.float32, device=pose.device)[None])[0]
  gaps.append(float(torch.linalg.norm(p1 - p2)))
log(f"closure gap at stand, max over 8 sites: {max(gaps):.2e} m")

# ---------------------------------------------------------------- 5. IMU 2 frame at stand
rear = robot.body_names.index(RC.YAW_REAR_BODY)
q_imu2 = quat_mul(pose[rear, 3:7][None], torch.tensor(RC.YAW_IMU2_QUAT_IN_REAR, device=pose.device)[None])[0].cpu().numpy()
dm_ = mj_at_stand(m)
q_mj = dm_.sensordata[m.sensor_adr[m.sensor("imu2_quat").id]:][:4]
dot = abs(float(np.dot(q_imu2 / np.linalg.norm(q_imu2), q_mj / np.linalg.norm(q_mj))))
log(f"IMU 2 orientation at stand: IsaacLab {np.round(q_imu2, 4).tolist()} vs MuJoCo imu2_quat {np.round(q_mj, 4).tolist()} -> angle {math.degrees(2 * math.acos(min(1.0, dot))):.3f} deg")

# ---------------------------------------------------------------- 3. + 4. dynamics in IsaacLab
fixed: Articulation = scene["robot_fixed"]
sj_f = fixed.joint_names.index(RC.SPINE_JOINT_NAME)
sj = robot.joint_names.index(RC.SPINE_JOINT_NAME)
stand_t = torch.tensor([[STAND[n] for n in robot.joint_names]], device=sim.device)
for art in (robot, vis, fixed):
  art.set_joint_position_target(stand_t)
rig: Articulation = scene["rigid"]
rig.set_joint_position_target(torch.tensor([[STAND[n] for n in rig.joint_names]], device=sim.device))
TAU = 0.05  # N m on the spine of the free-floating robot
spine_trace, root_z, pitch, spine_free = [], [], [], []
for k in range(int(3.0 / 0.0025)):
  if k < int(0.12 / 0.0025):
    eff = torch.zeros(1, fixed.num_joints, device=sim.device); eff[0, sj_f] = TAU
    fixed.set_joint_effort_target(eff)
  else:
    fixed.set_joint_effort_target(torch.zeros(1, fixed.num_joints, device=sim.device))
  scene.write_data_to_sim(); sim.step(); scene.update(0.0025)
  spine_trace.append(float(fixed.data.joint_pos[0, sj_f]))
  root_z.append(float(robot.data.root_link_pos_w[0, 2])); spine_free.append(float(robot.data.joint_pos[0, sj]))
  g = robot.data.projected_gravity_b[0]; pitch.append(math.degrees(math.asin(max(-1.0, min(1.0, float(g[0]))))))
tt = np.arange(1, int(0.12 / 0.0025) + 1) * 0.0025
alpha_il = 2 * np.polyfit(tt, np.array(spine_trace[:len(tt)]), 2)[0]


def mj_float_alpha(tau=TAU, T=0.12):
  sp_ = mujoco.MjSpec.from_file(str(XML))
  for name, (lo, hi) in RC.motor_joint_ranges().items():
    j = sp_.joint(name); j.limited = mujoco.mjtLimited.mjLIMITED_TRUE; j.range[:] = (lo, hi)
  mm = sp_.compile(); mm.opt.timestep = 0.0025; mm.opt.gravity[:] = 0
  sa = mm.jnt_dofadr[mm.joint(RC.SPINE_JOINT_NAME).id]; mm.dof_armature[sa] = RC.SPINE_ARMATURE; mm.dof_frictionloss[sa] = 0; mm.dof_damping[sa] = 0
  for n in RC.LEG_JOINT_NAMES:
    a_ = mm.jnt_dofadr[mm.joint(n).id]; mm.dof_armature[a_] = RC.ARMATURE; mm.dof_frictionloss[a_] = RC.FRICTIONLOSS_NM; mm.dof_damping[a_] = 0
  dd = mj_at_stand(mm, z=0.3); sq = mm.joint(RC.SPINE_JOINT_NAME).qposadr[0]; xs = []
  for _ in range(int(T / mm.opt.timestep)):
    dd.qfrc_applied[:] = 0; dd.qfrc_applied[sa] = tau
    for n in RC.LEG_JOINT_NAMES:
      qa, va = mm.joint(n).qposadr[0], mm.jnt_dofadr[mm.joint(n).id]
      dd.qfrc_applied[va] = np.clip(RC.KP * (STAND[n] - dd.qpos[qa]) - RC.KD * dd.qvel[va], -RC.EFFORT_LIMIT_NM, RC.EFFORT_LIMIT_NM)
    mujoco.mj_step(mm, dd); xs.append(dd.qpos[sq])
  return 2 * np.polyfit(np.arange(1, len(xs) + 1) * mm.opt.timestep, np.array(xs), 2)[0]


alpha_mj = mj_float_alpha()
log(f"spine actuator, free floating, gravity off, {TAU} N m: IsaacLab alpha {alpha_il:.3f} rad/s^2 (J_eff {TAU / alpha_il:.5f}), "
    f"MuJoCo {alpha_mj:.3f} (J_eff {TAU / alpha_mj:.5f}) -> diff {100 * (alpha_il / alpha_mj - 1):+.2f} %")
log("   note: free floating the link adds only ~3e-4 to the armature; with the front half held it adds 2.4e-3. The armature that"
    " reproduces the measured 0.0163 therefore depends on how the robot was suspended in the 2026-10-28 test (front held: 0.0139).")
rg = rig.data.projected_gravity_b[0]
rig_z, rig_p = float(rig.data.root_link_pos_w[0, 2]), math.degrees(math.asin(max(-1.0, min(1.0, float(rg[0])))))
log(f"IsaacLab standing 3 s: root z {root_z[-1] * 1e3:.2f} mm (spawn {YAW_NOMINAL_ROOT_HEIGHT_M * 1e3:.2f}), pitch {pitch[-1]:+.2f} deg (+ = nose down), "
    f"spine {spine_free[-1]:+.4f} rad")


# ---------------------------------------------------------------- 4. MuJoCo standing with the same motor model
def mj_stand(seconds=3.0, xml=XML, root_body=RC.YAW_ROOT_BODY, z0=YAW_NOMINAL_ROOT_HEIGHT_M):
  spec = mujoco.MjSpec.from_file(str(xml))
  for name, (lo, hi) in RC.motor_joint_ranges().items():
    j = spec.joint(name); j.limited = mujoco.mjtLimited.mjLIMITED_TRUE; j.range[:] = (lo, hi)
  spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(1, 1, 0.1), friction=(1.0, 0.005, 0.0001))
  mm = spec.compile()
  mm.opt.timestep = 0.0025
  for n in RC.LEG_JOINT_NAMES:
    a = mm.jnt_dofadr[mm.joint(n).id]; mm.dof_armature[a] = RC.ARMATURE; mm.dof_frictionloss[a] = RC.FRICTIONLOSS_NM; mm.dof_damping[a] = 0.0
  spine = RC.SPINE_JOINT_NAME in [mm.joint(j).name for j in range(mm.njnt)]
  gains = {n: (RC.KP, RC.KD, RC.EFFORT_LIMIT_NM) for n in RC.LEG_JOINT_NAMES}
  if spine:
    a = mm.jnt_dofadr[mm.joint(RC.SPINE_JOINT_NAME).id]
    mm.dof_armature[a] = RC.SPINE_ARMATURE; mm.dof_frictionloss[a] = RC.SPINE_FRICTIONLOSS_NM; mm.dof_damping[a] = 0.0
    gains[RC.SPINE_JOINT_NAME] = (RC.SPINE_KP, RC.SPINE_KD, RC.SPINE_EFFORT_LIMIT_NM)
  dd = mj_at_stand(mm, z=z0)
  idx = {n: (mm.joint(n).qposadr[0], mm.jnt_dofadr[mm.joint(n).id]) for n in gains}
  for _ in range(int(seconds / mm.opt.timestep)):
    dd.qfrc_applied[:] = 0
    for n, (kp, kd, cap) in gains.items():
      qa, va = idx[n]
      dd.qfrc_applied[va] = np.clip(kp * (STAND[n] - dd.qpos[qa]) - kd * dd.qvel[va], -cap, cap)
    mujoco.mj_step(mm, dd)
  root = mm.body(root_body).id
  R = dd.xmat[root].reshape(3, 3)
  return dd.xpos[root][2], math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0])))), (dd.qpos[idx[RC.SPINE_JOINT_NAME][0]] if spine else 0.0)


mz, mp_, ms = mj_stand()
log(f"MuJoCo standing 3 s (same PD, armature, friction, caps): root z {mz * 1e3:.2f} mm, pitch {mp_:+.2f} deg, spine {ms:+.4f} rad")
log(f"-> IsaacLab - MuJoCo: root z {(root_z[-1] - mz) * 1e3:+.2f} mm, pitch {pitch[-1] - mp_:+.2f} deg")
rz, rp, _ = mj_stand(xml=RC.XML_PATH, root_body=RC.ROOT_BODY, z0=NOMINAL_ROOT_HEIGHT_M)
log(f"control, rigid robot, same procedure: IsaacLab root z {rig_z * 1e3:.2f} mm pitch {rig_p:+.2f} deg; MuJoCo {rz * 1e3:.2f} mm {rp:+.2f} deg "
    f"-> difference {(rig_z - rz) * 1e3:+.2f} mm, {rig_p - rp:+.2f} deg")
LOG.close()
os._exit(0)
