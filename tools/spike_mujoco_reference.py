"""MuJoCo reference numbers for the rigid Microtaur closed-chain spike.

Loads rigid_microtaur/scene.xml with pure `mujoco` (no mjlab), applies the
config-time MjSpec patches that the training code applies (see
docs/ISAACLAB_GAPS_rigid.md section 1), puts the robot in the IK-consistent
stand pose, holds the 8 motors with the explicit DC-motor PD used in training,
and lets it settle. Prints masses/inertias, settled root height and pose, the 16
hinge angles, the 8 closing-site gaps, and the equality-constraint violation.

Usage:
  python tools/spike_mujoco_reference.py [--seconds 20] [--kd 0.045]
      [--timestep 0.005 --iterations 10 --ls-iterations 20] [--json out.json]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
MAIN = REPO / "Microtaur_RL-main"
XML_DIR = MAIN / "microtaur_xmls" / "rigid_microtaur"


def _load_kinematics():
  path = MAIN / "src" / "microtaur_velocity" / "microtaur_kinematics.py"
  spec = importlib.util.spec_from_file_location("microtaur_kinematics", path)
  mod = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = mod
  spec.loader.exec_module(mod)
  return mod


KIN = _load_kinematics()

# Values from Microtaur_RL-main/src/microtaur_velocity/env_cfgs.py:150-254 and
# microtaur_constants.py:107-121 (env-var defaults).
STAND_A = (0.45, -0.45, -0.45, 0.45)
STAND_E = (-0.45, 0.45, 0.45, -0.45)
HALF_RANGE = 0.75
ACT_ARMATURE = 2e-4
ACT_FRICTIONLOSS = 0.010
PAS_ARMATURE = 1e-6
PAS_FRICTIONLOSS = 0.001
PAS_DAMPING = 1e-4
KP = 1.0
EFFORT_LIMIT = 0.60 * 0.215  # 0.129 N*m
SATURATION_EFFORT = 0.215
VELOCITY_LIMIT = 0.80 * 40.11  # rad/s, approx 32.09

# Root mass patch: only documented in docs/ISAACLAB_GAPS_rigid.md section 1;
# no code in any branch applies it. It makes total mass exactly 0.540 kg.
ROOT_MASS = 0.4662146611
ROOT_IPOS = (-8.161907e-4, -7.185340e-4, 1.301072e-2)

MOTORS = KIN.MOTOR_JOINT_NAMES
FULL = KIN.FULL_JOINT_NAMES
PASSIVE = tuple(n for n in FULL if n not in MOTORS)


def build_model(args) -> mujoco.MjModel:
  spec = mujoco.MjSpec.from_file(str(XML_DIR / "scene.xml"))
  for leg, (sa, se) in enumerate(zip(STAND_A, STAND_E), start=1):
    for name, stand in ((f"leg{leg}_a_joint_act", sa), (f"leg{leg}_e_joint_act", se)):
      j = spec.joint(name)
      j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
      j.range[:] = (stand - HALF_RANGE, stand + HALF_RANGE)
      j.armature = ACT_ARMATURE
      j.frictionloss = ACT_FRICTIONLOSS
  for name in PASSIVE:
    j = spec.joint(name)
    j.armature = PAS_ARMATURE
    j.frictionloss = PAS_FRICTIONLOSS
    if np.ndim(j.damping) == 0:  # scalar in older mujoco, array in newer
      j.damping = PAS_DAMPING
    else:
      j.damping[:] = 0.0
      j.damping[0] = PAS_DAMPING
  if not args.no_mass_patch:
    b = spec.body("battery")
    # Read the unpatched values from a compiled copy.
    m0 = spec.compile()
    bid = m0.body("battery").id
    ratio = ROOT_MASS / float(m0.body_mass[bid])
    b.explicitinertial = True
    b.mass = ROOT_MASS
    b.ipos[:] = ROOT_IPOS
    b.iquat[:] = m0.body_iquat[bid]
    b.inertia[:] = m0.body_inertia[bid] * ratio
    b.fullinertia[:] = np.nan
  m = spec.compile()
  if args.no_frictionloss:
    m.dof_frictionloss[:] = 0.0
  m.opt.timestep = args.timestep
  m.opt.iterations = args.iterations
  m.opt.ls_iterations = args.ls_iterations
  m.opt.ccd_iterations = 50
  return m


def stand_state(m: mujoco.MjModel, d: mujoco.MjData) -> tuple[np.ndarray, float]:
  full_q = np.stack(
    [KIN.MicrotaurFiveBarKinematics().forward_numpy(STAND_A[i], STAND_E[i], leg_index=i + 1).full for i in range(4)]
  ).reshape(16)
  feet = np.stack(
    [KIN.MicrotaurFiveBarKinematics().forward_numpy(STAND_A[i], STAND_E[i], leg_index=i + 1).foot for i in range(4)]
  )
  nominal_root_z = float(KIN.FOOT_SPHERE_RADIUS_M - np.min(KIN.HIP_MIDPOINTS_ROOT_M[:, 2] + feet[:, 1]))
  mujoco.mj_resetData(m, d)
  d.qpos[0:3] = (0.0, 0.0, nominal_root_z)
  d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
  for name, q in zip(FULL, full_q):
    d.qpos[m.joint(name).qposadr[0]] = q
  mujoco.mj_forward(m, d)
  return full_q, nominal_root_z


def closing_pairs(m: mujoco.MjModel) -> list[tuple[str, int, int]]:
  out = []
  for i in range(m.neq):
    assert m.eq_type[i] == mujoco.mjtEq.mjEQ_CONNECT and m.eq_objtype[i] == mujoco.mjtObj.mjOBJ_SITE
    s1, s2 = int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
    out.append((m.site(s1).name, s1, s2))
  return out


def export_model(m: mujoco.MjModel, path: str) -> None:
  """Dump the patched, compiled model (at the IK stand pose) for the USD builder."""
  d = mujoco.MjData(m)
  full_q, nominal_root_z = stand_state(m, d)
  d.qpos[0:3] = 0.0  # author the USD with the root link at the origin
  mujoco.mj_kinematics(m, d)
  bodies = []
  for b in range(1, m.nbody):
    bodies.append({
      "name": m.body(b).name,
      "parent": m.body(m.body_parentid[b]).name,
      "world_pos": d.xpos[b].tolist(), "world_quat": d.xquat[b].tolist(),
      "pos": m.body_pos[b].tolist(), "quat": m.body_quat[b].tolist(),
      "mass": float(m.body_mass[b]), "ipos": m.body_ipos[b].tolist(),
      "iquat": m.body_iquat[b].tolist(), "inertia": m.body_inertia[b].tolist(),
    })
  joints = []
  for j in range(m.njnt):
    if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
      continue
    assert np.allclose(m.jnt_pos[j], 0.0)
    joints.append({
      "name": m.joint(j).name, "child": m.body(m.jnt_bodyid[j]).name,
      "parent": m.body(m.body_parentid[m.jnt_bodyid[j]]).name,
      "axis": m.jnt_axis[j].tolist(), "limited": bool(m.jnt_limited[j]),
      "range": m.jnt_range[j].tolist(),
    })
  geoms = []
  for g in range(m.ngeom):
    if m.geom_bodyid[g] == 0 or not (m.geom_contype[g] or m.geom_conaffinity[g]):
      continue
    geoms.append({
      "name": m.geom(g).name, "body": m.body(m.geom_bodyid[g]).name,
      "type": {int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere", int(mujoco.mjtGeom.mjGEOM_BOX): "box"}[int(m.geom_type[g])],
      "size": m.geom_size[g].tolist(), "pos": m.geom_pos[g].tolist(), "quat": m.geom_quat[g].tolist(),
      "friction": float(m.geom_friction[g][0]),
    })
  loops = []
  for name, s1, s2 in closing_pairs(m):
    loops.append({
      "name": name,
      "body0": m.body(m.site_bodyid[s1]).name, "pos0": m.site_pos[s1].tolist(),
      "body1": m.body(m.site_bodyid[s2]).name, "pos1": m.site_pos[s2].tolist(),
    })
  Path(path).write_text(json.dumps({
    "bodies": bodies, "joints": joints, "geoms": geoms, "loops": loops,
    "stand_q": dict(zip(FULL, full_q.tolist())), "nominal_root_z": nominal_root_z,
    "total_mass": float(m.body_mass.sum()),
  }, indent=1))
  print(f"exported {len(bodies)} bodies, {len(joints)} hinges, {len(geoms)} collision geoms, {len(loops)} loops -> {path}")


SIGNS = (1.0, -1.0, -1.0, 1.0)  # MINITAUR_SWING_SIGNS
TROT_COMMON, TROT_DIFF, TROT_HZ = 0.25, 0.15, 2.0


def motion_delta(t: float, motion: str) -> np.ndarray:
  """Motor-target offset from stand, ordered as MOTORS (leg1 a, leg1 e, ...).

  trot: diagonal pairs (1,3) and (2,4) in antiphase, using the training code's
  paired map da = sign*(common - diff), de = sign*(common + diff).
  """
  out = np.zeros(8)
  if motion != "trot":
    return out
  for leg in range(4):
    ph = 2 * math.pi * TROT_HZ * t + (0.0 if leg in (0, 2) else math.pi)
    common = TROT_COMMON * math.sin(ph)
    diff = TROT_DIFF * math.cos(ph)
    out[2 * leg] = SIGNS[leg] * (common - diff)
    out[2 * leg + 1] = SIGNS[leg] * (common + diff)
  return out


def dc_motor_torque(err, qd, kd):
  tau = KP * err - kd * qd
  tmax = np.clip(SATURATION_EFFORT * (1.0 - qd / VELOCITY_LIMIT), 0.0, EFFORT_LIMIT)
  tmin = np.clip(SATURATION_EFFORT * (-1.0 - qd / VELOCITY_LIMIT), -EFFORT_LIMIT, 0.0)
  return np.clip(tau, tmin, tmax)


def hf_rms(x: np.ndarray, dt: float, window_s: float = 0.1) -> np.ndarray:
  """RMS of x minus its centred moving average (window 0.1 s -> content >~10 Hz)."""
  w = max(3, int(round(window_s / dt)) | 1)
  k = np.ones(w) / w
  lo = np.stack([np.convolve(x[:, j], k, mode="same") for j in range(x.shape[1])], axis=1)
  h = (x - lo)[w:-w]
  return np.sqrt(np.mean(h**2, axis=0))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--seconds", type=float, default=20.0)
  ap.add_argument("--kd", type=float, default=0.045)
  ap.add_argument("--timestep", type=float, default=0.005)
  ap.add_argument("--iterations", type=int, default=10)
  ap.add_argument("--ls-iterations", type=int, default=20)
  ap.add_argument("--no-mass-patch", action="store_true")
  ap.add_argument("--no-frictionloss", action="store_true")
  ap.add_argument("--motion", choices=("stand", "trot", "drop"), default="stand")
  ap.add_argument("--drop-height", type=float, default=0.010)
  ap.add_argument("--replay", type=str, default="", help="npz from cpg_rl/openloop_spine_cpg.py; replays s_target open-loop")
  ap.add_argument("--replay-robot", type=int, default=0)
  ap.add_argument("--json", type=str, default="")
  ap.add_argument("--export-model", type=str, default="")
  args = ap.parse_args()

  m = build_model(args)
  if args.export_model:
    export_model(m, args.export_model)
    return
  d = mujoco.MjData(m)
  full_q, nominal_root_z = stand_state(m, d)
  pairs = closing_pairs(m)
  motor_qadr = np.array([m.joint(n).qposadr[0] for n in MOTORS])
  motor_dadr = np.array([m.joint(n).dofadr[0] for n in MOTORS])
  full_qadr = np.array([m.joint(n).qposadr[0] for n in FULL])
  target = np.array([full_q[FULL.index(n)] for n in MOTORS])
  if args.motion == "drop":
    d.qpos[2] += args.drop_height
    mujoco.mj_forward(m, d)
  replay = None
  if args.replay:
    Z = np.load(args.replay)
    assert tuple(Z["motor_names"][:8]) == MOTORS
    i = args.replay_robot
    replay = Z["s_target"][:, i, :8]
    kin = KIN.MicrotaurFiveBarKinematics()
    for leg in range(4):
      full = kin.forward_numpy(Z["s_q"][0, i, 2 * leg], Z["s_q"][0, i, 2 * leg + 1], leg_index=leg + 1).full
      for jn, q in zip(FULL[4 * leg:4 * leg + 4], full):
        d.qpos[m.joint(jn).qposadr[0]] = q
    gx, gy, gz = Z["s_imu_gravity"][0, i]
    pitch, roll = math.asin(max(-1.0, min(1.0, float(gx)))), math.atan2(-float(gy), -float(gz))
    cr, sr, cp, sp = math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    d.qpos[0:3] = (0.0, 0.0, float(Z["s_truth_pos"][0, i, 2]))
    d.qpos[3:7] = (cr * cp, sr * cp, cr * sp, -sr * sp)
    d.qvel[0:3] = Z["s_truth_vel_b"][0, i]  # MuJoCo free-joint linear velocity is world frame; yaw = 0 so ~body
    d.qvel[3:6] = Z["s_imu_gyro"][0, i]     # free-joint angular velocity is body frame
    mujoco.mj_forward(m, d)
    args.seconds = replay.shape[0] * float(Z["dt"])
    rec_speed = float(Z["m_forward_m_s"][i])
    print(f"replay {args.replay} robot {i}: {replay.shape[0]} frames, recorded forward {rec_speed:.4f} m/s")
  start = d.qpos[0:7].copy()

  print(f"mujoco {mujoco.__version__}  timestep {m.opt.timestep}  iterations {m.opt.iterations}"
        f"  ls_iterations {m.opt.ls_iterations}  cone {m.opt.cone}  solver {m.opt.solver}  kd {args.kd}")
  print(f"nbody {m.nbody}  njnt {m.njnt}  neq {m.neq}  nu {m.nu}")
  print(f"total mass {m.body_mass.sum():.10f} kg")
  print(f"{'body':32s} {'mass kg':>12s} {'Ixx':>11s} {'Iyy':>11s} {'Izz':>11s}   ipos")
  for b in range(1, m.nbody):
    I = m.body_inertia[b]
    print(f"{m.body(b).name:32s} {m.body_mass[b]:12.9f} {I[0]:11.4e} {I[1]:11.4e} {I[2]:11.4e}   {np.round(m.body_ipos[b], 6)}")
  print(f"nominal root z (code formula) {nominal_root_z:.5f} m")

  n = int(round(args.seconds / m.opt.timestep))
  decim = int(round(0.02 / m.opt.timestep))
  gaps = np.zeros((n, len(pairs)))
  q_hist = np.zeros((n, 16))
  root = np.zeros((n, 7))
  eq_viol = np.zeros(n)
  for t in range(n):
    q = d.qpos[motor_qadr]
    qd = d.qvel[motor_dadr]
    d.qfrc_applied[:] = 0.0
    if t % decim == 0:  # targets update at the 50 Hz control rate
      if replay is not None:
        tgt = replay[min(t // decim, replay.shape[0] - 1)]
      else:
        tgt = target + motion_delta(t * m.opt.timestep, args.motion)
    d.qfrc_applied[motor_dadr] = dc_motor_torque(tgt - q, qd, args.kd)
    mujoco.mj_step(m, d)
    for k, (_, s1, s2) in enumerate(pairs):
      gaps[t, k] = np.linalg.norm(d.site_xpos[s1] - d.site_xpos[s2])
    q_hist[t] = d.qpos[full_qadr]
    root[t] = d.qpos[0:7]
    eq_rows = d.efc_type[: d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    eq_viol[t] = np.max(np.abs(d.efc_pos[: d.nefc][eq_rows])) if eq_rows.any() else 0.0
    if not np.all(np.isfinite(d.qpos)):
      print(f"NaN at step {t}")
      break

  # Statistics over the second half (after settling).
  s = slice(n // 2, n)
  g = gaps[s]
  print(f"\nsettled window: {args.seconds/2:.1f}-{args.seconds:.1f} s")
  print(f"root z mean {root[s,2].mean():.5f} m  (drift last-first {root[-1,2]-root[n//2,2]:+.2e} m)"
        f"  xy drift {np.linalg.norm(root[-1,:2]-root[n//2,:2]):.2e} m")
  w, x, y, z = root[-1, 3:7]
  roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
  pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
  print(f"final roll {roll:+.3f} deg  pitch {pitch:+.3f} deg")
  print("hinge angles (settled mean, rad) vs IK stand:")
  for i, nme in enumerate(FULL):
    print(f"  {nme:18s} {q_hist[s, i].mean():+.5f}  IK {full_q[i]:+.5f}  diff {q_hist[s, i].mean()-full_q[i]:+.2e}")
  print("closing-site gaps (settled), metres:")
  for k, (nme, _, _) in enumerate(pairs):
    print(f"  {nme:26s} mean {g[:,k].mean():.3e}  p95 {np.percentile(g[:,k],95):.3e}  max {g[:,k].max():.3e}")
  w0, x0, y0, z0 = start[3:7]
  fwd0 = np.array([1 - 2 * (y0 * y0 + z0 * z0), 2 * (x0 * y0 + w0 * z0), 0.0])
  fwd0 /= np.linalg.norm(fwd0)
  fwd_speed = float(np.dot(root[-1, 0:3] - start[0:3], fwd0) / args.seconds)
  fell = bool((root[:, 2] < 0.05217).any())
  print(f"forward speed along initial heading {fwd_speed:.4f} m/s  fell {fell}")
  print(f"motion {args.motion}  FULL RUN: mean {gaps.mean():.3e}  p95 {np.percentile(gaps,95):.3e}  p99 {np.percentile(gaps,99):.3e}  max {gaps.max():.3e}"
        f"  root z min {root[:,2].min():.5f}")
  print(f"ALL pairs: mean {g.mean():.3e}  p95 {np.percentile(g,95):.3e}  max {g.max():.3e}  (full-run max {gaps.max():.3e})")
  print(f"equality efc_pos |max| settled: mean {eq_viol[s].mean():.3e}  max {eq_viol[s].max():.3e}")
  hf = hf_rms(q_hist[s], m.opt.timestep)
  print(f"joint HF RMS (>~10 Hz): max {hf.max():.3e} rad  mean {hf.mean():.3e} rad")

  # Fore/aft load split from contact normal forces.
  f = np.zeros(6)
  front = back = 0.0
  for i in range(d.ncon):
    c = d.contact[i]
    mujoco.mj_contactForce(m, d, i, f)
    gx = m.geom(c.geom1).name + "|" + m.geom(c.geom2).name
    xpos = c.pos[0] - d.qpos[0]
    if xpos > 0:
      front += abs(f[0])
    else:
      back += abs(f[0])
  tot = front + back
  print(f"contacts {d.ncon}  front(+x) {100*front/tot:.2f}%  rear {100*back/tot:.2f}%  sum Fn {tot:.4f} N  m*g {m.body_mass.sum()*9.81:.4f} N")

  if args.json:
    Path(args.json).write_text(json.dumps({
      "total_mass": float(m.body_mass.sum()),
      "root_z_mean": float(root[s, 2].mean()),
      "gap_mean": float(g.mean()), "gap_p95": float(np.percentile(g, 95)), "gap_max": float(g.max()),
      "hf_rms_max": float(hf.max()), "motion": args.motion, "forward_speed": fwd_speed, "fell": fell,
      "full_gap_mean": float(gaps.mean()), "full_gap_p95": float(np.percentile(gaps, 95)),
      "full_gap_p99": float(np.percentile(gaps, 99)), "full_gap_max": float(gaps.max()), "q_mean": q_hist[s].mean(0).tolist(), "joint_names": list(FULL),
      "stand_full_q": full_q.tolist(), "nominal_root_z": nominal_root_z,
    }, indent=1))


if __name__ == "__main__":
  main()
