"""Checks of the active-yaw training env (variant "yaw": 8 leg motors + the spine, two trunk halves).

  OMNI_KIT_ACCEPT_EULA=YES python tests/check_isaac_yaw_env.py --out <log> --part train|play [--num_envs 32]

(one env per process: building a second yaw env after env.close() hangs in Kit)

1. USD: foot sphere centres in their bodies (= YAW_FOOT_OFFSET_IN_BODY_M?), spine joint axis / limits.
2. Yaw-Teacher-Cur-Student, training cfg, leg_polar on: observation groups / terms / dims, 9 actions,
   joint order, encoder bias [N, 9], the spine actuator as built (gains, effort / velocity limit,
   armature, friction), reward terms (motor_energy: per-joint copper, mass, ramp), terminations;
   50 random steps stay finite.
3. Flat play cfg (no noise / randomisation; student group + leg_polar), 8 envs:
   a. reset + 60 zero-action steps: root height vs YAW_NOMINAL_ROOT_HEIGHT_M, both IMUs' gravity,
      IMU 2 frame vs the root frame, spine angle, virtual legs, terminations, closure gap;
   b. spine step: action 8 = +0.5 (target +0.175 rad) for 40 steps, then -0.5 for 40, legs at 0:
      spine angle against its target, rear - front gyro z against the spine velocity (sign, scale);
   c. 150 random steps (std 0.5): terminations, finite state, closure gap, spine torque cap,
      mechanical / copper power per motor; during a-c the five-bar FK on the yaw legs: foot-sphere
      centre relative to the motor-pivot midpoint, in the frame of the half carrying the leg, against
      FK(x, z) of the simulated motor angles.
Output goes to --out (Kit swallows stdout).
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
ap.add_argument("--num_envs", type=int, default=32)
ap.add_argument("--part", required=True, choices=("train", "play"), help="train = sections 1-2, play = section 3")
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
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

from microtaur_common.kinematics import five_bar_for  # noqa: E402
from microtaur_common.robot_constants import (  # noqa: E402
  LEG_JOINT_NAMES, SPINE_ARMATURE, SPINE_EFFORT_LIMIT_NM, SPINE_FRICTIONLOSS_NM, SPINE_JOINT_NAME, SPINE_KD, SPINE_KP,
  SPINE_VELOCITY_LIMIT_RAD_S, YAW_IMU2_QUAT_IN_REAR, YAW_REAR_BODY, YAW_ROOT_BODY, YAW_XML_PATH,
)
from microtaur_isaac import FOOT_BODY_NAMES, YAW_FOOT_OFFSET_IN_BODY_M  # noqa: E402
from microtaur_isaac import env_cfg as EC  # noqa: E402
from microtaur_isaac.robot import YAW_NOMINAL_ROOT_HEIGHT_M, YAW_USD_PATH  # noqa: E402
from microtaur_isaac.usd import compile_model  # noqa: E402

M = compile_model(YAW_XML_PATH)
LOOPS = [(M.body(int(M.site_bodyid[int(M.eq_obj1id[i])])).name, M.site_pos[int(M.eq_obj1id[i])],
          M.body(int(M.site_bodyid[int(M.eq_obj2id[i])])).name, M.site_pos[int(M.eq_obj2id[i])]) for i in range(M.neq)]
# motor pivots: the a / e joints sit at the origin of their child body (checked below)
PIVOTS = []
for leg in range(1, 5):
  pair = []
  for j in ("a", "e"):
    jid = M.joint(f"leg{leg}_{j}_joint_act").id
    assert np.allclose(M.jnt_pos[jid], 0.0), f"leg{leg}_{j}: joint not at its body origin"
    pair.append(M.body(int(M.jnt_bodyid[jid])).name)
  PIVOTS.append(pair)
K = five_bar_for(True)  # the yaw MJCF's foot angle (5.41 deg off D-C, rigid 4.32)
ok = True


def check(cond: bool, what: str):
  global ok
  ok &= bool(cond)
  log(f"  [{'ok' if cond else 'FAIL'}] {what}")


def closure_gap(robot) -> float:
  pose = robot.data.body_link_pose_w
  worst = 0.0
  for b0, p0, b1, p1 in LOOPS:
    i0, i1 = robot.body_names.index(b0), robot.body_names.index(b1)
    t0 = torch.tensor(p0, dtype=torch.float32, device=pose.device).expand(pose.shape[0], 3)
    t1 = torch.tensor(p1, dtype=torch.float32, device=pose.device).expand(pose.shape[0], 3)
    s0 = pose[:, i0, :3] + quat_apply(pose[:, i0, 3:7], t0)
    s1 = pose[:, i1, :3] + quat_apply(pose[:, i1, 3:7], t1)
    worst = max(worst, float(torch.linalg.norm(s0 - s1, dim=1).max()))
  return worst


def term_slices(env, group):
  om = env.observation_manager
  out, i = {}, 0
  for n, dims in zip(om.active_terms[group], om.group_obs_term_dim[group]):
    w = int(np.prod(dims)); out[n] = slice(i, i + w); i += w
  return out


# ---------------------------------------------------------------- 1. USD
log("== 1. USD ==")
stage = Usd.Stage.Open(str(YAW_USD_PATH))
for k, b in enumerate(FOOT_BODY_NAMES):
  found = None
  for prim in stage.Traverse():
    if prim.GetName() == b:
      for c in Usd.PrimRange(prim):
        if c.IsA(UsdGeom.Sphere):
          found = np.array(UsdGeom.Xformable(c).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()) \
            - np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation())
          body_rot = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractRotationMatrix()
          found = np.array(body_rot) @ found  # USD row-vector convention: local = world_delta * R^T -> R @ delta
          break
      break
  err = None if found is None else float(np.abs(found - np.array(YAW_FOOT_OFFSET_IN_BODY_M[k])).max())
  check(err is not None and err < 1e-6, f"leg {k + 1} foot sphere in {b}: USD {None if found is None else np.round(found, 6).tolist()} "
        f"vs YAW_FOOT_OFFSET_IN_BODY_M {YAW_FOOT_OFFSET_IN_BODY_M[k]} (max err {err})")
sj = M.joint(SPINE_JOINT_NAME).id
log(f"  MJCF spine: child {M.body(int(M.jnt_bodyid[sj])).name}, axis {M.jnt_axis[sj].tolist()}, range {M.jnt_range[sj].tolist()}")

# ---------------------------------------------------------------- 2. training cfg
def part_train():
  log("\n== 2. Yaw-Teacher-Cur-Student (training cfg, leg_polar on) ==")
  cfg = EC.MicrotaurYawTeacherCurStudentEnvCfg(leg_polar_obs=True)
  cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRLEnv(cfg)
  robot = env.scene["robot"]
  om = env.observation_manager
  for g in om.active_terms:
    log(f"  obs {g:8s} {tuple(om.group_obs_dim[g])}: " + ", ".join(f"{n} {int(np.prod(d))}" for n, d in zip(om.active_terms[g], om.group_obs_term_dim[g])))
  st = term_slices(env, "student")
  want = {"projected_gravity": 3, "base_ang_vel": 3, "imu2_projected_gravity": 3, "imu2_base_ang_vel": 3, "joint_pos": 9,
          "joint_vel": 9, "foot_fk": 8, "actions": 9, "command": 3, "leg_polar": 8}
  hist = int(getattr(cfg.observations.student, "history_length", 0) or 1)
  check(list(st) == list(want) and all(st[n].stop - st[n].start == w * hist for n, w in want.items()),
        f"student terms = {list(want)} x history {hist} (50 + 8 leg_polar = 58 per frame)")
  check(env.action_manager.total_action_dim == 9, f"action dim {env.action_manager.total_action_dim}")
  mot, _ = robot.find_joints(list(LEG_JOINT_NAMES) + [SPINE_JOINT_NAME], preserve_order=True)
  log(f"  joints (sim order) {robot.joint_names}; canonical motor ids {mot}")
  bias = getattr(env, "microtaur_encoder_bias", None)
  check(bias is not None and tuple(bias.shape) == (env.num_envs, 9),
        f"encoder bias {None if bias is None else tuple(bias.shape)}, |max| per motor {None if bias is None else np.round(bias.abs().amax(0).cpu().numpy(), 4).tolist()}")
  act = robot.actuators["spine"]
  sid = robot.joint_names.index(SPINE_JOINT_NAME)
  vals = {"stiffness": (float(act.stiffness[0, 0]), SPINE_KP), "damping": (float(act.damping[0, 0]), SPINE_KD),
          "effort_limit": (float(act.effort_limit[0, 0]), SPINE_EFFORT_LIMIT_NM),
          "velocity_limit": (float(act.velocity_limit[0, 0]), SPINE_VELOCITY_LIMIT_RAD_S),
          "armature (data)": (float(robot.data.joint_armature[0, sid]), SPINE_ARMATURE),
          "friction (data)": (float(getattr(robot.data, "joint_friction_coeff", torch.full_like(robot.data.joint_pos, float("nan")))[0, sid]),
                              SPINE_FRICTIONLOSS_NM)}
  log(f"  spine actuator {type(act).__name__} joints {act.joint_names}, sat effort {float(getattr(act.cfg, 'saturation_effort', float('nan')) or float('nan')):.3f}")
  for k, (v, w) in vals.items():
    check(abs(v - w) <= 1e-4 * max(1.0, abs(w)), f"spine {k}: {v:.5f} (want {w:.5f})")
  log(f"  spine joint limits {robot.data.joint_pos_limits[0, sid].tolist()}; leg actuator effort {float(robot.actuators[[k for k in robot.actuators if k != 'spine'][0]].effort_limit[0, 0]):.4f}")
  rm = env.reward_manager
  log("  rewards: " + ", ".join(f"{n} {rm.get_term_cfg(n).weight:g}" for n in rm.active_terms))
  me = rm.get_term_cfg("motor_energy").params
  log(f"  motor_energy: copper {me['copper_w_per_nm2']}, mass {me['mass_kg']}, joints {me['asset_cfg'].joint_names}")
  check(len(me["copper_w_per_nm2"]) == 9 and me["asset_cfg"].joint_names[-1] == SPINE_JOINT_NAME, "motor_energy over 9 motors")
  stages = cfg.curriculum.energy_weight.params["stages"]
  check(abs(stages[-1]["weight"] + 0.005) < 1e-9, f"energy ramp: {stages[0]} -> {stages[1]} ... {stages[-1]} (steps / 32 = iterations)")
  tm = env.termination_manager
  log("  terminations: " + ", ".join(tm.active_terms) + f"; illegal_contact bodies {tm.get_term_cfg('illegal_contact').params['sensor_cfg'].body_names}, "
      f"base_too_low {tm.get_term_cfg('base_too_low').params.get('min_height_m')}")
  obs, _ = env.reset()
  fin, falls = True, 0
  g = torch.Generator(device=env.device).manual_seed(0)
  for t in range(50):
    obs, r, te, tr, _ = env.step(0.5 * torch.randn(env.num_envs, 9, device=env.device, generator=g))
    fin &= all(bool(torch.isfinite(v).all()) for v in obs.values()) and bool(torch.isfinite(r).all())
    falls += int(te.sum())
  check(fin, f"50 random steps: obs / reward finite; terminations {falls}")


# ---------------------------------------------------------------- 3. flat play
def part_play():
  log("\n== 3. flat play cfg (yaw, student + leg_polar), 8 envs ==")
  cfg = EC.MicrotaurFlatEnvCfg(play=True, variant="yaw", student_obs=True, leg_polar_obs=True)
  cfg.scene.num_envs = 8
  cfg.commands.twist.resampling_time_range = (1e9, 1e9)
  env = ManagerBasedRLEnv(cfg)
  robot = env.scene["robot"]
  d = robot.data
  st = term_slices(env, "student")
  term = env.action_manager.get_term("joint_pos")
  mot, _ = robot.find_joints(list(LEG_JOINT_NAMES) + [SPINE_JOINT_NAME], preserve_order=True)
  sid = robot.joint_names.index(SPINE_JOINT_NAME)
  rid, rear = robot.body_names.index(YAW_ROOT_BODY), robot.body_names.index(YAW_REAR_BODY)
  fids = [robot.body_names.index(b) for b in FOOT_BODY_NAMES]
  pids = [[robot.body_names.index(b) for b in p] for p in PIVOTS]
  foff = torch.tensor(YAW_FOOT_OFFSET_IN_BODY_M, device=env.device)
  q2 = torch.tensor(YAW_IMU2_QUAT_IN_REAR, device=env.device)
  N = env.num_envs
  FK_ERR, FK_Y = [], []


  def fk_err():
    """[N, 4, 3] foot - pivot midpoint in the carrying half's frame, minus (FK x, ., FK z); y kept apart."""
    fp = d.body_link_pos_w[:, fids] + quat_apply(d.body_link_quat_w[:, fids].reshape(-1, 4), foff.repeat(N, 1)).reshape(N, 4, 3)
    mid = torch.stack([0.5 * (d.body_link_pos_w[:, a] + d.body_link_pos_w[:, e]) for a, e in pids], dim=1)
    qf = d.body_link_quat_w[:, rid]
    qr = quat_mul(d.body_link_quat_w[:, rear], q2.expand(N, 4))
    frames = torch.stack((qr, qr, qf, qf), dim=1)  # legs 1, 2 on the rear half; 3, 4 on the front (root)
    rel = quat_apply_inverse(frames.reshape(-1, 4), (fp - mid).reshape(-1, 3)).reshape(N, 4, 3)
    q = d.joint_pos[:, mot[:8]].double()
    fx, fz = K.forward_torch(q[:, 0::2], q[:, 1::2])
    FK_ERR.append(torch.stack((rel[..., 0] - fx.float(), rel[..., 2] - fz.float()), dim=-1).cpu().numpy())
    FK_Y.append(rel[..., 1].cpu().numpy())


  obs, _ = env.reset()
  z0 = float((d.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).mean())
  log(f"  after reset: root z {1e3 * z0:.2f} mm (YAW_NOMINAL_ROOT_HEIGHT_M {1e3 * YAW_NOMINAL_ROOT_HEIGHT_M:.3f}), spine q {float(d.joint_pos[:, sid].abs().max()):.2e}")
  # IMU 2 frame relative to the root frame (identity at spine 0 if YAW_IMU2_QUAT_IN_REAR is right)
  qr = quat_mul(d.body_link_quat_w[:, rear], q2.expand(N, 4))
  axes = quat_apply_inverse(d.body_link_quat_w[:, rid].repeat_interleave(3, 0), quat_apply(qr.repeat_interleave(3, 0), torch.eye(3, device=env.device).repeat(N, 1))).reshape(N, 3, 3)
  check(float((axes - torch.eye(3, device=env.device)).abs().max()) < 1e-3, f"IMU 2 frame axes in the root frame at spine 0 (rows x, y, z): {np.round(axes[0].cpu().numpy(), 4).tolist()}")
  gaps, falls = [], 0
  for t in range(60):
    obs, _, te, tr, _ = env.step(torch.zeros(N, 9, device=env.device))
    falls += int(te.sum()); gaps.append(closure_gap(robot)); fk_err()
  s = obs["student"]
  z = float((d.root_pos_w[:, 2] - env.scene.env_origins[:, 2]).mean())
  g1, g2 = s[:, st["projected_gravity"]].mean(0), s[:, st["imu2_projected_gravity"]].mean(0)
  lp = s[:, st["leg_polar"]].reshape(N, 4, 2).mean(0)
  log(f"  60 zero-action steps: root z {1e3 * z:.2f} mm, spine q {float(d.joint_pos[:, sid].mean()):+.4f} rad, terminations {falls}, max closure gap {1e3 * max(gaps):.3f} mm")
  # Not exactly (0, 0, -1): the rear hips sit 5.2 mm higher than the front ones (YAW_HIP_Z_ROOT_M), so with
  # equal legs the body stands nose-up by about atan(5.2 mm / hip spacing).
  pitch = np.degrees(np.arcsin(-float(g1[0])))
  check(float((g1 - g2).abs().max()) < 2e-3 and abs(pitch) < 3.0 and abs(float(g1[1])) < 0.01,
        f"gravity IMU 1 {np.round(g1.cpu().numpy(), 4).tolist()}, IMU 2 {np.round(g2.cpu().numpy(), 4).tolist()} (nose-up {pitch:+.2f} deg)")
  log(f"  virtual legs (r mm, theta deg) 1..4: " + ", ".join(f"({1e3 * float(r):.2f}, {np.degrees(float(th)):+.2f})" for r, th in lp))
  check(falls == 0 and max(gaps) < 1e-3, "standing: no termination, closure < 1 mm")

  # spine step
  qs, tg, dw, qd, yaw0 = [], [], [], [], None
  quat_yaw = lambda q: torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))  # noqa: E731
  yaw0 = quat_yaw(d.root_quat_w).clone()
  for t in range(80):
    a = torch.zeros(N, 9, device=env.device)
    a[:, 8] = 0.5 if t < 40 else -0.5
    obs, _, te, tr, _ = env.step(a)
    falls += int(te.sum()); fk_err()
    s = obs["student"]
    qs.append(float(d.joint_pos[:, sid].mean())); tg.append(float(term._processed[:, 8].mean()))
    dw.append((s[:, st["imu2_base_ang_vel"]][:, 2] - s[:, st["base_ang_vel"]][:, 2]).cpu().numpy())
    qd.append(d.joint_vel[:, sid].cpu().numpy())
  qs, tg, dw, qd = np.array(qs), np.array(tg), np.concatenate(dw), np.concatenate(qd)
  slope = float(np.dot(dw, qd) / max(np.dot(qd, qd), 1e-12))
  res = float(np.sqrt(np.mean((dw - slope * qd) ** 2)))
  log(f"  spine step +0.5 -> target {tg[5]:+.4f} rad: q after 1/3/10/39 steps {qs[0]:+.4f} {qs[2]:+.4f} {qs[9]:+.4f} {qs[39]:+.4f}; "
      f"-0.5 -> {tg[45]:+.4f}: q after 10/39 steps {qs[49]:+.4f} {qs[79]:+.4f}; front-half yaw moved {np.degrees(float((quat_yaw(d.root_quat_w) - yaw0).abs().max())):.2f} deg")
  check(abs(qs[39] - tg[39]) < 0.03 and abs(qs[79] - tg[79]) < 0.03, "spine settles within 0.03 rad of its target on the ground")
  check(abs(abs(slope) - 1.0) < 0.02 and res < 0.05, f"gyro z (rear - front) = {slope:+.4f} x spine qd (residual {res:.4f} rad/s): spine axis = "
        f"{'+' if slope > 0 else '-'}z of the body")
  log(f"  terminations so far {falls}")

  # random actions
  gaps, tau_s, P = [], [], []
  g = torch.Generator(device=env.device).manual_seed(1)
  fin = True
  for t in range(150):
    obs, _, te, tr, _ = env.step(0.5 * torch.randn(N, 9, device=env.device, generator=g))
    falls += int(te.sum()); gaps.append(closure_gap(robot)); fk_err()
    fin &= bool(torch.isfinite(d.root_state_w).all())
    tau = d.applied_torque[:, mot]; w = d.joint_vel[:, mot]
    tau_s.append(float(tau[:, 8].abs().max()))
    P.append(torch.stack(((tau * w).abs().mean(0), tau.square().mean(0)), dim=0).cpu().numpy())
  P = np.mean(P, axis=0)
  cu = np.array(env.reward_manager.get_term_cfg("motor_energy").params["copper_w_per_nm2"], dtype=float)
  log(f"  150 random steps: terminations {falls}, max closure gap {1e3 * max(gaps):.3f} mm, max |spine torque| {max(tau_s):.3f} N m")
  log(f"  power per motor (W): mech {np.round(P[0], 3).tolist()}; copper {np.round(cu * P[1], 3).tolist()}; spine share of copper {cu[8] * P[1][8] / (cu * P[1]).sum():.1%}")
  check(fin and max(tau_s) <= SPINE_EFFORT_LIMIT_NM + 1e-4, "finite state; spine torque within its 0.50 N m cap")
  E, Y = np.concatenate(FK_ERR), np.concatenate(FK_Y)
  for leg in range(4):
    e = np.abs(E[:, leg]) * 1e3
    log(f"  FK leg {leg + 1}: |x err| mean {e[..., 0].mean():.3f} max {e[..., 0].max():.3f} mm, |z err| mean {e[..., 1].mean():.3f} max {e[..., 1].max():.3f} mm, "
        f"lateral y {1e3 * Y[:, leg].mean():+.2f} +- {1e3 * Y[:, leg].std():.3f} mm")
  check(float(np.abs(E).max()) < 2e-3, f"five-bar FK matches the simulated yaw legs (max err {1e3 * float(np.abs(E).max()):.3f} mm over {len(E)} steps x {N} envs)")


part_train() if args.part == "train" else part_play()
log("\nALL OK" if ok else "\nFAILED")
LOG.close()
os._exit(0)
