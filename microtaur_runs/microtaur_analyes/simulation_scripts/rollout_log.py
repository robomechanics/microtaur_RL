"""Forward rollout + per-timestep logging for a trained Microtaur velocity policy.

Runs a frozen rsl-rl checkpoint in the mjlab play environment (no learning) and
records, at every control step: mechanical cost of transport (positive-work and
absolute-work conventions, plus raw per-joint tau / qd / qdd so any variant is
recomputable), trunk roll/pitch/yaw + position, and instantaneous linear /
angular velocity (world and body frame) alongside the commanded twist.

Works for every morphology variant:

    python scripts/rollout_log.py --variant rigid --checkpoint <path/model_XXXX.pt> \
        --vx 0.08 0.10 0.12 0.14 0.16 0.18 0.20 --yaw 0.0 --seeds 0 1 2 --sim-seconds 18

    python scripts/rollout_log.py --variant pitch --checkpoint variants/pitch/model_XXXX.pt ...
    python scripts/rollout_log.py --variant yaw   --checkpoint variants/yaw/model_XXXX.pt ...
    python scripts/rollout_log.py --variant roll  --checkpoint variants/roll/model_XXXX.pt ...

`--variant` installs that variant's `env_cfgs.py` (from its `*_env.txt`) for the
run and restores the shipped one afterwards (`--keep-env` to leave it in place),
sets `MICROTAUR_VARIANT` (+ `MICROTAUR_CONTROL_SPINE=1` for roll), and asserts the
loaded `ENV_CFG_REVISION` matches what that variant was trained against.

The 9-DoF active-spine variants (pitch / yaw / roll) add a `spine_pos` action
term and a `spine` joint; its tau / qd / qdd and target are logged alongside the
eight leg joints and included in the cost-of-transport sums.

Output: one CSV + meta.json per (vx, yaw, seed) rollout under
`rollouts/<variant>_<timestamp>/`, plus summary.json and quick plots.

Determinism: play mode disables observation corruption and randomised resets, so
a fixed (seed, command) reproduces the CSV byte-for-byte. Seeds still differ via
the startup domain-randomisation events (foot friction, base CoM, encoder bias).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from microtaur_variants import TASK_ID, VARIANTS, use_variant, verify_revision  # noqa: E402

os.environ["MICROTAUR_USE_JOYSTICK_COMMANDS"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK = TASK_ID
G = 9.81
V_EPS = 0.02


# --------------------------------------------------------------------------- #
# math helpers
# --------------------------------------------------------------------------- #
def quat_wxyz_to_rpy(q: np.ndarray) -> tuple[float, float, float]:
  """World-frame intrinsic XYZ Euler angles (roll, pitch, yaw) from (w, x, y, z)."""
  w, x, y, z = (float(v) for v in q)
  roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
  sinp = 2.0 * (w * y - z * x)
  pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
  yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  return roll, pitch, yaw


def short_joint(name: str) -> str:
  return name.replace("_joint_act", "").replace("_joint_passive", "").replace("_joint", "")


# --------------------------------------------------------------------------- #
# constants captured once
# --------------------------------------------------------------------------- #
def total_mass(env_u) -> float:
  sim = getattr(env_u, "sim", None)
  for path in ("mj_model", "model", "_mj_model", "mjw_model"):
    bm = getattr(getattr(sim, path, None), "body_mass", None)
    if bm is None:
      continue
    try:
      if torch.is_tensor(bm):
        arr = bm.detach().cpu().numpy()
      elif hasattr(bm, "numpy"):
        arr = bm.numpy()
      else:
        arr = np.asarray(bm)
      arr = np.asarray(arr, dtype=float).ravel()
    except Exception:  # noqa: BLE001
      continue
    if arr.size:
      return float(arr.sum())
  from microtaur_velocity.microtaur_constants import get_spec

  return float(np.asarray(get_spec().compile().body_mass).sum())


class JointModel:
  """Resolves canonical leg order + optional spine joint against the built env."""

  def __init__(self, env_u):
    from microtaur_velocity.microtaur_constants import LEG_JOINT_NAMES

    self.env_u = env_u
    robot = env_u.scene["robot"]
    self.leg_names = tuple(LEG_JOINT_NAMES)
    self.leg_term = env_u.action_manager.get_term("joint_pos")
    self.to_canon = self.leg_term.resolved_to_canonical
    self.leg_tids = self.leg_term._target_ids

    act_names = list(robot.actuator_names)
    self.leg_cols = [act_names.index(n) for n in self.leg_names]

    self.spine_term = None
    self.spine_name = None
    self.spine_tids = None
    self.spine_col = None
    for tname in env_u.action_manager.active_terms:
      if tname == "joint_pos":
        continue
      t = env_u.action_manager.get_term(tname)
      tgt = list(getattr(t, "_target_names", []) or [])
      if len(tgt) == 1 and "spine" in tgt[0]:
        self.spine_term = t
        self.spine_name = tgt[0]
        self.spine_tids = t._target_ids
        self.spine_col = act_names.index(tgt[0]) if tgt[0] in act_names else None
        break

    self.cols = [short_joint(n) for n in self.leg_names]
    if self.spine_term is not None:
      self.cols.append("spine")

    # Trunk body (== ROOT_BODY): free-base data is `root_link_*`; for spine
    # variants the reward-relevant trunk can be a different body.
    from microtaur_velocity.microtaur_constants import ROOT_BODY

    self.root_body = ROOT_BODY
    try:
      self.trunk_bi = list(robot.body_names).index(ROOT_BODY)
    except (ValueError, Exception):  # noqa: BLE001
      self.trunk_bi = None

  # -- per-step readback ------------------------------------------------------ #
  def joint_block(self, robot):
    q = self.to_canon(robot.data.joint_pos[:, self.leg_tids])[0].cpu().numpy()
    qd = self.to_canon(robot.data.joint_vel[:, self.leg_tids])[0].cpu().numpy()
    qdd = self.to_canon(robot.data.joint_acc[:, self.leg_tids])[0].cpu().numpy()
    tau = robot.data.actuator_force[0, self.leg_cols].cpu().numpy()
    if self.spine_term is not None:
      sq = robot.data.joint_pos[0, self.spine_tids].cpu().numpy()
      sqd = robot.data.joint_vel[0, self.spine_tids].cpu().numpy()
      sqdd = robot.data.joint_acc[0, self.spine_tids].cpu().numpy()
      stau = (robot.data.actuator_force[0, self.spine_col].cpu().numpy().reshape(1)
              if self.spine_col is not None else np.zeros(1))
      q = np.concatenate([q, sq]); qd = np.concatenate([qd, sqd])
      qdd = np.concatenate([qdd, sqdd]); tau = np.concatenate([tau, stau])
    return q, qd, qdd, tau

  def action_block(self, action_vec):
    """Canonical-order raw actions: legs via to_canon, spine appended."""
    a = torch.as_tensor(action_vec).reshape(1, -1)
    leg = self.to_canon(a[:, :8])[0].cpu().numpy()
    if self.spine_term is not None and a.shape[1] > 8:
      return np.concatenate([leg, a[0, 8:9].cpu().numpy()])
    return leg

  def applied_targets(self):
    leg = _term_vec(self.leg_term, "applied_targets", 8)
    if self.spine_term is not None:
      sp = getattr(self.spine_term, "applied_targets", None)
      if sp is None:
        sp = getattr(self.spine_term, "processed_actions", None)
      sp = np.zeros(1) if sp is None else np.asarray(sp[0].detach().cpu().numpy()).reshape(-1)[:1]
      return np.concatenate([leg, sp])
    return leg

  def trunk_pose(self, robot):
    """(pos_w[3], quat_w[4], v_w[3], w_w[3]) of ROOT_BODY, falling back to free base."""
    d = robot.data
    if self.trunk_bi is not None and hasattr(d, "body_link_quat_w"):
      bi = self.trunk_bi
      return (d.body_link_pos_w[0, bi].cpu().numpy(),
              d.body_link_quat_w[0, bi].cpu().numpy(),
              d.body_link_lin_vel_w[0, bi].cpu().numpy(),
              d.body_link_ang_vel_w[0, bi].cpu().numpy())
    return (d.root_link_pos_w[0].cpu().numpy(), d.root_link_quat_w[0].cpu().numpy(),
            d.root_link_lin_vel_w[0].cpu().numpy(), d.root_link_ang_vel_w[0].cpu().numpy())


# --------------------------------------------------------------------------- #
# single rollout
# --------------------------------------------------------------------------- #
def make_env_and_policy(task_id, checkpoint, device, clip_actions, cfg_hook):
  """Clean play env (1 robot, no obs corruption) with the checkpoint's policy.

  `clip_actions` must match training: the rough-terrain checkpoints were trained
  with --agent.clip-actions 1.0, which clamps the policy output before the env
  (and so before the last_action observation).
  """
  from dataclasses import asdict

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.observations["actor"].enable_corruption = False
  cfg_hook(env_cfg)
  agent_cfg = load_rl_cfg(task_id)
  agent_cfg.clip_actions = clip_actions

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=clip_actions)
  runner = load_runner_cls(task_id)(env, asdict(agent_cfg), device=device)
  runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
  return env, agent_cfg, runner.get_inference_policy(device=device)


def run_one(task_id, checkpoint, vx, yaw, seed, sim_seconds, device,
            terrain="flat", terrain_kw=None, weave_drive=False, lane_keep=False,
            raw_action=True, substep=False, clip_actions=None):
  from microtaur_velocity.distill_reliable.mjlab_utils import get_initial_obs, step_env
  from microtaur_velocity.env_cfgs import ENV_CFG_REVISION, set_joystick_twist_command
  from microtaur_terrains import apply_terrain, surface_drop_m

  torch.manual_seed(seed)
  np.random.seed(seed)

  # Always applied: on the rough task even "flat" has to replace the terrain.
  height_drop = surface_drop_m(terrain, **(terrain_kw or {}))
  def hook(env_cfg, _n=terrain, _kw=dict(terrain_kw or {})):  # noqa: E306
    apply_terrain(env_cfg, _n, **_kw)

  env, _agent_cfg, policy = make_env_and_policy(
    task_id, str(checkpoint), device, clip_actions, hook,
  )
  env_u = env.unwrapped
  try:
    env_u.seed(seed)
  except Exception:  # noqa: BLE001
    pass

  robot = env_u.scene["robot"]
  jm = JointModel(env_u)
  mass = total_mass(env_u)
  policy_dt = float(getattr(env_u, "step_dt", 0.02))
  n_steps = int(round(sim_seconds / policy_dt))
  action_dim = int(env_u.action_manager.action.shape[-1])

  try:
    feet = env_u.scene["feet_ground_contact"]
  except Exception:  # noqa: BLE001
    feet = None

  # mjlab expands the foot-geom pattern in model order, so channel i is NOT
  # leg i+1 on every variant. foot_contact_{i} below means leg{i+1}.
  from foot_index import leg_channels
  foot_chan = leg_channels(feet) if feet is not None else list(range(4))

  def contact_arrays():
    if feet is None:
      return np.zeros(4), np.zeros(4)
    found = getattr(feet.data, "found", None)
    force = getattr(feet.data, "force", None)
    f = np.zeros(4) if found is None else found[0].float().cpu().numpy().ravel()
    n = np.zeros(4)
    if force is not None:
      fv = force[0].float().cpu().numpy()
      n = np.linalg.norm(fv, axis=-1) if fv.ndim == 2 else np.abs(fv).ravel()
    f = np.pad(f, (0, max(0, 4 - f.size)))[:4]
    n = np.pad(n, (0, max(0, 4 - n.size)))[:4]
    return f[foot_chan], n[foot_chan]

  # These policies get a velocity command and no lateral-position or heading
  # reference, so heading bias integrates into unbounded lateral drift. Two
  # cases need that loop closed from outside:
  #   weave_drive -- the poles have no physics, so the slalom must come from
  #                  the yaw command;
  #   lane_keep   -- on the curb the robot otherwise wanders off the lip (at
  #                  any lip height), leaving nothing to measure.
  driver = None
  if weave_drive or lane_keep:
    from weave_drive import WeaveDriver, first_pole_x, yaw_from_quat_wxyz
    kw = dict(terrain_kw or {})
    origin_w = env_u.scene.env_origins[0].detach().cpu().numpy()
    if weave_drive:
      spacing = float(kw.get("spacing", 0.10))
      driver = WeaveDriver(x0=first_pole_x(env_u, spacing), spacing=spacing,
                           y_center=float(origin_w[1]))
    else:
      # amplitude 0 -> hold the spawn line (on the curb, that is the lip edge)
      driver = WeaveDriver(x0=float(origin_w[0]), spacing=1.0, amplitude=0.0,
                           y_center=float(origin_w[1]))
    meta_extra_driver = dict(mode="weave" if weave_drive else "lane_keep",
                             spacing=driver.spacing, amplitude=driver.amplitude,
                             y_center=driver.y_center,
                             lookahead_m=driver.lookahead_m, k_yaw=driver.k_yaw)

  def commanded_yaw():
    if driver is None:
      return yaw
    p = robot.data.root_link_pos_w[0].cpu().numpy()
    q = robot.data.root_link_quat_w[0].cpu().numpy()
    v = float(robot.data.root_link_lin_vel_b[0, 0])
    return driver.yaw_command(float(p[0]), float(p[1]), yaw_from_quat_wxyz(q),
                              speed=max(v, 0.02))

  # Reference line for "how far does it deviate from going straight": the line
  # through the spawn point along the spawn heading. `straight_dev_m` is the
  # signed perpendicular offset from it (+ = left), `straight_along_m` the
  # distance travelled down it. Path length vs |along| gives tortuosity.
  _p0 = robot.data.root_link_pos_w[0].cpu().numpy().copy()
  _yaw0 = quat_wxyz_to_rpy(robot.data.root_link_quat_w[0].cpu().numpy())[2]
  _c0, _s0 = math.cos(_yaw0), math.sin(_yaw0)
  _prev_xy = _p0[:2].copy()
  _path_len = 0.0

  # Exact joint work at physics rate (--substep-log). The logged tau*qd is one
  # instantaneous sample per 20 ms policy step, and the gait is phase-locked to
  # those ticks, so its time-average is a biased estimate of mechanical power.
  # Within one physics substep an explicit actuator's torque is constant, so
  # tau_k * (q_after - q_before) is that substep's work exactly.
  sub = None
  if substep:
    sub = dict(q=[], qd=[], tau=[], tgt=[], dq=[], k=[])
    _sim_step = env_u.sim.step
    _cur = {"k": -1}

    def _recorded_step(*a, **kw):
      q_pre = jm.joint_block(robot)[0]
      out = _sim_step(*a, **kw)
      q_, qd_, _, tau_ = jm.joint_block(robot)
      sub["q"].append(q_); sub["qd"].append(qd_); sub["tau"].append(tau_)
      sub["dq"].append(q_ - q_pre); sub["tgt"].append(jm.applied_targets())
      sub["k"].append(_cur["k"])
      return out
    env_u.sim.step = _recorded_step

  obs = get_initial_obs(env)
  set_joystick_twist_command(env, vx, commanded_yaw())

  rows = []
  for step in range(n_steps):
    if sub is not None:
      _cur["k"] = step
      _n0 = len(sub["k"])
    with torch.no_grad():
      pol_out = torch.nan_to_num(policy(obs), nan=0.0, posinf=6.0, neginf=-6.0)
      # The action term clamps internally before driving the motors, but the
      # `last_action` observation records whatever is passed to env.step(). These
      # policies were trained on the RAW previous action (pitch's reaches +6.4),
      # so clipping here feeds them an observation they never saw in training.
      action = pol_out if raw_action else torch.clamp(pol_out, -1.0, 1.0)
    action_raw = jm.action_block(action[0])

    obs, _reward, done, info = step_env(env, action)
    set_joystick_twist_command(env, vx, commanded_yaw())

    origin = env_u.scene.env_origins[0].cpu().numpy()
    cmd = env_u.command_manager.get_command("twist")[0].cpu().numpy()

    d = robot.data
    com_pos_w = d.root_com_pos_w[0].cpu().numpy()
    v_world = d.root_com_lin_vel_w[0].cpu().numpy()
    v_body = d.root_link_lin_vel_b[0].cpu().numpy()
    w_body = d.root_link_ang_vel_b[0].cpu().numpy()
    base_quat = d.root_link_quat_w[0].cpu().numpy()
    base_pos = d.root_link_pos_w[0].cpu().numpy()
    proj_g = d.projected_gravity_b[0].cpu().numpy()
    tr_pos, tr_quat, tr_vw, tr_ww = jm.trunk_pose(robot)

    q, qd, qdd, tau = jm.joint_block(robot)
    applied = jm.applied_targets()
    fblend = _term_vec(jm.leg_term, "filter_blend", 8)
    fcorr = _term_vec(jm.leg_term, "filter_correction", 8)
    foot_contact, foot_force = contact_arrays()

    roll, pitch, yaw_ang = quat_wxyz_to_rpy(base_quat)
    tr_roll, tr_pitch, tr_yaw = quat_wxyz_to_rpy(tr_quat)
    gn = proj_g / max(np.linalg.norm(proj_g), 1e-9)
    tilt = math.atan2(float(np.linalg.norm(gn[:2])), float(-gn[2]))

    _dx, _dy = float(base_pos[0] - _p0[0]), float(base_pos[1] - _p0[1])
    straight_along = _c0 * _dx + _s0 * _dy
    straight_dev = -_s0 * _dx + _c0 * _dy
    _path_len += float(np.hypot(base_pos[0] - _prev_xy[0], base_pos[1] - _prev_xy[1]))
    _prev_xy = base_pos[:2].copy()

    p_joint = tau * qd
    p_pos = float(np.sum(np.clip(p_joint, 0.0, None)))
    p_abs = float(np.sum(np.abs(p_joint)))
    p_net = float(np.sum(p_joint))
    p_legs = float(np.sum(np.clip(p_joint[:8], 0.0, None)))

    speed = float(np.linalg.norm(v_world))
    speed_h = float(np.linalg.norm(v_world[:2]))
    denom_x = mass * G * max(abs(float(v_body[0])), V_EPS)
    denom_h = mass * G * max(speed_h, V_EPS)

    done_b = bool(_scalar(done))
    time_out = _flag(getattr(env_u, "termination_manager", None), "time_outs", info,
                     ("time_outs", "time_out"))

    row = {
      "t": (step + 1) * policy_dt, "step": step,
      "episode_step": int(env_u.episode_length_buf[0].item()),
      "done": int(done_b), "terminated": int(done_b and not time_out), "time_out": int(bool(time_out)),
      "cmd_vx": cmd[0], "cmd_vy": cmd[1], "cmd_yaw": cmd[2],
      "com_x": com_pos_w[0] - origin[0], "com_y": com_pos_w[1] - origin[1], "com_z": com_pos_w[2] - origin[2],
      "base_x": base_pos[0] - origin[0], "base_y": base_pos[1] - origin[1], "base_z": base_pos[2] - origin[2],
      "roll": roll, "pitch": pitch, "yaw": yaw_ang, "tilt": tilt, "tilt_deg": math.degrees(tilt),
      "qw": base_quat[0], "qx": base_quat[1], "qy": base_quat[2], "qz": base_quat[3],
      "trunk_x": tr_pos[0] - origin[0], "trunk_y": tr_pos[1] - origin[1], "trunk_z": tr_pos[2] - origin[2],
      "trunk_roll": tr_roll, "trunk_pitch": tr_pitch, "trunk_yaw": tr_yaw,
      "trunk_v_world_x": tr_vw[0], "trunk_v_world_y": tr_vw[1], "trunk_v_world_z": tr_vw[2],
      "trunk_wx_w": tr_ww[0], "trunk_wy_w": tr_ww[1], "trunk_wz_w": tr_ww[2],
      "v_world_x": v_world[0], "v_world_y": v_world[1], "v_world_z": v_world[2],
      "v_body_x": v_body[0], "v_body_y": v_body[1], "v_body_z": v_body[2],
      "wx": w_body[0], "wy": w_body[1], "wz": w_body[2],
      "speed": speed, "speed_horiz": speed_h,
      "P_pos": p_pos, "P_abs": p_abs, "P_net": p_net, "P_pos_legs_only": p_legs,
      "CoT_pos": p_pos / denom_x, "CoT_abs": p_abs / denom_x, "CoT_net": p_net / denom_x,
      "CoT_pos_horiz": p_pos / denom_h, "CoT_abs_horiz": p_abs / denom_h,
      "CoT_pos_legs_only": p_legs / denom_x,
      "v_below_eps": int(abs(float(v_body[0])) < V_EPS),
    }
    for i, name in enumerate(jm.cols):
      row[f"q_{name}"] = q[i]; row[f"qd_{name}"] = qd[i]; row[f"qdd_{name}"] = qdd[i]
      row[f"tau_{name}"] = tau[i]; row[f"P_{name}"] = p_joint[i]
      row[f"action_raw_{name}"] = action_raw[i] if i < len(action_raw) else np.nan
      row[f"applied_target_{name}"] = applied[i] if i < len(applied) else np.nan
    for i in range(4):
      row[f"foot_contact_{i}"] = foot_contact[i]
      row[f"foot_force_{i}"] = foot_force[i]
    if driver is not None:
      row["weave_target_y"] = float(driver.target_y(float(base_pos[0])))
      row["weave_cross_track_m"] = float(
        driver.cross_track_error(float(base_pos[0]), float(base_pos[1])))
    row["straight_dev_m"] = straight_dev
    row["straight_along_m"] = straight_along
    row["path_len_m"] = _path_len
    row["filter_blend_mean"] = float(np.mean(fblend))
    row["filter_correction_norm"] = float(np.linalg.norm(fcorr))
    if sub is not None:
      # per-joint work over this policy step [J], summed over its substeps
      wk = np.asarray(sub["tau"][_n0:]) * np.asarray(sub["dq"][_n0:])
      for i, name in enumerate(jm.cols):
        row[f"Wx_{name}"] = float(wk[:, i].sum())
        row[f"Wxpos_{name}"] = float(np.clip(wk[:, i], 0, None).sum())
        row[f"Wxneg_{name}"] = float(np.clip(-wk[:, i], 0, None).sum())
      row["Wxpos_total"] = float(np.clip(wk, 0, None).sum())
    rows.append(row)

  env.close()
  substep_arrays = None
  if sub is not None:
    substep_arrays = {k: np.asarray(v, dtype=np.float32 if k != "k" else np.int32)
                      for k, v in sub.items()}
    substep_arrays["physics_dt"] = np.float32(getattr(env_u, "physics_dt", policy_dt / 10))

  meta = {
    "task_id": task_id, "checkpoint": str(checkpoint), "seed": seed,
    "cmd_vx": vx, "cmd_yaw": yaw, "mass_kg": mass, "g": G,
    "policy_dt": policy_dt, "n_steps": n_steps, "sim_seconds": sim_seconds,
    "action_dim": action_dim, "joint_order": list(jm.cols),
    "has_spine": jm.spine_term is not None, "spine_joint": jm.spine_name,
    "root_body": jm.root_body, "v_eps": V_EPS, "env_cfg_revision": ENV_CFG_REVISION,
    "terrain": terrain, "terrain_kw": dict(terrain_kw or {}),
    "height_termination_relaxed_m": height_drop,
    "weave_drive": bool(weave_drive),
    "lane_keep": bool(lane_keep),
    "raw_action": bool(raw_action),
    "clip_actions": clip_actions,
    "weave_driver": (meta_extra_driver if driver is not None else None),
    "substep_log": substep_arrays is not None,
  }
  if substep_arrays is not None:
    meta["physics_dt"] = float(substep_arrays["physics_dt"])
    meta["_substep_arrays"] = substep_arrays       # popped and saved by main()
  return rows, meta


# --------------------------------------------------------------------------- #
def _term_vec(term, name, n):
  v = getattr(term, name, None)
  return np.zeros(n) if v is None else v[0].detach().cpu().numpy()


def _scalar(t):
  if torch.is_tensor(t):
    return bool(t.reshape(-1)[0].item())
  return bool(np.asarray(t).reshape(-1)[0])


def _flag(term_mgr, attr, info, info_keys):
  v = getattr(term_mgr, attr, None) if term_mgr is not None else None
  if torch.is_tensor(v):
    return bool(v.reshape(-1)[0].item())
  if hasattr(info, "keys"):
    for k in info_keys:
      if k in info:
        val = info[k]
        return bool(val.reshape(-1)[0].item()) if torch.is_tensor(val) else bool(val)
  return False


# --------------------------------------------------------------------------- #
def _straightness_metrics(ss, meta):
  """How far the robot wandered off the line it started on.

  The reference is the line through the spawn point along the spawn heading, so
  this measures open-loop course-holding: with a zero yaw command the ideal
  trajectory is that straight line. Meaningless when something is deliberately
  steering (weave path following), so it is skipped there.
  """
  if meta.get("weave_drive") or "straight_dev_m" not in ss:
    return {}
  dev = ss["straight_dev_m"].to_numpy()
  along = ss["straight_along_m"].to_numpy()
  plen = ss["path_len_m"].to_numpy()
  yaw = np.unwrap(ss["yaw"].to_numpy())
  travelled = float(plen[-1] - plen[0]) if len(plen) > 1 else 0.0
  net = float(np.hypot(along[-1] - along[0], dev[-1] - dev[0])) if len(along) > 1 else 0.0
  return {
    "straight_dev_final_m": float(dev[-1]) if len(dev) else 0.0,
    "straight_dev_max_abs_m": float(np.max(np.abs(dev))) if len(dev) else 0.0,
    "straight_dev_rms_m": float(np.sqrt(np.mean(dev ** 2))) if len(dev) else 0.0,
    # deviation per metre travelled, so runs of different length compare
    "straight_dev_per_m": (float(dev[-1] - dev[0]) / travelled) if travelled > 1e-6 else 0.0,
    "straight_along_m": float(along[-1]) if len(along) else 0.0,
    "path_length_m": travelled,
    "path_tortuosity": (travelled / net) if net > 1e-6 else float("nan"),
    "heading_drift_deg": float(np.degrees(yaw[-1] - yaw[0])) if len(yaw) > 1 else 0.0,
  }


def _terrain_metrics(ss, meta):
  """Terrain-specific summary fields; empty dict on flat ground."""
  terrain = meta.get("terrain", "flat")
  if terrain == "curb":
    # The kerb edge sits at the spawn y, so base_y is the offset from the lip.
    # Feet sit ~50 mm either side of centre, so the robot is genuinely
    # straddling only while |base_y| stays inside ~40 mm.
    off = ss["base_y"].to_numpy()
    strad = np.abs(off) < 0.04
    frac = float(strad.mean())
    roll = ss["roll"].to_numpy()
    return {
      "curb_height_m": float(meta.get("terrain_kw", {}).get("curb_height", 0.015)),
      "curb_straddle_frac": frac,
      "curb_lateral_drift_m": float(off[-1] - off[0]) if len(off) else 0.0,
      "curb_mean_abs_roll_straddling_deg":
        float(np.degrees(np.abs(roll[strad]).mean())) if strad.any() else float("nan"),
    }
  if terrain == "weave" and "weave_cross_track_m" in ss:
    ct = ss["weave_cross_track_m"].to_numpy()
    return {
      "weave_cross_track_rms_m": float(np.sqrt(np.mean(ct ** 2))),
      "weave_cross_track_max_m": float(np.max(np.abs(ct))),
    }
  if terrain == "steps":
    return {"step_max_height_m": float(meta.get("terrain_kw", {}).get("max_height", 0.010))}
  return {}


def summarize(rows, meta, settle_s):
  import pandas as pd

  df = pd.DataFrame(rows)
  # A mid-rollout `done` is a RESET: the env teleports the robot back to spawn.
  # `out_of_terrain_bounds` is classified as a time-out (truncation), so it never
  # shows up in `terminated` -- the audit on 2026-09-10 found 11 such resets in
  # 384 rollouts, 6 of them in one cell (rigid / steps / 0.20). Averaging across
  # the reset silently mixed a second, easier episode (flat lead-in) into the
  # first. Summaries now use the FIRST EPISODE only and report the reset count.
  mid = df.index[(df["done"] > 0) & (df.index < len(df) - 1)]
  n_resets = int(len(mid))
  first_ep = df.loc[: mid[0] - 1] if n_resets else df
  ss = first_ep[(first_ep["t"] >= settle_s) & (first_ep["done"] == 0)]
  if len(ss) == 0:
    ss = first_ep if len(first_ep) else df
  yaw = np.unwrap(ss["yaw"].to_numpy())
  t = ss["t"].to_numpy()
  yaw_drift = float((yaw[-1] - yaw[0]) / max(t[-1] - t[0], 1e-9)) if len(yaw) > 1 else 0.0
  duty = float(np.mean(df[[f"foot_contact_{i}" for i in range(4)]].to_numpy() > 0.5))
  out = {
    **{k: meta[k] for k in ("cmd_vx", "cmd_yaw", "seed", "mass_kg", "policy_dt")},
    "variant": meta.get("variant"), "has_spine": meta.get("has_spine"),
    "env_cfg_revision": meta.get("env_cfg_revision"),
    "steady_state_rows": int(len(ss)),
    "mean_v_body_x": float(ss["v_body_x"].mean()),
    "mean_speed": float(ss["speed"].mean()),
    "tracking_err_vx": float((ss["v_body_x"] - ss["cmd_vx"]).abs().mean()),
    "yaw_rate_err": float((ss["wz"] - ss["cmd_yaw"]).abs().mean()),
    "mean_abs_roll_deg": float(np.degrees(ss["roll"].abs().mean())),
    "mean_abs_pitch_deg": float(np.degrees(ss["pitch"].abs().mean())),
    "mean_tilt_deg": float(ss["tilt_deg"].mean()),
    "yaw_drift_rate_rad_s": yaw_drift,
    "mean_CoT_pos": float(ss["CoT_pos"].mean()),
    "p95_CoT_pos": float(ss["CoT_pos"].quantile(0.95)),
    "mean_CoT_abs": float(ss["CoT_abs"].mean()),
    "p95_CoT_abs": float(ss["CoT_abs"].quantile(0.95)),
    "mean_CoT_net": float(ss["CoT_net"].mean()),
    "mean_CoT_pos_legs_only": float(ss["CoT_pos_legs_only"].mean()),
    "contact_duty_factor": duty,
    "terrain": meta.get("terrain", "flat"),
    **_straightness_metrics(ss, meta),
    **_terrain_metrics(ss, meta),
    "frac_below_v_eps": float(df["v_below_eps"].mean()),
    "any_terminated": int(df["terminated"].max()),
    "n_resets": n_resets,
    "n_timeout_resets": int(((df["done"] > 0) & (df.get("time_out", 0) > 0)
                             & (df.index < len(df) - 1)).sum()) if "time_out" in df else 0,
    "first_episode_s": float(first_ep["t"].iloc[-1]) if len(first_ep) else 0.0,
  }
  if meta.get("has_spine") and "q_spine" in ss:
    sp = ss["q_spine"].to_numpy()
    out["mean_abs_spine_deg"] = float(np.degrees(np.mean(np.abs(sp))))
    out["spine_range_deg"] = float(np.degrees(sp.max() - sp.min()))
    out["mean_abs_spine_power_W"] = float(ss["P_spine"].abs().mean())
  return out


def quick_plots(all_df, summaries, outdir):
  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
  except Exception as exc:  # noqa: BLE001
    print(f"[rollout_log] plotting skipped: {exc}")
    return
  sdf = pd.DataFrame(summaries)
  fig, ax = plt.subplots(1, 2, figsize=(11, 4))
  for conv, mk in (("mean_CoT_pos", "o-"), ("mean_CoT_abs", "s--")):
    g = sdf.groupby("cmd_vx")[conv].mean()
    ax[0].plot(g.index, g.values, mk, label=conv)
  ax[0].set_xlabel("cmd v_x [m/s]"); ax[0].set_ylabel("CoT"); ax[0].legend(); ax[0].set_title("CoT vs speed")
  g = sdf.groupby("cmd_vx")["mean_v_body_x"].mean()
  ax[1].plot(g.index, g.index, "k:"); ax[1].plot(g.index, g.values, "o-")
  ax[1].set_xlabel("cmd v_x [m/s]"); ax[1].set_ylabel("v_body_x [m/s]"); ax[1].set_title("Tracking")
  fig.tight_layout(); fig.savefig(outdir / "cot_and_tracking.png", dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--variant", choices=list(VARIANTS), default="rigid")
  ap.add_argument("--checkpoint", default=None,
                  help="path to model_XXXX.pt (default: the registered model_4499.pt for --variant)")
  ap.add_argument("--task-id", default=DEFAULT_TASK)
  ap.add_argument("--vx", type=float, nargs="+", default=[0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20])
  ap.add_argument("--yaw", type=float, nargs="+", default=[0.0])
  ap.add_argument("--seeds", type=int, nargs="+", default=[0])
  ap.add_argument("--sim-seconds", type=float, default=18.0)
  ap.add_argument("--settle-seconds", type=float, default=1.0)
  ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  ap.add_argument("--outdir", default=str(REPO_ROOT / "rollouts"))
  ap.add_argument("--keep-env", action="store_true", help="leave the variant env_cfgs.py active after the run")
  ap.add_argument("--allow-revision-mismatch", action="store_true")
  ap.add_argument("--terrain", default="flat",
                  help="flat | curb | steps | weave (see microtaur_terrains.py)")
  ap.add_argument("--curb-height", type=float, default=None, help="curb terrain: lip height (m)")
  ap.add_argument("--step-height", type=float, default=None, help="steps terrain: tallest tile (m)")
  ap.add_argument("--tile-size", type=float, default=None, help="steps terrain: tile edge (m)")
  ap.add_argument("--pole-spacing", type=float, default=None, help="weave terrain: pole pitch (m)")
  ap.add_argument("--num-poles", type=int, default=None, help="weave terrain: pole count")
  ap.add_argument("--clip-action", action="store_true",
                  help="clamp the policy output to +/-1 before env.step, so the last_action "
                       "observation is clipped. This is WRONG -- these policies were trained "
                       "on the raw previous action -- and is kept only to reproduce data "
                       "logged before 2026-09-09. Measured effect on the 15mm curb: straddle "
                       "0.10 vs 1.00, drift -0.567 m vs +0.042 m, CoT+ 1.81 vs 1.66.")
  ap.add_argument("--lane-keep", action="store_true",
                  help="hold the spawn line with a yaw controller. On --terrain curb this "
                       "keeps the robot straddling the lip; without it the robot drifts off "
                       "at any lip height, because the policy has no lateral-position "
                       "feedback, and there is nothing left to measure.")
  ap.add_argument("--weave-drive", action="store_true",
                  help="weave terrain: steer a slalom through the poles instead of "
                       "holding --yaw (the poles have no physics, so without this the "
                       "robot walks straight through them)")
  ap.add_argument("--substep-log", action="store_true",
                  help="also record every joint at physics rate (rollout_*.substep.npz) and "
                       "add exact per-step joint work columns Wx_/Wxpos_/Wxneg_. Read-only: "
                       "the simulation is unchanged.")
  args = ap.parse_args()

  terrain_kw = {}
  if args.terrain == "curb" and args.curb_height is not None:
    terrain_kw["curb_height"] = args.curb_height
  if args.terrain == "steps":
    if args.step_height is not None:
      terrain_kw["max_height"] = args.step_height
    if args.tile_size is not None:
      terrain_kw["tile_size"] = args.tile_size
  if args.terrain == "weave":
    if args.pole_spacing is not None:
      terrain_kw["spacing"] = args.pole_spacing
    if args.num_poles is not None:
      terrain_kw["num_poles"] = args.num_poles

  v = VARIANTS[args.variant]
  ckpt = Path(args.checkpoint) if args.checkpoint else Path(v.ckpt)
  if not ckpt.exists():
    ap.error(f"checkpoint not found: {ckpt}")
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  suffix = "" if args.terrain == "flat" else f"_{args.terrain}"
  outdir = Path(args.outdir) / f"{args.variant}{suffix}_{stamp}"
  outdir.mkdir(parents=True, exist_ok=True)
  print(f"[rollout_log] variant={args.variant}  checkpoint={ckpt.name}  "
        f"terrain={args.terrain}{terrain_kw or ''}  -> {outdir}")

  import pandas as pd

  summaries, all_df = [], {}
  with use_variant(args.variant, keep=args.keep_env):
    try:
      rev = verify_revision(v.revision)
      print(f"[rollout_log] ENV_CFG_REVISION={rev} OK")
    except RuntimeError as exc:
      if not args.allow_revision_mismatch:
        raise
      print(f"[rollout_log] WARNING {exc}")

    for vx in args.vx:
      for yaw in args.yaw:
        for seed in args.seeds:
          tag = f"vx{vx:.2f}_yaw{yaw:+.2f}_seed{seed}"
          print(f"[rollout_log] {tag} ...")
          rows, meta = run_one(args.task_id, str(ckpt), vx, yaw, seed, args.sim_seconds,
                               args.device, terrain=args.terrain, terrain_kw=terrain_kw,
                               weave_drive=args.weave_drive, lane_keep=args.lane_keep,
                               raw_action=not args.clip_action, substep=args.substep_log,
                               clip_actions=v.clip_actions)
          meta["variant"] = args.variant
          meta["mjcf_variant"] = v.mjcf_variant
          sa = meta.pop("_substep_arrays", None)
          if sa is not None:
            np.savez_compressed(outdir / f"rollout_{tag}.substep.npz", **sa)
          df = pd.DataFrame(rows)
          df.to_csv(outdir / f"rollout_{tag}.csv", index=False)
          (outdir / f"rollout_{tag}.meta.json").write_text(json.dumps(meta, indent=2))
          s = summarize(rows, meta, args.settle_seconds)
          s["variant"] = args.variant
          summaries.append(s)
          all_df[tag] = df
          extra = f"  spine|{s['mean_abs_spine_deg']:.1f}deg" if s.get("mean_abs_spine_deg") is not None else ""
          print(f"    v_body_x={s['mean_v_body_x']:.4f} (cmd {vx:.2f})  "
                f"CoT+={s['mean_CoT_pos']:.2f}  CoT_abs={s['mean_CoT_abs']:.2f}  "
                f"|roll|={s['mean_abs_roll_deg']:.2f}deg  term={s['any_terminated']}{extra}")

  (outdir / "summary.json").write_text(json.dumps(summaries, indent=2))
  quick_plots(all_df, summaries, outdir)
  print(f"[rollout_log] done. {len(summaries)} rollouts -> {outdir}/summary.json")


if __name__ == "__main__":
  main()
