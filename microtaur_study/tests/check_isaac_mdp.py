"""Check the IsaacLab MDP terms (microtaur_isaac.mdp) against hand computations.

Builds a minimal ManagerBasedRLEnv around the rigid Microtaur USD (plane
terrain, contact sensor, height scanner, a plain JointPositionAction as a
stand-in for MicrotaurWalkAction) with the real observation / reward /
termination / command / curriculum cfgs, steps it with random actions and
forced falls, and checks on the same state:

  1. every reward term (recorded inside the reward manager) against an
     independent recomputation from raw simulator data (own rotation matrices,
     own DC-motor clip, own gait kernels, own action buffers);
  2. applied_torque == hand DC-motor clip of the PD torque of the last substep;
  3. RewardManager dt scaling: reward_buf == sum(value * weight * step_dt);
  4. terminations by hand; is_terminated excludes time-outs;
  5. observation dims (actor 33, critic 60, teacher 60 + height scan) and every
     slice of every group by hand (actor within the stage's noise bounds);
  6. command ranges, zero-yaw mask, command and energy curricula, metrics log;
  7. validate_observation_contract rejects broken groups.

Usage (conda env spine):
  OMNI_KIT_ACCEPT_EULA=YES python tests/check_isaac_mdp.py \
      --usd microtaur_rigid_revolute.usda --model model.json [--num_envs 16] [--steps 160]
  (USD and model json from tools/spike_isaac_closed_chain.py / spike_mujoco_reference.py)
Output is also written to --out (Kit swallows stdout).
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--usd", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--num_envs", type=int, default=16)
ap.add_argument("--steps", type=int, default=160)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out", default="check_isaac_mdp.txt")
args = ap.parse_args()

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True, device=args.device).app

_log = open(args.out, "w")


def log(*a):
  s = " ".join(str(x) for x in a)
  _log.write(s + "\n")
  _log.flush()
  print(s, flush=True)


FAILS = []


def check(name, ok, detail=""):
  log(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
  if not ok:
    FAILS.append(name)


try:
  import torch

  import isaaclab.envs.mdp as il_mdp
  import isaaclab.sim as sim_utils
  from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
  from isaaclab.assets import ArticulationCfg
  from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
  from isaaclab.managers import CurriculumTermCfg
  from isaaclab.managers import EventTermCfg as EventTerm
  from isaaclab.scene import InteractiveSceneCfg
  from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
  from isaaclab.terrains import TerrainImporterCfg
  from isaaclab.utils import configclass

  from microtaur_common import robot_constants as RC
  from microtaur_common.sim2real import STAGES
  from microtaur_common.task_params import (
    COMMAND_STAGES, MAX_TILT_RAD, MIN_ROOT_HEIGHT_M, WEIGHTS, energy_weight_stages,
  )
  from microtaur_isaac import FOOT_BODY_NAMES, FOOT_OFFSET_IN_BODY_M
  from microtaur_isaac.mdp import commands as C
  from microtaur_isaac.mdp import observations as O
  from microtaur_isaac.mdp import rewards as R
  from microtaur_isaac.mdp import terminations as T

  M = json.load(open(args.model))
  N = args.num_envs
  DEC = 7
  STAGE = STAGES[1]
  TEST_ENERGY_STAGES = [{"step": 0, "weight": 0.0}, {"step": 25, "weight": -0.01}, {"step": 70, "weight": -0.025}]
  SCAN_SIZE, SCAN_RES = (0.40, 0.30), 0.05

  ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
      usd_path=args.usd,
      activate_contact_sensors=True,
      rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
      articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
      ),
      collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
      pos=(0.0, 0.0, M["nominal_root_z"] + 0.001), joint_pos=dict(M["stand_q"]), joint_vel={".*": 0.0}
    ),
    actuators={
      "motors": DCMotorCfg(
        joint_names_expr=list(RC.LEG_JOINT_NAMES), stiffness=RC.KP, damping=0.045,
        effort_limit=RC.EFFORT_LIMIT_NM, saturation_effort=RC.XL330_STALL_TORQUE_NM,
        velocity_limit=RC.VELOCITY_LIMIT_RAD_S, armature=RC.ARMATURE, friction=0.0,
      ),
      "passive": ImplicitActuatorCfg(
        joint_names_expr=list(RC.PASSIVE_JOINT_NAMES), stiffness=0.0, damping=RC.PASSIVE_DAMPING,
        armature=RC.PASSIVE_ARMATURE, friction=0.0,
      ),
    },
  )

  @configclass
  class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
      prim_path="/World/ground", terrain_type="plane",
      physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )
    robot = ROBOT_CFG
    contact_forces = ContactSensorCfg(
      prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=DEC, track_air_time=True,
      force_threshold=O.CONTACT_FORCE_THRESHOLD_N,
    )
    height_scanner = RayCasterCfg(
      prim_path="{ENV_REGEX_NS}/Robot/battery", ray_alignment="yaw",
      pattern_cfg=patterns.GridPatternCfg(resolution=SCAN_RES, size=list(SCAN_SIZE)),
      max_distance=1.0, mesh_prim_paths=["/World/ground"],
    )

  @configclass
  class ActionsCfg:
    joint_pos = il_mdp.JointPositionActionCfg(
      asset_name="robot", joint_names=list(RC.LEG_JOINT_NAMES), preserve_order=True, scale=0.25,
      use_default_offset=True,
    )

  @configclass
  class CommandsCfg:
    twist = C.make_twist_command_cfg()

  @configclass
  class EventsCfg:
    reset_base = EventTerm(
      func=il_mdp.reset_root_state_uniform, mode="reset",
      params={"pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-3.14, 3.14)}, "velocity_range": {}},
    )
    reset_joints = EventTerm(
      func=il_mdp.reset_joints_by_offset, mode="reset",
      params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )

  @configclass
  class CurriculumCfg:
    command_ranges = C.make_command_curriculum_term()
    energy_weight = CurriculumTermCfg(
      func=R.reward_weight_stages, params={"reward_name": "motor_energy", "stages": TEST_ENERGY_STAGES}
    )

  @configclass
  class EnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(num_envs=N, env_spacing=0.5)
    observations = O.make_observations_cfg(STAGE, rough=False, teacher=True)
    actions = ActionsCfg()
    commands = CommandsCfg()
    rewards = R.make_rewards_cfg()
    terminations = T.make_terminations_cfg()
    events = EventsCfg()
    curriculum = CurriculumCfg()

    def __post_init__(self):
      self.decimation = DEC
      self.episode_length_s = 1.5  # short, so time-outs happen
      self.sim.dt = 0.005
      self.sim.render_interval = DEC
      self.sim.device = args.device

  cfg = EnvCfg()

  # --- hand computations -------------------------------------------------------
  def rotmat(q):  # wxyz -> [N, 3, 3] (body -> world)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
      1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ), dim=-1).reshape(*q.shape[:-1], 3, 3)

  def to_body(q, v):
    return torch.einsum("nji,nj->ni", rotmat(q), v)

  HAND = {}
  RECORDED = {}
  CAP = {}  # actuator inputs of the last substep
  MY = {"act": None, "prev": None}
  COPPER = (5.0 / 1.47) / (0.215 / 1.47) ** 2  # R / k_t^2 from the XL330 datasheet stall point
  DENOM = 0.540 * 9.81 * 0.15

  def hand_torque(env):
    kp, kd = RC.KP, 0.045
    eff, sat, vl = RC.EFFORT_LIMIT_NM, RC.XL330_STALL_TORQUE_NM, RC.VELOCITY_LIMIT_RAD_S
    q, qd, qt = CAP["q"], CAP["qd"], CAP["qt"]
    tau_c = kp * (qt - q) + kd * (0.0 - qd)
    qdc = torch.clamp(qd, -vl * (1 + eff / sat), vl * (1 + eff / sat))
    hi = torch.clamp(sat * (1.0 - qdc / vl), max=eff)
    lo = torch.clamp(sat * (-1.0 - qdc / vl), min=-eff)
    return torch.clamp(tau_c, lo, hi), tau_c

  def hand_power(env):
    rob = env.scene["robot"]
    tau, _ = hand_torque(env)
    qd = rob.data.joint_vel[:, MOT]
    return torch.sum(torch.abs(tau * qd), 1), COPPER * torch.sum(tau * tau, 1)

  def hand_lin(env):
    rob = env.scene["robot"]
    v = to_body(rob.data.root_link_quat_w, rob.data.root_link_lin_vel_w)
    c = env.command_manager.get_command("twist")
    return torch.exp(-((c[:, 0] - v[:, 0]) ** 2 + (c[:, 1] - v[:, 1]) ** 2) / 0.10**2)

  def hand_yaw(env):
    rob = env.scene["robot"]
    w = to_body(rob.data.root_link_quat_w, rob.data.root_link_ang_vel_w)
    c = env.command_manager.get_command("twist")
    return torch.exp(-((c[:, 2] - w[:, 2]) ** 2) / 0.15**2)

  def hand_energy(env):
    mech, cu = hand_power(env)
    return (mech + cu) / DENOM

  def hand_action_rate(env):
    return torch.sum((MY["act"] - MY["prev"]) ** 2, 1)

  def hand_done(env):
    rob = env.scene["robot"]
    tilt = torch.acos(torch.clamp(rotmat(rob.data.root_link_quat_w)[:, 2, 2], -1, 1)) > math.radians(70.0)
    s = env.scene["contact_forces"]
    bid = s.body_names.index("battery")
    touch = torch.linalg.norm(s.data.net_forces_w_history[:, :, bid], dim=-1).max(1).values > 0.05
    low = rob.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] < MIN_ROOT_HEIGHT_M
    return tilt, touch, low

  def hand_term(env):
    tilt, touch, low = hand_done(env)
    HAND["causes"] = (tilt, touch, low)
    return (tilt | touch | low).float()

  def hand_trot(env):
    s = env.scene["contact_forces"]
    ids = [s.body_names.index(n) for n in FOOT_BODY_NAMES]
    air, con = s.data.current_air_time[:, ids], s.data.current_contact_time[:, ids]
    e2, std = (0.2 / 2) ** 2, 0.1 / 4
    out = torch.ones(env.num_envs, device=env.device)
    for a, b in ((0, 2), (1, 3)):
      se = torch.clamp((air[:, a] - air[:, b]) ** 2, max=e2) + torch.clamp((con[:, a] - con[:, b]) ** 2, max=e2)
      out = out * torch.exp(-se / std)
    for a, b in ((0, 1), (2, 3), (0, 3), (1, 2)):
      se = torch.clamp((air[:, a] - con[:, b]) ** 2, max=e2) + torch.clamp((con[:, a] - air[:, b]) ** 2, max=e2)
      out = out * torch.exp(-se / std)
    return out

  HAND_FNS = {
    "track_lin_vel_xy": hand_lin, "track_ang_vel_z": hand_yaw, "motor_energy": hand_energy,
    "action_rate": hand_action_rate, "termination": hand_term, "trot_gait": hand_trot,
  }
  METRIC_ACC = {}

  def recording(name, fn):
    @functools.wraps(fn)
    def wrapped(env, **kw):
      v = fn(env, **kw)
      RECORDED[name] = v.clone()
      HAND[name] = HAND_FNS[name](env)
      if name == "track_lin_vel_xy":  # always active: accumulate hand metrics once per step
        mech, cu = hand_power(env)
        spd = to_body(env.scene["robot"].data.root_link_quat_w, env.scene["robot"].data.root_link_lin_vel_w)[:, 0]
        METRIC_ACC["sum"] += torch.stack((mech, cu, spd), 1)
        METRIC_ACC["n"] += 1
        HAND["tau"], HAND["tau_c"] = hand_torque(env)
      return v
    return wrapped

  for name in HAND_FNS:
    t = getattr(cfg.rewards, name)
    t.func = recording(name, t.func)

  env = ManagerBasedRLEnv(cfg)
  dev = env.device
  rob = env.scene["robot"]
  sen = env.scene["contact_forces"]
  MOT = [rob.joint_names.index(n) for n in RC.LEG_JOINT_NAMES]
  FEET_R = [rob.body_names.index(n) for n in FOOT_BODY_NAMES]
  FEET_S = [sen.body_names.index(n) for n in FOOT_BODY_NAMES]
  log("joints", rob.joint_names)
  log("motor ids (canonical order)", MOT, " foot body ids robot", FEET_R, " sensor", FEET_S)
  log("sensor bodies", sen.body_names)
  log("step_dt", env.step_dt, " physics_dt", env.physics_dt, " max_episode_length", env.max_episode_length)
  METRIC_ACC["sum"] = torch.zeros(N, 3, device=dev)
  METRIC_ACC["n"] = torch.zeros(N, device=dev)

  orig_apply = rob._apply_actuator_model

  def capture_apply():
    CAP["q"] = rob.data.joint_pos[:, MOT].clone()
    CAP["qd"] = rob.data.joint_vel[:, MOT].clone()
    CAP["qt"] = rob.data.joint_pos_target[:, MOT].clone()
    orig_apply()

  rob._apply_actuator_model = capture_apply

  # --- 7. contract validator ---------------------------------------------------
  good = O.make_observations_cfg(STAGE, rough=False)
  O.validate_observation_contract(good, rough=False)
  O.validate_observation_contract(O.make_observations_cfg(STAGE, rough=True), rough=True)
  bad_cases = {}
  b = O.make_observations_cfg(STAGE); b.policy.base_lin_vel = O.ObsTerm(func=O.base_lin_vel)
  bad_cases["actor gets base_lin_vel"] = b
  b = O.make_observations_cfg(STAGE); b.policy.joint_pos.params["biased"] = False
  bad_cases["actor unbiased"] = b
  b = O.make_observations_cfg(STAGE); b.critic.joint_pos.params["biased"] = True
  bad_cases["critic biased"] = b
  b = O.make_observations_cfg(STAGE); b.critic.height_scan = O._height_scan_term()
  bad_cases["flat critic with height_scan"] = b
  for k, bc in bad_cases.items():
    try:
      O.validate_observation_contract(bc, rough=False)
      check(f"validator rejects: {k}", False)
    except RuntimeError as e:
      check(f"validator rejects: {k}", True, f"({str(e)[:70]})")

  # --- 6a. command + zero-yaw mask ----------------------------------------------
  mask = torch.arange(N, device=dev) % 2 == 0
  env.microtaur_zero_yaw_mask = mask
  env.microtaur_encoder_bias = (torch.rand(N, 8, device=dev) * 2 - 1) * STAGE.encoder_bias_rad
  obs, _ = env.reset()
  cmd = env.command_manager.get_command("twist")
  st0 = COMMAND_STAGES[0]
  check("reset: yaw cmd 0 on masked envs", bool((cmd[mask, 2] == 0).all()), f"max|wz| masked {cmd[mask, 2].abs().max():.3g}")
  check("reset: yaw cmd in range, nonzero on unmasked", bool(((cmd[~mask, 2].abs() > 0) & (cmd[~mask, 2].abs() <= st0["ang_vel_z"][1])).all()),
        f"unmasked wz {[round(float(x), 3) for x in cmd[~mask, 2][:4]]}")
  check("reset: vy == 0, vx in stage-0 range", bool((cmd[:, 1] == 0).all() and ((cmd[:, 0] >= st0["lin_vel_x"][0]) & (cmd[:, 0] <= st0["lin_vel_x"][1])).all()),
        f"vx {[round(float(x), 3) for x in cmd[:4, 0]]}")
  term = env.command_manager.get_term("twist")
  check("no standing envs, no heading", bool((~term.is_standing_env).all()) and not term.cfg.heading_command)

  # --- 5. observation dims --------------------------------------------------------
  dims = {g: tuple(v.shape) for g, v in obs.items()}
  log("obs dims", dims)
  log("obs terms", env.observation_manager.active_terms)
  log("obs term dims", env.observation_manager.group_obs_term_dim)
  R_SCAN = int(round(SCAN_SIZE[0] / SCAN_RES + 1) * round(SCAN_SIZE[1] / SCAN_RES + 1))
  check("actor dim 33", dims["policy"] == (N, 33))
  check("critic dim 60", dims["critic"] == (N, 60))
  check(f"teacher dim 60 + {R_SCAN}", dims["teacher"] == (N, 60 + R_SCAN))
  check("actor term order", tuple(env.observation_manager.active_terms["policy"]) == O.ACTOR_TERMS)
  check("critic term order", tuple(env.observation_manager.active_terms["critic"]) == O.CRITIC_TERMS)

  def hand_obs(env):
    d = rob.data
    qw = d.root_link_quat_w
    q = d.joint_pos[:, MOT] - d.default_joint_pos[:, MOT]
    pos, quat = d.body_link_pos_w[:, FEET_R], d.body_link_quat_w[:, FEET_R]
    off = torch.tensor(FOOT_OFFSET_IN_BODY_M, device=dev)
    centre = pos + torch.einsum("nkij,kj->nki", rotmat(quat), off)
    f = sen.data.net_forces_w[:, FEET_S]
    fl = -f.reshape(N, 12)
    return {
      "base_lin_vel": to_body(qw, d.root_link_lin_vel_w),
      "base_ang_vel": to_body(qw, d.root_link_ang_vel_w),
      "projected_gravity": to_body(qw, torch.tensor([[0.0, 0.0, -1.0]], device=dev).expand(N, 3)),
      "joint_pos": q, "joint_vel": d.joint_vel[:, MOT] - d.default_joint_vel[:, MOT],
      "actions": MY["obs_act"],
      "command": env.command_manager.get_command("twist"),
      "height_scan": torch.clamp(d.root_link_pos_w[:, 2:3] - 0.0, max=0.30).expand(N, R_SCAN) / 0.30,
      "foot_height": centre[..., 2] - env.scene.env_origins[:, 2:3],
      "foot_air_time": sen.data.current_air_time[:, FEET_S],
      "foot_contact": (torch.linalg.norm(f, dim=-1) > 0.05).float(),
      "foot_contact_forces": torch.sign(fl) * torch.log1p(fl.abs()),
    }

  NOISE = {"base_ang_vel": STAGE.ang_vel_noise, "projected_gravity": STAGE.gravity_noise,
           "joint_pos": STAGE.joint_pos_noise, "joint_vel": STAGE.joint_vel_noise}
  OBS_ERR = {}
  NOISE_SEEN = {}

  def check_obs(obs):
    h = hand_obs(env)
    for g in ("policy", "critic", "teacher"):
      names = env.observation_manager.active_terms[g]
      dimsg = env.observation_manager.group_obs_term_dim[g]
      i = 0
      for n, dd in zip(names, dimsg):
        w = dd[0]
        x = obs[g][:, i:i + w]
        i += w
        ref = h[n] + (env.microtaur_encoder_bias if (g == "policy" and n == "joint_pos") else 0.0)
        err = (x - ref).abs().max().item()
        if env.observation_manager._group_obs_term_cfgs[g][names.index(n)].noise is not None and n in NOISE:
          NOISE_SEEN[(g, n)] = max(NOISE_SEEN.get((g, n), 0.0), err)
          err = max(0.0, err - NOISE[n] * (1 + 1e-4))
        OBS_ERR[(g, n)] = max(OBS_ERR.get((g, n), 0.0), err)

  MY["obs_act"] = torch.zeros(N, 8, device=dev)
  check_obs(obs)
  # flat-ground fallback of foot_height == scanner nearest-hit version
  fh_scan = O.foot_height(env, O.feet_body_cfg().replace(body_ids=FEET_R), scanner_name="height_scanner")
  fh_flat = O.foot_height(env, O.feet_body_cfg().replace(body_ids=FEET_R), scanner_name=None)
  check("foot_height scanner == flat fallback on a plane", (fh_scan - fh_flat).abs().max().item() < 1e-5,
        f"max diff {(fh_scan - fh_flat).abs().max().item():.2e}")

  # --- main loop ------------------------------------------------------------------
  ERR = {k: 0.0 for k in HAND_FNS}
  DT_ERR = 0.0
  STEP_REWARD_ERR = 0.0
  TORQUE_ERR = 0.0
  CLIPPED = 0
  METRIC_ERR = 0.0
  N_METRIC = 0
  CURR_ERR = 0.0
  last_curr_counter = 0
  TIMEOUT_PEN = 0.0
  n_term, n_to = 0, 0
  TERM_BY = {"tilt": 0, "touch": 0, "low": 0}
  YAW_MASK_MAX = 0.0
  MAG = {k: [] for k in HAND_FNS}
  prev = torch.zeros(N, 8, device=dev)
  torch.manual_seed(0)
  names = env.reward_manager.active_terms
  for k in range(args.steps):
    act = 0.8 * torch.randn(N, 8, device=dev)
    MY["act"], MY["prev"] = act, prev
    if k in (30, 90):  # force falls: tilt env 0 by 80 deg, drop env 1 below the height limit
      pose = torch.cat((rob.data.root_link_pos_w, rob.data.root_link_quat_w), 1).clone()
      pose[0, 3:7] = torch.tensor([math.cos(math.radians(40)), math.sin(math.radians(40)), 0.0, 0.0], device=dev)
      pose[1, 2] = env.scene.env_origins[1, 2] + 0.03
      rob.write_root_pose_to_sim(pose[:2], env_ids=torch.tensor([0, 1], device=dev))
    RECORDED.clear()
    HAND.clear()
    # weights used by this step's reward (the curriculum may change them in the same step's reset)
    W = {n: env.reward_manager.get_term_cfg(n).weight for n in names}
    obs, rew, terminated, truncated, extras = env.step(act)
    dones = terminated | truncated
    # 1. reward terms vs hand (same state, recorded inside the manager)
    for n in HAND_FNS:
      if n in RECORDED:
        ERR[n] = max(ERR[n], (RECORDED[n] - HAND[n]).abs().max().item())
        MAG[n].append(RECORDED[n].mean().item())
    # 2. applied torque
    TORQUE_ERR = max(TORQUE_ERR, (rob.data.applied_torque[:, MOT] - HAND["tau"]).abs().max().item())
    CLIPPED += int(((HAND["tau_c"] - HAND["tau"]).abs() > 1e-6).sum())
    # 3. dt scaling
    total = torch.zeros(N, device=dev)
    for idx, n in enumerate(names):
      w = W[n]
      if w == 0.0:
        continue
      v = RECORDED.get(n, torch.zeros(N, device=dev))
      total += v * w * env.step_dt
      STEP_REWARD_ERR = max(STEP_REWARD_ERR, (env.reward_manager._step_reward[:, idx] - v * w).abs().max().item())
    DT_ERR = max(DT_ERR, (rew - total).abs().max().item())
    # 4. terminations
    for key, c in zip(("tilt", "touch", "low"), HAND["causes"]):
      TERM_BY[key] += int(c.sum())
    n_term += int(terminated.sum())
    n_to += int((truncated & ~terminated).sum())
    if (truncated & ~terminated).any():
      TIMEOUT_PEN = max(TIMEOUT_PEN, RECORDED["termination"][truncated & ~terminated].abs().max().item())
    check_term = (RECORDED["termination"] > 0) == terminated
    if not bool(check_term.all()):
      ERR["termination"] = max(ERR["termination"], 1.0)
    # 6. metrics log on reset
    if dones.any():
      ids = dones.nonzero().flatten()
      hand_m = (METRIC_ACC["sum"][ids] / METRIC_ACC["n"][ids].clamp(min=1)[:, None]).mean(0)
      lg = extras.get("log", {})
      got = torch.stack([torch.as_tensor(lg[f"Episode_Metrics/{m}"]) for m in R.METRIC_NAMES]).to(dev)
      METRIC_ERR = max(METRIC_ERR, ((got - hand_m).abs() / hand_m.abs().clamp(min=1e-3)).max().item())
      N_METRIC += 1
      METRIC_ACC["sum"][ids] = 0.0
      METRIC_ACC["n"][ids] = 0.0
      # the curriculum ran in this step's _reset_idx with the counter after increment
      last_curr_counter = env.common_step_counter
    exp_w = [s["weight"] for s in TEST_ENERGY_STAGES if last_curr_counter >= s["step"]][-1]
    CURR_ERR = max(CURR_ERR, abs(env.reward_manager.get_term_cfg("motor_energy").weight - exp_w))
    YAW_MASK_MAX = max(YAW_MASK_MAX, env.command_manager.get_command("twist")[mask, 2].abs().max().item())
    MY["obs_act"] = torch.where(dones[:, None], torch.zeros_like(act), act)  # ActionManager.reset zeroes action
    check_obs(obs)
    prev = torch.where(dones[:, None], torch.zeros_like(act), act)  # ActionManager.reset zeroes prev_action
    if k in (0, 31, 91):
      log(f"step {k}: terminated {terminated.nonzero().flatten().tolist()} truncated {truncated.nonzero().flatten().tolist()}"
          f" rew[0:4] {[round(float(x), 5) for x in rew[:4]]}")
      log("   termination log:", {kk: round(float(v), 3) for kk, v in extras.get("log", {}).items() if "Termination" in kk})

  log("")
  log("== reward terms vs hand (max abs err over all steps/envs; mean value) ==")
  for n in HAND_FNS:
    mv = sum(MAG[n]) / max(1, len(MAG[n]))
    check(f"reward {n}", ERR[n] < 1e-4, f"err {ERR[n]:.2e}  calls {len(MAG[n])}  mean {mv:.5f}  final weight*dt {env.reward_manager.get_term_cfg(n).weight * env.step_dt:.5f}")
  check("applied_torque == hand DC-motor clip of PD torque", TORQUE_ERR < 1e-5, f"err {TORQUE_ERR:.2e}, clipped entries {CLIPPED}")
  check("reward_buf == sum(term * weight * step_dt)", DT_ERR < 1e-5, f"err {DT_ERR:.2e} (step_dt {env.step_dt})")
  check("_step_reward == term * weight (value / dt)", STEP_REWARD_ERR < 1e-4, f"err {STEP_REWARD_ERR:.2e}")
  check("termination weight * dt == -2.0", abs(WEIGHTS["termination"] * env.step_dt + 2.0) < 1e-9, f"{WEIGHTS['termination'] * env.step_dt:.6f}")
  check("is_terminated == terminated (non-timeout) and 0 on pure time-outs", n_term > 0 and n_to > 0 and TIMEOUT_PEN == 0.0,
        f"terminated {n_term} (hand causes {TERM_BY}), time-outs {n_to}, max penalty on time-outs {TIMEOUT_PEN}")
  check("metrics log == hand episode means", N_METRIC > 0 and METRIC_ERR < 1e-3, f"resets {N_METRIC}, max rel err {METRIC_ERR:.2e}")
  check("energy curriculum weight == stage at last reset counter", CURR_ERR == 0.0,
        f"final counter {env.common_step_counter}, weight {env.reward_manager.get_term_cfg('motor_energy').weight}")
  check("yaw cmd stays 0 on masked envs through resamples", YAW_MASK_MAX == 0.0, f"max {YAW_MASK_MAX}")
  log("== observations vs hand (max abs err; actor minus the stage noise bound) ==")
  for (g, n), e in OBS_ERR.items():
    check(f"obs {g}/{n}", e < 1e-4, f"err {e:.2e}")
  for g in ("policy", "teacher"):
    for n, b in NOISE.items():
      seen = NOISE_SEEN.get((g, n), 0.0)
      check(f"{g} noise {n} present and within +-{b}", 0.5 * b < seen <= b * (1 + 1e-4) + 1e-6, f"max |obs - true| {seen:.5f}")
  check("critic has no noise", not any(g == "critic" for g, _ in NOISE_SEEN))
  log("sample critic foot_height", [round(float(x), 5) for x in obs["critic"][0, 36:40]],
      " contact", obs["critic"][0, 44:48].tolist(), " forces z", [round(float(x), 3) for x in obs["critic"][0, 48:60][2::3]])

  # --- 6b. command curriculum by step counter -------------------------------------
  saved = env.common_step_counter
  rows = []
  for s in (0, 7999, 8000, 19999, 20000, 90000):
    env.common_step_counter = s
    out = C.command_curriculum(env, None, "twist", COMMAND_STAGES)
    rng = env.command_manager.get_term("twist").cfg.ranges
    rows.append((s, int(out["stage"]), rng.lin_vel_x, rng.ang_vel_z, rng.lin_vel_y))
  expected = [0, 0, 1, 1, 2, 2]
  check("command curriculum stages by step", [r[1] for r in rows] == expected and all(r[4] == (0.0, 0.0) for r in rows), str(rows))
  env.common_step_counter = 0
  out = C.command_curriculum(env, None, "twist", (COMMAND_STAGES[-1],))
  rng = env.command_manager.get_term("twist").cfg.ranges
  check("play curriculum at step 0 falls back to its first (final) stage", rng.lin_vel_x == COMMAND_STAGES[-1]["lin_vel_x"], f"{rng.lin_vel_x} {rng.ang_vel_z}")
  env.common_step_counter = saved
  # zero_yaw_commands helper
  m2 = torch.zeros(N, dtype=torch.bool, device=dev); m2[1] = True
  C.zero_yaw_commands(env, "twist", m2)
  check("zero_yaw_commands helper", float(env.command_manager.get_command("twist")[1, 2]) == 0.0)
  log("real energy_weight_stages(-0.025):", energy_weight_stages(-0.025))
  # terminations on constructed states: tilt limit
  log("MAX_TILT_RAD", MAX_TILT_RAD, " MIN_ROOT_HEIGHT_M", MIN_ROOT_HEIGHT_M, " body contact N", T.BODY_CONTACT_FORCE_N)

  # --- rough-terrain code paths on synthetic data (the plane cannot exercise them) ---
  from types import SimpleNamespace as NS
  xs = torch.linspace(-0.2, 0.2, 9, device=dev)
  gx, gy = torch.meshgrid(xs, xs[:7] * 0.75, indexing="ij")
  hz = torch.where(gx > 0.001, torch.full_like(gx, 0.02), torch.zeros_like(gx))  # 2 cm step at x > 0
  hits = torch.stack((gx, gy, hz), -1).reshape(1, -1, 3).repeat(2, 1, 1)
  hits[1, :, :] = float("inf")  # env 1: every ray missed
  fake = NS(num_envs=2, device=dev, scene=NS(env_origins=torch.tensor([[0.0, 0.0, 0.005], [0.0, 0.0, 0.007]], device=dev),
                                          sensors={"height_scanner": NS(data=NS(ray_hits_w=hits))}))
  pts = torch.tensor([[[0.10, 0.0, 0.03], [-0.10, 0.0, 0.03], [1.0, 0.0, 0.03]]] * 2, device=dev)
  gz = O.ground_height_below(fake, pts, "height_scanner")
  check("ground_height_below: nearest hit / far point / all-miss fallback",
        torch.allclose(gz, torch.tensor([[0.02, 0.0, 0.005], [0.007, 0.007, 0.007]], device=dev)), str(gz.tolist()))
  class _Scene(dict):
    pass
  sc = _Scene(robot=NS(data=NS(root_link_pos_w=torch.tensor([[0.0, 0.0, 0.07], [0.0, 0.0, 0.07]], device=dev))))
  sc.sensors = {"height_scanner": NS(data=NS(ray_hits_w=hits))}
  hs = O.height_scan(NS(scene=sc), max_distance=0.30)
  check("height_scan: root z - hit z, misses -> max_distance",
        abs(float(hs[0].max()) - 0.07) < 1e-6 and abs(float(hs[0].min()) - 0.05) < 1e-6 and bool((hs[1] == 0.30).all()),
        f"env0 [{float(hs[0].min()):.3f}, {float(hs[0].max()):.3f}] env1 {float(hs[1].mean()):.2f}")
  sc2 = _Scene(robot=NS(data=NS(root_link_pos_w=torch.tensor([[3.6, 0.0, 0.07], [3.8, 0.0, 0.07], [0.0, 4.8, 0.07]], device=dev))))
  sc2.terrain = NS(cfg=NS(terrain_type="generator", terrain_generator=NS(size=(2.0, 2.0), border_width=1.0)),
                   terrain_origins=torch.zeros(3, 4, 3, device=dev))
  oob = T.out_of_terrain_bounds(NS(scene=sc2, num_envs=3, device=dev))
  check("out_of_terrain_bounds (3x4 grid of 2 m, border 1, margin 0.3 -> |x|>3.7, |y|>4.7)",
        oob.tolist() == [False, True, True], str(oob.tolist()))

  log("")
  log("FAILED:" if FAILS else "ALL PASS", FAILS)
except Exception:
  import traceback
  log(traceback.format_exc())
finally:
  _log.close()
  os._exit(0)
