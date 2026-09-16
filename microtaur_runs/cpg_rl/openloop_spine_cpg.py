"""Open-loop spine CPG for Microtaur, plus RL-teacher runs logged the same way.

Builds on the existing fitted CPG: the legs are the per-leg sines fitted to each flat
teacher (fit_<variant>.json from fit_cpg_from_flat.py), driven through cpg_action.py with
constant actions, so nothing reads a sensor. This script adds a spine sine locked to the
leg-1 phase:

    spine_target = offset + amp * sin(phase_leg1 + phase)

Logged signals are only what the robot can measure: the IMU on the root body (gravity
direction and gyro) and, for the 9 motors, position, velocity and torque. Base position
and velocity from the simulator are used only to score walking (speed, drift).

Modes
  cpg       legs = fitted CPG (cpg_action.py), spine = sine; one env per parameter combination
  harmonic  all 9 motor targets from a gait JSON of sine harmonics on one clock (the file the
            board player reads), fed through the RL's own action terms, so the safety filter,
            command delay and motor model are the ones the RL trained with; optional spine
            sine override grid for sweeps
  policy    an RL checkpoint (for the RL-vs-CPG comparison), same logging

Examples
  python openloop_spine_cpg.py pitch cpg --spine-amp-deg 0,10 --spine-phase-deg 0,90,180,270 --out sweep.npz
  python openloop_spine_cpg.py pitch policy --num-envs 8 --series --out rl.npz
  python openloop_spine_cpg.py pitch policy --delay-steps 1 --num-envs 32 --series --out rl_d1.npz
  python openloop_spine_cpg.py pitch harmonic --gait gait.json --repeats 16 --series --out h.npz
  python openloop_spine_cpg.py pitch harmonic --gait gait.json --h-spine-amp-deg 5,10 --h-spine-phase-deg 0,90 --out s.npz
  python openloop_spine_cpg.py pitch cpg --spine-amp-deg 10 --spine-phase-deg 90 --video cpg.mp4
"""

import argparse
import itertools
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1]
SPINES_HELP = RUNS.parent

# variant -> (folder, MICROTAUR_VARIANT, control spine, fit json, flat teacher checkpoint)
VARIANTS = {
  "rigid": ("rigid", "rigid_microtaur", "0", "fit_rigid.json", "rigid_flat_fixed_m077"),
  "roll": ("active_twist", "active_twist_microtaur", "1", "fit_roll.json", "active_twist_077_motors"),
  "pitch": ("active_pitch", "active_pitch_microtaur", "1", "fit_pitch.json", "pitch_all_motors_fixed_v1"),
  "yaw": ("active_yaw", "active_yaw_microtaur", "1", "fit_yaw.json", "yaw_motros_fixed_flat"),
}

LEG_JOINTS = tuple(f"leg{i}_{m}_joint_act" for i in range(1, 5) for m in ("a", "e"))
SPINE_JOINT = "spine_joint_act"
PARAM_NAMES = {
  "cpg": ["freq_hz", "leg_scale", "spine_amp_deg", "spine_phase_deg", "spine_offset_deg"],
  "harmonic": ["freq_scale", "spine_amp_deg", "spine_phase_deg", "lr_swing_trim"],
  "policy": [],
}


def floats(text: str) -> list[float]:
  return [float(x) for x in text.split(",")] if text else []


parser = argparse.ArgumentParser()
parser.add_argument("variant", choices=sorted(VARIANTS))
parser.add_argument("mode", choices=("cpg", "harmonic", "policy"))
parser.add_argument("--out", required=False, default=None, help=".npz output")
parser.add_argument("--checkpoint", default=None, help="policy mode; default: flat teacher")
parser.add_argument("--cpg-env", action="store_true", help="policy: checkpoint is a CPG-RL policy")
parser.add_argument("--delay-steps", type=int, default=None, help="force the motor command delay (policy steps)")
parser.add_argument("--vx", type=float, default=0.15, help="command (policy, and yaw spine bias)")
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--settle", type=float, default=4.0)
parser.add_argument("--num-envs", type=int, default=8, help="policy mode")
parser.add_argument("--freq-hz", default="", help="cpg: gait frequencies; default fitted")
parser.add_argument("--leg-scale", default="1.0", help="cpg: stride amplitude scale (0-1.5)")
parser.add_argument("--spine-amp-deg", default="0")
parser.add_argument("--spine-phase-deg", default="0")
parser.add_argument("--spine-offset-deg", default="0")
parser.add_argument("--gait", default=None, help="harmonic: gait JSON (board format)")
parser.add_argument("--freq-scale", default="1.0", help="harmonic: gait frequency multipliers")
parser.add_argument("--h-spine-amp-deg", default="", help="harmonic: replace the spine by sines of these amplitudes")
parser.add_argument("--h-spine-phase-deg", default="0", help="harmonic: spine sine phase on the gait clock")
parser.add_argument("--lr-swing-trim", default="0",
                    help="harmonic: steering trim; right-leg swing x (1 + trim), left-leg swing x (1 - trim)")
parser.add_argument("--repeats", type=int, default=1, help="cpg/harmonic: envs per combination")
parser.add_argument("--hold-steps", type=int, default=1,
                    help="cpg/harmonic: new motor command only every N control steps (2 = a 25 Hz board loop)")
parser.add_argument("--board-start", action="store_true",
                    help="harmonic: start like the board player (hold stand 1 s, ramp in over 2 s); use with --settle 0")
parser.add_argument("--cmd-lowpass-hz", type=float, default=0.0,
                    help="cpg/harmonic: first-order low-pass on motor commands, like a servo motion profile (0 = off)")
parser.add_argument("--series", action="store_true", help="save full time series")
parser.add_argument("--video", default=None, help="render env 0 to this mp4 (uses 1 env)")
args = parser.parse_args()
if args.mode == "harmonic" and not args.gait:
  parser.error("harmonic mode needs --gait")

folder, model_name, control_spine, fit_name, teacher = VARIANTS[args.variant]
fit_path = RUNS / "cpg_rl" / fit_name
os.environ.update(
  MUJOCO_GL="egl",
  MICROTAUR_USE_JOYSTICK_COMMANDS="1",
  MICROTAUR_VARIANT=model_name,
  MICROTAUR_CONTROL_SPINE=control_spine,
  MICROTAUR_CURRICULUM_START_STEP="0",
  MICROTAUR_CPG="1" if args.mode == "cpg" or (args.mode == "policy" and args.cpg_env) else "0",
  MICROTAUR_CPG_FIT=str(fit_path),
)
if args.delay_steps is not None:
  # Read by env_cfgs at import; applies to the leg and spine command delays alike.
  os.environ["MICROTAUR_OLYMPUS_MIN_ACTION_DELAY"] = str(args.delay_steps)
  os.environ["MICROTAUR_OLYMPUS_MAX_ACTION_DELAY"] = str(args.delay_steps)
sys.path.insert(0, str(RUNS / folder / "src"))

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import mjlab.tasks  # noqa: E402,F401
import microtaur_velocity  # noqa: E402,F401
from microtaur_velocity.microtaur_constants import MICROTAUR_XML  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402

TASK = "Mjlab-Velocity-Flat-microtaur_velocity"
DEVICE = "cuda:0"
fit = json.loads(fit_path.read_text())
has_spine = control_spine == "1"

# ---------------------------------------------------------------------------
# parameter grid (cpg / harmonic modes)
# ---------------------------------------------------------------------------
if args.mode == "cpg":
  freqs = floats(args.freq_hz) or [fit["frequency_hz"]]
  grid = list(itertools.product(
    freqs, floats(args.leg_scale), floats(args.spine_amp_deg),
    floats(args.spine_phase_deg), floats(args.spine_offset_deg),
  ))
elif args.mode == "harmonic":
  grid = list(itertools.product(
    floats(args.freq_scale), floats(args.h_spine_amp_deg) or [math.nan], floats(args.h_spine_phase_deg),
    floats(args.lr_swing_trim),
  ))
else:
  grid = []
grid = [g for g in grid for _ in range(args.repeats)]
num_envs = len(grid) if grid else args.num_envs
if args.video:
  num_envs = 1
  grid = grid[:1]

env_cfg = load_env_cfg(TASK, play=True)
env_cfg.scene.num_envs = num_envs
env_cfg.sim.njmax = 256
env_cfg.sim.nconmax = 64
if args.video:
  env_cfg.viewer.width, env_cfg.viewer.height = 960, 540
  for sensor in env_cfg.scene.sensors or ():
    if hasattr(sensor, "debug_vis"):
      sensor.debug_vis = False
env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE, render_mode="rgb_array" if args.video else None)

robot = env.scene["robot"]
leg_term = env.action_manager.get_term("joint_pos")
spine_term = env.action_manager.get_term("spine_pos") if has_spine else None
cmd = env.command_manager.get_command("twist")
cmd_term = env.command_manager.get_term("twist")
dt = float(env.step_dt)
print(f"[openloop] action terms {env.action_manager.active_terms}, total dim {env.action_manager.total_action_dim}, "
      f"delay steps {sorted(set(leg_term._delay_steps.tolist()))}")


def ids_by_name(found, wanted):
  ids, names = found
  lookup = dict(zip(names, ids))
  return torch.tensor([int(lookup[n]) for n in wanted], device=DEVICE)


motor_names = LEG_JOINTS + ((SPINE_JOINT,) if has_spine else ())
joint_ids = ids_by_name(robot.find_joints(list(motor_names), preserve_order=True), motor_names)
act_ids = ids_by_name(robot.find_actuators(list(motor_names), preserve_order=True), motor_names)
mass_kg = float(mujoco.MjModel.from_xml_path(str(MICROTAUR_XML)).body_subtreemass[1])

if has_spine and args.mode != "policy":
  # Give the open-loop spine its full target range: pitch/roll scale the action,
  # yaw adds it as a residual around the (zero at zero yaw command) steering bias.
  if hasattr(spine_term.cfg, "residual_scale_rad"):
    spine_term.cfg.residual_scale_rad = float(spine_term.cfg.target_limit_rad)
  else:
    spine_term.cfg.action_scale_rad = float(spine_term.cfg.target_limit_rad)


def spine_scale() -> float:
  # Read every step: the command curriculum may rewrite the scale at runtime.
  if hasattr(spine_term.cfg, "residual_scale_rad"):
    return float(spine_term.cfg.residual_scale_rad)
  return min(float(spine_term.cfg.action_scale_rad), float(spine_term.cfg.target_limit_rad))


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------
clock = torch.zeros(num_envs, device=DEVICE)  # seconds since episode start (harmonic mode)
clock_cmd = clock.clone()  # clock value the current motor command was computed at
# Harmonic mode plays gaits through the RL's leg action term, which clamps to stand +-30 deg (the RL's own command
# range). These count how much of a gait that clamp cuts off, so the clamp can be reproduced on the board.
clip_samples = torch.zeros(num_envs, device=DEVICE)
clip_calls = torch.zeros(num_envs, device=DEVICE)
clip_max = torch.zeros(num_envs, device=DEVICE)
if args.mode == "cpg":
  p = torch.tensor(grid, device=DEVICE, dtype=torch.float32)
  f_env, leg_scale = p[:, 0], p[:, 1]
  spine_amp, spine_phase, spine_offset = (torch.deg2rad(p[:, i]) for i in (2, 3, 4))
  gain_f = float(os.environ.get("MICROTAUR_CPG_FREQUENCY_GAIN", "0.5"))
  gain_a = float(os.environ.get("MICROTAUR_CPG_AMPLITUDE_GAIN", "0.5"))
  f_nom = float(fit["frequency_hz"])
  base_action = torch.zeros(num_envs, env.action_manager.total_action_dim, device=DEVICE)
  legs = base_action[:, :12].view(num_envs, 4, 3)
  legs[:, :, 0] = ((leg_scale - 1.0) / gain_a)[:, None]
  legs[:, :, 1] = ((leg_scale - 1.0) / gain_a)[:, None]
  legs[:, :, 2] = ((f_env / f_nom - 1.0) / gain_f)[:, None]
  if base_action[:, :12].abs().max() > 1.0:
    raise SystemExit("frequency or leg scale outside what the CPG action allows (+/-50%)")

  def next_action() -> torch.Tensor:
    action = base_action.clone()
    if has_spine:
      # leg phases advance inside this step, so aim the spine at the advanced phase
      phase1 = leg_term.phase[:, 0] + 2.0 * math.pi * f_env * dt
      target = spine_offset + spine_amp * torch.sin(phase1 + spine_phase)
      action[:, 12] = torch.clamp(target / spine_scale(), -1.0, 1.0)
    return action
elif args.mode == "harmonic":
  gait = json.loads(Path(args.gait).read_text())
  if tuple(gait["joint_order"]) != motor_names:
    raise SystemExit(f"gait joint_order {gait['joint_order']} does not match {motor_names}")
  p = torch.tensor(grid, device=DEVICE, dtype=torch.float32)
  f_env = float(gait["frequency_hz"]) * p[:, 0]
  h_amp, h_phase = torch.deg2rad(p[:, 1]), torch.deg2rad(p[:, 2])
  offset = torch.tensor(gait["offset_rad"], device=DEVICE, dtype=torch.float32)
  cos_c = torch.tensor(gait["cos_rad"], device=DEVICE, dtype=torch.float32)  # (motors, K)
  sin_c = torch.tensor(gait["sin_rad"], device=DEVICE, dtype=torch.float32)
  harmonics = torch.arange(1, cos_c.shape[1] + 1, device=DEVICE, dtype=torch.float32)
  lr_trim = p[:, 3]
  # Five-bar mapping per leg: a = off_a + sign*(swing - lift), e = off_e + sign*(swing + lift).
  leg_sign = torch.tensor([1.0, -1.0, -1.0, 1.0], device=DEVICE)
  right_side = torch.tensor([1.0, -1.0, -1.0, 1.0], device=DEVICE)  # leg1 back-right, leg2 back-left, leg3 front-left, leg4 front-right

  def next_action() -> torch.Tensor:
    angle = 2.0 * math.pi * (f_env * clock)[:, None] * harmonics[None, :]
    target = offset + torch.cos(angle) @ cos_c.T + torch.sin(angle) @ sin_c.T
    # Steering trim: scale only the oscillating swing of each leg, so the posture (offsets) stays.
    wave = (target[:, :8] - offset[:8]).view(num_envs, 4, 2)
    swing = (wave[..., 0] + wave[..., 1]) / (2.0 * leg_sign) * (1.0 + lr_trim[:, None] * right_side)
    lift = (wave[..., 1] - wave[..., 0]) / (2.0 * leg_sign)
    legs = torch.stack([leg_sign * (swing - lift), leg_sign * (swing + lift)], -1).view(num_envs, 8)
    target = torch.cat([offset[:8] + legs, target[:, 8:]], 1)
    if has_spine:
      sine = offset[8] + h_amp * torch.sin(2.0 * math.pi * f_env * clock + h_phase)
      target[:, 8] = torch.where(torch.isnan(h_amp), target[:, 8], sine)
    if args.board_start:
      # Same start as board/open_loop_cpg_9dof.py: hold the stand pose for 1 s, then over 2 s scale the waves
      # 0 -> 1 and blend the offsets from stand to the gait offsets (stand + ramp * (target - stand)).
      ramp = torch.clamp((clock - 1.0) / 2.0, 0.0, 1.0)[:, None]
      stand_pose = torch.cat([leg_term._stand, torch.zeros(target.shape[1] - 8, device=DEVICE)])
      target = stand_pose + ramp * (target - stand_pose)
    action = torch.zeros(num_envs, env.action_manager.total_action_dim, device=DEVICE)
    # Direct leg action term: requested = stand + action_scale * action, with the action clamped to +-1.
    leg_action = (target[:, :8] - leg_term._stand) / float(leg_term.cfg.action_scale_rad)
    beyond = (leg_action.abs() - 1.0).clamp(min=0.0) * float(leg_term.cfg.action_scale_rad)
    clip_samples.add_((beyond > 0).float().sum(1))
    clip_calls.add_(8.0)
    clip_max.copy_(torch.maximum(clip_max, beyond.amax(1)))
    action[:, :8] = torch.clamp(leg_action, -1.0, 1.0)
    if has_spine:
      action[:, 8] = torch.clamp(target[:, 8] / spine_scale(), -1.0, 1.0)
    return action
else:
  agent_cfg = load_rl_cfg(TASK)
  agent_cfg.clip_actions = None
  vec = RslRlVecEnvWrapper(env, clip_actions=None)
  runner = load_runner_cls(TASK)(vec, asdict(agent_cfg), device=DEVICE)
  ckpt = args.checkpoint or str(SPINES_HELP / "old_distilling" / teacher / "model_4499.pt")
  runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=DEVICE)
  policy = runner.get_inference_policy(device=DEVICE)

# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------
steps = int(round(args.seconds / dt))
settle = int(round(args.settle / dt))
env.reset()
obs = vec.get_observations() if args.mode == "policy" else None

series = {k: [] for k in (
  "imu_gravity", "imu_gyro", "q", "qd", "tau", "target", "requested", "safe", "filter_correction", "phase_leg1", "action",
)}
truth = {k: [] for k in ("pos", "yaw", "vel_b")}
falls = torch.zeros(num_envs, device=DEVICE)
writer = imageio.get_writer(args.video, fps=round(1 / dt), codec="libx264", quality=8) if args.video else None


def yaw_of(quat):
  w, x, y, z = quat.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


for k in range(steps):
  cmd[:, 0] = args.vx
  cmd[:, 1] = 0.0
  cmd[:, 2] = 0.0
  cmd_term.time_left.fill_(1.0e9)
  if args.mode == "policy":
    with torch.no_grad():
      action = policy(obs)
    obs, _, dones, _ = vec.step(action)
  else:
    if k % args.hold_steps == 0:
      fresh = next_action()
      clock_cmd = clock.clone()
      if args.cmd_lowpass_hz <= 0.0 or k == 0:
        action = fresh
    if args.cmd_lowpass_hz > 0.0 and k > 0:
      # Actions are affine in the joint targets, so smoothing actions smooths the targets.
      action = action + (1.0 - math.exp(-2.0 * math.pi * args.cmd_lowpass_hz * dt)) * (fresh - action)
    _, _, terminated, truncated, _ = env.step(action)
    dones = terminated | truncated
    clock = torch.where(dones, torch.zeros_like(clock), clock + dt)
  if spine_term is not None and args.mode != "policy":
    # the robot writes all 9 servos in one packet, so give the spine the legs' latency
    spine_term._delay_steps.copy_(leg_term._delay_steps)
  if k >= settle:
    falls += dones.float()

  # requested -> safety filter -> safe -> 1-2 step delay -> target (applied to the motors).
  # The board has no safety filter, so a deployable gait needs filter_correction ~ 0.
  target = leg_term._applied_targets[:, :8]
  requested = leg_term._requested_targets[:, :8]
  safe = leg_term._safe_targets[:, :8]
  filter_correction = leg_term._filter_correction[:, :8]
  if spine_term is not None:
    target = torch.cat([target, spine_term._applied_targets], 1)
    requested = torch.cat([requested, spine_term._requested_targets], 1)
    safe = torch.cat([safe, spine_term._safe_targets], 1)
    filter_correction = torch.cat([filter_correction, spine_term._filter_correction], 1)
  if hasattr(leg_term, "phase"):
    phase_leg1 = leg_term.phase[:, 0]
  elif args.mode == "harmonic":
    phase_leg1 = torch.remainder(2.0 * math.pi * f_env * clock_cmd, 2.0 * math.pi)
  else:
    phase_leg1 = torch.zeros(num_envs, device=DEVICE)
  row = {
    "imu_gravity": robot.data.projected_gravity_b,
    "imu_gyro": robot.data.root_link_ang_vel_b,
    "q": robot.data.joint_pos[:, joint_ids],
    "qd": robot.data.joint_vel[:, joint_ids],
    "tau": robot.data.actuator_force[:, act_ids],
    "target": target,
    "requested": requested,
    "safe": safe,
    "filter_correction": filter_correction,
    "phase_leg1": phase_leg1,
    "action": action,
  }
  for key, value in row.items():
    series[key].append(value.detach().float().clone())
  truth["pos"].append((robot.data.root_link_pos_w - env.scene.env_origins).detach().clone())
  truth["yaw"].append(yaw_of(robot.data.root_link_quat_w).detach().clone())
  truth["vel_b"].append(robot.data.root_link_lin_vel_b.detach().clone())
  if writer is not None:
    writer.append_data(env.render())

if writer is not None:
  writer.close()
delay_steps = leg_term._delay_steps.detach().cpu().numpy()
spine_scale_end = spine_scale() if has_spine else 0.0
env.close()

# ---------------------------------------------------------------------------
# per-env walking metrics (after settle)
# ---------------------------------------------------------------------------
S = {k: torch.stack(v)[settle:] for k, v in series.items()}  # (T, N, ...)
pos = torch.stack(truth["pos"])[settle:]
yaw = torch.stack(truth["yaw"])[settle:]
vel_b = torch.stack(truth["vel_b"])[settle:]
T = (pos.shape[0] - 1) * dt  # positions are sampled at both ends

heading0 = yaw[0]
disp = pos[-1, :, :2] - pos[0, :, :2]
forward = (disp[:, 0] * torch.cos(heading0) + disp[:, 1] * torch.sin(heading0)) / T
lateral = (-disp[:, 0] * torch.sin(heading0) + disp[:, 1] * torch.cos(heading0)) / T
yaw_steps = torch.remainder(yaw[1:] - yaw[:-1] + math.pi, 2 * math.pi) - math.pi
heading_drift = torch.rad2deg(yaw_steps.sum(0))
grav = S["imu_gravity"]
tilt = torch.rad2deg(torch.acos(torch.clamp(-grav[..., 2], -1.0, 1.0)))
power = torch.clamp(S["tau"] * S["qd"], min=0.0).sum(-1)  # positive mechanical power, W
# Per metre walked along the body axis, so a curving path is not penalized as "less efficient".
v_ok = torch.clamp(vel_b[..., 0].mean(0), min=1e-3)
cot = power.mean(0) / (mass_kg * 9.81 * v_ok)
target_speed = (S["target"][1:] - S["target"][:-1]).abs() / dt

metrics = {
  "forward_m_s": forward,
  "lateral_m_s": lateral,
  "heading_drift_deg": heading_drift,
  "body_vx_mean_m_s": vel_b[..., 0].mean(0),
  "falls": falls,
  "imu_tilt_mean_deg": tilt.mean(0),
  "imu_tilt_max_deg": tilt.amax(0),
  "gyro_rms_rad_s": S["imu_gyro"].pow(2).mean((0, 2)).sqrt(),
  "leg_torque_rms_nm": S["tau"][..., :8].pow(2).mean((0, 2)).sqrt(),
  "leg_torque_peak_nm": S["tau"][..., :8].abs().amax((0, 2)),
  "power_w": power.mean(0),
  "cot": cot,
  "leg_target_speed_peak_rad_s": target_speed[..., :8].amax((0, 2)),
  "leg_filter_correction_deg": torch.rad2deg(S["filter_correction"][..., :8].abs().mean((0, 2))),
  "leg_filter_correction_max_deg": torch.rad2deg(S["filter_correction"][..., :8].abs().amax((0, 2))),
  "leg_clip_share": clip_samples / clip_calls.clamp(min=1.0),
  "leg_clip_max_deg": torch.rad2deg(clip_max),
}
if has_spine:
  sq = S["q"][..., 8]
  metrics.update(
    spine_mean_deg=torch.rad2deg(sq.mean(0)),
    spine_amp_deg=torch.rad2deg((torch.quantile(sq, 0.95, dim=0) - torch.quantile(sq, 0.05, dim=0)) / 2),
    spine_target_amp_deg=torch.rad2deg(
      (torch.quantile(S["target"][..., 8], 0.95, dim=0) - torch.quantile(S["target"][..., 8], 0.05, dim=0)) / 2
    ),
    spine_torque_rms_nm=S["tau"][..., 8].pow(2).mean(0).sqrt(),
    spine_torque_peak_nm=S["tau"][..., 8].abs().amax(0),
  )

out = {f"m_{k}": v.cpu().numpy() for k, v in metrics.items()}
out["params"] = np.asarray(grid, dtype=np.float32) if grid else np.zeros((num_envs, 0), np.float32)
out["param_names"] = np.asarray(PARAM_NAMES[args.mode])
out["motor_names"] = np.asarray(motor_names)
out["delay_steps"] = delay_steps
out["dt"] = dt
if args.series:
  for key, value in S.items():
    out[f"s_{key}"] = value.cpu().numpy()
  out["s_truth_pos"] = pos.cpu().numpy()
  out["s_truth_vel_b"] = vel_b.cpu().numpy()
meta = dict(variant=args.variant, mode=args.mode, fit=str(fit_path), mass_kg=mass_kg, vx_command=args.vx,
            seconds=args.seconds, settle=args.settle, imu="root body projected gravity + gyro",
            checkpoint=ckpt if args.mode == "policy" else None, gait=args.gait, delay_steps_forced=args.delay_steps,
            hold_steps=args.hold_steps, cmd_lowpass_hz=args.cmd_lowpass_hz,
            spine_scale_rad=spine_scale_end, xml=str(MICROTAUR_XML))
out["meta"] = json.dumps(meta)
if args.out:
  Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  np.savez(args.out, **out)

# ---------------------------------------------------------------------------
# summary table
# ---------------------------------------------------------------------------
cols = ["forward_m_s", "falls", "imu_tilt_max_deg", "cot", "leg_target_speed_peak_rad_s", "leg_filter_correction_deg"]
if has_spine:
  cols += ["spine_amp_deg", "spine_mean_deg", "spine_torque_peak_nm"]
rows = []
for i in range(num_envs):
  rows.append([*(grid[i] if grid else ()), *(float(metrics[c][i]) for c in cols)])
header = PARAM_NAMES[args.mode] if grid else []
header = [h[:10] for h in header] + cols
order = sorted(range(num_envs), key=lambda i: (float(metrics["falls"][i]) > 0, -float(metrics["forward_m_s"][i])))
print(" | ".join(f"{h:>10.10s}" for h in header))
for i in order[:40]:
  print(" | ".join(f"{v:10.3f}" for v in rows[i]))
print(f"[openloop] {args.variant} {args.mode}: {num_envs} envs, {args.seconds:.0f} s, mass {mass_kg:.3f} kg, "
      f"spine scale {math.degrees(spine_scale_end):.1f} deg"
      + (f", saved {args.out}" if args.out else "") + (f", video {args.video}" if args.video else ""))
