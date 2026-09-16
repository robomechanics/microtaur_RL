"""Fit a per-leg CPG to the gait of a trained flat-terrain Microtaur policy.

Rolls the flat policy out on flat ground at a fixed command, converts the motor
targets it produces into the repo's five-bar (swing, lift) coordinates

    a = STAND_A + sign * (swing - lift)
    e = STAND_E + sign * (swing + lift)

and fits, per leg, a single sinusoid to each of swing and lift:

    swing_i(t) = c_swing_i + A_swing_i * sin(2*pi*f*t + phi_i)
    lift_i(t)  = c_lift_i  + A_lift_i  * sin(2*pi*f*t + phi_i + psi_i)

One shared frequency f (the gait frequency) is estimated from the swing signals;
the per-leg phases phi_i give the gait pattern (trot = 0, pi, 0, pi) and psi_i the
lift-vs-swing lag. These numbers become the nominal CPG the RL policy modulates.

Usage (variant env vars must already be set, as in the training scripts):
    python fit_cpg_from_flat.py <checkpoint.pt> <out.json> [--vx 0.15] [--seconds 20]
"""

import argparse
import json
import os
from dataclasses import asdict

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MICROTAUR_USE_JOYSTICK_COMMANDS", "1")

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import microtaur_velocity  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Flat-microtaur_velocity"
STAND_A = np.array([0.45, -0.45, -0.45, 0.45])
STAND_E = np.array([-0.45, 0.45, 0.45, -0.45])
LEG_SIGNS = np.array([+1.0, -1.0, -1.0, +1.0])


def targets_to_swing_lift(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """(T, 8) motor targets [a1, e1, a2, e2, ...] -> (T, 4) swing and lift."""
  a = targets[:, 0::2] - STAND_A
  e = targets[:, 1::2] - STAND_E
  swing = (a + e) / (2.0 * LEG_SIGNS)
  lift = (e - a) / (2.0 * LEG_SIGNS)
  return swing, lift


def dominant_frequency(swing: np.ndarray, lift: np.ndarray, t: np.ndarray) -> float:
  """Gait frequency: the one whose single-sinusoid fit explains the most variance.

  A power-spectrum peak picks up the policy's high-frequency jitter instead of the
  stride, so scan the plausible band and keep the frequency that fits best.
  """
  best_f, best_score = 0.0, -np.inf
  for f in np.arange(0.4, 5.0, 0.005):
    score = 0.0
    for signal in (swing, lift):
      for i in range(signal.shape[1]):
        amp, phi, c = fit_sinusoid(signal[:, i], t, f)
        score += r_squared(signal[:, i], c + amp * np.sin(2 * np.pi * f * t + phi))
    if score > best_score:
      best_f, best_score = float(f), score
  return best_f


def fit_sinusoid(y: np.ndarray, t: np.ndarray, f: float) -> tuple[float, float, float]:
  """Least-squares fit of c + A*sin(2*pi*f*t + phi); returns (A, phi, c)."""
  design = np.stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t), np.ones_like(t)], axis=1)
  (a_sin, a_cos, c), *_ = np.linalg.lstsq(design, y, rcond=None)
  return float(np.hypot(a_sin, a_cos)), float(np.arctan2(a_cos, a_sin)), float(c)


def r_squared(y: np.ndarray, fit: np.ndarray) -> float:
  ss_res = float(((y - fit) ** 2).sum())
  ss_tot = float(((y - y.mean()) ** 2).sum())
  return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint")
  parser.add_argument("out")
  parser.add_argument("--vx", type=float, default=0.15)
  parser.add_argument("--yaw", type=float, default=0.0)
  parser.add_argument("--seconds", type=float, default=20.0)
  parser.add_argument("--settle", type=float, default=4.0, help="discard this much before fitting")
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--clip-actions", type=float, default=None)
  args = parser.parse_args()

  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = args.num_envs
  agent_cfg = load_rl_cfg(TASK)
  agent_cfg.clip_actions = args.clip_actions

  device = "cuda:0"
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(TASK)(vec_env, asdict(agent_cfg), device=device)
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  action_term = env.action_manager.get_term("joint_pos")
  command = env.command_manager.get_command("twist")
  cmd_term = env.command_manager.get_term("twist")
  obs = vec_env.get_observations()

  dt = env.step_dt
  targets = []
  for step in range(int(args.seconds / dt)):
    command[:, 0] = args.vx
    command[:, 1] = 0.0
    command[:, 2] = args.yaw
    if hasattr(cmd_term, "time_left"):
      cmd_term.time_left.fill_(1.0e9)
    with torch.no_grad():
      actions = policy(obs)
    obs, _, _, _ = vec_env.step(actions)
    targets.append(action_term._applied_targets[:, :8].detach().cpu().numpy())
  env.close()

  data = np.stack(targets)  # (T, num_envs, 8)
  skip = int(args.settle / dt)
  data = data[skip:]
  t = np.arange(len(data)) * dt

  # Fit each environment separately, then average (phases via circular mean).
  per_env = []
  for e in range(data.shape[1]):
    swing, lift = targets_to_swing_lift(data[:, e, :])
    f = dominant_frequency(swing, lift, t)
    legs = []
    for i in range(4):
      a_s, phi, c_s = fit_sinusoid(swing[:, i], t, f)
      a_l, phi_l, c_l = fit_sinusoid(lift[:, i], t, f)
      fit_s = c_s + a_s * np.sin(2 * np.pi * f * t + phi)
      fit_l = c_l + a_l * np.sin(2 * np.pi * f * t + phi_l)
      legs.append({
        "swing_amp": a_s, "swing_offset": c_s, "phase": phi, "swing_r2": r_squared(swing[:, i], fit_s),
        "lift_amp": a_l, "lift_offset": c_l, "lift_lag": float(np.angle(np.exp(1j * (phi_l - phi)))),
        "lift_r2": r_squared(lift[:, i], fit_l),
      })
    per_env.append({"frequency_hz": f, "legs": legs})

  def mean_of(key, leg):
    return float(np.mean([p["legs"][leg][key] for p in per_env]))

  def circmean_of(key, leg):
    angles = np.array([p["legs"][leg][key] for p in per_env])
    return float(np.angle(np.exp(1j * angles).mean()))

  # Phases are reported relative to leg 1 so the gait pattern is readable.
  phase_abs = [circmean_of("phase", i) for i in range(4)]
  fit = {
    "checkpoint": args.checkpoint,
    "command": {"vx": args.vx, "yaw": args.yaw},
    "frequency_hz": float(np.median([p["frequency_hz"] for p in per_env])),
    "legs": [
      {
        "swing_amp": mean_of("swing_amp", i),
        "swing_offset": mean_of("swing_offset", i),
        "lift_amp": mean_of("lift_amp", i),
        "lift_offset": mean_of("lift_offset", i),
        "phase_offset": float(np.angle(np.exp(1j * (phase_abs[i] - phase_abs[0])))),
        "lift_lag": circmean_of("lift_lag", i),
        "swing_r2": mean_of("swing_r2", i),
        "lift_r2": mean_of("lift_r2", i),
      }
      for i in range(4)
    ],
  }
  np.savez(args.out.replace(".json", "_signals.npz"), targets=data, dt=dt)
  with open(args.out, "w") as f:
    json.dump(fit, f, indent=2)
  print(json.dumps(fit, indent=2))


if __name__ == "__main__":
  main()
