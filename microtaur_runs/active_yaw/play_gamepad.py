"""Play the trained Microtaur velocity policy with an 8BitDo gamepad."""

from __future__ import annotations

# Must be set before importing the Microtaur task/env module.
import os
os.environ["MICROTAUR_USE_JOYSTICK_COMMANDS"] = "1"

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

try:
  import pygame
except ImportError as exc:
  raise RuntimeError(
    "pygame is not installed in the Microtaur venv. Run: uv add pygame"
  ) from exc

import microtaur_velocity  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer import NativeMujocoViewer

from microtaur_velocity.env_cfgs import set_joystick_twist_command


DEFAULT_TASK = "Mjlab-Velocity-Flat-microtaur_velocity"
DEFAULT_CHECKPOINT = (
  "logs/rsl_rl/microtaur_minitaur_velocity/"
  "olympus_v5_train_on_rough/model_2999.pt"
)

DEFAULT_DEADZONE = 0.15
DEFAULT_FORWARD_MIN = 0.08
DEFAULT_FORWARD_MAX = 0.20
DEFAULT_YAW_MAX = 0.30


def _remove_deadzone(value: float, deadzone: float) -> float:
  value = float(value)
  if abs(value) <= deadzone:
    return 0.0
  magnitude = (abs(value) - deadzone) / (1.0 - deadzone)
  magnitude = max(0.0, min(magnitude, 1.0))
  return math.copysign(magnitude, value)


class EightBitDoController:
  """Read the tested left-stick mapping from the 8BitDo 2.4 GHz controller."""

  def __init__(
    self,
    joystick_index: int,
    deadzone: float,
    forward_min: float,
    forward_max: float,
    yaw_max: float,
  ) -> None:
    if not 0.0 <= deadzone < 1.0:
      raise ValueError("deadzone must be in [0, 1).")
    if not 0.0 <= forward_min <= forward_max:
      raise ValueError("Require 0 <= forward_min <= forward_max.")
    if yaw_max < 0.0:
      raise ValueError("yaw_max must be non-negative.")

    self.deadzone = float(deadzone)
    self.forward_min = float(forward_min)
    self.forward_max = float(forward_max)
    self.yaw_max = float(yaw_max)

    pygame.init()
    pygame.joystick.init()
    pygame.event.pump()

    count = pygame.joystick.get_count()
    if count == 0:
      raise RuntimeError(
        "No gamepad detected. Connect the 8BitDo receiver/controller first."
      )
    if not 0 <= joystick_index < count:
      raise ValueError(
        f"joystick_index={joystick_index} is invalid; detected {count} controller(s)."
      )

    self.joystick = pygame.joystick.Joystick(joystick_index)
    self.joystick.init()

    if self.joystick.get_numaxes() < 2:
      raise RuntimeError(
        f"Controller exposes only {self.joystick.get_numaxes()} axes; "
        "Axis 0 and Axis 1 are required."
      )

    print(
      f"[GAMEPAD] Connected: {self.joystick.get_name()} "
      f"(index {joystick_index}, axes={self.joystick.get_numaxes()})"
    )
    print("[GAMEPAD] UP=forward, LEFT/RIGHT=yaw, DOWN=no reverse.")

  def read(self) -> tuple[float, float]:
    pygame.event.pump()

    # Tested mapping:
    # Axis 0: left=-1, right=+1
    # Axis 1: up=-1, down=+1
    stick_x = float(self.joystick.get_axis(0))
    stick_y = float(self.joystick.get_axis(1))

    # Forward-only.
    forward_input = max(0.0, -stick_y)
    if forward_input <= self.deadzone:
      vx = 0.0
    else:
      normalized = (
        (forward_input - self.deadzone)
        / (1.0 - self.deadzone)
      )
      normalized = max(0.0, min(normalized, 1.0))
      vx = (
        self.forward_min
        + normalized * (self.forward_max - self.forward_min)
      )

    # Left -> +yaw, right -> -yaw.
    yaw_input = _remove_deadzone(stick_x, self.deadzone)
    yaw = -yaw_input * self.yaw_max

    return vx, yaw

  def close(self) -> None:
    try:
      self.joystick.quit()
    finally:
      pygame.joystick.quit()
      pygame.quit()


class GamepadPolicy:
  """Inject joystick twist, recompute observations, then run learned policy."""

  def __init__(self, env, learned_policy, controller: EightBitDoController):
    self.env = env
    self.learned_policy = learned_policy
    self.controller = controller
    self._last_print_time = 0.0

  def __call__(self, stale_obs):
    # MJLab's viewer computes observations before calling policy().
    # Ignore those stale observations, replace the command, then recompute.
    del stale_obs

    vx, yaw = self.controller.read()

    set_joystick_twist_command(
      self.env,
      forward_velocity=vx,
      yaw_velocity=yaw,
      command_name="twist",
    )

    # Fresh actor observation contains THIS cycle's joystick command.
    obs = self.env.get_observations()

    now = time.monotonic()
    if now - self._last_print_time >= 0.25:
      print(
        f"\\r[GAMEPAD] vx={vx:+.3f} m/s   yaw={yaw:+.3f} rad/s   ",
        end="",
        flush=True,
      )
      self._last_print_time = now

    return self.learned_policy(obs)

  def reset(self) -> None:
    pass


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Play the trained Microtaur policy with an 8BitDo gamepad."
  )
  parser.add_argument("--task", default=DEFAULT_TASK)
  parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
  parser.add_argument("--device", default=None)
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--joystick-index", type=int, default=0)
  parser.add_argument("--deadzone", type=float, default=DEFAULT_DEADZONE)
  parser.add_argument("--forward-min", type=float, default=DEFAULT_FORWARD_MIN)
  parser.add_argument("--forward-max", type=float, default=DEFAULT_FORWARD_MAX)
  parser.add_argument("--yaw-max", type=float, default=DEFAULT_YAW_MAX)
  return parser.parse_args()


def main() -> None:
  args = _parse_args()

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  checkpoint = Path(args.checkpoint).expanduser().resolve()
  if not checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

  controller = EightBitDoController(
    joystick_index=args.joystick_index,
    deadzone=args.deadzone,
    forward_min=args.forward_min,
    forward_max=args.forward_max,
    yaw_max=args.yaw_max,
  )

  env = None
  try:
    env_cfg = load_env_cfg(args.task, play=True)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = int(args.num_envs)

    env = ManagerBasedRlEnv(
      cfg=env_cfg,
      device=device,
      render_mode=None,
    )
    env = RslRlVecEnvWrapper(
      env,
      clip_actions=agent_cfg.clip_actions,
    )

    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(
      env,
      asdict(agent_cfg),
      device=device,
    )
    runner.load(
      str(checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    learned_policy = runner.get_inference_policy(device=device)

    policy = GamepadPolicy(
      env=env,
      learned_policy=learned_policy,
      controller=controller,
    )

    print(f"[INFO] Checkpoint: {checkpoint}")
    print("[INFO] Joystick command mode: ON")
    print("[INFO] Center stick -> [vx=0, yaw=0]")
    print("[INFO] Close viewer or Ctrl+C to exit.")

    NativeMujocoViewer(env, policy).run()

  finally:
    print()
    controller.close()
    if env is not None:
      env.close()


if __name__ == "__main__":
  main()