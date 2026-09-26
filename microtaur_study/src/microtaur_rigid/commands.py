"""Forward-velocity / yaw-rate commands and their curriculum.

Microtaur has no ab/ad joint, so lateral velocity is always commanded as zero.
"""

from __future__ import annotations

import torch
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

COMMAND_NAME = "twist"
RESAMPLE_S = (6.0, 10.0)

# Stages are keyed by env.common_step_counter (policy steps across all envs).
# Note: yaw commands stay at zero for the first 25k steps.
COMMAND_STAGES = (
  {"step": 0, "lin_vel_x": (0.10, 0.18), "ang_vel_z": (0.00, 0.00), "standing": 0.00},
  {"step": 25_000, "lin_vel_x": (0.08, 0.20), "ang_vel_z": (-0.10, 0.10), "standing": 0.05},
  {"step": 50_000, "lin_vel_x": (0.08, 0.20), "ang_vel_z": (-0.15, 0.15), "standing": 0.05},
  {"step": 70_000, "lin_vel_x": (0.08, 0.20), "ang_vel_z": (-0.20, 0.20), "standing": 0.10},
  {"step": 90_000, "lin_vel_x": (0.08, 0.20), "ang_vel_z": (-0.25, 0.25), "standing": 0.10},
)


def _apply_stage(cmd: UniformVelocityCommandCfg, stage: dict) -> None:
  cmd.ranges.lin_vel_x = tuple(stage["lin_vel_x"])
  cmd.ranges.lin_vel_y = (0.0, 0.0)
  cmd.ranges.ang_vel_z = tuple(stage["ang_vel_z"])
  cmd.rel_standing_envs = float(stage["standing"])


def command_curriculum(env, env_ids, command_name: str, stages: tuple[dict, ...], start_step: int = 0):
  del env_ids
  step = start_step + int(getattr(env, "common_step_counter", 0))
  # Play passes only the final stage (step 90k); before that step, still use it.
  idx = max((i for i, s in enumerate(stages) if step >= s["step"]), default=0)
  _apply_stage(env.command_manager.get_term(command_name).cfg, stages[idx])
  as_t = lambda v: torch.tensor(float(v), device=env.device)
  return {
    "stage": as_t(idx),
    "lin_vel_x_max": as_t(stages[idx]["lin_vel_x"][1]),
    "ang_vel_z_max": as_t(stages[idx]["ang_vel_z"][1]),
    "standing_fraction": as_t(stages[idx]["standing"]),
  }


def configure_commands(cfg, play: bool, start_step: int = 0) -> None:
  cmd = cfg.commands[COMMAND_NAME]
  if not isinstance(cmd, UniformVelocityCommandCfg):
    raise TypeError(f"Command '{COMMAND_NAME}' must be UniformVelocityCommandCfg")
  cmd.viz.z_offset = 0.12
  cmd.heading_command = False
  cmd.rel_heading_envs = 0.0
  cmd.rel_forward_envs = 0.0
  cmd.init_velocity_prob = 0.0
  cmd.ranges.heading = None
  cmd.resampling_time_range = RESAMPLE_S

  stages = (COMMAND_STAGES[-1],) if play else COMMAND_STAGES
  _apply_stage(cmd, stages[0])
  cfg.curriculum.pop("command_vel", None)
  cfg.curriculum["command_ranges"] = CurriculumTermCfg(
    func=command_curriculum,
    params={"command_name": COMMAND_NAME, "stages": stages, "start_step": start_step},
  )
