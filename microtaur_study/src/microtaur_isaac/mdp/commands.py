"""Forward-velocity / yaw-rate command "twist" and its curriculum (IsaacLab).
Port of microtaur_rigid/commands.py.

Microtaur has no ab/ad joint, so lateral velocity is always commanded as zero.
Heading control is off (yaw rate is sampled directly) and there are no
standing envs.

Terrain C (the step course) is walked straight: envs whose entry in the bool
mask env.microtaur_zero_yaw_mask ([num_envs], set by env_cfg) is True get
their yaw-rate command zeroed every time the command is resampled, inside the
command term, so observation and reward see the same zeroed command.
zero_yaw_commands() does the same on demand for an explicit mask.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass

from microtaur_common.task_params import COMMAND_NAME, COMMAND_STAGES, RESAMPLE_S  # noqa: F401

from .observations import ROBOT

if TYPE_CHECKING:
  from isaaclab.envs import ManagerBasedRLEnv

ZERO_YAW_MASK_ATTR = "microtaur_zero_yaw_mask"


def zero_yaw_commands(env: ManagerBasedRLEnv, command_name: str, mask: torch.Tensor) -> None:
  """Set the yaw-rate command to 0 for the envs where mask ([num_envs] bool) is True."""
  term = env.command_manager.get_term(command_name)
  term.vel_command_b[mask.to(device=term.vel_command_b.device, dtype=torch.bool), 2] = 0.0


class MicrotaurVelocityCommand(UniformVelocityCommand):
  """UniformVelocityCommand plus: no standing envs when rel_standing_envs <= 0
  (the parent's `uniform <= 0.0` test can still fire on an exact 0 draw), and
  yaw zeroed for masked envs after every resample."""

  cfg: MicrotaurVelocityCommandCfg

  def _resample_command(self, env_ids: Sequence[int]):
    super()._resample_command(env_ids)
    if self.cfg.rel_standing_envs <= 0.0:
      self.is_standing_env[env_ids] = False
    attr = self.cfg.zero_yaw_mask_attr
    mask = getattr(self._env, attr, None) if attr else None
    if mask is not None:
      ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long) if not isinstance(env_ids, slice) \
        else torch.arange(self.num_envs, device=self.device)[env_ids]
      m = mask.to(device=self.device, dtype=torch.bool)[ids]
      self.vel_command_b[ids[m], 2] = 0.0


@configclass
class MicrotaurVelocityCommandCfg(UniformVelocityCommandCfg):
  class_type: type = MicrotaurVelocityCommand
  zero_yaw_mask_attr: str | None = ZERO_YAW_MASK_ATTR
  """Name of an env attribute holding a [num_envs] bool mask of envs whose yaw
  command is forced to 0 (terrain C). Absent attribute or None: no masking."""


def apply_stage(cmd: UniformVelocityCommandCfg, stage: dict) -> None:
  cmd.ranges.lin_vel_x = tuple(stage["lin_vel_x"])
  cmd.ranges.lin_vel_y = (0.0, 0.0)
  cmd.ranges.ang_vel_z = tuple(stage["ang_vel_z"])
  cmd.rel_standing_envs = float(stage["standing"])


def make_twist_command_cfg(play: bool = False, debug_vis: bool = False) -> MicrotaurVelocityCommandCfg:
  """The "twist" command at the first curriculum stage (the final one for play)."""
  stage = COMMAND_STAGES[-1] if play else COMMAND_STAGES[0]
  cmd = MicrotaurVelocityCommandCfg(
    asset_name=ROBOT,
    resampling_time_range=RESAMPLE_S,
    heading_command=False,
    rel_heading_envs=0.0,
    rel_standing_envs=0.0,
    debug_vis=debug_vis,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=None
    ),
  )
  apply_stage(cmd, stage)
  return cmd


def command_curriculum(
  env: ManagerBasedRLEnv, env_ids, command_name: str, stages: Sequence[dict], start_step: int = 0
) -> dict:
  """Apply the last stage whose "step" <= start_step + common_step_counter
  (policy steps); before the first stage's step, use the first given stage.
  New ranges take effect at each env's next resample. IsaacLab calls this
  from _reset_idx, i.e. whenever some env resets."""
  del env_ids
  step = start_step + int(getattr(env, "common_step_counter", 0))
  idx = max((i for i, s in enumerate(stages) if step >= s["step"]), default=0)
  apply_stage(env.command_manager.get_term(command_name).cfg, stages[idx])
  as_t = lambda v: torch.tensor(float(v))  # noqa: E731
  return {
    "stage": as_t(idx),
    "lin_vel_x_max": as_t(stages[idx]["lin_vel_x"][1]),
    "ang_vel_z_max": as_t(stages[idx]["ang_vel_z"][1]),
    "standing_fraction": as_t(stages[idx]["standing"]),
  }


def make_command_curriculum_term(play: bool = False, start_step: int = 0) -> CurriculumTermCfg:
  stages = (COMMAND_STAGES[-1],) if play else COMMAND_STAGES
  return CurriculumTermCfg(
    func=command_curriculum,
    params={"command_name": COMMAND_NAME, "stages": stages, "start_step": start_step},
  )
