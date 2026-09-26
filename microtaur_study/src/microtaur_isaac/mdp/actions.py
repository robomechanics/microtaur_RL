"""IsaacLab wrapper around the shared walk action (microtaur_common.walk_action).

The joints are resolved with preserve_order=True, so IsaacLab hands them over
in canonical LEG_JOINT_NAMES order and no reordering is needed. The encoder
bias (env.microtaur_encoder_bias) is subtracted from the target before it is
written, as mjlab's JointPositionAction does.
"""

from __future__ import annotations

from dataclasses import MISSING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from microtaur_common.robot_constants import LEG_JOINT_NAMES
from microtaur_common.walk_action import WalkActionCore, WalkActionParams


class MicrotaurWalkAction(ActionTerm):
  cfg: "MicrotaurWalkActionCfg"
  _asset: Articulation

  def __init__(self, cfg: "MicrotaurWalkActionCfg", env):
    super().__init__(cfg, env)
    self._walk_env = env
    ids, names = self._asset.find_joints(list(LEG_JOINT_NAMES), preserve_order=True)
    if tuple(names) != tuple(LEG_JOINT_NAMES):
      raise ValueError(f"Expected the eight leg motors in canonical order, got {names}")
    self._joint_ids = ids
    params = WalkActionParams(target_gain_range=tuple(cfg.target_gain_range), target_bias_rad=float(cfg.target_bias_rad))
    self.core = WalkActionCore(params, self.num_envs, self.device)
    self._raw = torch.zeros(self.num_envs, 8, device=self.device)
    self._processed = torch.zeros_like(self._raw)

  @property
  def action_dim(self) -> int:
    return 8

  @property
  def raw_actions(self) -> torch.Tensor:
    return self._raw

  @property
  def processed_actions(self) -> torch.Tensor:
    return self._processed

  def process_actions(self, actions: torch.Tensor):
    d = self._asset.data
    raw, applied = self.core.step(
      actions, d.joint_pos[:, self._joint_ids], d.joint_vel[:, self._joint_ids], self._walk_env.episode_length_buf <= 1
    )
    self._raw[:] = raw
    self._processed[:] = applied

  def apply_actions(self):
    bias = getattr(self._walk_env, "microtaur_encoder_bias", None)
    target = self._processed if bias is None else self._processed - bias
    self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)

  def reset(self, env_ids=None) -> None:
    self._raw[env_ids] = 0.0


@configclass
class MicrotaurWalkActionCfg(ActionTermCfg):
  class_type: type = MicrotaurWalkAction
  asset_name: str = "robot"
  # Per-episode command-side uncertainty (sim2real stage); the rest of the
  # pipeline uses the WalkActionParams defaults shared with mjlab.
  target_gain_range: tuple[float, float] = (1.0, 1.0)
  target_bias_rad: float = 0.0
