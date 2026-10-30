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
from microtaur_common.walk_action import ACTION_SCALE_RAD, WalkActionCore, WalkActionParams


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
    params = WalkActionParams(target_gain_range=tuple(cfg.target_gain_range), target_bias_rad=float(cfg.target_bias_rad),
                              action_scale_rad=float(cfg.action_scale_rad))
    self.core = WalkActionCore(params, self.num_envs, self.device)
    # Optional 9th action: the yaw robot's spine (target = clip(a) x spine_action_scale_rad, clamped to
    # +-spine_limit_rad, held one policy step like the legs' targets).
    self._spine = cfg.spine_joint_name is not None
    if self._spine:
      sid, _ = self._asset.find_joints([cfg.spine_joint_name])
      self._spine_id = sid
      self._spine_hold = torch.zeros(self.num_envs, 1, device=self.device)
    n = 9 if self._spine else 8
    self._raw = torch.zeros(self.num_envs, n, device=self.device)
    self._processed = torch.zeros_like(self._raw)

  @property
  def action_dim(self) -> int:
    return 9 if self._spine else 8

  @property
  def raw_actions(self) -> torch.Tensor:
    return self._raw

  @property
  def processed_actions(self) -> torch.Tensor:
    return self._processed

  def process_actions(self, actions: torch.Tensor):
    d = self._asset.data
    reset = self._walk_env.episode_length_buf <= 1
    raw, applied = self.core.step(
      actions[:, :8], d.joint_pos[:, self._joint_ids], d.joint_vel[:, self._joint_ids], reset
    )
    self._raw[:, :8] = raw
    self._processed[:, :8] = applied
    if self._spine:
      a = torch.clamp(torch.nan_to_num(actions[:, 8:9], nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
      lim = float(self.cfg.spine_limit_rad)
      requested = torch.clamp(a * float(self.cfg.spine_action_scale_rad), -lim, lim)
      # seed the hold from the actual reset pose, then apply the previous step's request (1-step latency)
      self._spine_hold = torch.where(reset[:, None], d.joint_pos[:, self._spine_id], self._spine_hold)
      self._processed[:, 8:9] = self._spine_hold
      self._spine_hold = requested
      self._raw[:, 8:9] = a

  def apply_actions(self):
    bias = getattr(self._walk_env, "microtaur_encoder_bias", None)
    legs = self._processed[:, :8] if bias is None else self._processed[:, :8] - bias[:, :8]
    self._asset.set_joint_position_target(legs, joint_ids=self._joint_ids)
    if self._spine:
      spine = self._processed[:, 8:9] if bias is None or bias.shape[1] < 9 else self._processed[:, 8:9] - bias[:, 8:9]
      self._asset.set_joint_position_target(spine, joint_ids=self._spine_id)

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
  # Motor offset from the stand pose at |action| = 1 (actions are clipped to +-1).
  # The hardware controller must use the same value as the policy was trained with.
  action_scale_rad: float = ACTION_SCALE_RAD
  # Yaw robot: the spine joint as a 9th action (None: the rigid robot's 8 actions).
  spine_joint_name: str | None = None
  spine_action_scale_rad: float = 0.35  # |action| = 1 -> the joint's full +-0.35 rad range
  spine_limit_rad: float = 0.33  # kept inside the 0.35 rad joint limit
