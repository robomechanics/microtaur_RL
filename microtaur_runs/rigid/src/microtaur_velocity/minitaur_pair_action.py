"""Swing/lift action adapter for the four Microtaur legs.

The policy outputs two leg-space values per leg: ``swing`` for the main
forward/backward sweep and ``lift`` for the paired-joint shape/extension mode.
These eight values are converted to the eight physical ``a``/``e`` motor
position targets using the per-leg mirror signs and nominal stand offsets.
"""

from __future__ import annotations

from dataclasses import MISSING

import torch

try:
  from mjlab.utils import configclass
except Exception:
  from dataclasses import dataclass as configclass

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


@configclass
class MicrotaurSwingLiftActionCfg(JointPositionActionCfg):
  """Config for Microtaur leg-space swing/lift action."""

  class_type: type = MISSING

  # Policy order is (swing, lift) for legs 1 through 4.

  swing_scale: float = 0.45
  lift_scale: float = 0.45

  swing_offset: float = 0.0
  lift_offset: float = 0.0

  # Mirror signs convert the shared leg-space convention into each leg frame.
  leg_signs: tuple[float, float, float, float] = (+1.0, -1.0, -1.0, +1.0)

  # A zero policy action maps exactly to the nominal stand pose.
  stand_a_offsets: tuple[float, float, float, float] = (0.45, -0.45, -0.45, 0.45)
  stand_e_offsets: tuple[float, float, float, float] = (-0.45, 0.45, 0.45, -0.45)

  def __init__(
    self,
    asset_cfg=None,
    scale=None,
    offset=None,
    preserve_order: bool = True,
    use_default_offset: bool = False,
    swing_scale: float = 0.45,
    lift_scale: float = 0.45,
    swing_offset: float = 0.0,
    lift_offset: float = 0.0,
    leg_signs: tuple[float, float, float, float] = (+1.0, -1.0, -1.0, +1.0),
    stand_a_offsets: tuple[float, float, float, float] = (0.45, -0.45, -0.45, 0.45),
    stand_e_offsets: tuple[float, float, float, float] = (-0.45, 0.45, 0.45, -0.45),
    **kwargs,
  ):
    try:
      super().__init__()
    except TypeError:
      pass

    self.class_type = MicrotaurSwingLiftAction

    self.entity_name = "robot"
    self.actuator_names = None
    self.preserve_order = preserve_order
    self.use_default_offset = use_default_offset

    if asset_cfg is not None:
      self.asset_cfg = asset_cfg

      if hasattr(asset_cfg, "name"):
        self.entity_name = asset_cfg.name
      elif hasattr(asset_cfg, "entity_name"):
        self.entity_name = asset_cfg.entity_name

      if hasattr(asset_cfg, "joint_names"):
        self.actuator_names = asset_cfg.joint_names
      elif hasattr(asset_cfg, "actuator_names"):
        self.actuator_names = asset_cfg.actuator_names

    if self.actuator_names is None:
      self.actuator_names = (
        "leg1_a_joint_act",
        "leg1_e_joint_act",
        "leg2_a_joint_act",
        "leg2_e_joint_act",
        "leg3_a_joint_act",
        "leg3_e_joint_act",
        "leg4_a_joint_act",
        "leg4_e_joint_act",
      )

    # Leave the parent transform as identity; physical scaling happens below.
    self.scale = 1.0 if scale is None else scale
    self.offset = 0.0 if offset is None else offset

    self.swing_scale = swing_scale
    self.lift_scale = lift_scale

    self.swing_offset = swing_offset
    self.lift_offset = lift_offset

    self.leg_signs = leg_signs

    self.stand_a_offsets = stand_a_offsets
    self.stand_e_offsets = stand_e_offsets

    for key, value in kwargs.items():
      setattr(self, key, value)

  def build(self, env):
    return MicrotaurSwingLiftAction(self, env)


class MicrotaurSwingLiftAction(JointPositionAction):
  """Map normalized swing/lift actions to Microtaur motor targets."""

  cfg: MicrotaurSwingLiftActionCfg

  def process_actions(self, actions: torch.Tensor):
    if actions.shape[-1] != 8:
      raise ValueError(
        f"MicrotaurSwingLiftAction expects 8 actions "
        f"(swing/lift for 4 legs), got {actions.shape[-1]}."
      )

    # Clamp policy output before converting it to motor targets.
    actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
    actions = torch.clamp(actions, -1.0, 1.0)

    processed = torch.zeros_like(actions)

    stand_a_offsets = torch.tensor(
      self.cfg.stand_a_offsets,
      device=actions.device,
      dtype=actions.dtype,
    )
    stand_e_offsets = torch.tensor(
      self.cfg.stand_e_offsets,
      device=actions.device,
      dtype=actions.dtype,
    )
    leg_signs = torch.tensor(
      self.cfg.leg_signs,
      device=actions.device,
      dtype=actions.dtype,
    )

    for leg_idx in range(4):
      swing_raw = actions[:, 2 * leg_idx]
      lift_raw = actions[:, 2 * leg_idx + 1]

      # Convert normalized policy output into leg-space swing/lift coordinates.
      swing = self.cfg.swing_offset + self.cfg.swing_scale * swing_raw
      lift = self.cfg.lift_offset + self.cfg.lift_scale * lift_raw

      sign = leg_signs[leg_idx]

      # Swing moves both motors in the mirrored leg direction; lift changes
      # their difference to shape and raise/lower the leg.
      a_target = stand_a_offsets[leg_idx] + sign * swing - sign * lift
      e_target = stand_e_offsets[leg_idx] + sign * swing + sign * lift

      processed[:, 2 * leg_idx] = a_target
      processed[:, 2 * leg_idx + 1] = e_target

    return super().process_actions(processed)


# Keep the old class names available for configs that still import them.
MinitaurPairPositionActionCfg = MicrotaurSwingLiftActionCfg
MinitaurPairPositionAction = MicrotaurSwingLiftAction

MicrotaurSwingLiftActionCfg.class_type = MicrotaurSwingLiftAction