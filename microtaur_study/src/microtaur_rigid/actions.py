"""mjlab wrapper around the shared walk action (microtaur_common.walk_action).

mjlab resolves actuators in model order; this wrapper maps to and from the
canonical LEG_JOINT_NAMES order the core uses. mjlab's
JointPositionAction.apply_actions subtracts the encoder bias before writing the
joint target.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from microtaur_common.walk_action import WalkActionCore, WalkActionParams

from .robot import LEG_JOINT_NAMES

_PARAM_NAMES = tuple(f.name for f in fields(WalkActionParams))


@dataclass(kw_only=True)
class MicrotaurWalkActionCfg(JointPositionActionCfg, WalkActionParams):
  entity_name: str = "robot"
  actuator_names: tuple[str, ...] = LEG_JOINT_NAMES
  preserve_order: bool = True
  use_default_offset: bool = False

  def params(self) -> WalkActionParams:
    return WalkActionParams(**{k: getattr(self, k) for k in _PARAM_NAMES})

  def build(self, env) -> "MicrotaurWalkAction":
    return MicrotaurWalkAction(self, env)


class MicrotaurWalkAction(JointPositionAction):
  cfg: MicrotaurWalkActionCfg

  def __init__(self, cfg: MicrotaurWalkActionCfg, env):
    super().__init__(cfg, env)
    self._walk_env = env
    device = env.device
    resolved = tuple(self._target_names)
    if sorted(resolved) != sorted(LEG_JOINT_NAMES):
      raise ValueError(f"Expected the eight leg motors, got {resolved}")
    res_idx = {name: i for i, name in enumerate(resolved)}
    can_idx = {name: i for i, name in enumerate(LEG_JOINT_NAMES)}
    self._canonical_from_resolved = torch.tensor([res_idx[x] for x in LEG_JOINT_NAMES], device=device)
    self._resolved_from_canonical = torch.tensor([can_idx[x] for x in resolved], device=device)
    self.core = WalkActionCore(cfg.params(), env.num_envs, device)

  # --- exposed for rewards / logging / tests ---------------------------------

  @property
  def stand(self) -> torch.Tensor:
    return self.core.stand

  @property
  def requested_targets(self) -> torch.Tensor:
    return self.core.requested

  @property
  def safe_targets(self) -> torch.Tensor:
    return self.core.safe

  @property
  def applied_targets(self) -> torch.Tensor:
    return self.core.applied

  @property
  def filter_correction(self) -> torch.Tensor:
    return self.core.filter_correction

  @property
  def filter_blend(self) -> torch.Tensor:
    return self.core.filter_blend

  def _filter_targets(self, position, velocity, requested):
    return self.core.filter_targets(position, velocity, requested)

  def resolved_to_canonical(self, value: torch.Tensor) -> torch.Tensor:
    return value[:, self._canonical_from_resolved]

  def canonical_to_resolved(self, value: torch.Tensor) -> torch.Tensor:
    return value[:, self._resolved_from_canonical]

  # --- ActionTerm interface --------------------------------------------------

  def process_actions(self, actions: torch.Tensor):
    position = self.resolved_to_canonical(self._entity.data.joint_pos[:, self._target_ids])
    velocity = self.resolved_to_canonical(self._entity.data.joint_vel[:, self._target_ids])
    raw, applied = self.core.step(actions, position, velocity, self._walk_env.episode_length_buf <= 1)
    # Raw actions feed the `actions` observation and action_rate; the parent
    # class applies the processed targets.
    self._raw_actions[:] = raw
    self._processed_actions[:] = self.canonical_to_resolved(applied)


def walk_action_term(env, name: str = "joint_pos") -> MicrotaurWalkAction:
  term = env.action_manager.get_term(name)
  if not isinstance(term, MicrotaurWalkAction):
    raise TypeError(f"Action term '{name}' is {type(term).__name__}, not MicrotaurWalkAction")
  return term
