"""Eight direct motor-position actions with a predictive five-bar safety filter.

Pipeline per policy step (unchanged from upstream 2026-09-20):

  requested = stand + gain * action_scale * clip(action, -1, 1) + bias
  safe      = filter(requested)   # 1) per-joint time-to-limit blend
                                  # 2) common / diff limits in paired coords
                                  # 3) |common| + |diff| <= diamond (L1)
                                  # 4) exact per-joint clamp
  applied   = safe delayed by `delay_steps` policy steps

Paired coordinates per leg: da = sign*(common - diff), de = sign*(common + diff).
mjlab's JointPositionAction.apply_actions subtracts the encoder bias before
writing the joint target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from .robot import JOINT_HALF_RANGE_RAD, LEG_JOINT_NAMES, LEG_SIGNS, STAND_A, STAND_E

ACTION_SCALE_RAD = math.radians(30.0)


@dataclass(kw_only=True)
class MicrotaurWalkActionCfg(JointPositionActionCfg):
  entity_name: str = "robot"
  actuator_names: tuple[str, ...] = LEG_JOINT_NAMES
  preserve_order: bool = True
  use_default_offset: bool = False

  action_scale_rad: float = ACTION_SCALE_RAD
  filter_horizon_s: float = 0.10
  velocity_floor_rad_s: float = math.radians(60.0)
  joint_margin_rad: float = math.radians(8.0)
  coupled_margin_rad: float = math.radians(10.0)
  common_limit_rad: float = 0.52
  diff_limit_rad: float = 0.52
  diamond_limit_rad: float = 0.70
  # Hardware command-to-response latency is modelled as one policy-step hold
  # plus the identified servo dynamics.
  min_delay_steps: int = 1
  max_delay_steps: int = 1
  target_gain_range: tuple[float, float] = (1.0, 1.0)
  target_bias_rad: float = 0.0

  stand_a: tuple[float, ...] = STAND_A
  stand_e: tuple[float, ...] = STAND_E
  leg_signs: tuple[float, ...] = LEG_SIGNS
  joint_half_range_rad: float = JOINT_HALF_RANGE_RAD

  def build(self, env) -> "MicrotaurWalkAction":
    return MicrotaurWalkAction(self, env)


class MicrotaurWalkAction(JointPositionAction):
  cfg: MicrotaurWalkActionCfg

  def __init__(self, cfg: MicrotaurWalkActionCfg, env):
    super().__init__(cfg, env)
    self._walk_env = env
    device = env.device
    n = env.num_envs

    # mjlab resolves actuators in model order; keep the policy interface in
    # canonical LEG_JOINT_NAMES order regardless.
    resolved = tuple(self._target_names)
    if sorted(resolved) != sorted(LEG_JOINT_NAMES):
      raise ValueError(f"Expected the eight leg motors, got {resolved}")
    res_idx = {name: i for i, name in enumerate(resolved)}
    can_idx = {name: i for i, name in enumerate(LEG_JOINT_NAMES)}
    self._canonical_from_resolved = torch.tensor([res_idx[x] for x in LEG_JOINT_NAMES], device=device)
    self._resolved_from_canonical = torch.tensor([can_idx[x] for x in resolved], device=device)

    stand = torch.empty(8, device=device)
    stand[0::2] = torch.tensor(cfg.stand_a, device=device)
    stand[1::2] = torch.tensor(cfg.stand_e, device=device)
    self._stand = stand
    self._leg_signs = torch.tensor(cfg.leg_signs, device=device)
    self._joint_lower = stand - cfg.joint_half_range_rad
    self._joint_upper = stand + cfg.joint_half_range_rad

    self._requested = stand[None, :].repeat(n, 1)
    self._safe = self._requested.clone()
    self._applied = self._requested.clone()
    self._filter_correction = torch.zeros_like(self._requested)
    self._filter_blend = torch.zeros_like(self._requested)
    self._gain = torch.ones_like(self._requested)
    self._bias = torch.zeros_like(self._requested)

    self._delay_buffer = stand[None, None, :].repeat(n, max(1, cfg.max_delay_steps + 1), 1)
    self._delay_steps = torch.full((n,), cfg.min_delay_steps, device=device, dtype=torch.long)
    self._delay_write = 0

  # --- exposed for rewards / logging ----------------------------------------

  @property
  def stand(self) -> torch.Tensor:
    return self._stand

  @property
  def requested_targets(self) -> torch.Tensor:
    return self._requested

  @property
  def safe_targets(self) -> torch.Tensor:
    return self._safe

  @property
  def applied_targets(self) -> torch.Tensor:
    return self._applied

  @property
  def filter_correction(self) -> torch.Tensor:
    return self._filter_correction

  @property
  def filter_blend(self) -> torch.Tensor:
    return self._filter_blend

  def resolved_to_canonical(self, value: torch.Tensor) -> torch.Tensor:
    return value[:, self._canonical_from_resolved]

  def canonical_to_resolved(self, value: torch.Tensor) -> torch.Tensor:
    return value[:, self._resolved_from_canonical]

  # --- per-episode state -----------------------------------------------------

  def _reset_episode_state(self, reset: torch.Tensor, position: torch.Tensor) -> None:
    if not torch.any(reset):
      return
    count = int(reset.sum().item())
    dev = self._gain.device

    lo, hi = self.cfg.target_gain_range
    self._gain[reset] = lo + (hi - lo) * torch.rand((count, 8), device=dev) if hi > lo else float(lo)
    b = float(self.cfg.target_bias_rad)
    self._bias[reset] = (2.0 * torch.rand((count, 8), device=dev) - 1.0) * b if b > 0.0 else 0.0
    self._delay_steps[reset] = torch.randint(
      self.cfg.min_delay_steps, self.cfg.max_delay_steps + 1, (count,), device=dev
    )

    # Seed every delay slot from the actual reset pose, not the nominal stand.
    p = position[reset]
    self._delay_buffer[reset] = p[:, None, :].expand(-1, self._delay_buffer.shape[1], -1)
    self._requested[reset] = p
    self._safe[reset] = p
    self._applied[reset] = p
    self._filter_correction[reset] = 0.0
    self._filter_blend[reset] = 0.0

  # --- safety filter ---------------------------------------------------------

  def _axis_filter(self, position, velocity, requested, lower, upper, margin):
    """Olympus-style time-to-limit blending for one set of coordinates."""
    lower = torch.as_tensor(lower, device=position.device, dtype=position.dtype)
    upper = torch.as_tensor(upper, device=position.device, dtype=position.dtype)
    m_lo = position - lower
    m_hi = upper - position
    outside = (m_lo < 0.0) | (m_hi < 0.0)

    floor = float(self.cfg.velocity_floor_rad_s)
    v = torch.where(velocity.abs() < floor, torch.sign(velocity) * floor, velocity)
    inf = torch.full_like(position, float("inf"))
    t_hi = torch.where((v > 0.0) & (m_hi > 0.0), m_hi / v.clamp_min(1e-6), inf)
    t_lo = torch.where((v < 0.0) & (m_lo > 0.0), m_lo / (-v).clamp_min(1e-6), inf)
    horizon = max(float(self.cfg.filter_horizon_s), 1e-6)
    blend = torch.clamp(1.0 - torch.minimum(t_hi, t_lo) / horizon, 0.0, 1.0)
    blend = torch.where(outside, torch.ones_like(blend), blend)

    # Near a limit, block commands that move farther out; leave inward ones.
    near_hi = m_hi < margin
    near_lo = m_lo < margin
    blend = torch.where(near_hi & (requested > position), torch.ones_like(blend), blend)
    blend = torch.where(near_lo & (requested < position), torch.ones_like(blend), blend)
    blend = torch.where(near_hi & (requested < position), torch.zeros_like(blend), blend)
    blend = torch.where(near_lo & (requested > position), torch.zeros_like(blend), blend)

    clamped = torch.minimum(torch.maximum(requested, lower), upper)
    return (1.0 - blend) * requested + blend * clamped, blend

  def _filter_targets(self, position, velocity, requested):
    cfg = self.cfg
    individual, ind_blend = self._axis_filter(
      position, velocity, requested, self._joint_lower, self._joint_upper, cfg.joint_margin_rad
    )

    q, qd, qr = position.reshape(-1, 4, 2), velocity.reshape(-1, 4, 2), individual.reshape(-1, 4, 2)
    stand = self._stand.reshape(1, 4, 2)
    s = self._leg_signs.reshape(1, 4)
    da, de = q[..., 0] - stand[..., 0], q[..., 1] - stand[..., 1]
    rda, rde = qr[..., 0] - stand[..., 0], qr[..., 1] - stand[..., 1]
    common, diff = 0.5 * s * (da + de), 0.5 * s * (de - da)
    common_dot = 0.5 * s * (qd[..., 0] + qd[..., 1])
    diff_dot = 0.5 * s * (qd[..., 1] - qd[..., 0])
    req_common, req_diff = 0.5 * s * (rda + rde), 0.5 * s * (rde - rda)

    safe_common, c_blend = self._axis_filter(
      common, common_dot, req_common, -cfg.common_limit_rad, cfg.common_limit_rad, cfg.coupled_margin_rad
    )
    safe_diff, d_blend = self._axis_filter(
      diff, diff_dot, req_diff, -cfg.diff_limit_rad, cfg.diff_limit_rad, cfg.coupled_margin_rad
    )

    # L1 diamond |common| + |diff| <= limit bounds both motor offsets.
    radius = common.abs() + diff.abs()
    req_radius = safe_common.abs() + safe_diff.abs()
    radial_v = torch.sign(common) * common_dot + torch.sign(diff) * diff_dot
    dist = cfg.diamond_limit_rad - radius
    inf = torch.full_like(radius, float("inf"))
    t = torch.where((radial_v > 0.0) & (dist > 0.0), dist / radial_v.clamp_min(1e-6), inf)
    horizon = max(float(cfg.filter_horizon_s), 1e-6)
    dia_blend = torch.clamp(1.0 - t / horizon, 0.0, 1.0)
    dia_blend = torch.where(radius > cfg.diamond_limit_rad, torch.ones_like(dia_blend), dia_blend)
    near = dist < cfg.coupled_margin_rad
    outward = req_radius > radius
    dia_blend = torch.where(near & outward, torch.ones_like(dia_blend), dia_blend)
    dia_blend = torch.where(near & ~outward, torch.zeros_like(dia_blend), dia_blend)
    proj = torch.clamp(cfg.diamond_limit_rad / (req_radius + 1e-6), max=1.0)
    # Keep upstream's operation order: results are then bit-identical, which is
    # what tests/check_equivalence.py relies on (contact onset is chaotic).
    safe_common = (1.0 - dia_blend) * safe_common + dia_blend * (safe_common * proj)
    safe_diff = (1.0 - dia_blend) * safe_diff + dia_blend * (safe_diff * proj)

    pairs = torch.empty_like(qr)
    pairs[..., 0] = stand[..., 0] + s * (safe_common - safe_diff)
    pairs[..., 1] = stand[..., 1] + s * (safe_common + safe_diff)
    safe = torch.minimum(torch.maximum(pairs.reshape(-1, 8), self._joint_lower), self._joint_upper)

    pair_blend = torch.maximum(torch.maximum(c_blend, d_blend), dia_blend)
    pair_blend = pair_blend[:, :, None].expand(-1, -1, 2).reshape(-1, 8)
    return safe, torch.maximum(ind_blend, pair_blend)

  # --- ActionTerm interface --------------------------------------------------

  def process_actions(self, actions: torch.Tensor):
    if actions.shape[-1] != 8:
      raise ValueError(f"Expected 8 motor actions, got {actions.shape[-1]}")
    actions = torch.clamp(torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)

    position = self.resolved_to_canonical(self._entity.data.joint_pos[:, self._target_ids])
    velocity = self.resolved_to_canonical(self._entity.data.joint_vel[:, self._target_ids])
    self._reset_episode_state(self._walk_env.episode_length_buf <= 1, position)

    requested = self._stand[None, :] + self._gain * self.cfg.action_scale_rad * actions + self._bias
    safe, blend = self._filter_targets(position, velocity, requested)
    self._requested = requested
    self._safe = safe
    self._filter_correction = requested - safe
    self._filter_blend = blend

    self._delay_buffer[:, self._delay_write, :] = safe
    read = torch.remainder(self._delay_write - self._delay_steps, self._delay_buffer.shape[1])
    applied = self._delay_buffer[torch.arange(self._delay_buffer.shape[0], device=actions.device), read]
    self._delay_write = (self._delay_write + 1) % self._delay_buffer.shape[1]
    self._applied = applied

    # Raw actions feed the `actions` observation and action_rate; the parent
    # class applies the processed targets.
    self._raw_actions[:] = actions
    self._processed_actions[:] = self.canonical_to_resolved(applied)


def walk_action_term(env, name: str = "joint_pos") -> MicrotaurWalkAction:
  term = env.action_manager.get_term(name)
  if not isinstance(term, MicrotaurWalkAction):
    raise TypeError(f"Action term '{name}' is {type(term).__name__}, not MicrotaurWalkAction")
  return term
