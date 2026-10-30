"""The walk action pipeline, framework-free: filter, per-episode randomisation, delay.

Both the mjlab and the IsaacLab action terms are thin wrappers around
WalkActionCore, so the two simulators run literally the same action code.

Per policy step (unchanged from upstream 2026-09-20):

  requested = stand + gain * action_scale * clip(action, -1, 1) + bias
  safe      = filter(requested)   # 1) per-joint time-to-limit blend
                                  # 2) common / diff limits in paired coords
                                  # 3) |common| + |diff| <= diamond (L1)
                                  # 4) exact per-joint clamp
  applied   = safe delayed by `delay_steps` policy steps

Paired coordinates per leg: da = sign*(common - diff), de = sign*(common + diff).
All tensors are in canonical LEG_JOINT_NAMES order: (a, e) for legs 1-4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .robot_constants import JOINT_HALF_RANGE_RAD, LEG_SIGNS, STAND_A, STAND_E

ACTION_SCALE_RAD = math.radians(30.0)


@dataclass
class WalkActionParams:
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


class WalkActionCore:
  def __init__(self, p: WalkActionParams, num_envs: int, device):
    self.p = p
    n = num_envs
    stand = torch.empty(8, device=device)
    stand[0::2] = torch.tensor(p.stand_a, device=device)
    stand[1::2] = torch.tensor(p.stand_e, device=device)
    self.stand = stand
    self._leg_signs = torch.tensor(p.leg_signs, device=device)
    self._joint_lower = stand - p.joint_half_range_rad
    self._joint_upper = stand + p.joint_half_range_rad

    self.requested = stand[None, :].repeat(n, 1)
    self.safe = self.requested.clone()
    self.applied = self.requested.clone()
    self.filter_correction = torch.zeros_like(self.requested)
    self.filter_blend = torch.zeros_like(self.requested)
    self._gain = torch.ones_like(self.requested)
    self._bias = torch.zeros_like(self.requested)

    self._delay_buffer = stand[None, None, :].repeat(n, max(1, p.max_delay_steps + 1), 1)
    self._delay_steps = torch.full((n,), p.min_delay_steps, device=device, dtype=torch.long)
    self._delay_write = 0

  # --- per-episode state -----------------------------------------------------

  def reset_episode_state(self, reset: torch.Tensor, position: torch.Tensor) -> None:
    if not torch.any(reset):
      return
    count = int(reset.sum().item())
    dev = self._gain.device

    lo, hi = self.p.target_gain_range
    self._gain[reset] = lo + (hi - lo) * torch.rand((count, 8), device=dev) if hi > lo else float(lo)
    b = float(self.p.target_bias_rad)
    self._bias[reset] = (2.0 * torch.rand((count, 8), device=dev) - 1.0) * b if b > 0.0 else 0.0
    self._delay_steps[reset] = torch.randint(
      self.p.min_delay_steps, self.p.max_delay_steps + 1, (count,), device=dev
    )

    # Seed every delay slot from the actual reset pose, not the nominal stand.
    q = position[reset]
    self._delay_buffer[reset] = q[:, None, :].expand(-1, self._delay_buffer.shape[1], -1)
    self.requested[reset] = q
    self.safe[reset] = q
    self.applied[reset] = q
    self.filter_correction[reset] = 0.0
    self.filter_blend[reset] = 0.0

  # --- safety filter ---------------------------------------------------------

  def axis_filter(self, position, velocity, requested, lower, upper, margin):
    """Olympus-style time-to-limit blending for one set of coordinates."""
    lower = torch.as_tensor(lower, device=position.device, dtype=position.dtype)
    upper = torch.as_tensor(upper, device=position.device, dtype=position.dtype)
    m_lo = position - lower
    m_hi = upper - position
    outside = (m_lo < 0.0) | (m_hi < 0.0)

    floor = float(self.p.velocity_floor_rad_s)
    v = torch.where(velocity.abs() < floor, torch.sign(velocity) * floor, velocity)
    inf = torch.full_like(position, float("inf"))
    t_hi = torch.where((v > 0.0) & (m_hi > 0.0), m_hi / v.clamp_min(1e-6), inf)
    t_lo = torch.where((v < 0.0) & (m_lo > 0.0), m_lo / (-v).clamp_min(1e-6), inf)
    horizon = max(float(self.p.filter_horizon_s), 1e-6)
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

  def filter_targets(self, position, velocity, requested):
    p = self.p
    individual, ind_blend = self.axis_filter(
      position, velocity, requested, self._joint_lower, self._joint_upper, p.joint_margin_rad
    )

    q, qd, qr = position.reshape(-1, 4, 2), velocity.reshape(-1, 4, 2), individual.reshape(-1, 4, 2)
    stand = self.stand.reshape(1, 4, 2)
    s = self._leg_signs.reshape(1, 4)
    da, de = q[..., 0] - stand[..., 0], q[..., 1] - stand[..., 1]
    rda, rde = qr[..., 0] - stand[..., 0], qr[..., 1] - stand[..., 1]
    common, diff = 0.5 * s * (da + de), 0.5 * s * (de - da)
    common_dot = 0.5 * s * (qd[..., 0] + qd[..., 1])
    diff_dot = 0.5 * s * (qd[..., 1] - qd[..., 0])
    req_common, req_diff = 0.5 * s * (rda + rde), 0.5 * s * (rde - rda)

    safe_common, c_blend = self.axis_filter(
      common, common_dot, req_common, -p.common_limit_rad, p.common_limit_rad, p.coupled_margin_rad
    )
    safe_diff, d_blend = self.axis_filter(
      diff, diff_dot, req_diff, -p.diff_limit_rad, p.diff_limit_rad, p.coupled_margin_rad
    )

    # L1 diamond |common| + |diff| <= limit bounds both motor offsets.
    radius = common.abs() + diff.abs()
    req_radius = safe_common.abs() + safe_diff.abs()
    radial_v = torch.sign(common) * common_dot + torch.sign(diff) * diff_dot
    dist = p.diamond_limit_rad - radius
    inf = torch.full_like(radius, float("inf"))
    t = torch.where((radial_v > 0.0) & (dist > 0.0), dist / radial_v.clamp_min(1e-6), inf)
    horizon = max(float(p.filter_horizon_s), 1e-6)
    dia_blend = torch.clamp(1.0 - t / horizon, 0.0, 1.0)
    dia_blend = torch.where(radius > p.diamond_limit_rad, torch.ones_like(dia_blend), dia_blend)
    near = dist < p.coupled_margin_rad
    outward = req_radius > radius
    dia_blend = torch.where(near & outward, torch.ones_like(dia_blend), dia_blend)
    dia_blend = torch.where(near & ~outward, torch.zeros_like(dia_blend), dia_blend)
    proj = torch.clamp(p.diamond_limit_rad / (req_radius + 1e-6), max=1.0)
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

  # --- one policy step -------------------------------------------------------

  def step(self, actions: torch.Tensor, position: torch.Tensor, velocity: torch.Tensor,
           reset: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (clipped raw actions, applied targets). Inputs in canonical order."""
    if actions.shape[-1] != 8:
      raise ValueError(f"Expected 8 motor actions, got {actions.shape[-1]}")
    actions = torch.clamp(torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    self.reset_episode_state(reset, position)

    requested = self.stand[None, :] + self._gain * self.p.action_scale_rad * actions + self._bias
    safe, blend = self.filter_targets(position, velocity, requested)
    self.requested = requested
    self.safe = safe
    self.filter_correction = requested - safe
    self.filter_blend = blend

    self._delay_buffer[:, self._delay_write, :] = safe
    read = torch.remainder(self._delay_write - self._delay_steps, self._delay_buffer.shape[1])
    applied = self._delay_buffer[torch.arange(self._delay_buffer.shape[0], device=actions.device), read]
    self._delay_write = (self._delay_write + 1) % self._delay_buffer.shape[1]
    self.applied = applied
    return actions, applied
