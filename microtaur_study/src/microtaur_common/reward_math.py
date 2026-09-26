"""Reward math on plain tensors, shared by both simulators' reward terms."""

from __future__ import annotations

import torch


def lin_vel_tracking(cmd: torch.Tensor, lin_vel_b: torch.Tensor, sigma: float) -> torch.Tensor:
  err2 = torch.sum(torch.square(cmd[:, :2] - lin_vel_b[:, :2]), dim=1)
  return torch.exp(-err2 / sigma**2)


def yaw_rate_tracking(cmd: torch.Tensor, ang_vel_b: torch.Tensor, sigma: float) -> torch.Tensor:
  return torch.exp(-torch.square(cmd[:, 2] - ang_vel_b[:, 2]) / sigma**2)


def motor_power(tau: torch.Tensor, qd: torch.Tensor, copper_w_per_nm2: float) -> tuple[torch.Tensor, torch.Tensor]:
  """(|mechanical power|, copper loss) in W, summed over the given joints.

  XL330s do not regenerate, so braking costs too (absolute value). Copper loss
  R/k_t^2 * tau^2 is what a motor spends holding torque at zero speed.
  """
  return torch.sum(torch.abs(tau * qd), dim=1), copper_w_per_nm2 * torch.sum(torch.square(tau), dim=1)


def trot_gait(air: torch.Tensor, contact: torch.Tensor, pairs, std: float, max_err: float) -> torch.Tensor:
  """IsaacLab Spot's GaitReward for a trot, on running air / contact times [N, 4].

  Product of six kernels: the two feet of each diagonal pair should have the
  same air and contact time (in phase), and each foot's air time should match
  the contact time of the feet in the other pair (out of phase). 1 for a
  perfect trot. ⚠ Alone it also scores fast foot chatter highly (short, equal
  times); Spot suppresses that with a separate air-time term.
  """
  e2 = max_err**2

  def sync(a, b):
    se_air = torch.clamp(torch.square(air[:, a] - air[:, b]), max=e2)
    se_con = torch.clamp(torch.square(contact[:, a] - contact[:, b]), max=e2)
    return torch.exp(-(se_air + se_con) / std)

  def anti(a, b):
    se_0 = torch.clamp(torch.square(air[:, a] - contact[:, b]), max=e2)
    se_1 = torch.clamp(torch.square(contact[:, a] - air[:, b]), max=e2)
    return torch.exp(-(se_0 + se_1) / std)

  (a0, a1), (b0, b1) = pairs
  return sync(a0, a1) * sync(b0, b1) * anti(a0, b0) * anti(a1, b1) * anti(a0, b1) * anti(b0, a1)
