"""IK-consistent reset, computed without a simulator.

The planar five-bar has an analytical IK, so a reset can solve active and
passive joints together and write them in one go: the closure constraints start
satisfied and there is no reset impulse. The simulator-specific part (writing
the root and joint state) stays in each framework's events module.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .kinematics import FOOT_SPHERE_RADIUS_M, HIP_MIDPOINTS_ROOT_M, MicrotaurFiveBarKinematics
from .robot_constants import STAND_A, STAND_E

KIN = MicrotaurFiveBarKinematics()

# Reset targets are built around the foot positions of the stand pose, so a
# zero action after reset does not jump.
NOMINAL_FOOT_XZ = np.stack([KIN.forward_numpy(STAND_A[i], STAND_E[i], leg_index=i + 1).foot for i in range(4)])
NOMINAL_FULL_Q = np.stack([KIN.forward_numpy(STAND_A[i], STAND_E[i], leg_index=i + 1).full for i in range(4)])

# Root height with the lowest foot sphere resting on the ground (0.070173 m).
NOMINAL_ROOT_HEIGHT_M = float(FOOT_SPHERE_RADIUS_M - np.min(HIP_MIDPOINTS_ROOT_M[:, 2] + NOMINAL_FOOT_XZ[:, 1]))

RESET_ROOT_XY_M = (-0.10, 0.10)
RESET_YAW_RAD = (-math.pi, math.pi)
RESET_FOOT_X_M = (-0.006, 0.006)
RESET_SHARED_Z_M = (-0.004, 0.004)  # body-height offset
RESET_PER_LEG_Z_M = (-0.001, 0.001)  # zero-mean per leg
RESET_LOWEST_FOOT_Z_M = (FOOT_SPHERE_RADIUS_M, FOOT_SPHERE_RADIUS_M + 0.002)


def uniform(count: int, tail: tuple[int, ...], bounds: tuple[float, float], device, dtype=torch.float32):
  lo, hi = float(bounds[0]), float(bounds[1])
  return lo + (hi - lo) * torch.rand((count, *tail), device=device, dtype=dtype)


def sample_reset(n: int, randomize: bool, motor_limits: torch.Tensor, device, dtype) -> dict[str, torch.Tensor]:
  """Sample n reset states.

  motor_limits: [n, 4, 2, 2] (leg, motor a/e, lower/upper), canonical order.
  Returns full_q [n, 16] in kinematics.FULL_JOINT_NAMES order (a, b, e, d per
  leg), motor_q [n, 8] in LEG_JOINT_NAMES order, root_height [n] above the
  ground, root_xy [n, 2] offset from the env origin, yaw [n], valid [n, 4].
  The torch.rand call order is the upstream one (kept for bit-exact tests).
  """
  nominal = torch.tensor(NOMINAL_FOOT_XZ, device=device, dtype=dtype)[None].repeat(n, 1, 1)
  reference = torch.stack(
    (torch.tensor(STAND_A, device=device, dtype=dtype), torch.tensor(STAND_E, device=device, dtype=dtype)), dim=-1
  )[None].repeat(n, 1, 1)

  targets = nominal.clone()
  if randomize:
    targets[:, :, 0] += uniform(n, (4,), RESET_FOOT_X_M, device, dtype)
    shared = uniform(n, (1,), RESET_SHARED_Z_M, device, dtype)
    per_leg = uniform(n, (4,), RESET_PER_LEG_Z_M, device, dtype)
    per_leg -= per_leg.mean(dim=1, keepdim=True)
    targets[:, :, 1] += shared + per_leg

  full_q, valid = KIN.solve_all_legs_torch(targets, reference, motor_limits=motor_limits)
  full_q = full_q.reshape(n, 4, 4)
  # Fall back to the nominal closed-chain state on IK / joint-limit edge cases.
  full_q = torch.where(valid[:, :, None], full_q, torch.tensor(NOMINAL_FULL_Q, device=device, dtype=dtype)[None])
  targets = torch.where(valid[:, :, None], targets, nominal)

  hip_z = torch.tensor(HIP_MIDPOINTS_ROOT_M[:, 2], device=device, dtype=dtype)[None]
  if randomize:
    lowest = uniform(n, (), RESET_LOWEST_FOOT_Z_M, device, dtype)
  else:
    lowest = torch.full((n,), RESET_LOWEST_FOOT_Z_M[0], device=device, dtype=dtype)
  root_height = lowest - torch.amin(hip_z + targets[:, :, 1], dim=1)

  root_xy = torch.zeros(n, 2, device=device, dtype=dtype)
  yaw = torch.zeros(n, device=device, dtype=dtype)
  if randomize:
    root_xy[:, 0] = uniform(n, (), RESET_ROOT_XY_M, device, dtype)
    root_xy[:, 1] = uniform(n, (), RESET_ROOT_XY_M, device, dtype)
    yaw = uniform(n, (), RESET_YAW_RAD, device, dtype)

  return {
    "full_q": full_q.reshape(n, 16), "motor_q": full_q[:, :, (0, 2)].reshape(n, 8),
    "root_height": root_height, "root_xy": root_xy, "yaw": yaw, "valid": valid,
  }
