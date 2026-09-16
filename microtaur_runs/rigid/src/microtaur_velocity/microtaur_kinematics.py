"""Analytical planar kinematics for the rigid Microtaur five-bar legs.

Each leg is solved in the same 2-D frame: the origin is midway between the
``a`` and ``e`` motor pivots, +x follows the robot's forward axis, and +z points
up. The linkage is labeled A-B-C and E-D-C, with the foot point F rigidly
extended from the D-C branch. F is the foot-site/sphere center, not the closure
point C.

Both NumPy and Torch implementations are provided. Inverse kinematics checks
all four assembly candidates and chooses the valid solution closest to the
reference motor angles, which keeps resets and replay on the intended physical
branch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Geometry measured from robot_modified.xml
# ---------------------------------------------------------------------------

MOTOR_PIVOT_DISTANCE_M = 0.0205000
PROXIMAL_LINK_M = 0.0464129
DISTAL_LINK_M = 0.0708302
DISTAL_TO_FOOT_M = 0.08296474
FOOT_SPHERE_RADIUS_M = 0.0062

# Fixed angle from the D-C link to the D-F foot extension.
FOOT_EXTENSION_ANGLE_RAD = math.radians(4.31513)

# Link orientations at zero joint position in the canonical x-z frame.
THETA_AB_ZERO_RAD = -3.0 * math.pi / 4.0
THETA_ED_ZERO_RAD = -math.pi / 4.0
THETA_BC_ZERO_RAD = math.radians(-52.55077750641945)
THETA_DC_ZERO_RAD = math.radians(-127.44922249358055)

# Hip midpoints expressed in the battery/root frame.
HIP_MIDPOINTS_ROOT_M = np.asarray(
  [
    [-0.0844769, -0.0429600, 0.0095000],
    [-0.0844769, +0.0429600, 0.0095000],
    [+0.0835731, +0.0429600, 0.0095000],
    [+0.0835731, -0.0429600, 0.0095000],
  ],
  dtype=np.float64,
)

LEG_SIGNS = np.asarray((+1.0, -1.0, -1.0, +1.0), dtype=np.float64)

# The linkage is layered outboard of the actuator-center plane, so each foot
# site is offset 7 mm laterally. XML leg order is rear-left, rear-right,
# front-right, front-left.
FOOT_LATERAL_OFFSETS_M = np.asarray(
  (-0.007, +0.007, +0.007, -0.007),
  dtype=np.float64,
)

# Joint order used when writing a complete closed-chain state to simulation.
FULL_JOINT_NAMES = tuple(
  name
  for leg_idx in range(1, 5)
  for name in (
    f"leg{leg_idx}_a_joint_act",
    f"leg{leg_idx}_b_joint",
    f"leg{leg_idx}_e_joint_act",
    f"leg{leg_idx}_d_joint",
  )
)

MOTOR_JOINT_NAMES = tuple(
  name
  for leg_idx in range(1, 5)
  for name in (
    f"leg{leg_idx}_a_joint_act",
    f"leg{leg_idx}_e_joint_act",
  )
)

FOOT_SITE_NAMES = tuple(f"leg{i}_foot_site" for i in range(1, 5))


@dataclass(frozen=True)
class FiveBarSolution:
  """One NumPy IK/FK result for a single leg."""

  q_a: float
  q_b: float
  q_e: float
  q_d: float
  foot_x: float
  foot_z: float
  closure_x: float
  closure_z: float
  valid: bool

  @property
  def motor(self) -> np.ndarray:
    return np.asarray((self.q_a, self.q_e), dtype=np.float64)

  @property
  def full(self) -> np.ndarray:
    return np.asarray(
      (self.q_a, self.q_b, self.q_e, self.q_d),
      dtype=np.float64,
    )

  @property
  def foot(self) -> np.ndarray:
    return np.asarray((self.foot_x, self.foot_z), dtype=np.float64)


class MicrotaurFiveBarKinematics:
  """Analytical forward and inverse kinematics for all four Microtaur legs."""

  def __init__(self) -> None:
    self.l0 = MOTOR_PIVOT_DISTANCE_M
    self.l1 = PROXIMAL_LINK_M
    self.l2 = DISTAL_LINK_M
    self.lf = DISTAL_TO_FOOT_M
    self.foot_delta = FOOT_EXTENSION_ANGLE_RAD

    self.a = np.asarray((-0.5 * self.l0, 0.0), dtype=np.float64)
    self.e = np.asarray((+0.5 * self.l0, 0.0), dtype=np.float64)

  # -------------------------------------------------------------------------
  # NumPy implementation
  # -------------------------------------------------------------------------

  @staticmethod
  def _unit_np(theta: float) -> np.ndarray:
    return np.asarray((math.cos(theta), math.sin(theta)), dtype=np.float64)

  @staticmethod
  def _circle_intersections_np(
    c0: np.ndarray,
    r0: float,
    c1: np.ndarray,
    r1: float,
  ) -> tuple[np.ndarray, np.ndarray, bool]:
    delta = c1 - c0
    distance = float(np.linalg.norm(delta))
    valid = (
      distance > 1e-12
      and distance <= r0 + r1 + 1e-10
      and distance >= abs(r0 - r1) - 1e-10
    )
    if not valid:
      nan = np.full(2, np.nan, dtype=np.float64)
      return nan, nan, False

    ex = delta / distance
    along = (r0 * r0 - r1 * r1 + distance * distance) / (
      2.0 * distance
    )
    height_sq = max(0.0, r0 * r0 - along * along)
    height = math.sqrt(height_sq)

    center = c0 + along * ex
    normal = np.asarray((-ex[1], ex[0]), dtype=np.float64)
    return center + height * normal, center - height * normal, True

  @staticmethod
  def _solve_2r_np(
    base: np.ndarray,
    target: np.ndarray,
    first_length: float,
    second_length: float,
    elbow_sign: float,
  ) -> tuple[float, float, bool]:
    offset = target - base
    distance_sq = float(np.dot(offset, offset))
    cosine = (
      distance_sq
      - first_length * first_length
      - second_length * second_length
    ) / (2.0 * first_length * second_length)

    valid = -1.0 - 1e-9 <= cosine <= 1.0 + 1e-9
    cosine = float(np.clip(cosine, -1.0, 1.0))
    relative = float(elbow_sign) * math.acos(cosine)

    first = math.atan2(offset[1], offset[0]) - math.atan2(
      second_length * math.sin(relative),
      first_length + second_length * math.cos(relative),
    )
    second_absolute = first + relative
    return first, second_absolute, valid

  def forward_numpy(
    self,
    q_a: float,
    q_e: float,
    leg_index: int,
  ) -> FiveBarSolution:
    """Solve foot position and passive joints from the two motor angles.

    The returned state uses the lower, non-crossed assembly branch.
    ``leg_index`` is 1-based.
    """
    sign = float(LEG_SIGNS[leg_index - 1])

    theta_ab = THETA_AB_ZERO_RAD - sign * float(q_a)
    theta_ed = THETA_ED_ZERO_RAD - sign * float(q_e)

    b = self.a + self.l1 * self._unit_np(theta_ab)
    d = self.e + self.l1 * self._unit_np(theta_ed)

    c0, c1, valid = self._circle_intersections_np(
      b,
      self.l2,
      d,
      self.l2,
    )
    if not valid:
      return FiveBarSolution(
        q_a=float(q_a),
        q_b=float("nan"),
        q_e=float(q_e),
        q_d=float("nan"),
        foot_x=float("nan"),
        foot_z=float("nan"),
        closure_x=float("nan"),
        closure_z=float("nan"),
        valid=False,
      )

    # Select the physical assembly branch, where the closure lies below both proximal tips.
    c = c0 if c0[1] <= c1[1] else c1

    theta_bc = math.atan2(c[1] - b[1], c[0] - b[0])
    theta_dc = math.atan2(c[1] - d[1], c[0] - d[0])
    theta_df = theta_dc + self.foot_delta
    foot = d + self.lf * self._unit_np(theta_df)

    q_b = sign * (THETA_BC_ZERO_RAD - theta_bc) - float(q_a)
    q_d = sign * (THETA_DC_ZERO_RAD - theta_dc) - float(q_e)

    return FiveBarSolution(
      q_a=float(q_a),
      q_b=float(q_b),
      q_e=float(q_e),
      q_d=float(q_d),
      foot_x=float(foot[0]),
      foot_z=float(foot[1]),
      closure_x=float(c[0]),
      closure_z=float(c[1]),
      valid=True,
    )

  def inverse_numpy(
    self,
    foot_x: float,
    foot_z: float,
    leg_index: int,
    reference_motor: Iterable[float] | None = None,
    motor_limits: np.ndarray | None = None,
  ) -> FiveBarSolution:
    """Solve one leg for a target foot position.

    All elbow combinations are tested, then the valid lower assembly closest
    to ``reference_motor=(q_a, q_e)`` is selected.
    """
    sign = float(LEG_SIGNS[leg_index - 1])
    target = np.asarray((foot_x, foot_z), dtype=np.float64)

    if reference_motor is None:
      reference = np.zeros(2, dtype=np.float64)
    else:
      reference = np.asarray(tuple(reference_motor), dtype=np.float64)
      if reference.shape != (2,):
        raise ValueError("reference_motor must contain (q_a, q_e).")

    best: FiveBarSolution | None = None
    best_score = float("inf")

    for e_elbow in (-1.0, +1.0):
      theta_ed, theta_df, valid_e = self._solve_2r_np(
        self.e,
        target,
        self.l1,
        self.lf,
        e_elbow,
      )

      d = self.e + self.l1 * self._unit_np(theta_ed)
      theta_dc = theta_df - self.foot_delta
      c = d + self.l2 * self._unit_np(theta_dc)

      for a_elbow in (-1.0, +1.0):
        theta_ab, theta_bc, valid_a = self._solve_2r_np(
          self.a,
          c,
          self.l1,
          self.l2,
          a_elbow,
        )

        b = self.a + self.l1 * self._unit_np(theta_ab)
        lower_branch = c[1] <= min(b[1], d[1]) + 1e-5

        q_a = sign * (THETA_AB_ZERO_RAD - theta_ab)
        q_e = sign * (THETA_ED_ZERO_RAD - theta_ed)
        q_b = sign * (THETA_BC_ZERO_RAD - theta_bc) - q_a
        q_d = sign * (THETA_DC_ZERO_RAD - theta_dc) - q_e

        valid = bool(valid_e and valid_a and lower_branch)
        score = (q_a - reference[0]) ** 2 + (q_e - reference[1]) ** 2

        if motor_limits is not None:
          limits = np.asarray(motor_limits, dtype=np.float64)
          if limits.shape != (2, 2):
            raise ValueError("motor_limits must have shape (2, 2).")
          violation = (
            max(0.0, limits[0, 0] - q_a)
            + max(0.0, q_a - limits[0, 1])
            + max(0.0, limits[1, 0] - q_e)
            + max(0.0, q_e - limits[1, 1])
          )
          score += 1e4 * violation * violation
          valid = valid and violation <= 1e-9

        if not valid:
          score += 1e9

        if score < best_score:
          best_score = score
          best = FiveBarSolution(
            q_a=float(q_a),
            q_b=float(q_b),
            q_e=float(q_e),
            q_d=float(q_d),
            foot_x=float(foot_x),
            foot_z=float(foot_z),
            closure_x=float(c[0]),
            closure_z=float(c[1]),
            valid=valid,
          )

    assert best is not None
    return best

  # -------------------------------------------------------------------------
  # Torch implementation used by vectorized environment resets
  # -------------------------------------------------------------------------

  @staticmethod
  def _unit_torch(theta: torch.Tensor) -> torch.Tensor:
    return torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)

  @staticmethod
  def _solve_2r_torch(
    base: torch.Tensor,
    target: torch.Tensor,
    first_length: float,
    second_length: float,
    elbow_sign: float,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offset = target - base
    distance_sq = torch.sum(torch.square(offset), dim=-1)
    cosine = (
      distance_sq
      - first_length * first_length
      - second_length * second_length
    ) / (2.0 * first_length * second_length)

    valid = (cosine >= -1.0) & (cosine <= 1.0)
    relative = float(elbow_sign) * torch.acos(
      torch.clamp(cosine, -1.0, 1.0)
    )

    first = torch.atan2(offset[..., 1], offset[..., 0]) - torch.atan2(
      second_length * torch.sin(relative),
      first_length + second_length * torch.cos(relative),
    )
    return first, first + relative, valid

  def inverse_torch(
    self,
    foot_xz: torch.Tensor,
    leg_sign: torch.Tensor | float,
    reference_motor: torch.Tensor,
    motor_limits: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized IK.

    Args:
      foot_xz: ``[N, 2]`` canonical foot targets.
      leg_sign: scalar or ``[N]`` values from ``(+1,-1,-1,+1)``.
      reference_motor: ``[N, 2]`` reference ``(q_a,q_e)``.
      motor_limits: optional ``[N,2,2]`` or ``[2,2]`` lower/upper limits.

    Returns:
      full_joint_state: ``[N,4]`` in ``(q_a,q_b,q_e,q_d)`` order.
      valid: ``[N]`` mask.
    """
    if foot_xz.ndim != 2 or foot_xz.shape[1] != 2:
      raise ValueError("foot_xz must have shape [N, 2].")
    if reference_motor.shape != foot_xz.shape:
      raise ValueError("reference_motor must have shape [N, 2].")

    device = foot_xz.device
    dtype = foot_xz.dtype
    count = foot_xz.shape[0]

    a = torch.tensor(self.a, device=device, dtype=dtype)
    e = torch.tensor(self.e, device=device, dtype=dtype)
    sign = torch.as_tensor(leg_sign, device=device, dtype=dtype)
    if sign.ndim == 0:
      sign = sign.expand(count)

    candidate_q: list[torch.Tensor] = []
    candidate_valid: list[torch.Tensor] = []
    candidate_score: list[torch.Tensor] = []

    for e_elbow in (-1.0, +1.0):
      theta_ed, theta_df, valid_e = self._solve_2r_torch(
        e,
        foot_xz,
        self.l1,
        self.lf,
        e_elbow,
      )

      d = e + self.l1 * self._unit_torch(theta_ed)
      theta_dc = theta_df - self.foot_delta
      c = d + self.l2 * self._unit_torch(theta_dc)

      for a_elbow in (-1.0, +1.0):
        theta_ab, theta_bc, valid_a = self._solve_2r_torch(
          a,
          c,
          self.l1,
          self.l2,
          a_elbow,
        )

        b = a + self.l1 * self._unit_torch(theta_ab)
        lower_branch = c[..., 1] <= (
          torch.minimum(b[..., 1], d[..., 1]) + 1e-5
        )

        q_a = sign * (THETA_AB_ZERO_RAD - theta_ab)
        q_e = sign * (THETA_ED_ZERO_RAD - theta_ed)
        q_b = sign * (THETA_BC_ZERO_RAD - theta_bc) - q_a
        q_d = sign * (THETA_DC_ZERO_RAD - theta_dc) - q_e
        q = torch.stack((q_a, q_b, q_e, q_d), dim=-1)

        valid = valid_e & valid_a & lower_branch
        score = (
          torch.square(q_a - reference_motor[:, 0])
          + torch.square(q_e - reference_motor[:, 1])
        )

        if motor_limits is not None:
          limits = torch.as_tensor(
            motor_limits,
            device=device,
            dtype=dtype,
          )
          if limits.ndim == 2:
            limits = limits[None, :, :].expand(count, -1, -1)
          if limits.shape != (count, 2, 2):
            raise ValueError(
              "motor_limits must have shape [2,2] or [N,2,2]."
            )

          lower_violation = torch.clamp(
            limits[:, :, 0] - torch.stack((q_a, q_e), dim=1),
            min=0.0,
          )
          upper_violation = torch.clamp(
            torch.stack((q_a, q_e), dim=1) - limits[:, :, 1],
            min=0.0,
          )
          violation = torch.sum(lower_violation + upper_violation, dim=1)
          valid = valid & (violation <= 1e-8)
          score = score + 1e4 * torch.square(violation)

        score = score + torch.where(
          valid,
          torch.zeros_like(score),
          torch.full_like(score, 1e9),
        )

        candidate_q.append(q)
        candidate_valid.append(valid)
        candidate_score.append(score)

    q_stack = torch.stack(candidate_q, dim=1)
    valid_stack = torch.stack(candidate_valid, dim=1)
    score_stack = torch.stack(candidate_score, dim=1)

    best_index = torch.argmin(score_stack, dim=1)
    rows = torch.arange(count, device=device)
    best_q = q_stack[rows, best_index]
    best_valid = valid_stack[rows, best_index]
    return best_q, best_valid

  def solve_all_legs_torch(
    self,
    foot_targets_xz: torch.Tensor,
    reference_motor: torch.Tensor,
    motor_limits: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve all four legs.

    Args:
      foot_targets_xz: ``[N,4,2]``.
      reference_motor: ``[N,4,2]`` in per-leg ``(q_a,q_e)`` order.
      motor_limits: optional ``[N,4,2,2]`` or ``[4,2,2]``.

    Returns:
      full_joint_state: ``[N,16]`` ordered as ``FULL_JOINT_NAMES``.
      valid_per_leg: ``[N,4]``.
    """
    if foot_targets_xz.ndim != 3 or foot_targets_xz.shape[1:] != (4, 2):
      raise ValueError("foot_targets_xz must have shape [N,4,2].")
    if reference_motor.shape != foot_targets_xz.shape:
      raise ValueError("reference_motor must have shape [N,4,2].")

    count = foot_targets_xz.shape[0]
    device = foot_targets_xz.device
    dtype = foot_targets_xz.dtype
    signs = torch.tensor(LEG_SIGNS, device=device, dtype=dtype)

    q_per_leg: list[torch.Tensor] = []
    valid_per_leg: list[torch.Tensor] = []

    for leg in range(4):
      leg_limits = None
      if motor_limits is not None:
        limits = torch.as_tensor(
          motor_limits,
          device=device,
          dtype=dtype,
        )
        if limits.ndim == 3:
          leg_limits = limits[leg]
        elif limits.ndim == 4:
          leg_limits = limits[:, leg]
        else:
          raise ValueError(
            "motor_limits must have shape [4,2,2] or [N,4,2,2]."
          )

      q_leg, valid_leg = self.inverse_torch(
        foot_targets_xz[:, leg],
        signs[leg],
        reference_motor[:, leg],
        motor_limits=leg_limits,
      )
      q_per_leg.append(q_leg)
      valid_per_leg.append(valid_leg)

    return (
      torch.cat(q_per_leg, dim=1),
      torch.stack(valid_per_leg, dim=1),
    )


def self_test() -> None:
  """Check deterministic and randomized FK/IK round trips."""
  kin = MicrotaurFiveBarKinematics()

  zero = kin.forward_numpy(0.0, 0.0, leg_index=1)
  expected = np.asarray((-0.0022797, -0.1022930))
  if not zero.valid or not np.allclose(zero.foot, expected, atol=2e-6):
    raise AssertionError(
      f"XML-zero FK mismatch: got {zero.foot}, expected {expected}."
    )

  rng = np.random.default_rng(7)
  for leg in range(1, 5):
    sign = LEG_SIGNS[leg - 1]
    reference = np.asarray((0.45 * sign, -0.45 * sign))
    for _ in range(250):
      motor = reference + rng.uniform(-0.15, 0.15, size=2)
      fk = kin.forward_numpy(motor[0], motor[1], leg)
      if not fk.valid:
        raise AssertionError("Random FK sample became invalid.")
      ik = kin.inverse_numpy(
        fk.foot_x,
        fk.foot_z,
        leg,
        reference_motor=motor,
      )
      if not ik.valid or not np.allclose(ik.motor, motor, atol=3e-6):
        raise AssertionError(
          f"FK/IK mismatch on leg {leg}: {motor} -> {ik.motor}."
        )


if __name__ == "__main__":
  self_test()
  print("Microtaur five-bar kinematics self-test passed.")