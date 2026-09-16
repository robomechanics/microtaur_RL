"""Steer the robot along a target line -- a weave sinusoid, or a straight lane.

Both uses exist for the same reason: **these policies have no lateral-position
or heading reference**. The observation carries a velocity command (forward
speed and yaw rate) and nothing that says where the robot is or which way it
should be pointing. So any small heading bias integrates into unbounded lateral
drift -- measured at ~-39 deg of heading and -0.59 m of lateral travel over 12 s
for the rigid variant. On flat ground that is harmless. On the curb course it
means the robot always eventually wanders off the lip, at *any* lip height
(it still slides off a 4 mm lip), so "cost of walking along a lip" cannot be
measured without closing the position loop outside the policy.

``WeaveDriver`` with ``amplitude=0`` is therefore a straight-lane keeper; with
the default amplitude it is the weave follower described below.

Steer the robot through the weave-pole course.

The poles carry no physics, so nothing pushes the robot off a straight line --
the slalom has to come from the yaw command. This drives the robot along a
sinusoid that threads the pole line: one full left-right cycle per two poles,
with a lateral amplitude of half the pole pitch, which puts the body on
alternating sides of consecutive poles.

    y_target(x) = A * sin(2*pi * (x - x0) / (2 * spacing))

A pure-pursuit controller converts that into a yaw-rate command. The forward
command stays fixed, so CoT stays comparable with the straight-line runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


DEFAULT_EXCURSION_M = 0.06
"""Lateral half-excursion for a weave, metres.

The robot must pass *beside* each pole, so the offset at a pole has to exceed
its own half-width (feet reach ~50 mm either side of centre) plus the pole
radius (4 mm). 60 mm gives a small clearance margin.
"""

ENV_MAX_YAW_RAD_S = 0.25
"""What `set_joystick_twist_command` will actually pass through (the final
training stage's `ang_vel_z` range). Anything larger is clamped silently."""


def required_yaw_rate(amplitude: float, spacing: float, speed: float) -> float:
    """Peak yaw rate the weave path demands, rad/s.

    The path is a sinusoid of half-excursion ``amplitude`` and wavelength
    ``2 * spacing``, so its peak curvature is ``A * (pi / spacing)**2`` and the
    yaw rate needed to follow it at ``speed`` is that times the speed.
    """
    return amplitude * (math.pi / spacing) ** 2 * speed


def min_trackable_spacing(speed: float, amplitude: float = DEFAULT_EXCURSION_M,
                          max_yaw: float = ENV_MAX_YAW_RAD_S) -> float:
    """Tightest pole pitch whose weave stays inside the yaw-rate limit, m."""
    return math.pi * math.sqrt(amplitude * speed / max_yaw)


@dataclass
class WeaveDriver:
  """Pure-pursuit follower for the weave path.

  All positions are world-frame metres; ``x0`` is the x of the first pole.
  """

  x0: float
  spacing: float = 0.10
  amplitude: float | None = None
  """Lateral half-excursion (m).

  Defaults to :data:`DEFAULT_EXCURSION_M`, a FIXED value -- what a weave needs
  is for the body to clear each pole either side, which is set by the robot's
  half-width, not by the pole pitch. An earlier version defaulted this to
  ``spacing / 2``; that made the swing grow with the spacing, so widening the
  course barely reduced the demanded curvature and no spacing was trackable.
  """
  lookahead_m: float = 0.12
  """Pure-pursuit lookahead. Roughly one wheelbase works well."""
  k_yaw: float = 3.0
  """Heading-error gain (rad/s per rad)."""
  max_yaw: float = 0.25
  """Clamp on the commanded yaw rate (rad/s).

  Defaults to the env's own limit: ``set_joystick_twist_command`` clamps the
  yaw command to the final training stage's ``ang_vel_z`` range, ±0.25 rad/s.
  Asking for more than this is silently truncated, so keep them equal or the
  controller will think it is commanding turns it never gets.
  """
  y_center: float = 0.0

  def __post_init__(self):
    if self.amplitude is None:
      self.amplitude = DEFAULT_EXCURSION_M

  def target_y(self, x: float) -> float:
    """Lateral target at ``x``: a cosine, peaking AT each pole.

    Phase matters, and getting it wrong is not subtle. With a *sine* the path
    is zero at every pole and maximal between them, so the robot drives
    straight at each pole and swings wide in the gaps -- the opposite of a
    weave. (That was the original bug here, spotted from the render: "why does
    it look like the robot is aiming at the poles?")

    A cosine puts the peak at each pole: +A at pole 0, -A at pole 1, +A at pole
    2, so the robot passes alternate sides of consecutive poles, and crosses
    the pole line midway between them, which is what a weave is.
    """
    return self.y_center + self.amplitude * math.cos(
      math.pi * (x - self.x0) / self.spacing)

  def path_curvature(self, x: float) -> float:
    """Curvature of the target path at ``x`` (1/m), small-slope approximation.

    Second derivative of :meth:`target_y`, so it must use the same phase.
    """
    k = math.pi / self.spacing
    return -self.amplitude * k * k * math.cos(k * (x - self.x0))

  def yaw_command(self, x: float, y: float, heading: float,
                  speed: float | None = None) -> float:
    """Yaw rate to steer along the path, given the robot's world pose.

    Feedback alone (pure pursuit) is inherently late on a sinusoid: with a
    lookahead that is a small fraction of the wavelength the robot tracks the
    right amplitude roughly 90 deg behind the path. Measured at 0.5 m pitch:
    amplitude ratio 0.96, phase lag +91 deg. So the dominant term here is
    curvature feed-forward -- the yaw rate the path itself demands, v * kappa --
    with pure pursuit left to trim the residual.
    """
    ff = self.path_curvature(x) * speed if speed is not None else 0.0
    xa = x + self.lookahead_m
    dy = self.target_y(xa) - y
    desired_heading = math.atan2(dy, self.lookahead_m)
    err = math.atan2(math.sin(desired_heading - heading),
                     math.cos(desired_heading - heading))
    return max(-self.max_yaw, min(self.max_yaw, ff + self.k_yaw * err))

  def cross_track_error(self, x: float, y: float) -> float:
    return y - self.target_y(x)


def yaw_from_quat_wxyz(q) -> float:
  w, x, y, z = (float(v) for v in q)
  return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def first_pole_x(env_unwrapped, spacing: float, lead_in: float = 0.40) -> float:
  """World x of the first weave pole, derived from the env's spawn origin."""
  del spacing
  origin = env_unwrapped.scene.env_origins[0].detach().cpu().numpy()
  return float(origin[0]) + lead_in
