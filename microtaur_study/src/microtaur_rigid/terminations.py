"""Episode terminations.

With the D0 reward these are the only posture guards: there is no upright or
body-height reward term.
"""

from __future__ import annotations

import math

import torch
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp

from .events import NOMINAL_ROOT_HEIGHT_M
from .sensors import BODY_CONTACT

MIN_ROOT_HEIGHT_M = NOMINAL_ROOT_HEIGHT_M - 0.018  # 0.052173 m
MAX_TILT_RAD = math.radians(70.0)


def root_too_low(env, min_height_m: float) -> torch.Tensor:
  robot = env.scene["robot"]
  return robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] < min_height_m


def configure_terminations(cfg) -> None:
  t = cfg.terminations
  t["fell_over"] = TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": MAX_TILT_RAD})
  t["illegal_contact"] = TerminationTermCfg(func=mdp.illegal_contact, params={"sensor_name": BODY_CONTACT})
  t["base_too_low"] = TerminationTermCfg(func=root_too_low, params={"min_height_m": MIN_ROOT_HEIGHT_M})
