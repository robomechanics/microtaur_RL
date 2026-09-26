"""Task definition shared by both simulators: control rate, commands, reward
weights and parameters, terminations. Numbers only; no framework imports.
"""

from __future__ import annotations

import math

from .reset import NOMINAL_ROOT_HEIGHT_M

# --- control -------------------------------------------------------------------

# 35 ms = 28.6 Hz, the policy period the hardware characterisation chose
# (Test 09: 25-33 Hz reliable, 40 Hz not deterministic) and that the one-step
# action delay and 0.10 s filter horizon were tuned for. Every 2026-09 policy
# before this trained at 50 Hz: upstream's 30 Hz override tested for a
# non-existent cfg.sim.dt and never ran.
POLICY_DT_S = 0.035
EPISODE_S = 20.0

# --- commands ------------------------------------------------------------------

COMMAND_NAME = "twist"
RESAMPLE_S = (6.0, 10.0)
# Stages are keyed by the env's common step counter (policy steps; 32 per
# iteration). Three front-loaded stages (TERRAIN_DESIGN.md sec. 5): turning from
# step 0, finished by iteration ~625. Forward speed never goes to 0 and there
# are no standing envs: standing is not trained separately. The upstream
# schedule (0.08-0.20 m/s, yaw 0 for 25k steps, 5-10% standing) is in git history.
COMMAND_STAGES = (
  {"step": 0, "lin_vel_x": (0.10, 0.20), "ang_vel_z": (-0.10, 0.10), "standing": 0.0},
  {"step": 8_000, "lin_vel_x": (0.10, 0.30), "ang_vel_z": (-0.18, 0.18), "standing": 0.0},
  {"step": 20_000, "lin_vel_x": (0.10, 0.35), "ang_vel_z": (-0.25, 0.25), "standing": 0.0},
)

# --- reward --------------------------------------------------------------------

GRAVITY = 9.81
LIN_VEL_SIGMA_M_S = 0.10
YAW_RATE_SIGMA_RAD_S = 0.10  # 0.15 in flat_pilot2: not turning still earned most of the term
ENERGY_REF_SPEED_M_S = 0.15

WEIGHTS = {
  "track_lin_vel_xy": 1.0,
  "track_ang_vel_z": 0.5,
  # Final value of the energy curriculum. Off for now: get a good gait first
  # (the 2026-09-25 sweep showed 0.005-0.018 all keep walking with the ramp).
  "motor_energy": 0.0,
  "action_rate": -0.05 / 8,
  # Both frameworks multiply reward terms by the policy dt; this keeps the
  # penalty at -2.0 per non-timeout termination.
  "termination": -2.0 / POLICY_DT_S,
  # Spot's weighting (IsaacLab config/spot): gait 10 against velocity tracking 5.
  "trot_gait": 2.0,
}

# Energy curriculum: weight 0 while PPO discovers walking, then a linear ramp.
# Pilot 3 (weight on from step 0) walked by iteration 100 and gave it up by 200.
ENERGY_WARMUP_ITERS = 200
ENERGY_RAMP_ITERS = 300
ENERGY_RAMP_STAGES = 10
STEPS_PER_ITER = 32  # PPO num_steps_per_env; the step counter counts these

# Trot = diagonal pairs in phase: (leg1 RR, leg3 FL) and (leg2 RL, leg4 FR),
# indices in FOOT_GEOM_NAMES / foot order.
TROT_PAIRS = ((0, 2), (1, 3))
# IsaacLab Spot uses std 0.1 s^2 and max_err 0.2 s for a 0.3 s air-time target.
# Microtaur's target is ~0.15 s (upstream AIR_TIME_TARGET_S), half the time
# scale, so the squared-time std is quartered and the clip halved.
GAIT_STD_S2 = 0.1 / 4
GAIT_MAX_ERR_S = 0.2 / 2


def energy_weight_stages(target: float) -> list[dict]:
  """0 for ENERGY_WARMUP_ITERS, then a linear ramp to `target` in equal steps."""
  stages = [{"step": 0, "weight": 0.0}]
  for k in range(1, ENERGY_RAMP_STAGES + 1):
    it = ENERGY_WARMUP_ITERS + ENERGY_RAMP_ITERS * k / ENERGY_RAMP_STAGES
    stages.append({"step": int(it * STEPS_PER_ITER), "weight": target * k / ENERGY_RAMP_STAGES})
  return stages


# --- terminations --------------------------------------------------------------

MIN_ROOT_HEIGHT_M = NOMINAL_ROOT_HEIGHT_M - 0.018  # 0.052173 m
MAX_TILT_RAD = math.radians(70.0)
