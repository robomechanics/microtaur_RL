from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  microtaur_velocity_flat_env_cfg,
  microtaur_velocity_rough_env_cfg,
  microtaur_velocity_gait_flat_env_cfg,
  microtaur_velocity_gait_rough_env_cfg,
  microtaur_velocity_trot_flat_env_cfg,
  microtaur_velocity_trot_rough_env_cfg,
  microtaur_velocity_bound_flat_env_cfg,
  microtaur_velocity_bound_rough_env_cfg,
)
from .env_cfgs import microtaur_velocity_yaw_flat_env_cfg
from .rl_cfg import microtaur_velocity_ppo_runner_cfg


# ---------------------------------------------------------------------------
# Base Microtaur velocity tasks
# ---------------------------------------------------------------------------

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-microtaur_velocity",
  env_cfg=microtaur_velocity_rough_env_cfg(),
  play_env_cfg=microtaur_velocity_rough_env_cfg(play=True),
  rl_cfg=microtaur_velocity_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-microtaur_velocity",
  env_cfg=microtaur_velocity_flat_env_cfg(),
  play_env_cfg=microtaur_velocity_flat_env_cfg(play=True),
  rl_cfg=microtaur_velocity_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)


# ---------------------------------------------------------------------------
# Boolean-switch gait tasks
# ---------------------------------------------------------------------------
#
# These use MICROTAUR_BOUND_GAIT:
#
#   MICROTAUR_BOUND_GAIT=0 -> trot
#   MICROTAUR_BOUND_GAIT=1 -> bound
#
# Useful when you want one task name and switch gait by env var.

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-microtaur_gait",
  env_cfg=microtaur_velocity_gait_rough_env_cfg(),
  play_env_cfg=microtaur_velocity_gait_rough_env_cfg(play=True),
  rl_cfg=microtaur_velocity_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-microtaur_gait",
  env_cfg=microtaur_velocity_gait_flat_env_cfg(),
  play_env_cfg=microtaur_velocity_gait_flat_env_cfg(play=True),
  rl_cfg=microtaur_velocity_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)


# ---------------------------------------------------------------------------
# Explicit trot-shaped Microtaur velocity tasks
# ---------------------------------------------------------------------------

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-microtaur_trot",
  env_cfg=microtaur_velocity_trot_rough_env_cfg(),
  play_env_cfg=microtaur_velocity_trot_rough_env_cfg(play=True),
  rl_cfg=microtaur_velocity_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-microtaur_trot",
  env_cfg=microtaur_velocity_trot_flat_env_cfg(),
  play_env_cfg=microtaur_velocity_trot_flat_env_cfg(play=True),
  rl_cfg=microtaur_velocity_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)


# ---------------------------------------------------------------------------
# Explicit bound-shaped Microtaur velocity tasks
# ---------------------------------------------------------------------------

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-microtaur_bound",
  env_cfg=microtaur_velocity_bound_rough_env_cfg(),
  play_env_cfg=microtaur_velocity_bound_rough_env_cfg(play=True),
  rl_cfg=microtaur_velocity_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-microtaur_bound",
  env_cfg=microtaur_velocity_bound_flat_env_cfg(),
  play_env_cfg=microtaur_velocity_bound_flat_env_cfg(play=True),
  rl_cfg=microtaur_velocity_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
    task_id="Mjlab-Velocity-Yaw-Flat-Microtaur",
    env_cfg=microtaur_velocity_yaw_flat_env_cfg(),
    play_env_cfg=microtaur_velocity_yaw_flat_env_cfg(play=True),
    rl_cfg=microtaur_velocity_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)