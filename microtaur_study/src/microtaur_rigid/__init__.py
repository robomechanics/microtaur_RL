"""Rigid Microtaur velocity task for mjlab (spine study, D0 reward).

Module map:
  robot.py         names, XL330 motor + electrical model, MjSpec patch, entity
  kinematics.py    closed-form five-bar FK/IK (unchanged from upstream)
  sim2real.py      stage 0/1/2 randomisation table
  actions.py       8 motor actions, predictive five-bar safety filter, delay
  observations.py  actor/critic terms and the deployment contract check
  sensors.py       contact and height sensors
  events.py        IK-consistent reset, randomisation events
  commands.py      command stages and curriculum
  rewards.py       D0 reward
  terminations.py  terminations (the only posture guards)
  terrain.py       rough-terrain generator
  env_cfg.py       assembly
  rl_cfg.py        PPO runner
"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import make_env_cfg
from .rl_cfg import microtaur_velocity_ppo_runner_cfg

for _terrain, _rough in (("Flat", False), ("Rough", True)):
  for _stage in (0, 1, 2):
    register_mjlab_task(
      task_id=f"Microtaur-Rigid-{_terrain}-S{_stage}",
      env_cfg=make_env_cfg(rough=_rough, stage=_stage),
      play_env_cfg=make_env_cfg(rough=_rough, stage=_stage, play=True),
      rl_cfg=microtaur_velocity_ppo_runner_cfg(),
      runner_cls=VelocityOnPolicyRunner,
    )
