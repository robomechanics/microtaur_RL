"""Assemble the rigid Microtaur velocity environment.

Everything robot- or task-specific lives in its own module; this file only
starts from mjlab's velocity task and applies them in order.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from .actions import MicrotaurWalkActionCfg
from .commands import configure_commands
from .events import configure_events
from .observations import configure_observations, validate_observation_contract
from .rewards import configure_rewards
from .robot import ROOT_BODY, make_robot_cfg
from .sensors import configure_sensors
from .sim2real import STAGES
from .terminations import configure_terminations
from .terrain import MICRO_ROUGH_TERRAINS_CFG

PHYSICS_DT_S = 0.005
# 7 x 5 ms = 35 ms = 28.6 Hz, the policy period the hardware characterisation
# chose (Test 09: 25-33 Hz reliable, 40 Hz not deterministic) and that the
# one-step action delay and 0.10 s filter horizon were tuned for. Every 2026-09
# policy before this trained at 50 Hz: upstream's 30 Hz override tested for a
# non-existent cfg.sim.dt and never ran.
DECIMATION = 7


def make_env_cfg(
  *, play: bool = False, rough: bool = False, stage: int = 0, energy_weight: float | None = None
) -> ManagerBasedRlEnvCfg:
  s2r = STAGES[stage]
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.timestep = PHYSICS_DT_S
  cfg.decimation = DECIMATION
  cfg.sim.njmax = 1024
  cfg.sim.nconmax = 256
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 256

  cfg.scene.entities = {"robot": make_robot_cfg()}
  cfg.viewer.body_name = ROOT_BODY
  cfg.viewer.distance = 0.7
  cfg.viewer.elevation = -15.0

  configure_sensors(cfg, play=play, rough=rough)
  cfg.actions.clear()
  cfg.actions["joint_pos"] = MicrotaurWalkActionCfg(
    target_gain_range=(1.0, 1.0) if play else s2r.action_gain_range,
    target_bias_rad=0.0 if play else s2r.action_bias_rad,
  )
  configure_observations(cfg, s2r, rough=rough)
  validate_observation_contract(cfg, rough=rough)
  configure_commands(cfg, play=play)
  configure_rewards(cfg, energy_weight=energy_weight)
  configure_terminations(cfg)
  configure_events(cfg, s2r, play=play, rough=rough)

  if rough:
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = replace(MICRO_ROUGH_TERRAINS_CFG)
    cfg.scene.terrain.max_init_terrain_level = 1
  else:
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.curriculum.pop("terrain_levels", None)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
  return cfg
