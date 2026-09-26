"""Foot contact per policy step, and air / contact timers built on it.

Why not the ContactSensor's own current_air_time / current_contact_time:
PhysX reports a force per physics substep, and a foot resting on the ground
gets zero contact impulse in some substeps. Measured on the flat_pilot2 policy
(2026-09-26): with the foot within 0.5 mm of the ground, only 57% of the 14
substeps had |F| > 0.05 N (p10 0 N), so the sensor's timers restarted every
~10 ms and the trot GaitReward saw near-zero, near-equal times for every foot
(i.e. it scored foot chatter as a perfect trot). mjlab's contact is geometric
(no flicker).

Here a foot is in contact for a policy step if |F| > the sensor's
force_threshold in ANY substep of that step (net_forces_w_history, which must
hold >= decimation entries). With the foot down this is 98.5% (foot up: 0%).
Timers advance by step_dt per policy step. State lives on the env and is
updated once per step (keyed by env.common_step_counter), whichever of the
reward / observation / script asks first; envs reset in this step
(episode_length_buf == 0) read as zero until their next step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
  from isaaclab.envs import ManagerBasedRLEnv

_ATTR = "_microtaur_foot_timers"


def foot_contact_timers(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> tuple[torch.Tensor, ...]:
  """(contact bool [N, F], air_time s [N, F], contact_time s [N, F]) for the
  sensor_cfg bodies (the feet, in the order of sensor_cfg.body_ids)."""
  sensor = env.scene.sensors[sensor_cfg.name]
  ids = sensor_cfg.body_ids
  n_feet = len(ids) if not isinstance(ids, slice) else sensor.num_bodies
  step = int(getattr(env, "common_step_counter", 0))
  st = getattr(env, _ATTR, None)
  if st is None:
    z = torch.zeros(env.num_envs, n_feet, device=env.device)
    st = {"step": -1, "contact": z.bool(), "air": z.clone(), "con": z.clone()}
    setattr(env, _ATTR, st)
  if st["step"] != step:
    hist = sensor.data.net_forces_w_history[:, : env.cfg.decimation][:, :, ids]  # [N, H, F, 3]
    contact = (torch.linalg.norm(hist, dim=-1) > sensor.cfg.force_threshold).any(dim=1)
    fresh = (env.episode_length_buf <= 1)[:, None]  # first step of an episode: timers restart
    air = torch.where(fresh, torch.zeros_like(st["air"]), st["air"])
    con = torch.where(fresh, torch.zeros_like(st["con"]), st["con"])
    dt = env.step_dt
    st.update(step=step, contact=contact,
              air=torch.where(contact, torch.zeros_like(air), air + dt),
              con=torch.where(contact, con + dt, torch.zeros_like(con)))
  reset = (env.episode_length_buf == 0)[:, None]
  zero = torch.zeros_like(st["air"])
  return (st["contact"] & ~reset, torch.where(reset, zero, st["air"]), torch.where(reset, zero, st["con"]))
