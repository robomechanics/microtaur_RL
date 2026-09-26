"""Reset and randomisation events (IsaacLab)."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_from_euler_xyz

from microtaur_common.kinematics import FULL_JOINT_NAMES
from microtaur_common.reset import sample_reset
from microtaur_common.robot_constants import LEG_JOINT_NAMES


def _ids(env, env_ids) -> torch.Tensor:
  if env_ids is None or isinstance(env_ids, slice):
    return torch.arange(env.num_envs, device=env.device)
  return env_ids.to(env.device, dtype=torch.long)


def encoder_bias(env, env_ids, bias_rad: float) -> None:
  """Startup: per-env motor encoder zero error, U(-bias, bias), canonical order.

  Stored as env.microtaur_encoder_bias [N, 8]; the actor's joint_pos adds it and
  the action term subtracts it from the target (the mjlab convention).
  """
  if not hasattr(env, "microtaur_encoder_bias"):
    env.microtaur_encoder_bias = torch.zeros(env.num_envs, 8, device=env.device)
  ids = _ids(env, env_ids)
  env.microtaur_encoder_bias[ids] = (2.0 * torch.rand(len(ids), 8, device=env.device) - 1.0) * bias_rad


def ik_consistent_reset(env, env_ids, randomize: bool, spawn_fn=None) -> None:
  """Reset root and all 16 leg joints from reachable foot targets (microtaur_common.reset).

  spawn_fn(env, env_ids, xy, yaw) -> (xy [n, 2], yaw [n], ground_z [n]) lets the
  terrain override the spawn: terrain C starts at the course start heading +x,
  terrain B spawns anywhere on its fixed map and lifts the root by the highest
  cell under the footprint. Without it: the sampled offset, heading, ground 0.
  """
  robot: Articulation = env.scene["robot"]
  ids = _ids(env, env_ids)
  n = len(ids)
  if n == 0:
    return
  dev, dt = env.device, robot.data.default_joint_pos.dtype
  motor_ids, _ = robot.find_joints(list(LEG_JOINT_NAMES), preserve_order=True)
  full_ids, _ = robot.find_joints(list(FULL_JOINT_NAMES), preserve_order=True)
  limits = robot.data.joint_pos_limits[ids][:, motor_ids, :].reshape(n, 4, 2, 2)
  r = sample_reset(n, randomize, limits, dev, dt)

  xy, yaw = r["root_xy"], r["yaw"]
  ground = torch.zeros_like(yaw)
  if spawn_fn is not None:
    xy, yaw, ground = spawn_fn(env, ids, xy, yaw)

  pos = env.scene.env_origins[ids].clone()
  pos[:, :2] += xy
  pos[:, 2] += r["root_height"] + ground
  # Remembered for the terrain curriculum: distance walked is measured from the
  # spawn, not the env origin (C and B spawn away from the tile centre).
  if not hasattr(env, "microtaur_spawn_xy"):
    env.microtaur_spawn_xy = env.scene.env_origins[:, :2].clone()
  env.microtaur_spawn_xy[ids] = pos[:, :2]
  zeros = torch.zeros_like(yaw)
  quat = quat_from_euler_xyz(zeros, zeros, yaw)
  robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=ids)
  # write_root_velocity_to_sim (the CoM version) leaves the cached
  # data.root_link_vel_w of this step stale, so the reset observation would
  # carry the previous episode's velocity (NaN after a solver blow-up); the
  # link version updates both buffers.
  robot.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=ids)
  robot.write_joint_state_to_sim(r["full_q"], torch.zeros_like(r["full_q"]), joint_ids=full_ids, env_ids=ids)
  # Seed motor targets from the reset pose so the first delayed command does
  # not pull the mechanism away.
  robot.set_joint_position_target(r["motor_q"], joint_ids=motor_ids, env_ids=ids)
