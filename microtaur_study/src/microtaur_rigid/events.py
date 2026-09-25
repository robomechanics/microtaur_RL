"""Reset and randomisation events."""

from __future__ import annotations

import math

import numpy as np
import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ._util import env_id_tensor, ordered_joint_ids, safe_log, uniform
from .kinematics import FOOT_SPHERE_RADIUS_M, FULL_JOINT_NAMES, HIP_MIDPOINTS_ROOT_M, MicrotaurFiveBarKinematics
from .robot import FOOT_GEOM_NAMES, LEG_JOINT_NAMES, ROOT_BODY, STAND_A, STAND_E
from .sim2real import PUSH_FLAT_SCALE, PUSH_INTERVAL_S, PUSH_VELOCITY_RANGE, Sim2RealStage

_KIN = MicrotaurFiveBarKinematics()

# Reset targets are built around the foot positions of the stand pose, so a
# zero action after reset does not jump.
NOMINAL_FOOT_XZ = np.stack([_KIN.forward_numpy(STAND_A[i], STAND_E[i], leg_index=i + 1).foot for i in range(4)])
NOMINAL_FULL_Q = np.stack([_KIN.forward_numpy(STAND_A[i], STAND_E[i], leg_index=i + 1).full for i in range(4)])

# Root height with the lowest foot sphere resting on the ground (0.070173 m).
NOMINAL_ROOT_HEIGHT_M = float(FOOT_SPHERE_RADIUS_M - np.min(HIP_MIDPOINTS_ROOT_M[:, 2] + NOMINAL_FOOT_XZ[:, 1]))

RESET_ROOT_XY_M = (-0.10, 0.10)
RESET_YAW_RAD = (-math.pi, math.pi)
RESET_FOOT_X_M = (-0.006, 0.006)
RESET_SHARED_Z_M = (-0.004, 0.004)  # body-height offset
RESET_PER_LEG_Z_M = (-0.001, 0.001)  # zero-mean per leg
RESET_LOWEST_FOOT_Z_M = (FOOT_SPHERE_RADIUS_M, FOOT_SPHERE_RADIUS_M + 0.002)


def ik_consistent_reset(env, env_ids, *, randomize: bool) -> None:
  """Reset root and all 16 leg joints from reachable foot targets.

  The five-bar has an analytical IK, so active and passive joints are written
  together and the closure constraints start satisfied (no reset impulse).
  """
  robot = env.scene["robot"]
  env_ids = env_id_tensor(env, env_ids)
  n = int(env_ids.numel())
  if n == 0:
    return
  dev, dt = env.device, robot.data.default_joint_pos.dtype

  nominal = torch.tensor(NOMINAL_FOOT_XZ, device=dev, dtype=dt)[None].repeat(n, 1, 1)
  reference = torch.stack(
    (torch.tensor(STAND_A, device=dev, dtype=dt), torch.tensor(STAND_E, device=dev, dtype=dt)), dim=-1
  )[None].repeat(n, 1, 1)

  targets = nominal.clone()
  if randomize:
    targets[:, :, 0] += uniform(n, (4,), RESET_FOOT_X_M, dev, dt)
    shared = uniform(n, (1,), RESET_SHARED_Z_M, dev, dt)
    per_leg = uniform(n, (4,), RESET_PER_LEG_Z_M, dev, dt)
    per_leg -= per_leg.mean(dim=1, keepdim=True)
    targets[:, :, 1] += shared + per_leg

  motor_ids = ordered_joint_ids(robot, LEG_JOINT_NAMES, dev)
  limits = robot.data.joint_pos_limits[env_ids][:, motor_ids, :].reshape(n, 4, 2, 2)
  full_q, valid = _KIN.solve_all_legs_torch(targets, reference, motor_limits=limits)
  full_q = full_q.reshape(n, 4, 4)
  # Fall back to the nominal closed-chain state on IK / joint-limit edge cases.
  full_q = torch.where(valid[:, :, None], full_q, torch.tensor(NOMINAL_FULL_Q, device=dev, dtype=dt)[None])
  targets = torch.where(valid[:, :, None], targets, nominal)

  hip_z = torch.tensor(HIP_MIDPOINTS_ROOT_M[:, 2], device=dev, dtype=dt)[None]
  if randomize:
    lowest = uniform(n, (), RESET_LOWEST_FOOT_Z_M, dev, dt)
  else:
    lowest = torch.full((n,), RESET_LOWEST_FOOT_Z_M[0], device=dev, dtype=dt)
  root_height = lowest - torch.amin(hip_z + targets[:, :, 1], dim=1)

  root = robot.data.default_root_state[env_ids].clone()
  root[:, 0:3] = env.scene.env_origins[env_ids].to(dt)
  root[:, 2] += root_height
  root[:, 7:13] = 0.0
  yaw = torch.zeros(n, device=dev, dtype=dt)
  if randomize:
    root[:, 0] += uniform(n, (), RESET_ROOT_XY_M, dev, dt)
    root[:, 1] += uniform(n, (), RESET_ROOT_XY_M, dev, dt)
    yaw = uniform(n, (), RESET_YAW_RAD, dev, dt)
  root[:, 3], root[:, 4], root[:, 5], root[:, 6] = torch.cos(0.5 * yaw), 0.0, 0.0, torch.sin(0.5 * yaw)

  full_flat = full_q.reshape(n, 16)
  robot.write_root_state_to_sim(root, env_ids=env_ids)
  robot.write_joint_state_to_sim(
    full_flat, torch.zeros_like(full_flat),
    joint_ids=ordered_joint_ids(robot, FULL_JOINT_NAMES, dev), env_ids=env_ids,
  )
  # Seed motor targets from the reset pose so the first delayed command does
  # not pull the mechanism away. mjlab >= 1.6 outer-indexes 1-D ids itself.
  robot.set_joint_position_target(full_q[:, :, (0, 2)].reshape(n, 8), joint_ids=motor_ids, env_ids=env_ids)

  safe_log(env, "Metrics/ik_reset/valid_fraction", valid.float().mean())
  safe_log(env, "Metrics/ik_reset/root_height", root_height.mean())


def configure_events(cfg, stage: Sim2RealStage, play: bool, rough: bool) -> None:
  ev = cfg.events
  ev.pop("reset_base", None)
  ev.pop("reset_robot_joints", None)
  ev["ik_consistent_reset"] = EventTermCfg(func=ik_consistent_reset, mode="reset", params={"randomize": not play})

  # Test 20 did not justify changing the nominal contact model: stage 0 keeps
  # the XML friction exactly.
  if play or stage.foot_friction_range is None:
    ev.pop("foot_friction", None)
  else:
    ev["foot_friction"].params["asset_cfg"] = SceneEntityCfg("robot", geom_names=list(FOOT_GEOM_NAMES))
    ev["foot_friction"].params["ranges"] = stage.foot_friction_range

  # The nominal COM is already the measured one; only residual uncertainty.
  if play or stage.root_com_xy_m is None:
    ev.pop("base_com", None)
  else:
    xy, z = stage.root_com_xy_m, stage.root_com_z_m
    ev["base_com"].params["asset_cfg"] = SceneEntityCfg("robot", body_names=[ROOT_BODY])
    ev["base_com"].params["ranges"] = {0: (-xy, xy), 1: (-xy, xy), 2: (-z, z)}

  if play:
    ev.pop("encoder_bias", None)
  else:
    ev["encoder_bias"].params["asset_cfg"] = SceneEntityCfg("robot", joint_names=list(LEG_JOINT_NAMES))
    ev["encoder_bias"].params["bias_range"] = (-stage.encoder_bias_rad, stage.encoder_bias_rad)

  if play or not (rough or stage.push):
    ev.pop("push_robot", None)
  else:
    ev["push_robot"].interval_range_s = PUSH_INTERVAL_S
    k = 1.0 if rough else PUSH_FLAT_SCALE
    ev["push_robot"].params["velocity_range"] = {a: (-k * v, k * v) for a, v in PUSH_VELOCITY_RANGE.items()}
