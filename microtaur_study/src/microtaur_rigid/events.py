"""Reset and randomisation events."""

from __future__ import annotations

import torch
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from microtaur_common.kinematics import FULL_JOINT_NAMES
from microtaur_common.reset import NOMINAL_ROOT_HEIGHT_M, sample_reset  # noqa: F401  (re-exported)
from microtaur_common.sim2real import PUSH_FLAT_SCALE, PUSH_INTERVAL_S, PUSH_VELOCITY_RANGE, Sim2RealStage

from ._util import env_id_tensor, ordered_joint_ids, safe_log
from .robot import FOOT_GEOM_NAMES, LEG_JOINT_NAMES, ROOT_BODY

def ik_consistent_reset(env, env_ids, *, randomize: bool) -> None:
  """Reset root and all 16 leg joints from reachable foot targets (see microtaur_common.reset)."""
  robot = env.scene["robot"]
  env_ids = env_id_tensor(env, env_ids)
  n = int(env_ids.numel())
  if n == 0:
    return
  dev, dt = env.device, robot.data.default_joint_pos.dtype
  motor_ids = ordered_joint_ids(robot, LEG_JOINT_NAMES, dev)
  limits = robot.data.joint_pos_limits[env_ids][:, motor_ids, :].reshape(n, 4, 2, 2)
  r = sample_reset(n, randomize, limits, dev, dt)

  root = robot.data.default_root_state[env_ids].clone()
  root[:, 0:3] = env.scene.env_origins[env_ids].to(dt)
  root[:, 2] += r["root_height"]
  root[:, 7:13] = 0.0
  root[:, 0:2] += r["root_xy"]
  yaw = r["yaw"]
  root[:, 3], root[:, 4], root[:, 5], root[:, 6] = torch.cos(0.5 * yaw), 0.0, 0.0, torch.sin(0.5 * yaw)

  robot.write_root_state_to_sim(root, env_ids=env_ids)
  robot.write_joint_state_to_sim(
    r["full_q"], torch.zeros_like(r["full_q"]),
    joint_ids=ordered_joint_ids(robot, FULL_JOINT_NAMES, dev), env_ids=env_ids,
  )
  # Seed motor targets from the reset pose so the first delayed command does
  # not pull the mechanism away. mjlab >= 1.6 outer-indexes 1-D ids itself.
  robot.set_joint_position_target(r["motor_q"], joint_ids=motor_ids, env_ids=env_ids)

  safe_log(env, "Metrics/ik_reset/valid_fraction", r["valid"].float().mean())
  safe_log(env, "Metrics/ik_reset/root_height", r["root_height"].mean())


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
