"""Episode terminations (IsaacLab). Port of microtaur_rigid/terminations.py
plus the time_out / out_of_terrain_bounds terms of mjlab's velocity base cfg.

With the D1 reward these are the only posture guards: there is no upright or
body-height reward term.

IsaacLab vs mjlab:
  * illegal_contact: mjlab terminates on ANY geometric contact between the two
    body collision boxes and the terrain (sensor "found", no force threshold).
    PhysX reports forces only, so here: |net contact force| on the root body
    "battery" (which carries both boxes) > BODY_CONTACT_FORCE_N at any entry
    of the contact sensor's force history (IsaacLab's illegal_contact
    convention). With history_length >= decimation this sees every physics
    substep of the policy step. net_forces_w includes contacts with the robot's
    own links; the articulation must keep self-collisions disabled (as the
    MJCF's contype/conaffinity do) for this to mean "body touched terrain".
  * base_too_low: relative to the median height-scan hit under the body when a
    height scanner exists (rough / teacher), else to the env origin (as mjlab).
  * physics_unstable (IsaacLab only): under a fresh policy's N(0,1) actions,
    GPU PhysX blows up about one env in 65k env-steps (root velocity NaN or
    ~1e12 while the root position stays finite, so no posture term fires).
    Such an env is truncated (time_out=True: no termination penalty, the
    solver's fault rather than the policy's) and reset before observations
    are computed; its last reward is cleaned in rewards.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as il_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from microtaur_common.task_params import MAX_TILT_RAD, MIN_ROOT_HEIGHT_M

from .. import ROOT_BODY_NAME
from .observations import CONTACT_SENSOR, ROBOT

if TYPE_CHECKING:
  from isaaclab.envs import ManagerBasedRLEnv

# Any real touch: ~1% of body weight (0.54 kg -> 5.3 N), above PhysX's
# zero-force proximity contacts.
BODY_CONTACT_FORCE_N = 0.05
OUT_OF_BOUNDS_MARGIN_M = 0.3
# Far above anything physical for a 0.54 kg robot 7 cm tall.
UNSTABLE_LIN_VEL_M_S = 5.0
UNSTABLE_ANG_VEL_RAD_S = 100.0


def root_too_low(
  env: ManagerBasedRLEnv, min_height_m: float, asset_cfg: SceneEntityCfg = SceneEntityCfg(ROBOT)
) -> torch.Tensor:
  """Root height above the ground under the body < min_height_m. The ground is the
  median height-scan hit when a height scanner exists (rough terrain: a low B cell
  or the low side of C must not read as "too low"), else the env origin."""
  robot = env.scene[asset_cfg.name]
  ground = env.scene.env_origins[:, 2]
  sensors = getattr(env.scene, "sensors", {})
  if "height_scanner" in sensors:
    hz = sensors["height_scanner"].data.ray_hits_w[..., 2]
    hz = torch.where(torch.isfinite(hz), hz, ground[:, None].expand_as(hz))
    ground = torch.median(hz, dim=1).values
  return robot.data.root_link_pos_w[:, 2] - ground < min_height_m


def body_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
  """|net force| on the sensor_cfg bodies > threshold anywhere in the sensor's
  force history (identical to IsaacLab's mdp.illegal_contact)."""
  forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids]  # [N, H, B, 3]
  return torch.any(torch.amax(torch.norm(forces, dim=-1), dim=1) > threshold, dim=1)


def out_of_terrain_bounds(
  env: ManagerBasedRLEnv, margin: float = OUT_OF_BOUNDS_MARGIN_M, asset_cfg: SceneEntityCfg = SceneEntityCfg(ROBOT)
) -> torch.Tensor:
  """Truncate (time_out=True) when the root leaves the generated terrain
  footprint minus margin; all False on non-generator terrain. IsaacLab centres
  the generated grid on the world origin and adds border_width around it, as
  mjlab does."""
  terrain = getattr(env.scene, "terrain", None)
  none = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  if terrain is None or terrain.cfg.terrain_type != "generator" or terrain.terrain_origins is None:
    return none
  gen = terrain.cfg.terrain_generator
  if gen is None:
    return none
  rows, cols = terrain.terrain_origins.shape[:2]
  limit_x = max(0.0, 0.5 * rows * gen.size[0] + gen.border_width - margin)
  limit_y = max(0.0, 0.5 * cols * gen.size[1] + gen.border_width - margin)
  xy = env.scene[asset_cfg.name].data.root_link_pos_w[:, :2]
  return (xy[:, 0].abs() > limit_x) | (xy[:, 1].abs() > limit_y)


def physics_unstable(
  env: ManagerBasedRLEnv, max_lin_vel: float, max_ang_vel: float, asset_cfg: SceneEntityCfg = SceneEntityCfg(ROBOT)
) -> torch.Tensor:
  d = env.scene[asset_cfg.name].data
  state = torch.cat((d.root_link_pos_w, d.root_link_quat_w, d.root_link_lin_vel_w, d.root_link_ang_vel_w,
                     d.joint_pos, d.joint_vel), dim=1)
  bad = ~torch.isfinite(state).all(dim=1)
  lin = torch.nan_to_num(torch.linalg.norm(d.root_link_lin_vel_w, dim=1), nan=0.0)
  ang = torch.nan_to_num(torch.linalg.norm(d.root_link_ang_vel_w, dim=1), nan=0.0)
  return bad | (lin > max_lin_vel) | (ang > max_ang_vel)


@configclass
class MicrotaurTerminationsCfg:
  time_out = DoneTerm(func=il_mdp.time_out, time_out=True)
  fell_over = DoneTerm(func=il_mdp.bad_orientation, params={"limit_angle": MAX_TILT_RAD})
  illegal_contact = DoneTerm(
    func=body_contact,
    params={"threshold": BODY_CONTACT_FORCE_N, "sensor_cfg": SceneEntityCfg(CONTACT_SENSOR, body_names=ROOT_BODY_NAME)},
  )
  base_too_low = DoneTerm(func=root_too_low, params={"min_height_m": MIN_ROOT_HEIGHT_M})
  out_of_terrain_bounds = DoneTerm(func=out_of_terrain_bounds, time_out=True)
  physics_unstable = DoneTerm(
    func=physics_unstable, time_out=True,
    params={"max_lin_vel": UNSTABLE_LIN_VEL_M_S, "max_ang_vel": UNSTABLE_ANG_VEL_RAD_S},
  )


def make_terminations_cfg() -> MicrotaurTerminationsCfg:
  return MicrotaurTerminationsCfg()
