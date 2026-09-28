"""Actor / critic / teacher observations and the deployment contract (IsaacLab).

Port of microtaur_rigid/observations.py plus the mjlab velocity-task terms it
relies on. Same term names, same order, same noise per Sim2RealStage.

Actor (33, deployable): base_ang_vel 3, projected_gravity 3, joint_pos 8
(encoder-biased), joint_vel 8, actions 8, command 3. No history, no delay.

Critic (60, privileged): base_lin_vel 3 + the actor terms with unbiased
joint_pos, [height_scan when rough], foot_height 4, foot_air_time 4,
foot_contact 4, foot_contact_forces 12.

Teacher (privileged actor): the critic's terms plus, optionally, the
height_scan from "height_scanner".

Group names: mjlab calls the actor group "actor"; rsl-rl 3.0.1 on IsaacLab
needs a "policy" entry in obs_groups, so the builders below use "policy"
(ACTOR_GROUP) and the runner maps {"policy": ["policy"], "critic": ["critic"]}.

Semantics vs mjlab (checked on the mjlab env, 2026-09-25):
  * base_ang_vel / base_lin_vel: mjlab reads gyro / velocimeter at imu_site,
    which sits at the battery origin with identity orientation, so they equal
    root_link_ang_vel_b / root_link_lin_vel_b used here.
  * foot_height: mjlab's TerrainHeightSensor returns foot-site z minus terrain
    z, i.e. the foot-sphere CENTRE height (0.0057 m on a standing robot: the
    0.0062 m radius minus 0.5 mm soft-contact penetration). This port returns
    the same quantity (offset_m=0). Pass offset_m=FOOT_SPHERE_RADIUS_M to get
    clearance instead.
  * foot_contact_forces: mjlab's netforce is the force the foot exerts on the
    terrain (z < 0 when standing); IsaacLab's net_forces_w is the force on the
    foot (z > 0). The term negates it to keep mjlab's sign, then applies the
    same sign(f) * log1p(|f|) compression.
  * foot_contact / foot_air_time: mjlab uses geometric contact ("found"); PhysX
    only reports forces, and a resting foot gets zero impulse in some physics
    substeps. Both terms come from mdp.contact.foot_contact_timers (|force| >
    the sensor's force_threshold in any substep of the policy step; timers at
    the policy rate), the same source as the trot_gait reward.
    IsaacLab's default (1.0 N) is ~20% of the robot's weight (0.54 kg -> 5.3 N,
    ~1.3 N per foot in 4-foot stance); set it to CONTACT_FORCE_THRESHOLD_N.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as il_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from isaaclab.utils.noise import UniformNoiseCfg

from microtaur_common.robot_constants import LEG_JOINT_NAMES
from microtaur_common.sim2real import Sim2RealStage
from microtaur_common.task_params import COMMAND_NAME

from .. import FOOT_BODY_NAMES, FOOT_OFFSET_IN_BODY_M
from .contact import foot_contact_timers

if TYPE_CHECKING:
  from isaaclab.envs import ManagerBasedRLEnv

ACTOR_TERMS = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command")
CRITIC_TERMS = (
  "base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command",
  "foot_height", "foot_air_time", "foot_contact", "foot_contact_forces",
)
# Nothing a real Microtaur cannot measure may reach the actor.
SIM_ONLY_TERMS = {
  "base_lin_vel", "height_scan", "foot_height", "foot_air_time", "foot_contact", "foot_contacts", "foot_contact_forces",
}

ACTOR_GROUP = "policy"
CRITIC_GROUP = "critic"
TEACHER_GROUP = "teacher"

ROBOT = "robot"
CONTACT_SENSOR = "contact_forces"
HEIGHT_SCANNER = "height_scanner"
ENCODER_BIAS_ATTR = "microtaur_encoder_bias"

# Recommended ContactSensorCfg.force_threshold for the foot contact flag and
# air-time timers: ~1% of body weight, far above PhysX's zero-force proximity
# contacts, far below one foot's stance load (~1.3 N).
CONTACT_FORCE_THRESHOLD_N = 0.05
FOOT_SPHERE_RADIUS_M = 0.0062
# mjlab height_scan on the rough critic: RayCastSensor max_distance 0.30 m and
# scale 1 / max_distance.
HEIGHT_SCAN_MAX_DISTANCE_M = 0.30


def legs_cfg() -> SceneEntityCfg:
  """The 8 motor joints in canonical LEG_JOINT_NAMES order."""
  return SceneEntityCfg(ROBOT, joint_names=list(LEG_JOINT_NAMES), preserve_order=True)


def feet_body_cfg() -> SceneEntityCfg:
  """The 4 foot bodies of the articulation, leg order 1..4."""
  return SceneEntityCfg(ROBOT, body_names=list(FOOT_BODY_NAMES), preserve_order=True)


def feet_sensor_cfg() -> SceneEntityCfg:
  """The 4 foot bodies of the contact sensor, leg order 1..4."""
  return SceneEntityCfg(CONTACT_SENSOR, body_names=list(FOOT_BODY_NAMES), preserve_order=True)


# -----------------------------------------------------------------------------
# Terms
# -----------------------------------------------------------------------------


def base_lin_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg(ROBOT)) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.root_link_lin_vel_b


def base_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg(ROBOT)) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.root_link_ang_vel_b


def projected_gravity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg(ROBOT)) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.projected_gravity_b


def encoder_bias(env: ManagerBasedRLEnv, num_joints: int) -> torch.Tensor:
  """env.microtaur_encoder_bias [N, 8] (canonical order), zeros if absent."""
  bias = getattr(env, ENCODER_BIAS_ATTR, None)
  if bias is None:
    return torch.zeros(env.num_envs, num_joints, device=env.device)
  return bias


def joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, biased: bool = False) -> torch.Tensor:
  """Motor angles relative to the default pose, in asset_cfg order (canonical
  when built with legs_cfg()). biased=True adds the per-episode encoder bias
  (hardware calibration error), as mjlab's joint_pos_biased."""
  d = env.scene[asset_cfg.name].data
  q = d.joint_pos[:, asset_cfg.joint_ids] - d.default_joint_pos[:, asset_cfg.joint_ids]
  if biased:
    q = q + encoder_bias(env, q.shape[1])
  return q


def joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  d = env.scene[asset_cfg.name].data
  return d.joint_vel[:, asset_cfg.joint_ids] - d.default_joint_vel[:, asset_cfg.joint_ids]


def foot_centres_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Foot sphere centres in world [N, 4, 3] from the foot body poses."""
  d = env.scene[asset_cfg.name].data
  pos = d.body_link_pos_w[:, asset_cfg.body_ids]
  quat = d.body_link_quat_w[:, asset_cfg.body_ids]
  offset = torch.tensor(FOOT_OFFSET_IN_BODY_M, device=pos.device, dtype=pos.dtype).expand_as(pos)
  return pos + quat_apply(quat.reshape(-1, 4), offset.reshape(-1, 3)).reshape(pos.shape)


def ground_height_below(
  env: ManagerBasedRLEnv, points_w: torch.Tensor, scanner_name: str | None, max_xy_m: float = 0.05
) -> torch.Tensor:
  """Terrain z under each point [N, P].

  With a height scanner in the scene: z of the scanner ray hit nearest in xy
  (grid 0.05 m -> <= 0.035 m away; wrong only near block/step edges). Rays that
  missed, or no hit within max_xy_m, fall back to the env origin z. Without a
  scanner: env origin z, which is exact on flat ground only.
  """
  n, p = points_w.shape[:2]
  base = env.scene.env_origins[:, 2:3].expand(n, p)
  if scanner_name is None or scanner_name not in env.scene.sensors:
    return base
  hits = env.scene.sensors[scanner_name].data.ray_hits_w  # [N, R, 3], inf on miss
  finite = torch.isfinite(hits).all(dim=-1)  # [N, R]
  hits_xy = torch.where(finite[..., None], hits[..., :2], torch.zeros_like(hits[..., :2]))
  d2 = torch.sum(torch.square(points_w[:, :, None, :2] - hits_xy[:, None, :, :]), dim=-1)  # [N, P, R]
  d2 = torch.where(finite[:, None, :], d2, torch.full_like(d2, float("inf")))
  best, idx = torch.min(d2, dim=-1)
  z = torch.gather(torch.where(finite, hits[..., 2], torch.zeros_like(hits[..., 2])), 1, idx)
  return torch.where(best <= max_xy_m**2, z, base)


def foot_height(
  env: ManagerBasedRLEnv,
  asset_cfg: SceneEntityCfg,
  offset_m: float = 0.0,
  scanner_name: str | None = HEIGHT_SCANNER,
) -> torch.Tensor:
  """Foot sphere centre height above the terrain minus offset_m, [N, 4].
  offset_m=0 is mjlab's foot_height (see module docstring)."""
  c = foot_centres_w(env, asset_cfg)
  return c[..., 2] - ground_height_below(env, c, scanner_name) - offset_m


def foot_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
  return foot_contact_timers(env, sensor_cfg)[1]


def foot_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
  """1.0 where the foot touched the ground in any substep of this policy step."""
  return foot_contact_timers(env, sensor_cfg)[0].float()


def foot_contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
  """Log-compressed net foot forces [N, 12], mjlab sign (force on the ground)."""
  f = -env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids].flatten(start_dim=1)
  return torch.sign(f) * torch.log1p(torch.abs(f))


_KIN = None


def foot_fk(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, biased: bool = True) -> torch.Tensor:
  """Five-bar forward kinematics of each leg from its two (encoder-biased) motor angles: foot
  (x, z) in the leg plane, metres, [N, 8] = (x1, z1, ..., x4, z4). The hardware computes the same
  closed form (microtaur_common.kinematics.forward_torch) from its encoders."""
  global _KIN
  if _KIN is None:
    from microtaur_common.kinematics import MicrotaurFiveBarKinematics
    _KIN = MicrotaurFiveBarKinematics()
  d = env.scene[asset_cfg.name].data
  q = d.joint_pos[:, asset_cfg.joint_ids]
  if biased:
    q = q + encoder_bias(env, q.shape[1])
  fx, fz = _KIN.forward_torch(q[:, 0::2], q[:, 1::2])
  return torch.stack((fx, fz), dim=-1).reshape(q.shape[0], 8)


def zeros_obs(env: ManagerBasedRLEnv, dim: int) -> torch.Tensor:
  """Placeholder filled with zeros (reserved slots, e.g. the second IMU of the spine robot)."""
  return torch.zeros(env.num_envs, dim, device=env.device)


def height_scan(
  env: ManagerBasedRLEnv,
  sensor_name: str = HEIGHT_SCANNER,
  max_distance: float = HEIGHT_SCAN_MAX_DISTANCE_M,
  asset_cfg: SceneEntityCfg = SceneEntityCfg(ROBOT),
) -> torch.Tensor:
  """Root height above each ray hit [N, R]; misses (inf) and hits farther
  than max_distance read max_distance, like mjlab's height_scan. Unlike
  IsaacLab's mdp.height_scan there is no 0.5 m offset."""
  hits_z = env.scene.sensors[sensor_name].data.ray_hits_w[..., 2]
  root_z = env.scene[asset_cfg.name].data.root_link_pos_w[:, 2:3]
  h = root_z - hits_z
  return torch.where(torch.isfinite(h) & (h <= max_distance), h, torch.full_like(h, max_distance))


# -----------------------------------------------------------------------------
# Groups
# -----------------------------------------------------------------------------


def _obs_legs(func, **params) -> ObsTerm:
  return ObsTerm(func=func, params={"asset_cfg": legs_cfg(), **params})


def _height_scan_term() -> ObsTerm:
  return ObsTerm(
    func=height_scan,
    params={"sensor_name": HEIGHT_SCANNER, "max_distance": HEIGHT_SCAN_MAX_DISTANCE_M},
    scale=1.0 / HEIGHT_SCAN_MAX_DISTANCE_M,
  )


@configclass
class ActorObsCfg(ObsGroup):
  """Deployable actor observations (33). Field order = observation order."""

  base_ang_vel: ObsTerm = ObsTerm(func=base_ang_vel)
  projected_gravity: ObsTerm = ObsTerm(func=projected_gravity)
  joint_pos: ObsTerm = _obs_legs(joint_pos, biased=True)
  joint_vel: ObsTerm = _obs_legs(joint_vel)
  actions: ObsTerm = ObsTerm(func=il_mdp.last_action)
  command: ObsTerm = ObsTerm(func=il_mdp.generated_commands, params={"command_name": COMMAND_NAME})
  # Frames stacked per term (0 = none). An int default (IsaacLab's is None) so Hydra can set it:
  # env.observations.policy.history_length=15 for an MLP student.
  history_length: int = 0

  def __post_init__(self):
    self.enable_corruption = True
    self.concatenate_terms = True


@configclass
class CriticObsCfg(ObsGroup):
  """Privileged critic observations (60, +height_scan when rough). Field order
  = observation order; height_scan is None (skipped) on flat ground."""

  base_lin_vel: ObsTerm = ObsTerm(func=base_lin_vel)
  base_ang_vel: ObsTerm = ObsTerm(func=base_ang_vel)
  projected_gravity: ObsTerm = ObsTerm(func=projected_gravity)
  joint_pos: ObsTerm = _obs_legs(joint_pos)
  joint_vel: ObsTerm = _obs_legs(joint_vel)
  actions: ObsTerm = ObsTerm(func=il_mdp.last_action)
  command: ObsTerm = ObsTerm(func=il_mdp.generated_commands, params={"command_name": COMMAND_NAME})
  height_scan: ObsTerm | None = None
  foot_height: ObsTerm = ObsTerm(func=foot_height, params={"asset_cfg": feet_body_cfg()})
  foot_air_time: ObsTerm = ObsTerm(func=foot_air_time, params={"sensor_cfg": feet_sensor_cfg()})
  foot_contact: ObsTerm = ObsTerm(func=foot_contact, params={"sensor_cfg": feet_sensor_cfg()})
  foot_contact_forces: ObsTerm = ObsTerm(func=foot_contact_forces, params={"sensor_cfg": feet_sensor_cfg()})

  def __post_init__(self):
    self.enable_corruption = False
    self.concatenate_terms = True


def _stage_noise(stage: Sim2RealStage) -> dict[str, float]:
  return {
    "base_ang_vel": stage.ang_vel_noise,
    "projected_gravity": stage.gravity_noise,
    "joint_pos": stage.joint_pos_noise,
    "joint_vel": stage.joint_vel_noise,
  }


def _apply_noise(group: ObsGroup, stage: Sim2RealStage) -> None:
  for name, bound in _stage_noise(stage).items():
    getattr(group, name).noise = UniformNoiseCfg(n_min=-bound, n_max=bound)


def actor_obs_cfg(stage: Sim2RealStage) -> ActorObsCfg:
  group = ActorObsCfg()
  _apply_noise(group, stage)
  return group


def critic_obs_cfg(rough: bool = False) -> CriticObsCfg:
  group = CriticObsCfg()
  if rough:
    group.height_scan = _height_scan_term()
  return group


def teacher_obs_cfg(height_scan: bool = True, stage: Sim2RealStage | None = None) -> CriticObsCfg:
  """Privileged teacher actor: the critic's terms (unbiased joint_pos) plus
  the "height_scanner" height map when height_scan. With a stage, the four
  terms the student will also see get that stage's noise."""
  group = critic_obs_cfg(rough=height_scan)
  if stage is not None:
    _apply_noise(group, stage)
    group.enable_corruption = True
  return group


@configclass
class StudentObsCfg(ObsGroup):
  """Deployable student observations, 47 per frame. Layout shared with the future active-spine
  robot (two body segments, one IMU each; the spine robot appends spine pos/vel and a 9th action):
  front IMU (gravity 3, ang vel 3), rear IMU (gravity 3, ang vel 3: zeros on the rigid robot),
  8 encoder-biased motor angles, 8 motor velocities, five-bar FK foot (x, z) x 4, last action 8,
  command 3. No base linear velocity, no terrain height. Field order = observation order."""

  projected_gravity: ObsTerm = ObsTerm(func=projected_gravity)
  base_ang_vel: ObsTerm = ObsTerm(func=base_ang_vel)
  imu2_projected_gravity: ObsTerm = ObsTerm(func=zeros_obs, params={"dim": 3})
  imu2_base_ang_vel: ObsTerm = ObsTerm(func=zeros_obs, params={"dim": 3})
  joint_pos: ObsTerm = _obs_legs(joint_pos, biased=True)
  joint_vel: ObsTerm = _obs_legs(joint_vel)
  foot_fk: ObsTerm = _obs_legs(foot_fk, biased=True)
  actions: ObsTerm = ObsTerm(func=il_mdp.last_action)
  command: ObsTerm = ObsTerm(func=il_mdp.generated_commands, params={"command_name": COMMAND_NAME})
  history_length: int = 0  # int default so Hydra can set env.observations.student.history_length

  def __post_init__(self):
    self.enable_corruption = True
    self.concatenate_terms = True


STUDENT_TERMS = ("projected_gravity", "base_ang_vel", "imu2_projected_gravity", "imu2_base_ang_vel",
                 "joint_pos", "joint_vel", "foot_fk", "actions", "command")
STUDENT_OBS_DIM = 47
FK_NOISE_LEVER_M = 0.08  # foot distance from the motor pivots: joint-angle noise x lever -> foot noise


def student_obs_cfg(stage: Sim2RealStage | None) -> StudentObsCfg:
  group = StudentObsCfg()
  if stage is not None:
    _apply_noise(group, stage)
    group.foot_fk.noise = UniformNoiseCfg(n_min=-stage.joint_pos_noise * FK_NOISE_LEVER_M,
                                          n_max=stage.joint_pos_noise * FK_NOISE_LEVER_M)
  return group


@configclass
class MicrotaurObservationsCfg:
  policy: ObsGroup = ActorObsCfg()
  critic: ObsGroup = CriticObsCfg()
  teacher: ObsGroup | None = None
  student: ObsGroup | None = None


OBS_CLIP = 100.0  # far above any normal value (joint speed <= 32 rad/s, log forces < 10, air time <= 20 s)


def make_observations_cfg(
  stage: Sim2RealStage, rough: bool = False, teacher: bool = False, play: bool = False, student: bool = False
) -> MicrotaurObservationsCfg:
  """Observation groups policy (actor), critic and optionally teacher.
  play disables actor noise, as mjlab's play config."""
  cfg = MicrotaurObservationsCfg(policy=actor_obs_cfg(stage), critic=critic_obs_cfg(rough))
  if teacher:
    cfg.teacher = teacher_obs_cfg(height_scan=True, stage=None if play else stage)
  if student:
    cfg.student = student_obs_cfg(stage)
  if play:
    cfg.policy.enable_corruption = False
    for g in (cfg.teacher, cfg.student):
      if g is not None:
        g.enable_corruption = False
  # Clip every term: a finite PhysX glitch (e.g. a leg joint of the closed chain at 1e4 rad/s
  # while the root is calm) must not reach the networks (p3 ablation: critic loss 0.07 -> inf).
  for group in (cfg.policy, cfg.critic, cfg.teacher, cfg.student):
    if group is not None:
      for term in group_terms(group).values():
        term.clip = (-OBS_CLIP, OBS_CLIP)
  if cfg.student is not None and tuple(group_terms(cfg.student)) != STUDENT_TERMS:
    raise RuntimeError(f"Student observation layout changed: {tuple(group_terms(cfg.student))}")
  validate_observation_contract(cfg, rough=rough)
  return cfg


# -----------------------------------------------------------------------------
# Contract
# -----------------------------------------------------------------------------


_GROUP_SETTINGS = {"enable_corruption", "concatenate_terms", "history_length", "flatten_history_dim", "concatenate_dim"}


def group_terms(group) -> dict[str, ObsTerm]:
  """Active terms of a group in the order the ObservationManager uses."""
  items = group.items() if isinstance(group, dict) else group.__dict__.items()
  return {k: v for k, v in items if k not in _GROUP_SETTINGS and v is not None}


def validate_observation_contract(
  obs_cfg, rough: bool, actor_group: str = ACTOR_GROUP, critic_group: str = CRITIC_GROUP
) -> None:
  get = (lambda k: obs_cfg[k]) if isinstance(obs_cfg, dict) else (lambda k: getattr(obs_cfg, k))
  actor = group_terms(get(actor_group))
  critic = group_terms(get(critic_group))
  expected_critic = CRITIC_TERMS[:7] + (("height_scan",) if rough else ()) + CRITIC_TERMS[7:]
  if tuple(actor) != ACTOR_TERMS:
    raise RuntimeError(f"Actor observation contract changed: expected {ACTOR_TERMS}, got {tuple(actor)}")
  if tuple(critic) != expected_critic:
    raise RuntimeError(f"Critic observation contract changed: expected {expected_critic}, got {tuple(critic)}")
  if (actor["joint_pos"].params or {}).get("biased") is not True:
    raise RuntimeError("Actor joint_pos must be encoder-biased")
  if (critic["joint_pos"].params or {}).get("biased", False):
    raise RuntimeError("Critic joint_pos must be unbiased")
  for name in ("joint_pos", "joint_vel"):
    for group_name, group in ((actor_group, actor), (critic_group, critic)):
      ac = group[name].params.get("asset_cfg")
      if ac is None or tuple(ac.joint_names or ()) != LEG_JOINT_NAMES or not ac.preserve_order:
        raise RuntimeError(f"{group_name}/{name} must select LEG_JOINT_NAMES in canonical order")
  leaked = SIM_ONLY_TERMS.intersection(actor)
  if leaked:
    raise RuntimeError(f"Simulator-only observations in the actor: {sorted(leaked)}")
