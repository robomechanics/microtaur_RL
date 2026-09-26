"""Actor / critic observations and the deployment contract.

Actor (33, deployable): base_ang_vel 3, projected_gravity 3, joint_pos 8
(encoder-biased), joint_vel 8, actions 8, command 3. No history, no delay
(the command hold in the action term models the latency).

Critic (60, privileged): actor terms with unbiased joint_pos, plus base_lin_vel
3, foot_height 4, foot_air_time 4, foot_contact 4, foot_contact_forces 12.
"""

from __future__ import annotations

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg

from .robot import LEG_JOINT_NAMES
from microtaur_common.sim2real import Sim2RealStage

ACTOR_TERMS = ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command")
CRITIC_TERMS = (
  "base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command",
  "foot_height", "foot_air_time", "foot_contact", "foot_contact_forces",
)
# Nothing a real Microtaur cannot measure may reach the actor.
SIM_ONLY_TERMS = {"base_lin_vel", "height_scan", "foot_height", "foot_air_time", "foot_contact", "foot_contacts", "foot_contact_forces"}


def _legs() -> SceneEntityCfg:
  return SceneEntityCfg("robot", joint_names=list(LEG_JOINT_NAMES))


def configure_observations(cfg, stage: Sim2RealStage, rough: bool) -> None:
  actor = cfg.observations["actor"].terms
  critic = cfg.observations["critic"].terms

  actor.pop("base_lin_vel", None)  # no hardware sensor
  actor.pop("height_scan", None)  # no hardware sensor
  if not rough:
    critic.pop("height_scan", None)

  noise = {
    "base_ang_vel": stage.ang_vel_noise,
    "projected_gravity": stage.gravity_noise,
    "joint_pos": stage.joint_pos_noise,
    "joint_vel": stage.joint_vel_noise,
  }
  for name, bound in noise.items():
    actor[name].noise = UniformNoiseCfg(n_min=-bound, n_max=bound)
  for term in actor.values():
    if hasattr(term, "delay_min_lag"):
      term.delay_min_lag = term.delay_max_lag = 0

  # Actor sees encoder-biased motor angles (hardware calibration error);
  # critic sees the true angles.
  actor["joint_pos"].params = {"asset_cfg": _legs(), "biased": True}
  critic["joint_pos"].params = {"asset_cfg": _legs()}
  actor["joint_vel"].params = {"asset_cfg": _legs()}
  critic["joint_vel"].params = {"asset_cfg": _legs()}


def validate_observation_contract(cfg, rough: bool) -> None:
  actor = cfg.observations["actor"].terms
  critic = cfg.observations["critic"].terms
  expected_critic = CRITIC_TERMS[:7] + (("height_scan",) if rough else ()) + CRITIC_TERMS[7:]
  if tuple(actor) != ACTOR_TERMS:
    raise RuntimeError(f"Actor observation contract changed: expected {ACTOR_TERMS}, got {tuple(actor)}")
  if tuple(critic) != expected_critic:
    raise RuntimeError(f"Critic observation contract changed: expected {expected_critic}, got {tuple(critic)}")
  if (actor["joint_pos"].params or {}).get("biased") is not True:
    raise RuntimeError("Actor joint_pos must be encoder-biased")
  if (critic["joint_pos"].params or {}).get("biased", False):
    raise RuntimeError("Critic joint_pos must be unbiased")
  leaked = SIM_ONLY_TERMS.intersection(actor)
  if leaked:
    raise RuntimeError(f"Simulator-only observations in the actor: {sorted(leaked)}")
