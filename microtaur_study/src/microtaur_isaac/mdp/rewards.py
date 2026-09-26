"""D1 reward (IsaacLab): command tracking, motor energy, action rate,
termination penalty and a trot-gait kernel. Port of microtaur_rigid/rewards.py.

  r = + 1.0   * exp(-|v_xy_cmd - v_xy|^2 / 0.10^2)          track_lin_vel_xy
      + 0.5   * exp(-(w_z_cmd - w_z)^2 / 0.15^2)            track_ang_vel_z
      + w_E   * sum_j (|tau_j qd_j| + c_j tau_j^2) / (m g v_ref)   motor_energy (curriculum)
      - 0.05/8 * sum_j (a_j - a_j,prev)^2                   action_rate
      - 2.0/dt per non-timeout termination                  termination
      + 2.0   * trot kernel (Spot GaitReward)               trot_gait
      + w_A   * Spot air_time_reward (phase durations)      feet_air_time (0 = off by default)

Weights come from task_params.WEIGHTS; the maths from reward_math. IsaacLab's
RewardManager.compute multiplies every term by weight * step_dt (checked in
isaaclab/managers/reward_manager.py and by tests/check_isaac_mdp.py), the same
as mjlab, so task_params' weights carry over unchanged (termination:
-2.0/0.035 * 0.035 = -2.0 per termination). Terms with weight 0 are skipped.

IsaacLab vs mjlab:
  * motor torque: mjlab uses qfrc_actuator; here robot.data.applied_torque on
    the 8 motor joints. For explicit actuators (DCMotorCfg) applied_torque is
    the PD torque AFTER the DC-motor torque-speed clip and effort_limit, the
    value written to PhysX as joint effort (Articulation._apply_actuator_model).
    It is from the last physics substep of the policy step, and joint_vel is
    after that substep, as in mjlab. For an ImplicitActuatorCfg it would only be
    an estimate (PhysX computes the drive torque itself), so the motors must
    use an explicit actuator. Joint friction (frictionloss) is not included in
    either simulator.
  * is_terminated: termination_manager.terminated = OR of the terms with
    time_out=False, computed in the same step before the reward, same as mjlab.
  * trot_gait: air / contact times from mdp.contact.foot_contact_timers, NOT
    the sensor's current_air_time / current_contact_time: PhysX gives a resting
    foot zero impulse in some substeps, which restarted the sensor's timers
    every ~10 ms and let foot chatter score as a perfect trot (flat_pilot2).
    Contact = |net force| > force_threshold in any substep of the policy step;
    timers at the policy rate (mjlab: geometric contact, per substep).
  * finite inputs: an env whose PhysX state blew up (terminations.physics_unstable)
    still gets one reward before its reset; the simulator quantities read here
    pass through _finite() so that reward, and the metric sums, stay finite.
  * metrics: IsaacLab has no metrics manager. energy_speed_metrics is a reward
    term that returns zeros (weight 1, so the manager does not skip it),
    accumulates per-step mechanical / copper power and forward speed, and on
    reset writes the episode means of the resetting envs to
    env.extras["log"]["Episode_Metrics/<name>"] (mjlab's key format).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.envs.mdp as il_mdp
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from microtaur_common import reward_math as RM
from microtaur_common.robot_constants import EXPECTED_TOTAL_MASS_KG, XL330_COPPER_W_PER_NM2
from microtaur_common.task_params import (  # noqa: F401  (re-exported)
  AIR_TIME_MODE_S, AIR_TIME_VELOCITY_THRESHOLD_M_S, AIR_TIME_WEIGHT, COMMAND_NAME, ENERGY_REF_SPEED_M_S, GAIT_MAX_ERR_S, GAIT_STD_S2, GRAVITY, LIN_VEL_SIGMA_M_S, TROT_PAIRS,
  WEIGHTS, YAW_RATE_SIGMA_RAD_S, energy_weight_stages,
)

from .contact import foot_contact_timers
from .terminations import UNSTABLE_ANG_VEL_RAD_S, UNSTABLE_LIN_VEL_M_S, physics_unstable
from .observations import ROBOT, feet_sensor_cfg, legs_cfg

if TYPE_CHECKING:
  from isaaclab.envs import ManagerBasedRLEnv


def _finite(x: torch.Tensor) -> torch.Tensor:
  return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def track_lin_vel_xy(
  env: ManagerBasedRLEnv, command_name: str, sigma: float, asset_cfg: SceneEntityCfg = SceneEntityCfg(ROBOT)
) -> torch.Tensor:
  return RM.lin_vel_tracking(
    env.command_manager.get_command(command_name), _finite(env.scene[asset_cfg.name].data.root_link_lin_vel_b), sigma
  )


def track_ang_vel_z(
  env: ManagerBasedRLEnv, command_name: str, sigma: float, asset_cfg: SceneEntityCfg = SceneEntityCfg(ROBOT)
) -> torch.Tensor:
  return RM.yaw_rate_tracking(
    env.command_manager.get_command(command_name), _finite(env.scene[asset_cfg.name].data.root_link_ang_vel_b), sigma
  )


def motor_power_w(
  env: ManagerBasedRLEnv, copper_w_per_nm2: float, asset_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  """Per-env (|mechanical|, copper) power in W summed over the asset_cfg joints
  (the 8 motors). Torque = applied_torque (after the DC-motor clip)."""
  d = env.scene[asset_cfg.name].data
  ids = asset_cfg.joint_ids
  return RM.motor_power(_finite(d.applied_torque[:, ids]), _finite(d.joint_vel[:, ids]), copper_w_per_nm2)


def motor_energy(
  env: ManagerBasedRLEnv, copper_w_per_nm2: float, mass_kg: float, ref_speed_m_s: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Electrical power as cost of transport at ref_speed_m_s (positive; the
  curriculum's weight is negative)."""
  mech, copper = motor_power_w(env, copper_w_per_nm2, asset_cfg)
  return (mech + copper) / (mass_kg * GRAVITY * ref_speed_m_s)


def trot_gait(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, pairs, std: float, max_err: float) -> torch.Tensor:
  _, air, con = foot_contact_timers(env, sensor_cfg)
  return RM.trot_gait(air, con, pairs, std, max_err)


def feet_air_time(
  env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, mode_time_s: float, velocity_threshold_m_s: float,
  command_name: str = COMMAND_NAME, asset_cfg: SceneEntityCfg = SceneEntityCfg(ROBOT),
) -> torch.Tensor:
  """IsaacLab Spot air_time_reward on the policy-step contact timers of mdp.contact
  (Spot reads the sensor's timers, which flicker here; see contact.py)."""
  _, air, con = foot_contact_timers(env, sensor_cfg)
  cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
  speed = torch.linalg.norm(_finite(env.scene[asset_cfg.name].data.root_link_lin_vel_b[:, :2]), dim=1)
  return RM.air_time_spot(air, con, cmd, speed, mode_time_s, velocity_threshold_m_s)


def forward_speed_m_s(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg(ROBOT)) -> torch.Tensor:
  return _finite(env.scene[asset_cfg.name].data.root_link_lin_vel_b[:, 0])


METRIC_NAMES = ("mechanical_power_w", "copper_power_w", "forward_speed_m_s")


class energy_speed_metrics(ManagerTermBase):
  """Logging-only reward term (returns zeros; give it weight 1.0).

  Keeps per-env episode sums of mechanical power, copper power (W) and forward
  speed (m/s) and, on reset, writes the mean over the resetting envs of their
  per-step episode average to env.extras["log"]["Episode_Metrics/<name>"].
  RewardManager.reset runs after ManagerBasedRLEnv._reset_idx has created
  extras["log"], so the values reach the runner's episode logging.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
    super().__init__(cfg, env)
    self._sums = torch.zeros(env.num_envs, len(METRIC_NAMES), device=env.device)
    self._count = torch.zeros(env.num_envs, device=env.device)
    self.last = torch.zeros(env.num_envs, len(METRIC_NAMES), device=env.device)

  def reset(self, env_ids: Sequence[int] | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    counts = torch.clamp(self._count[env_ids], min=1.0)
    means = torch.mean(self._sums[env_ids] / counts[:, None], dim=0)
    log = getattr(self._env, "extras", {}).get("log")
    if isinstance(log, dict):
      for k, name in enumerate(METRIC_NAMES):
        log[f"Episode_Metrics/{name}"] = means[k]
    self._sums[env_ids] = 0.0
    self._count[env_ids] = 0.0

  def __call__(
    self, env: ManagerBasedRLEnv, copper_w_per_nm2: float, asset_cfg: SceneEntityCfg
  ) -> torch.Tensor:
    mech, copper = motor_power_w(env, copper_w_per_nm2, asset_cfg)
    self.last = torch.stack((mech, copper, forward_speed_m_s(env)), dim=1)
    # A step whose PhysX state blew up (finite but ~1e11 velocities) is left out.
    ok = ~physics_unstable(env, UNSTABLE_LIN_VEL_M_S, UNSTABLE_ANG_VEL_RAD_S)
    self._sums += torch.where(ok[:, None], self.last, torch.zeros_like(self.last))
    self._count += ok.float()
    return torch.zeros(env.num_envs, device=env.device)


@configclass
class MicrotaurRewardsCfg:
  track_lin_vel_xy = RewardTermCfg(
    func=track_lin_vel_xy, weight=WEIGHTS["track_lin_vel_xy"],
    params={"command_name": COMMAND_NAME, "sigma": LIN_VEL_SIGMA_M_S},
  )
  track_ang_vel_z = RewardTermCfg(
    func=track_ang_vel_z, weight=WEIGHTS["track_ang_vel_z"],
    params={"command_name": COMMAND_NAME, "sigma": YAW_RATE_SIGMA_RAD_S},
  )
  motor_energy = RewardTermCfg(
    func=motor_energy, weight=0.0,  # set by the energy curriculum
    params={
      "copper_w_per_nm2": XL330_COPPER_W_PER_NM2,
      "mass_kg": EXPECTED_TOTAL_MASS_KG,
      "ref_speed_m_s": ENERGY_REF_SPEED_M_S,
      "asset_cfg": legs_cfg(),
    },
  )
  action_rate = RewardTermCfg(func=il_mdp.action_rate_l2, weight=WEIGHTS["action_rate"])
  termination = RewardTermCfg(func=il_mdp.is_terminated, weight=WEIGHTS["termination"])
  trot_gait = RewardTermCfg(
    func=trot_gait, weight=WEIGHTS["trot_gait"],
    params={"sensor_cfg": feet_sensor_cfg(), "pairs": TROT_PAIRS, "std": GAIT_STD_S2, "max_err": GAIT_MAX_ERR_S},
  )
  feet_air_time = RewardTermCfg(
    func=feet_air_time, weight=AIR_TIME_WEIGHT,
    params={"sensor_cfg": feet_sensor_cfg(), "mode_time_s": AIR_TIME_MODE_S,
            "velocity_threshold_m_s": AIR_TIME_VELOCITY_THRESHOLD_M_S},
  )
  metrics = RewardTermCfg(
    func=energy_speed_metrics, weight=1.0,
    params={"copper_w_per_nm2": XL330_COPPER_W_PER_NM2, "asset_cfg": legs_cfg()},
  )


def make_rewards_cfg() -> MicrotaurRewardsCfg:
  return MicrotaurRewardsCfg()


def reward_weight_stages(env: ManagerBasedRLEnv, env_ids, reward_name: str, stages: Sequence[dict]) -> dict:
  """Curriculum: the weight of the last stage whose "step" the env's
  common_step_counter has reached (>=, as mjlab's reward_curriculum; IsaacLab's
  modify_reward_weight uses > and a single step). Before the first stage the
  cfg weight is left unchanged. IsaacLab calls curriculum terms from
  _reset_idx only, i.e. whenever some env resets."""
  del env_ids
  step = int(env.common_step_counter)
  term = env.reward_manager.get_term_cfg(reward_name)
  reached = [s for s in stages if step >= s["step"]]
  if reached:
    term.weight = float(reached[-1]["weight"])
    env.reward_manager.set_term_cfg(reward_name, term)
  return {"weight": torch.tensor(float(term.weight)), "step": torch.tensor(float(step))}


def make_energy_curriculum_term(energy_weight: float | None = None) -> CurriculumTermCfg:
  """energy_weight is the ramp's final value (default WEIGHTS['motor_energy'];
  negative, since motor_energy returns a positive cost)."""
  if energy_weight is None:
    energy_weight = WEIGHTS["motor_energy"]
  return CurriculumTermCfg(
    func=reward_weight_stages,
    params={"reward_name": "motor_energy", "stages": energy_weight_stages(energy_weight)},
  )
