"""D0 reward: command tracking plus one motor-cost model applied to every motor.

  r = + 1.0   * exp(-|v_xy_cmd - v_xy|^2 / 0.10^2)          track_lin_vel_xy
      + 0.5   * exp(-(w_z_cmd - w_z)^2 / 0.15^2)            track_ang_vel_z (always on)
      - 0.025 * sum_j (|tau_j qd_j| + c_j tau_j^2) / (m g v_ref)   motor_energy
      - 0.05/8 * sum_j (a_j - a_j,prev)^2                   action_rate
      - 2.0 per non-timeout termination                     termination

  (mjlab then multiplies every term by weight * step_dt.)

Design rules:
  * Only two kinds of quantity appear: how well the robot follows the command,
    and what its motors spend. There are no posture, gait-shape or
    motor-specific terms; posture is bounded by terminations only.
  * motor_energy and action_rate SUM over motors with one shared formula, so an
    extra motor (e.g. a spine) pays exactly what a leg motor would pay for the
    same work. Nothing is spine-specific.
  * motor_energy is an electrical-power estimate: |mechanical power| (XL330s do
    not regenerate, so braking costs too) plus copper loss R/k_t^2 * tau^2,
    which is what a motor spends holding torque at zero speed. It is divided
    by m g v_ref, so the term is a cost-of-transport at v_ref and is comparable
    across robots of different mass.

Why the termination penalty: without it (pure D0, pilot 2026-09-25) PPO learned
to end every episode in 0.1 s by dropping below the height limit. Under the
initial exploration noise (std 1.0) living costs ~15 W of copper loss per
step, i.e. ~-1.0 of discounted return, while dying costs 0. -2.0 per
termination makes failure strictly worse. It is a statement that failing has a
cost, not a posture term.

Why sigma = 0.10 m/s: with the upstream 0.04 (pilot 2) a standing robot scores
exp(-(0.14/0.04)^2) = 6e-6 on forward tracking, so there is no gradient
towards walking and PPO settled on standing still for 900 iterations.
legged_gym uses sigma ~ 50% of the command magnitude; 0.10 gives a standing
robot 0.14 and keeps precision near the command.

Magnitudes on the 2026-09 rigid RL gait (0.465 kg recording, 0.147 m/s):
mechanical 1.32 W, copper 5.07 W -> ~8 in CoT units -> motor_energy ~ -0.2 per
step against +1.5 maximum positive. The yaw term is also the de-facto alive
bonus early in training, when forward tracking is ~0.
"""

from __future__ import annotations

import torch
from mjlab.envs import mdp as env_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity import mdp

from .commands import COMMAND_NAME
from .robot import EXPECTED_TOTAL_MASS_KG, XL330_COPPER_W_PER_NM2

GRAVITY = 9.81

LIN_VEL_SIGMA_M_S = 0.10
YAW_RATE_SIGMA_RAD_S = 0.15
ENERGY_REF_SPEED_M_S = 0.15

# Energy curriculum: the energy weight is 0 while PPO discovers walking, then
# ramps linearly to its target. Pilot 3 (weight on from step 0) learned to walk
# by iteration 100 and abandoned it by 200: for the first, inefficient gait,
# walking gained +0.40 of tracking and cost 22.5 x weight of energy, so any
# weight above ~0.018 made standing still optimal.
ENERGY_WARMUP_ITERS = 200
ENERGY_RAMP_ITERS = 300
ENERGY_RAMP_STAGES = 10
STEPS_PER_ITER = 32  # rl_cfg num_steps_per_env; common_step_counter counts these

WEIGHTS = {
  "track_lin_vel_xy": 1.0,
  "track_ang_vel_z": 0.5,
  # Off for now: get a good gait first, then reintroduce energy (the sweep of
  # 2026-09-25 showed 0.005-0.018 all keep walking with the curriculum).
  "motor_energy": 0.0,
  "action_rate": -0.05 / 8,
  # mjlab multiplies by step_dt (0.02 s): -100 * 0.02 = -2.0 per termination.
  "termination": -100.0,
}


def track_lin_vel_xy(env, command_name: str, sigma: float) -> torch.Tensor:
  cmd = env.command_manager.get_command(command_name)
  v = env.scene["robot"].data.root_link_lin_vel_b
  err2 = torch.sum(torch.square(cmd[:, :2] - v[:, :2]), dim=1)
  return torch.exp(-err2 / sigma**2)


def track_ang_vel_z(env, command_name: str, sigma: float) -> torch.Tensor:
  cmd = env.command_manager.get_command(command_name)
  w = env.scene["robot"].data.root_link_ang_vel_b
  return torch.exp(-torch.square(cmd[:, 2] - w[:, 2]) / sigma**2)


def motor_power_w(env, copper_w_per_nm2: float) -> tuple[torch.Tensor, torch.Tensor]:
  """Per-env (mechanical, copper) power in W, summed over all actuated joints.

  qfrc_actuator is the joint-space actuator torque, zero on passive joints, so
  every motor in the model is included without listing it.
  """
  robot = env.scene["robot"]
  tau = robot.data.qfrc_actuator
  mech = torch.sum(torch.abs(tau * robot.data.joint_vel), dim=1)
  copper = copper_w_per_nm2 * torch.sum(torch.square(tau), dim=1)
  return mech, copper


def motor_energy(env, copper_w_per_nm2: float, mass_kg: float, ref_speed_m_s: float) -> torch.Tensor:
  mech, copper = motor_power_w(env, copper_w_per_nm2)
  return (mech + copper) / (mass_kg * GRAVITY * ref_speed_m_s)


# Metrics are computed every step whatever the reward weight is (a reward term
# with weight 0 is skipped by mjlab), so energy stays visible during warm-up.
def mechanical_power_w(env, copper_w_per_nm2: float) -> torch.Tensor:
  return motor_power_w(env, copper_w_per_nm2)[0]


def copper_power_w(env, copper_w_per_nm2: float) -> torch.Tensor:
  return motor_power_w(env, copper_w_per_nm2)[1]


def forward_speed_m_s(env) -> torch.Tensor:
  return env.scene["robot"].data.root_link_lin_vel_b[:, 0]


def energy_weight_stages(target: float) -> list[dict]:
  """0 for ENERGY_WARMUP_ITERS, then a linear ramp to `target` in equal steps."""
  stages = [{"step": 0, "weight": 0.0}]
  for k in range(1, ENERGY_RAMP_STAGES + 1):
    it = ENERGY_WARMUP_ITERS + ENERGY_RAMP_ITERS * k / ENERGY_RAMP_STAGES
    stages.append({"step": int(it * STEPS_PER_ITER), "weight": target * k / ENERGY_RAMP_STAGES})
  return stages


def configure_rewards(cfg, energy_weight: float | None = None) -> None:
  """energy_weight is the curriculum's final value (default WEIGHTS['motor_energy'])."""
  if energy_weight is None:
    energy_weight = WEIGHTS["motor_energy"]
  cfg.rewards.clear()
  cfg.rewards["track_lin_vel_xy"] = RewardTermCfg(
    func=track_lin_vel_xy, weight=WEIGHTS["track_lin_vel_xy"],
    params={"command_name": COMMAND_NAME, "sigma": LIN_VEL_SIGMA_M_S},
  )
  cfg.rewards["track_ang_vel_z"] = RewardTermCfg(
    func=track_ang_vel_z, weight=WEIGHTS["track_ang_vel_z"],
    params={"command_name": COMMAND_NAME, "sigma": YAW_RATE_SIGMA_RAD_S},
  )
  cfg.rewards["motor_energy"] = RewardTermCfg(
    func=motor_energy, weight=0.0,  # set by the energy curriculum
    params={
      "copper_w_per_nm2": XL330_COPPER_W_PER_NM2,
      "mass_kg": EXPECTED_TOTAL_MASS_KG,
      "ref_speed_m_s": ENERGY_REF_SPEED_M_S,
    },
  )
  cfg.rewards["action_rate"] = RewardTermCfg(func=mdp.action_rate_l2, weight=WEIGHTS["action_rate"])
  cfg.rewards["termination"] = RewardTermCfg(func=mdp.is_terminated, weight=WEIGHTS["termination"])

  cfg.curriculum["energy_weight"] = CurriculumTermCfg(
    func=env_mdp.reward_curriculum,
    params={"reward_name": "motor_energy", "stages": energy_weight_stages(energy_weight)},
  )
  cu = {"copper_w_per_nm2": XL330_COPPER_W_PER_NM2}
  cfg.metrics["mechanical_power_w"] = MetricsTermCfg(func=mechanical_power_w, params=cu)
  cfg.metrics["copper_power_w"] = MetricsTermCfg(func=copper_power_w, params=cu)
  cfg.metrics["forward_speed_m_s"] = MetricsTermCfg(func=forward_speed_m_s)
