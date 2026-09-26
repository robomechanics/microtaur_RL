"""Check the D0 reward terms against hand-computed values.

  1. qfrc_actuator (used by motor_energy) equals the DC-motor output torque on
     the 8 motor joints and is zero on the 8 passive joints.
  2. Each reward function equals the formula in rewards.py recomputed here from
     raw simulator state, evaluated on the same state. (The reward manager
     itself sees body-derived quantities one physics substep stale -- mjlab
     calls forward() only before observations -- so its values cannot be
     matched against post-step state; the manager is instead checked against
     the value each term returned inside the manager, which tests the weight
     scaling exactly.)
  3. Prints the magnitude of every term for a standing robot and a random-action
     robot, so the weights can be judged.

Usage (mjlab 1.6 venv):  python tests/check_rewards.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from mjlab.envs import ManagerBasedRlEnv

from microtaur_rigid import rewards as R
from microtaur_rigid.env_cfg import make_env_cfg
from microtaur_rigid.robot import EXPECTED_TOTAL_MASS_KG, LEG_JOINT_NAMES, PASSIVE_JOINT_NAMES, XL330_COPPER_W_PER_NM2

DEV = "cuda:0"
N = 64
cfg = make_env_cfg()
cfg.scene.num_envs = N
cfg.seed = 0
# The energy curriculum starts the weight at 0 and would reset it on every env
# reset; remove it and pin the final weight so the manager check below has a
# nonzero weight to divide by.
cfg.curriculum.pop("energy_weight")
TEST_ENERGY_WEIGHT = -0.025  # the default target is 0 (energy off); test at a nonzero weight
cfg.rewards["motor_energy"].weight = TEST_ENERGY_WEIGHT
# Record what each term returned inside the reward manager, so the manager-side
# check compares against the value computed at reward time (body quantities are
# one physics substep stale then; post-step state is not the same state).
RECORDED = {}


def _recording(name, fn):
  def wrapped(env, **kw):
    RECORDED[name] = fn(env, **kw).clone()
    return RECORDED[name]
  return wrapped


for _n in ("motor_energy", "action_rate"):
  cfg.rewards[_n].func = _recording(_n, cfg.rewards[_n].func)
env = ManagerBasedRlEnv(cfg, device=DEV)
env.reset()
robot = env.scene["robot"]
motor_ids, _ = robot.find_joints(list(LEG_JOINT_NAMES), preserve_order=True)
passive_ids, _ = robot.find_joints(list(PASSIVE_JOINT_NAMES), preserve_order=True)
act_names = list(robot.actuator_names) if hasattr(robot, "actuator_names") else None

ok = True


def check(name, a, b, tol):
  global ok
  d = float((a - b).abs().max())
  ok &= d <= tol
  print(f"  [{'PASS' if d <= tol else 'FAIL'}] {name:46s} max diff {d:.3e}")


def function_terms():
  cmd_name, p = "twist", {}
  return {
    "track_lin_vel_xy": R.track_lin_vel_xy(env, cmd_name, R.LIN_VEL_SIGMA_M_S),
    "track_ang_vel_z": R.track_ang_vel_z(env, cmd_name, R.YAW_RATE_SIGMA_RAD_S),
    "motor_energy": R.motor_energy(env, XL330_COPPER_W_PER_NM2, EXPECTED_TOTAL_MASS_KG, R.ENERGY_REF_SPEED_M_S),
    "action_rate": R.mdp.action_rate_l2(env),
    "termination": R.mdp.is_terminated(env),
  }


def manual_terms(cmd):
  d = robot.data
  v, w = d.root_link_lin_vel_b, d.root_link_ang_vel_b
  tau, qd = d.qfrc_actuator, d.joint_vel
  am = env.action_manager
  return {
    "track_lin_vel_xy": torch.exp(-((cmd[:, :2] - v[:, :2]) ** 2).sum(1) / R.LIN_VEL_SIGMA_M_S**2),
    "track_ang_vel_z": torch.exp(-((cmd[:, 2] - w[:, 2]) ** 2) / R.YAW_RATE_SIGMA_RAD_S**2),
    "motor_energy": ((tau * qd).abs().sum(1) + XL330_COPPER_W_PER_NM2 * (tau**2).sum(1))
    / (EXPECTED_TOTAL_MASS_KG * R.GRAVITY * R.ENERGY_REF_SPEED_M_S),
    "action_rate": ((am.action - am.prev_action) ** 2).sum(1),
    "termination": env.termination_manager.terminated.float(),
  }


def run(label, action_fn, steps=150):
  print(f"\n{label}")
  sums = {k: 0.0 for k in R.WEIGHTS}
  mech = copper = 0.0
  for t in range(steps):
    _, _, term, trunc, _ = env.step(action_fn())
    cmd = env.command_manager.get_command("twist")
    # Rewards are computed before terminated envs are reset, so compare only
    # envs that did not reset this step. _step_reward holds raw * weight.
    live = ~(term | trunc)
    rm = env.reward_manager
    man = manual_terms(cmd)
    if t == steps - 1:
      tau = robot.data.qfrc_actuator
      check("qfrc_actuator == 0 on passive joints", tau[:, passive_ids], torch.zeros_like(tau[:, passive_ids]), 0.0)
      if hasattr(robot.data, "actuator_force"):
        check("qfrc_actuator == actuator_force on motors", tau[:, motor_ids], robot.data.actuator_force[:, :8], 1e-6)
      fun = function_terms()
      for name in R.WEIGHTS:
        check(f"{name} function == manual formula", fun[name], man[name], 1e-5 * max(1.0, float(man[name].abs().max())))
      for name in ("motor_energy", "action_rate"):
        raw = rm._step_reward[:, rm.active_terms.index(name)] / rm.get_term_cfg(name).weight
        check(f"{name} manager step_reward / weight == term value", raw, RECORDED[name], 1e-5 * max(1.0, float(RECORDED[name].abs().max())))
    for k in sums:
      sums[k] += float(man[k].mean())
    m, c = R.motor_power_w(env, XL330_COPPER_W_PER_NM2)
    mech += float(m.mean()); copper += float(c.mean())
  print(f"  mean over {steps} steps (raw term, then x weight):")
  for k, v in sums.items():
    v /= steps
    print(f"    {k:18s} raw {v:8.4f}   weighted {v * env.reward_manager.get_term_cfg(k).weight:+8.4f}")
  print(f"    motor power: mechanical {mech / steps:.3f} W, copper {copper / steps:.3f} W")


run("standing, zero action", lambda: torch.zeros(N, 8, device=DEV))
g = torch.Generator(device=DEV).manual_seed(1)
run("random actions N(0, 0.5)", lambda: 0.5 * torch.randn(N, 8, device=DEV, generator=g))
env.close()
print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
