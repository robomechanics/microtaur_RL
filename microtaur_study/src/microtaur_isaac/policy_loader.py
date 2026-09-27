"""Load a teacher (PPO) or student (distillation) checkpoint for evaluation scripts.

The kind of run is read from <checkpoint dir>/params/agent.yaml (runner / policy class) and
env.yaml (actor observation history), so every eval script handles teachers, MLP students
and GRU students the same way:

  cfg = MicrotaurTeacherCurPlayEnvCfg()
  prepare_env_cfg(cfg, checkpoint)          # before creating the env (history length)
  env = ManagerBasedRLEnv(cfg); w = RslRlVecEnvWrapper(env)
  act, reset = load_policy(w, checkpoint)
  obs = w.get_observations()
  obs, _, dones, _ = w.step(act(obs)); reset(dones)   # reset clears GRU memory of done envs
"""

from __future__ import annotations

import re
from pathlib import Path


def _params(checkpoint: str, name: str) -> str:
  p = Path(checkpoint).parent / "params" / name
  return p.read_text() if p.exists() else ""


def is_student(checkpoint: str) -> bool:
  return "DistillationRunner" in _params(checkpoint, "agent.yaml")


def policy_history(checkpoint: str) -> int:
  """history_length of the actor ("policy") observation group the run was trained with."""
  env = _params(checkpoint, "env.yaml")
  m = re.search(r"^observations:\n(?:.*\n)*?  policy:\n(?:    .*\n)*?    history_length: (\d+)", env, re.M)
  return int(m.group(1)) if m else 0


def prepare_env_cfg(cfg, checkpoint: str):
  """Apply what the policy's observations need (actor history length, action scale) to an env cfg."""
  h = policy_history(checkpoint)
  if h > 0:
    cfg.observations.policy.history_length = h
    cfg.observations.policy.flatten_history_dim = True
  m = re.search(r"action_scale_rad: ([0-9.eE+-]+)", _params(checkpoint, "env.yaml"))
  if m:
    cfg.actions.joint_pos.action_scale_rad = float(m.group(1))
  return cfg


def load_policy(wrapped_env, checkpoint: str, device: str | None = None):
  """Returns (act, reset): act(obs) -> actions; reset(dones) clears recurrent memory (no-op for MLPs)."""
  from rsl_rl.runners import DistillationRunner, OnPolicyRunner

  from .agents import MicrotaurStudentGRURunnerCfg, MicrotaurStudentMLPRunnerCfg, MicrotaurTeacherPPORunnerCfg

  dev = device or wrapped_env.unwrapped.device
  if is_student(checkpoint):
    rnn = "StudentTeacherRecurrent" in _params(checkpoint, "agent.yaml")
    cfg = (MicrotaurStudentGRURunnerCfg if rnn else MicrotaurStudentMLPRunnerCfg)()
    runner = DistillationRunner(wrapped_env, cfg.to_dict(), log_dir=None, device=dev)
  else:
    runner = OnPolicyRunner(wrapped_env, MicrotaurTeacherPPORunnerCfg().to_dict(), log_dir=None, device=dev)
  runner.load(checkpoint)
  act = runner.get_inference_policy(device=dev)
  module = runner.alg.policy

  def reset(dones):
    if getattr(module, "is_recurrent", False):
      module.reset(dones.bool())

  return act, reset
