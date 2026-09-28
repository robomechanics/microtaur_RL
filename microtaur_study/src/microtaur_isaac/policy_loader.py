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


def student_group(checkpoint: str) -> str:
  """Observation group the (student) policy reads: "student" for the 47-D layout, else "policy"."""
  a = _params(checkpoint, "agent.yaml")
  m = re.search(r"obs_groups:\n(?:  .*\n)*?  policy:\n  - (\w+)", a)
  return m.group(1) if m else "policy"


def policy_history(checkpoint: str) -> int:
  """history_length of the observation group the policy reads."""
  env = _params(checkpoint, "env.yaml")
  g = student_group(checkpoint)
  m = re.search(r"^observations:\n(?:.*\n)*?  " + g + r":\n(?:    .*\n)*?    history_length: (\d+)", env, re.M)
  return int(m.group(1)) if m else 0


def prepare_env_cfg(cfg, checkpoint: str):
  """Apply what the policy's observations need (actor history length, action scale) to an env cfg."""
  g = student_group(checkpoint)
  if g == "student" and getattr(cfg.observations, "student", None) is None:
    from .mdp import observations as O
    cfg.observations.student = O.student_obs_cfg(None)  # eval: no noise, like the play actor group
    cfg.observations.student.enable_corruption = False
    for term in O.group_terms(cfg.observations.student).values():
      term.clip = (-O.OBS_CLIP, O.OBS_CLIP)
  h = policy_history(checkpoint)
  if h > 0:
    grp = getattr(cfg.observations, g)
    grp.history_length = h
    grp.flatten_history_dim = True
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
    cfg.obs_groups = {"policy": [student_group(checkpoint)], "teacher": ["teacher"]}
    hd = re.search(r"student_hidden_dims:\n((?:  - \d+\n)+)", _params(checkpoint, "agent.yaml"))
    if hd:
      cfg.policy.student_hidden_dims = [int(x) for x in re.findall(r"\d+", hd.group(1))]
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
