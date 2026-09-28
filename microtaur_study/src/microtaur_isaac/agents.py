"""rsl_rl PPO runner cfg for IsaacLab (isaaclab_rl 0.4.4 / rsl-rl-lib 3.0.1).

Hyper-parameters are the mjlab ones (microtaur_rigid/rl_cfg.py, unchanged from
upstream): 512-256-128 ELU actor and critic, init std 1.0, 32-step rollouts.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
  RslRlDistillationAlgorithmCfg,
  RslRlDistillationRunnerCfg,
  RslRlDistillationStudentTeacherCfg,
  RslRlDistillationStudentTeacherRecurrentCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
  RslRlSymmetryCfg,
)

from .mdp.symmetry import compute_symmetric_states


@configclass
class MicrotaurPPORunnerCfg(RslRlOnPolicyRunnerCfg):
  num_steps_per_env = 32
  max_iterations = 3000
  save_interval = 50
  experiment_name = "microtaur_isaac"
  obs_groups = {"policy": ["policy"], "critic": ["critic"]}
  policy = RslRlPpoActorCriticCfg(
    init_noise_std=1.0,
    actor_obs_normalization=False,
    critic_obs_normalization=False,
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[512, 256, 128],
    activation="elu",
  )
  algorithm = RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.005,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
  )


@configclass
class MicrotaurTeacherPPORunnerCfg(MicrotaurPPORunnerCfg):
  """Privileged teacher: actor and critic both read the teacher group
  (critic set + 35 mm height map)."""
  experiment_name = "microtaur_isaac_teacher"
  obs_groups = {"policy": ["teacher"], "critic": ["teacher"]}


def _with_symmetry(alg: RslRlPpoAlgorithmCfg) -> RslRlPpoAlgorithmCfg:
  """Left/right mirror data augmentation (mdp/symmetry.py), as IsaacLab's ANYmal
  ...WithSymmetryCfg but mirror only (no front/back copies: the five-bar legs are
  not front/back symmetric)."""
  return alg.replace(symmetry_cfg=RslRlSymmetryCfg(use_data_augmentation=True,
                                                   data_augmentation_func=compute_symmetric_states))


@configclass
class MicrotaurPPORunnerSymCfg(MicrotaurPPORunnerCfg):
  """MicrotaurPPORunnerCfg + left/right symmetry augmentation (train with --agent rsl_rl_sym_cfg_entry_point)."""
  algorithm = _with_symmetry(MicrotaurPPORunnerCfg().algorithm)


@configclass
class MicrotaurTeacherPPORunnerSymCfg(MicrotaurTeacherPPORunnerCfg):
  algorithm = _with_symmetry(MicrotaurTeacherPPORunnerCfg().algorithm)


# -----------------------------------------------------------------------------
# Student distillation (rsl_rl DistillationRunner, DAgger-style: the student acts,
# loss = MSE to the frozen teacher's action). The teacher is loaded from a PPO
# checkpoint of MicrotaurTeacherPPORunnerCfg (its "actor." weights); the student
# reads only the deployment-compatible "policy" group (no height map, no base
# linear velocity). experiment_name is the teacher's, so --load_run / --checkpoint
# (agent.load_run / agent.load_checkpoint) find the teacher run directly.
# -----------------------------------------------------------------------------
_TEACHER_DIMS = [512, 256, 128]


@configclass
class MicrotaurStudentMLPRunnerCfg(RslRlDistillationRunnerCfg):
  """MLP student; give it history with env.observations.policy.history_length=<n>."""
  num_steps_per_env = 64
  max_iterations = 1500
  save_interval = 50
  experiment_name = "microtaur_isaac_teacher"
  obs_groups = {"policy": ["policy"], "teacher": ["teacher"]}
  policy = RslRlDistillationStudentTeacherCfg(
    init_noise_std=0.1,
    noise_std_type="scalar",
    student_obs_normalization=True,
    teacher_obs_normalization=False,
    student_hidden_dims=[512, 256, 128],
    teacher_hidden_dims=_TEACHER_DIMS,
    activation="elu",
  )
  algorithm = RslRlDistillationAlgorithmCfg(
    num_learning_epochs=2,
    learning_rate=1.0e-3,
    gradient_length=15,
    max_grad_norm=1.0,
  )


@configclass
class MicrotaurStudentGRURunnerCfg(MicrotaurStudentMLPRunnerCfg):
  """Recurrent (GRU) student: the memory replaces an explicit observation history."""
  policy = RslRlDistillationStudentTeacherRecurrentCfg(
    init_noise_std=0.1,
    noise_std_type="scalar",
    student_obs_normalization=True,
    teacher_obs_normalization=False,
    student_hidden_dims=[256, 128],
    teacher_hidden_dims=_TEACHER_DIMS,
    activation="elu",
    rnn_type="gru",
    rnn_hidden_dim=256,
    rnn_num_layers=1,
    teacher_recurrent=False,
  )


@configclass
class MicrotaurStudentV2RunnerCfg(MicrotaurStudentMLPRunnerCfg):
  """MLP student on the 47-D "student" group (two IMU slots, FK); give it history with
  env.observations.student.history_length=<n>. 60 steps per rollout (a multiple of any BPTT
  window; critic audit 2026-09-27)."""
  num_steps_per_env = 60
  obs_groups = {"policy": ["student"], "teacher": ["teacher"]}


@configclass
class MicrotaurStudentV2SmallRunnerCfg(MicrotaurStudentV2RunnerCfg):
  """Smaller student (256-128) for on-board inference."""
  policy = MicrotaurStudentV2RunnerCfg().policy.replace(student_hidden_dims=[256, 128])
