"""rsl_rl PPO runner cfg for IsaacLab (isaaclab_rl 0.4.4 / rsl-rl-lib 3.0.1).

Hyper-parameters are the mjlab ones (microtaur_rigid/rl_cfg.py, unchanged from
upstream): 512-256-128 ELU actor and critic, init std 1.0, 32-step rollouts.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

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
