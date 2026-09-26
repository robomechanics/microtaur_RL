"""Left/right mirror of observations and actions for rsl_rl symmetry augmentation.

  RslRlSymmetryCfg(use_data_augmentation=True, data_augmentation_func=compute_symmetric_states)

Only the left/right (sagittal-plane) mirror: the five-bar legs are not front/back
symmetric, so IsaacLab ANYmal's front/back and diagonal copies do not apply.

Mirror of the robot (reflection y -> -y):
  legs     leg1 RR <-> leg2 RL, leg4 FR <-> leg3 FL; every joint angle of the partner
           leg is negated (LEG_SIGNS flip between mirror legs, so FK(leg2, -a, -e) is
           FK(leg1, a, e) in the shared leg-plane frame; checked in tests/check_isaac_symmetry.py)
  vectors  linear velocity / gravity / forces (x, y, z) -> (x, -y, z);
           angular velocity (x, y, z) -> (-x, y, -z); command (vx, vy, wz) -> (vx, -vy, -wz)
The per-group layout is read from the observation manager (term names and dims), so
the flat, rough and teacher groups share this function; an unknown term raises.
"""

from __future__ import annotations

import torch

# canonical motor order: leg1 a, leg1 e, leg2 a, leg2 e, leg3 a, leg3 e, leg4 a, leg4 e
JOINT_PERM = (2, 3, 0, 1, 6, 7, 4, 5)
FOOT_PERM = (1, 0, 3, 2)  # feet in leg order RR, RL, FL, FR

_VEC_SIGN = {
  "base_lin_vel": (1.0, -1.0, 1.0),
  "base_ang_vel": (-1.0, 1.0, -1.0),
  "projected_gravity": (1.0, -1.0, 1.0),
  "command": (1.0, -1.0, -1.0),
}
_JOINT_TERMS = ("joint_pos", "joint_vel", "actions")
_FOOT_TERMS = ("foot_height", "foot_air_time", "foot_contact")


def mirror_joints(x: torch.Tensor) -> torch.Tensor:
  """[..., 8] motor values in canonical order -> mirrored robot."""
  return -x[..., list(JOINT_PERM)]


def _mirror_term(name: str, x: torch.Tensor, env) -> torch.Tensor:
  if name in _VEC_SIGN:
    return x * torch.tensor(_VEC_SIGN[name], device=x.device, dtype=x.dtype)
  if name in _JOINT_TERMS:
    return mirror_joints(x)
  if name in _FOOT_TERMS:
    return x[:, list(FOOT_PERM)]
  if name == "foot_contact_forces":  # [N, 4 feet x (fx, fy, fz)]
    f = x.reshape(-1, 4, 3)[:, list(FOOT_PERM)]
    return (f * torch.tensor((1.0, -1.0, 1.0), device=x.device, dtype=x.dtype)).reshape(-1, 12)
  if name == "height_scan":
    return _mirror_height_scan(x, env)
  raise NotImplementedError(f"no left/right mirror for observation term '{name}'")


def _mirror_height_scan(x: torch.Tensor, env) -> torch.Tensor:
  """Grid ray pattern, ordering 'xy' (x varies fastest): flip the y rows."""
  pattern = env.scene.sensors["height_scanner"].cfg.pattern_cfg
  if getattr(pattern, "ordering", "xy") != "xy":
    raise NotImplementedError("height_scan mirror assumes GridPatternCfg ordering 'xy'")
  nx = int(round(pattern.size[0] / pattern.resolution)) + 1
  ny = int(round(pattern.size[1] / pattern.resolution)) + 1
  return x.reshape(-1, ny, nx).flip(dims=[1]).reshape(-1, nx * ny)


def mirror_group(env, group: str, obs: torch.Tensor) -> torch.Tensor:
  om = env.observation_manager
  out, i = [], 0
  for name, dims in zip(om.active_terms[group], om.group_obs_term_dim[group]):
    w = 1
    for d in dims:
      w *= int(d)
    out.append(_mirror_term(name, obs[:, i:i + w], env))
    i += w
  if i != obs.shape[1]:
    raise ValueError(f"group '{group}': terms cover {i} of {obs.shape[1]} values")
  return torch.cat(out, dim=1)


@torch.no_grad()
def compute_symmetric_states(env, obs=None, actions=None):
  """rsl_rl data_augmentation_func: (obs TensorDict [B], actions [B, 8]) ->
  (original stacked with the left/right mirror [2B], same for actions)."""
  env = env.unwrapped
  obs_aug = None
  if obs is not None:
    b = obs.batch_size[0]
    obs_aug = obs.repeat(2)
    for group in obs.keys():
      obs_aug[group][b:] = mirror_group(env, group, obs[group])
  act_aug = None
  if actions is not None:
    act_aug = torch.cat((actions, mirror_joints(actions)), dim=0)
  return obs_aug, act_aug
