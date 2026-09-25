"""Small helpers shared by events, rewards and actions."""

from __future__ import annotations

import torch


def safe_log(env, name: str, value: torch.Tensor) -> None:
  """Write a scalar into env.extras['log'] if the runner provides one."""
  extras = getattr(env, "extras", None)
  if isinstance(extras, dict) and isinstance(extras.get("log"), dict):
    extras["log"][name] = value


def env_id_tensor(env, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
  if isinstance(env_ids, torch.Tensor):
    return env_ids.to(device=env.device, dtype=torch.long)
  if env_ids is None:
    env_ids = slice(None)
  return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]


def uniform(count: int, tail: tuple[int, ...], bounds: tuple[float, float], device, dtype=torch.float32):
  lo, hi = float(bounds[0]), float(bounds[1])
  return lo + (hi - lo) * torch.rand((count, *tail), device=device, dtype=dtype)


def ordered_joint_ids(robot, names: tuple[str, ...], device) -> torch.Tensor:
  """Joint ids in exactly the requested order (find_joints may reorder)."""
  ids, resolved = robot.find_joints(list(names), preserve_order=True)
  if len(ids) != len(names) or set(resolved) != set(names):
    raise ValueError(f"Joint resolution mismatch: expected {names}, got {tuple(resolved)}")
  by_name = {n: int(i) for n, i in zip(resolved, ids)}
  return torch.tensor([by_name[n] for n in names], device=device, dtype=torch.long)
