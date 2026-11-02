"""Terrain-level curriculum logic (pure torch; the IsaacLab glue is env_cfg.terrain_levels_progress).

Per env, at the end of each episode: promote one level if the episode's progress ratio (body progress along the
command / commanded distance) is >= promote_frac, demote if < demote_frac, clamped to [min_level, cap].

The cap (highest unlocked level) comes from a gate:
  time     cap = step // steps_per_level for every terrain type (300 iterations per level; the original rule).
  success  one cap per terrain type, raised by one when the type's episodes that ENDED AT ITS CAP succeed often
           enough (Ben, 2026-11-01: advance a terrain once it has converged on its level). A level that a robot masters
           costs about one episode, a hard one gets the time it needs, and a level it cannot master stays locked.
           Unlock when, since the last unlock of that type: at least min_episodes episodes ended at the cap, at least
           min_steps policy steps passed, and their success rate (bias-corrected exponential mean over the last
           ~horizon episodes) is >= success; never before start_step (the gait forms first). Only episodes
           update() is told are valid count (terrain_levels_progress: commanded distance >= gate_min_cmd_m, so a
           near-zero speed command, whose progress ratio is noise, can neither pass nor fail a level). start_level
           may differ per terrain type (Ben, 2026-11-02: a fine-tune restarts each type at its first level below
           100 %).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def progress_levels(levels: torch.Tensor, ratio: torch.Tensor, cap: torch.Tensor, promote_frac: float,
                    demote_frac: float, min_level: int) -> torch.Tensor:
  """New levels of the envs whose episodes just ended; cap is a scalar or one value per env."""
  up = (ratio >= promote_frac) & (levels < cap)
  down = ratio < demote_frac
  new = (levels + up.long() - down.long()).clamp_min(min_level)
  return torch.minimum(new, torch.as_tensor(cap, device=levels.device).clamp_min(min_level))


class SuccessGate:
  """Per terrain type cap, unlocked by the success rate of the episodes that ended at the cap."""

  def __init__(self, n_types: int, n_levels: int, device, start_level: int | Sequence[int] = 0, success: float = 0.8,
               min_episodes: int = 64, min_steps: int = 20 * 32, start_step: int = 200 * 32, horizon: int = 64):
    self.n_levels, self.success, self.min_episodes = n_levels, success, min_episodes
    self.min_steps, self.start_step, self.decay = min_steps, start_step, 1.0 - 1.0 / horizon
    start = torch.as_tensor(start_level, dtype=torch.long, device=device)
    self.cap = start.expand(n_types).clone()  # per type: the levels up to the cap start unlocked
    self.mean = torch.zeros(n_types, device=device)     # exponential mean of success (biased towards 0)
    self.weight = torch.zeros(n_types, device=device)   # its total weight: mean / weight is unbiased
    self.count = torch.zeros(n_types, dtype=torch.long, device=device)
    self.last_unlock = torch.zeros(n_types, dtype=torch.long, device=device)

  def rate(self) -> torch.Tensor:
    return torch.where(self.weight > 0, self.mean / self.weight.clamp_min(1e-9), torch.zeros_like(self.mean))

  def update(self, step: int, types: torch.Tensor, levels: torch.Tensor, success: torch.Tensor,
             valid: torch.Tensor | None = None) -> torch.Tensor:
    """Feed the episodes that just ended (terrain type, level they ended on, success, optionally which of them count);
    returns the caps per type."""
    for t in torch.unique(types).tolist():
      m = (types == t) & (levels == self.cap[t])
      if valid is not None:
        m &= valid
      k = int(m.sum())
      if k == 0:
        continue
      d = self.decay ** k
      self.mean[t] = self.mean[t] * d + success[m].float().mean() * (1.0 - d)
      self.weight[t] = self.weight[t] * d + (1.0 - d)
      self.count[t] += k
      if (step >= self.start_step and step - int(self.last_unlock[t]) >= self.min_steps
          and int(self.count[t]) >= self.min_episodes and float(self.rate()[t]) >= self.success
          and int(self.cap[t]) < self.n_levels - 1):
        self.cap[t] += 1
        self.mean[t] = self.weight[t] = 0.0
        self.count[t] = 0
        self.last_unlock[t] = step
    return self.cap
