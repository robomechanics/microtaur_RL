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
           100 %). window > 0: the rate is instead the exact success rate of the last `window` counted episodes at
           the cap, and an unlock needs at least that many since the last unlock (Ben, 2026-11-03: 98 % over 128
           episodes; the exponential mean over ~64 episodes unlocks on short peaks).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def step_levels(levels: torch.Tensor, up: torch.Tensor, down: torch.Tensor, cap: torch.Tensor,
                min_level: int) -> torch.Tensor:
  """One level up where up, one down where down, clamped to [min_level, cap]; cap is a scalar or one per env."""
  up = up & (levels < cap)
  new = (levels + up.long() - down.long()).clamp_min(min_level)
  return torch.minimum(new, torch.as_tensor(cap, device=levels.device).clamp_min(min_level))


def progress_levels(levels: torch.Tensor, ratio: torch.Tensor, cap: torch.Tensor, promote_frac: float,
                    demote_frac: float, min_level: int) -> torch.Tensor:
  """New levels of the envs whose episodes just ended; cap is a scalar or one value per env."""
  return step_levels(levels, ratio >= promote_frac, ratio < demote_frac, cap, min_level)


class SuccessGate:
  """Per terrain type cap, unlocked by the success rate of the episodes that ended at the cap."""

  def __init__(self, n_types: int, n_levels: int, device, start_level: int | Sequence[int] = 0, success: float = 0.8,
               min_episodes: int = 64, min_steps: int = 20 * 32, start_step: int = 200 * 32, horizon: int = 64,
               window: int = 0):
    self.n_levels, self.success, self.window = n_levels, success, window
    self.min_episodes = max(min_episodes, window)
    self.min_steps, self.start_step, self.decay = min_steps, start_step, 1.0 - 1.0 / horizon
    start = torch.as_tensor(start_level, dtype=torch.long, device=device)
    self.cap = start.expand(n_types).clone()  # per type: the levels up to the cap start unlocked
    self.mean = torch.zeros(n_types, device=device)     # exponential mean of success (biased towards 0)
    self.weight = torch.zeros(n_types, device=device)   # its total weight: mean / weight is unbiased
    self.count = torch.zeros(n_types, dtype=torch.long, device=device)
    self.last_unlock = torch.zeros(n_types, dtype=torch.long, device=device)
    self.buf = torch.zeros(n_types, max(window, 1), dtype=torch.bool, device=device)  # window: ring of outcomes
    self.pos = torch.zeros(n_types, dtype=torch.long, device=device)

  def rate(self) -> torch.Tensor:
    if self.window > 0:
      return self.buf.float().sum(1) / self.count.clamp(1, self.window).float()
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
      if self.window > 0:
        s = success[m][-self.window:]  # (the order inside one step is arbitrary)
        idx = (self.pos[t] + torch.arange(len(s), device=s.device)) % self.window
        self.buf[t, idx] = s
        self.pos[t] = (self.pos[t] + len(s)) % self.window
      else:
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
        self.buf[t] = False
        self.pos[t] = 0
        self.last_unlock[t] = step
    return self.cap
