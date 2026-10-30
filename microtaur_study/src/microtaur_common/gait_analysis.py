"""Gait statistics from foot-contact sequences (framework-free; numpy only).

Same definitions as scripts/eval_gait.py (mjlab): stride frequency from leg-1
touchdowns, each leg's touchdown phase relative to leg 1 (circular mean), and
the gait whose in-phase pairs have the smallest phase gap.
"""

from __future__ import annotations

import numpy as np

LEGS = ("leg1 RR", "leg2 RL", "leg3 FL", "leg4 FR")
PAIRS = {"trot": ((0, 2), (1, 3)), "pace": ((0, 3), (1, 2)), "bound": ((0, 1), (2, 3))}


def onset_phases(contact: np.ndarray, dt: float) -> tuple[float, list[float]]:
  """contact [T, 4] bool -> (stride Hz, touchdown phase of each leg vs leg 1)."""
  onsets = [np.flatnonzero(~contact[:-1, k] & contact[1:, k]) + 1 for k in range(4)]
  if len(onsets[0]) < 3:
    return float("nan"), [float("nan")] * 4
  period = float(np.median(np.diff(onsets[0])))
  phases = []
  for k in range(4):
    rel = []
    for t in onsets[k]:
      prev = onsets[0][onsets[0] <= t]
      if len(prev):
        rel.append(((t - prev[-1]) / period) % 1.0)
    a = 2 * np.pi * np.asarray(rel)
    phases.append(float((np.angle(np.mean(np.exp(1j * a))) / (2 * np.pi)) % 1.0) if len(rel) else float("nan"))
  return 1.0 / (period * dt), phases


def classify(phases) -> str:
  p = np.asarray(phases)
  if np.any(np.isnan(p)):
    return "unclear"

  def gap(a, b):
    d = abs(p[a] - p[b]) % 1.0
    return min(d, 1 - d)

  score = {name: gap(*x) + gap(*y) + abs(0.5 - gap(x[0], y[0])) for name, (x, y) in PAIRS.items()}
  best = min(score, key=score.get)
  return best if score[best] < 0.25 else f"irregular (closest {best}, score {score[best]:.2f})"
