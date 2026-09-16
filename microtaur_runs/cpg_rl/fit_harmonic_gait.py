"""Fit an open-loop gait (offset + K sine harmonics per motor, one shared clock) to an RL rollout.

Input is an RL rollout from openloop_spine_cpg.py (policy mode, --series). The fit uses the
RL's safe motor targets: after the simulator's safety filter and before the command delay,
i.e. the angles the motors are asked to reach. A gait built from them needs no filter on
the board, where there is none.

For every robot: find its stride rate (the best single-sine fit of its 8 leg targets), fit
offset + K harmonics per motor at that rate, and shift time so leg1_a's first harmonic has
phase 0. Robots are grouped by stride rate (the pitch RL settles into two); the gait is the
average of one group. The output is the board gait JSON read by board/open_loop_cpg_9dof.py
and by openloop_spine_cpg.py harmonic mode:

    angle_j(t) = offset_j + sum_k cos_rad[j][k-1] cos(2 pi k f t) + sin_rad[j][k-1] sin(2 pi k f t)

  python fit_harmonic_gait.py pitch_match/rl_d1.npz pitch_match/gait_k3.json --harmonics 3
"""

import argparse
import json

import numpy as np


def design(t: np.ndarray, f: float, k: int) -> np.ndarray:
  cols = [np.ones_like(t)]
  for h in range(1, k + 1):
    cols += [np.cos(2 * np.pi * h * f * t), np.sin(2 * np.pi * h * f * t)]
  return np.stack(cols, 1)


def r_squared(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
  """Per column: 1 = identical, 0 = no better than the column mean."""
  return 1 - ((y - y_hat) ** 2).sum(0) / np.maximum(((y - y.mean(0)) ** 2).sum(0), 1e-12)


def stride_rate(legs: np.ndarray, t: np.ndarray, lo=2.5, hi=5.0, step=0.002) -> float:
  best_f, best_score = lo, -np.inf
  for f in np.arange(lo, hi, step):
    x = design(t, f, 1)
    coef, *_ = np.linalg.lstsq(x, legs, rcond=None)
    score = r_squared(legs, x @ coef).mean()
    if score > best_score:
      best_f, best_score = float(f), score
  return best_f


def fit_robot(y: np.ndarray, t: np.ndarray, f: float, k: int, ref: int):
  """Offset (M,) and complex harmonics C (k, M), with y = offset + Re(C_k e^{i k w t}),
  time-shifted so the reference channel's first harmonic has phase 0. Also returns that
  phase, which puts the robot back on its own clock."""
  coef, *_ = np.linalg.lstsq(design(t, f, k), y, rcond=None)
  c = coef[1::2] - 1j * coef[2::2]  # a cos + b sin = Re((a - ib) e^{iwt})
  phase = np.angle(c[0, ref])
  c = c * np.exp(-1j * np.arange(1, k + 1)[:, None] * phase)
  return coef[0], c, phase


def rebuild(offset, c, t, f, phase):
  k = np.arange(1, c.shape[0] + 1)[:, None, None]
  wave = np.exp(1j * k * (2 * np.pi * f * t[None, :, None] + phase))  # (k, T, 1)
  return offset + np.real((c[:, None, :] * wave).sum(0))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("rollout")
  parser.add_argument("out")
  parser.add_argument("--harmonics", type=int, default=3)
  parser.add_argument("--group", choices=("largest", "slow", "fast"), default="largest")
  parser.add_argument("--max-harmonic-hz", type=float, default=24.5,
                      help="hardware limit: the board sends commands at 50 Hz, so every sine term must stay below 25 Hz "
                           "(with a margin); harmonics above this are dropped")
  parser.add_argument("--leg-scale", type=float, default=1.0,
                      help="multiply the leg waves (not offsets); averaging robots with slightly different timing "
                           "shrinks them, and 1.04 restored the RL's speed for pitch")
  args = parser.parse_args()

  z = np.load(args.rollout)
  y = z["s_safe"].astype(float)  # (T, N, M)
  dt = float(z["dt"])
  t = np.arange(y.shape[0]) * dt
  names = [str(n) for n in z["motor_names"]]
  ref = names.index("leg1_a_joint_act")
  fell = z["m_falls"] > 0
  n_robots = y.shape[1]

  rates = np.array([stride_rate(y[:, r, :8], t) for r in range(n_robots)])
  groups = {"all": ~fell}
  ordered = np.sort(rates[~fell])
  gap = int(np.argmax(np.diff(ordered))) if len(ordered) > 1 else 0
  if len(ordered) > 1 and ordered[gap + 1] - ordered[gap] > 0.1:
    # Split only at a real gap between two stride-rate clusters (e.g. robots with a 20 vs 40 ms delay),
    # not inside one spread-out cluster.
    cut = (ordered[gap] + ordered[gap + 1]) / 2
    groups = {"slow": ~fell & (rates <= cut), "fast": ~fell & (rates > cut)}
  if args.group == "largest":
    name = max(groups, key=lambda g: groups[g].sum())
  else:
    name = args.group if args.group in groups else "all"
  chosen = groups[name]

  harmonics = min(args.harmonics, int(args.max_harmonic_hz // float(np.median(rates[chosen]))))
  if harmonics < args.harmonics:
    print(f"[fit] stride rate {np.median(rates[chosen]):.3f} Hz: keeping {harmonics} harmonics "
          f"(harmonic {harmonics + 1} would be at or above {args.max_harmonic_hz} Hz)")
  args.harmonics = harmonics
  fits = {r: fit_robot(y[:, r, :], t, rates[r], args.harmonics, ref) for r in np.flatnonzero(~fell)}
  offset = np.mean([fits[r][0] for r in np.flatnonzero(chosen)], 0)
  c = np.mean([fits[r][1] for r in np.flatnonzero(chosen)], 0)
  f = float(np.median(rates[chosen]))
  legs = [i for i, n in enumerate(names) if n != "spine_joint_act"]
  c[:, legs] *= args.leg_scale

  # How well the averaged gait matches each robot on its own clock (same rate and phase).
  match = {g: np.mean([r_squared(y[:, r, :], rebuild(offset, c, t, rates[r], fits[r][2]))
                       for r in np.flatnonzero(mask)], 0) for g, mask in groups.items() if mask.any()}

  gait = {
    "robot": json.loads(str(z["meta"]))["variant"],
    "source": args.rollout,
    "harmonics": args.harmonics,
    "max_harmonic_hz": args.max_harmonic_hz,
    "leg_scale": args.leg_scale,
    "frequency_hz": f,
    "control_hz": 50,
    "joint_order": names,
    "offset_rad": offset.tolist(),
    "cos_rad": np.real(c).T.tolist(),
    "sin_rad": (-np.imag(c)).T.tolist(),
    # Same wave, readable: angle = offset + sum_k amplitude cos(2 pi k f t + phase)
    "amplitude_deg": np.degrees(np.abs(c)).T.round(2).tolist(),
    "phase_deg": np.degrees(np.angle(c)).T.round(1).tolist(),
    "fit": {
      "group": name,
      "robots_used": int(chosen.sum()),
      "robots_fell": int(fell.sum()),
      "stride_rates_hz": {g: np.round(rates[m], 3).tolist() for g, m in groups.items()},
      "match_r2_per_motor": {g: np.round(v, 3).tolist() for g, v in match.items()},
      "match_r2_legs_mean": {g: round(float(v[:8].mean()), 3) for g, v in match.items()},
    },
  }
  with open(args.out, "w") as handle:
    json.dump(gait, handle, indent=1)
  print(f"[fit] {args.out}: K={args.harmonics}, group {name} ({chosen.sum()} robots), f={f:.3f} Hz")
  for g, m in groups.items():
    print(f"  {g}: rates {np.round(np.sort(rates[m]), 2).tolist()}")
  for g, v in match.items():
    print(f"  match vs {g} robots: legs {v[:8].mean():.3f}, per motor {np.round(v, 2).tolist()}")


if __name__ == "__main__":
  main()
