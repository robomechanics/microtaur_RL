"""Bake a left/right steering trim into a gait JSON (the board player has no trim option).

Same transform as openloop_spine_cpg.py --lr-swing-trim: per leg, the oscillating part of the
five-bar swing is scaled by (1 + trim) on the right legs (leg1, leg4) and (1 - trim) on the
left legs (leg2, leg3); lift, offsets and the spine are unchanged. Everything is linear, so it
is applied to each harmonic's cos and sin coefficients directly.

  python apply_lr_trim.py pitch_match/spine15_d1_gait.json pitch_match/spine15_d1_trim_gait.json --trim 0.15
"""

import argparse
import json

import numpy as np

LEG_SIGN = np.array([1.0, -1.0, -1.0, 1.0])
RIGHT_SIDE = np.array([1.0, -1.0, -1.0, 1.0])  # leg1 back-right, leg2 back-left, leg3 front-left, leg4 front-right


def trim_legs(coef: np.ndarray, trim: float) -> np.ndarray:
  """coef: (motors, K) leg coefficients in [leg1_a, leg1_e, ..., leg4_e, spine] order."""
  out = coef.copy()
  a, e = coef[0:8:2], coef[1:8:2]  # (4, K)
  swing = (a + e) / (2 * LEG_SIGN[:, None]) * (1 + trim * RIGHT_SIDE[:, None])
  lift = (e - a) / (2 * LEG_SIGN[:, None])
  out[0:8:2] = LEG_SIGN[:, None] * (swing - lift)
  out[1:8:2] = LEG_SIGN[:, None] * (swing + lift)
  return out


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("gait")
  parser.add_argument("out")
  parser.add_argument("--trim", type=float, required=True)
  args = parser.parse_args()

  gait = json.load(open(args.gait))
  cos_c = trim_legs(np.array(gait["cos_rad"]), args.trim)
  sin_c = trim_legs(np.array(gait["sin_rad"]), args.trim)
  gait["cos_rad"], gait["sin_rad"] = cos_c.tolist(), sin_c.tolist()
  # Readable form: angle = offset + sum_k amplitude cos(2 pi k f t + phase)
  gait["amplitude_deg"] = np.degrees(np.hypot(cos_c, sin_c)).round(2).tolist()
  gait["phase_deg"] = np.degrees(np.arctan2(-sin_c, cos_c)).round(1).tolist()
  gait["lr_swing_trim"] = args.trim
  with open(args.out, "w") as handle:
    json.dump(gait, handle, indent=1)
  print(f"[trim] wrote {args.out} with lr_swing_trim {args.trim:+.3f}")


if __name__ == "__main__":
  main()
