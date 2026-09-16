"""Score CPG rollouts against one or more RL reference rollouts, stride by stride.

Uses the stride curves of compare_rl_cpg.py. Every path may carry a time window, "path@t0:t1" (seconds of
the logged series, after settle), so runs of different lengths can be compared on equal windows: a longer
window averages over more stride-to-stride drift and smears the average stride. Put RL windows in the
candidate list too: RL vs an independent RL window of the same length is the ceiling a gait can reach.

  python score_gaits.py --ref "RL=pitch_match/rl_d1.npz" \
      "RL fresh 0-20 s=pitch_match/rl_fresh.npz@0:20" "Final 0-20 s=pitch_match/long_final.npz@0:20"

Each score cell is R² of the stride-averaged curves against each --ref in order (1 = identical). "periodic"
is the median share of each robot's raw signal explained by its own repeating stride (legs / IMU / torque):
what an open-loop gait can reproduce at most, stride to stride.
"""

import argparse

import numpy as np

from compare_rl_cpg import r2_curve, stride_curves

SIGNALS = (("legs", "angle", slice(0, 8)), ("spine", "angle", slice(8, 9)), ("IMU", "imu", slice(0, 2)),
           ("gyro", "imu", slice(2, 5)), ("torque", "torque", slice(0, 8)))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--ref", action="append", required=True, help="label=path[@t0:t1] of an RL rollout")
  parser.add_argument("candidates", nargs="+", help="label=path[@t0:t1]")
  args = parser.parse_args()

  refs = [(label, stride_curves(spec)) for label, spec in (r.split("=", 1) for r in args.ref)]
  motors = len(refs[0][1]["names"])
  signals = [s for s in SIGNALS if s[0] != "spine" or motors > 8]
  width = 6 * len(refs)
  print(f"scores: R² vs {' / '.join(label for label, _ in refs)}")
  print(f"{'candidate':26s} " + " ".join(f"{name:>{width}s}" for name, _, _ in signals)
        + f" {'body vx':>8s} {'periodic legs/IMU/torque':>25s} {'falls':>5s} {'tilt max':>8s}")
  for label, spec in (c.split("=", 1) for c in args.candidates):
    run = stride_curves(spec)
    cells = []
    for _, key, part in signals:
      scores = [r2_curve(ref["curves"][key][:, part], run["curves"][key][:, part]).mean() for _, ref in refs]
      cells.append(" / ".join(f"{s:4.2f}" for s in scores))
    periodic = run["periodic"]
    print(f"{label:26s} " + " ".join(f"{cell:>{width}s}" for cell in cells)
          + f" {run['body_vx']:8.3f} {periodic['legs']:>11.2f} / {periodic['imu']:.2f} / {periodic['torque']:.2f}"
          + f" {run['table']['falls']:>5s} {run['table']['IMU tilt max (deg)']:>8s}")


if __name__ == "__main__":
  main()
