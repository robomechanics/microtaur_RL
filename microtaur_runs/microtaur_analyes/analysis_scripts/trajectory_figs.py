"""Plot how far a rollout wandered off the line it started on.

    python scripts/trajectory_figs.py rollouts/rigid_steps_2026-... [more dirs]

Left panel is the ground track in the spawn frame (x = along the spawn heading,
y = left), with the ideal straight line at y = 0. Right panel is the signed
deviation against distance travelled, which is the quantity summarised as
`straight_dev_*` in summary.json.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("rollouts/trajectory_figs")
COL = ["#33688f", "#bd6f1c", "#2f8a61", "#8f4bbf", "#b0392f"]


def main():
  dirs = [Path(d) for d in sys.argv[1:]]
  if not dirs:
    cands = sorted(glob.glob("rollouts/rigid_steps_*"))
    if not cands:
      raise SystemExit("no rollout dir given and no rollouts/rigid_steps_* found")
    dirs = [Path(cands[-1])]
  OUT.mkdir(parents=True, exist_ok=True)

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
  label_bits = []
  n = 0
  for d in dirs:
    meta_files = sorted(glob.glob(str(d / "rollout_*.meta.json")))
    for mf in meta_files:
      meta = json.loads(Path(mf).read_text())
      csv = mf.replace(".meta.json", ".csv")
      if not Path(csv).exists():
        continue
      df = pd.read_csv(csv)
      if "straight_dev_m" not in df:
        print(f"skip {csv}: no straight_dev_m (logged before the metric existed)")
        continue
      df = df[df.done == 0]
      c = COL[n % len(COL)]
      lbl = f"{meta.get('terrain','flat')} seed {meta.get('seed')}"
      ax1.plot(df.straight_along_m, df.straight_dev_m * 1000, color=c, lw=1.3, label=lbl)
      ax2.plot(df.t, df.straight_dev_m * 1000, color=c, lw=1.3, label=lbl)
      n += 1
      label_bits.append(meta.get("terrain", "flat"))

  for ax in (ax1, ax2):
    ax.axhline(0, color="#444", lw=1.0, ls="--", zorder=0)
    ax.grid(alpha=0.25)
  ax1.set_xlabel("distance along the spawn heading [m]")
  ax1.set_ylabel("deviation from the straight line [mm]   (+ = left)")
  ax1.set_title("Ground track vs the line it started on")
  ax2.set_xlabel("t [s]")
  ax2.set_ylabel("deviation [mm]")
  ax2.set_title("Deviation over time")
  ax1.legend(fontsize=8, loc="best")
  ax1.set_aspect("auto")
  terr = "-".join(sorted(set(label_bits))) or "run"
  fig.suptitle(f"Straight-line deviation — {terr}  (zero yaw command, so the "
               f"ideal track is the dashed line)", fontsize=10)
  fig.tight_layout(rect=(0, 0, 1, 0.93))
  p = OUT / f"deviation_{terr}.png"
  fig.savefig(p, dpi=130)
  print(f"wrote {p}")


if __name__ == "__main__":
  main()
