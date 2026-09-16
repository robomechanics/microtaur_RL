"""Compare one variant across the task terrains at several commanded speeds.

    python scripts/compare_terrains.py                    # auto-discover rigid
    python scripts/compare_terrains.py --variant rigid
    python scripts/compare_terrains.py rollouts/rigid_flat_... rollouts/rigid_curb_...

Reads the summary.json of each terrain sweep produced by run_terrain_sweep.sh
and writes rollouts/terrain_compare/<variant>/ with a markdown table, a CSV and
figures.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ORDER = ["flat", "curb", "steps", "weave"]
COL = {"flat": "#5a6570", "curb": "#33688f", "steps": "#bd6f1c", "weave": "#2f8a61"}
OUT = Path("rollouts/terrain_compare")


def discover(variant: str) -> list[Path]:
  """Newest sweep directory per terrain for this variant."""
  found: dict[str, Path] = {}
  for d in sorted(glob.glob(f"rollouts/{variant}_*"), key=os.path.getmtime):
    p = Path(d)
    sj = p / "summary.json"
    if not sj.exists():
      continue
    try:
      rows = json.loads(sj.read_text())
    except Exception:  # noqa: BLE001
      continue
    if not rows:
      continue
    t = rows[0].get("terrain", "flat")
    # only take sweeps that actually span several speeds
    if len({r["cmd_vx"] for r in rows}) >= 2:
      found[t] = p
  return [found[t] for t in ORDER if t in found]


def load(dirs: list[Path]) -> pd.DataFrame:
  recs = []
  for d in dirs:
    for r in json.loads((d / "summary.json").read_text()):
      r["_dir"] = d.name
      recs.append(r)
  if not recs:
    raise SystemExit("no summaries found")
  df = pd.DataFrame(recs)
  # Deviation and heading drift are SIGNED, and their sign is seed-dependent
  # (the robot wanders left or right at random). Averaging the signed value
  # across seeds cancels the wander and understates it badly -- flat at cmd 0.14
  # reads -0.075 signed but 0.103 in magnitude. Aggregate magnitudes.
  for src, dst in (("straight_dev_final_m", "abs_dev_final_m"),
                   ("straight_dev_per_m", "abs_dev_per_m"),
                   ("heading_drift_deg", "abs_heading_drift_deg")):
    if src in df:
      df[dst] = df[src].abs()
  return df


def agg(df: pd.DataFrame) -> pd.DataFrame:
  keep = ["mean_CoT_pos", "mean_CoT_abs", "mean_v_body_x", "mean_abs_roll_deg",
          "tilt_deg_mean", "any_terminated"]
  extra = ["abs_dev_final_m", "straight_dev_max_abs_m", "straight_dev_rms_m",
           "abs_dev_per_m", "path_tortuosity", "abs_heading_drift_deg",
           "weave_cross_track_rms_m", "curb_straddle_frac",
           "curb_mean_abs_roll_straddling_deg"]
  cols = [c for c in keep + extra if c in df.columns]
  g = df.groupby(["terrain", "cmd_vx"])[cols].agg(["mean", "std"])
  return g


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("dirs", nargs="*")
  ap.add_argument("--variant", default="rigid")
  args = ap.parse_args()

  dirs = [Path(d) for d in args.dirs] or discover(args.variant)
  if not dirs:
    raise SystemExit(f"no terrain sweeps found for variant {args.variant!r}")
  print("using:")
  for d in dirs:
    print("   ", d)
  df = load(dirs)
  outdir = OUT / args.variant
  outdir.mkdir(parents=True, exist_ok=True)
  df.to_csv(outdir / "rollouts.csv", index=False)

  present = [t for t in ORDER if t in set(df.terrain)]
  speeds = sorted(df.cmd_vx.unique())

  def m(t, v, col):
    s = df[(df.terrain == t) & (df.cmd_vx == v)][col]
    return (float(s.mean()), float(s.std())) if len(s) and col in df else (np.nan, np.nan)

  # ---- markdown table ---- #
  lines = [f"# Terrain comparison — {args.variant}", "",
           f"Speeds {speeds}, seeds {sorted(df.seed.unique())}, "
           f"{len(df)} rollouts, {int(df.any_terminated.sum())} terminations.", ""]
  lines += ["| terrain | cmd vx | achieved vx | CoT⁺ | CoT_abs | roll° | "
            "|dev| final (m) | |dev|/m | tortuosity | |heading drift|° |",
            "|---|---|---|---|---|---|---|---|---|---|"]
  for t in present:
    for v in speeds:
      def f(c, k=2, pct=False):
        a, s = m(t, v, c)
        if np.isnan(a):
          return "—"
        return f"{a:.{k}f} ± {s:.{k}f}" if not np.isnan(s) else f"{a:.{k}f}"
      lines.append(
        f"| {t} | {v:.2f} | {f('mean_v_body_x',3)} | {f('mean_CoT_pos')} | "
        f"{f('mean_CoT_abs')} | {f('mean_abs_roll_deg',1)} | "
        f"{f('abs_dev_final_m')} | {f('abs_dev_per_m',3)} | "
        f"{f('path_tortuosity',3)} | {f('abs_heading_drift_deg',0)} |")
  (outdir / "TERRAIN_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf8")

  # ---- figures ---- #
  panels = [("mean_CoT_pos", "CoT⁺  [-]", "Cost of transport"),
            ("mean_v_body_x", "achieved v_body_x  [m/s]", "Speed tracking"),
            ("mean_abs_roll_deg", "mean |roll|  [deg]", "Trunk roll"),
            ("abs_dev_per_m", "|deviation| per metre travelled  [m/m]",
             "Course holding (0 = perfectly straight)")]
  fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
  for ax, (col, ylab, title) in zip(axes, panels):
    for t in present:
      if col not in df:
        continue
      xs, ys, es = [], [], []
      for v in speeds:
        a, s = m(t, v, col)
        if not np.isnan(a):
          xs.append(v); ys.append(a); es.append(0.0 if np.isnan(s) else s)
      if not xs:
        continue
      ax.errorbar(xs, ys, yerr=es, marker="o", ms=5, capsize=3, lw=1.6,
                  color=COL.get(t, "#888"), label=t)
    ax.set_xlabel("commanded v_x  [m/s]"); ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=11); ax.grid(alpha=0.25)
  axes[1].plot(speeds, speeds, ls=":", color="#444", lw=1.2, label="perfect")
  axes[0].legend(fontsize=9)
  fig.suptitle(f"{args.variant}: flat vs task terrains, slow / medium / peak "
               f"(mean ± sd over seeds)", fontsize=12)
  fig.tight_layout(rect=(0, 0, 1, 0.92))
  fig.savefig(outdir / "t1_overview.png", dpi=130)
  plt.close(fig)

  print("\n".join(lines[4:]))
  print(f"\nwrote {outdir}/TERRAIN_COMPARISON.md, t1_overview.png, rollouts.csv")


if __name__ == "__main__":
  main()
