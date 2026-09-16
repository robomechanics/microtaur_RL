"""Figure 2: distance traveled on the step field at 0.20 m/s, per run.

    python scripts/paper_figures/fig2_steps_distance.py
      -> rollouts/paper_figures/fig2_steps_distance.pdf

Source (no re-derivation): rollouts/audit_2026-09-11/task6_long_format.csv,
filtered to terrain=steps, speed=0.20, metric=dist_BL. Individual per-seed
points (n=8) plus mean +/- SD, since a bar-only summary would hide the
variance collapse that is the point of this figure.

Single column, 3.5 in wide. No title -- caption carries it:
  "Distance traveled on the step field at 0.20 m/s (n=8 seeds/variant,
  body lengths, BL=0.168 m). Pitch (15.55+/-1.59 BL) both travels farther
  and is far more consistent than rigid (7.41+/-4.86 BL)."

Axis is distance, not speed -- the achieved-vs-commanded distinction does
not apply to this figure (speed is fixed at the 0.20 m/s condition, not
plotted as a data axis).
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "rollouts/audit_2026-09-11/task6_long_format.csv"
OUT = ROOT / "rollouts/paper_figures/fig2_steps_distance.pdf"

VARIANTS = ["rigid", "pitch", "yaw", "roll"]
VCOL = {"rigid": "#0072B2", "pitch": "#E69F00", "yaw": "#009E73", "roll": "#CC79A7"}

RNG = np.random.default_rng(0)  # fixed seed: jitter is cosmetic only, must be reproducible


def load():
    rows = list(csv.DictReader(open(SRC)))
    d = defaultdict(list)
    for r in rows:
        if r["terrain"] == "steps" and r["speed"] == "0.2" and r["metric"] == "dist_BL":
            d[r["variant"]].append(float(r["value"]))
    return d


def main():
    d = load()
    for v in VARIANTS:
        assert len(d[v]) == 8, f"expected n=8 for {v}, got {len(d[v])}"

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 8,
                         "ytick.labelsize": 8})
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    for i, v in enumerate(VARIANTS):
        vals = np.asarray(d[v])
        jitter = RNG.uniform(-0.09, 0.09, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=14, color=VCOL[v],
                  alpha=0.75, edgecolors="none", zorder=3)
        m, sd = vals.mean(), vals.std(ddof=1)  # sample SD, matches the quoted 15.55+/-1.59 etc.
        ax.errorbar(i, m, yerr=sd, fmt="_", ms=16, mew=1.6, color=VCOL[v],
                   ecolor="black", elinewidth=1.0, capsize=3, zorder=4)

    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels(VARIANTS)
    ax.set_ylabel("distance traveled (BL)")
    ax.set_xlim(-0.5, len(VARIANTS) - 0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", lw=0.5, color="#d9d9d9", zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    means = {v: (np.mean(d[v]), np.std(d[v], ddof=1)) for v in VARIANTS}
    print(f"wrote {OUT}  means: " + ", ".join(f"{v}={m:.2f}+/-{s:.2f}" for v, (m, s) in means.items()))


if __name__ == "__main__":
    main()
