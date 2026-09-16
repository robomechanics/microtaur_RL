"""Figure 3: weave cross-track error, median + IQR, 4 variants x 3 speeds.

    python scripts/paper_figures/fig3_weave_xtrack.py
      -> rollouts/paper_figures/fig3_weave_xtrack.pdf

Source (no re-derivation): rollouts/audit_2026-09-11/task6_long_format.csv,
terrain=weave, metric=xtrack_rms_mm. This file's weave rows were rebuilt
from the 0.55 m aggressive resweep by scripts/rebuild_task6_weave.py (run
that first; it backs up the prior 0.80 m-derived file to
task6_long_format_pre055m.csv). Every other terrain's rows are untouched.

Single column, 3.5 in wide. No title -- caption carries it (numbers below
are illustrative -- regenerate this docstring's numbers from the current
PDF/data if rebuild_task6_weave.py is re-run):
  "Weave cross-track error at the aggressive 0.55 m pole spacing, median
  +/- IQR (n=8 seeds/cell)."

Speed groups are labeled by the commanded value (the experimental factor);
this figure plots cross-track error, not speed, on its data axis.
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
OUT = ROOT / "rollouts/paper_figures/fig3_weave_xtrack.pdf"

VARIANTS = ["rigid", "pitch", "yaw", "roll"]
SPEEDS = ["0.08", "0.14", "0.2"]
SPEED_LABEL = {"0.08": "0.08", "0.14": "0.14", "0.2": "0.20"}
VCOL = {"rigid": "#0072B2", "pitch": "#E69F00", "yaw": "#009E73", "roll": "#CC79A7"}


def load():
    rows = list(csv.DictReader(open(SRC)))
    d = defaultdict(list)
    for r in rows:
        if r["terrain"] == "weave" and r["metric"] == "xtrack_rms_mm":
            d[(r["variant"], r["speed"])].append(float(r["value"]))
    return d


def main():
    d = load()
    for v in VARIANTS:
        for s in SPEEDS:
            assert len(d[(v, s)]) == 8, f"expected n=8 for {v}/{s}, got {len(d[(v, s)])}"

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 7.5})
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    ngroup = len(SPEEDS)
    width = 0.19
    group_x = np.arange(ngroup)
    for i, v in enumerate(VARIANTS):
        meds, lo, hi = [], [], []
        for s in SPEEDS:
            vals = np.asarray(d[(v, s)])
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            meds.append(med); lo.append(med - q1); hi.append(q3 - med)
        xs = group_x + (i - 1.5) * width
        ax.errorbar(xs, meds, yerr=[lo, hi], fmt="o", ms=3.2, color=VCOL[v],
                   ecolor=VCOL[v], elinewidth=1.1, capsize=2.2, label=v, zorder=3)

    ax.set_xticks(group_x)
    ax.set_xticklabels([f"{SPEED_LABEL[s]} m/s" for s in SPEEDS])
    ax.set_ylabel("cross-track RMS (mm)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", lw=0.5, color="#d9d9d9", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, ncol=2, columnspacing=1.0,
             handlelength=1.0, handletextpad=0.4)

    fig.tight_layout(pad=0.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
