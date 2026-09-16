"""Figure 1: curb fall counts, grouped bars, 4 variants x 3 speeds.

    python scripts/paper_figures/fig1_curb_falls.py
      -> rollouts/paper_figures/fig1_curb_falls.pdf

Source (no re-derivation): rollouts/audit_2026-09-11/task3_curb_falloff_per_run.csv
(per-run fall outcome + distance-to-fall, already computed by the data audit).

Single column, 3.5 in wide. No title -- caption carries it:
  "Curb fall incidence by commanded speed (n=8 seeds/cell). Pitch fails at
  every speed tested, including the slowest; rigid/yaw/roll only fail at
  0.20 m/s. Inset: mean +/- SD distance traveled before falling (fallers
  only, pooled across speeds), in body lengths (BL = 0.168 m)."

Speed groups are labeled by the commanded value (the experimental factor);
no axis in this figure plots speed as a continuous quantity, so the
achieved-vs-commanded distinction does not apply here.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42   # embed real (subsetted TrueType) fonts
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "rollouts/audit_2026-09-11/task3_curb_falloff_per_run.csv"
OUT = ROOT / "rollouts/paper_figures/fig1_curb_falls.pdf"

VARIANTS = ["rigid", "pitch", "yaw", "roll"]
SPEEDS = ["0.08", "0.14", "0.2"]
SPEED_LABEL = {"0.08": "0.08", "0.14": "0.14", "0.2": "0.20"}

# Fixed, colorblind-safe variant palette (Okabe-Ito), reused identically
# across every figure in this set.
VCOL = {"rigid": "#0072B2", "pitch": "#E69F00", "yaw": "#009E73", "roll": "#CC79A7"}

BL_M = 0.168  # body length, m (see microtaur_kinematics.HIP_MIDPOINTS_ROOT_M)


def load():
    rows = list(csv.DictReader(open(SRC)))
    n = defaultdict(int)
    fell = defaultdict(int)
    dist_fall_bl = defaultdict(list)  # variant -> [BL, ...] pooled across speed, fallers only
    for r in rows:
        v, s = r["variant"], r["speed"]
        n[(v, s)] += 1
        if r["fell"] == "True":
            fell[(v, s)] += 1
            dist_fall_bl[v].append(float(r["dist_fall_BL"]))
    return n, fell, dist_fall_bl


def main():
    n, fell, dist_fall_bl = load()
    for v in VARIANTS:
        for s in SPEEDS:
            assert n[(v, s)] == 8, f"expected n=8 for {v}/{s}, got {n[(v, s)]}"

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 7.5})
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    ngroup = len(SPEEDS)
    width = 0.19
    group_x = np.arange(ngroup)
    for i, v in enumerate(VARIANTS):
        xs = group_x + (i - 1.5) * width
        heights = [fell[(v, s)] for s in SPEEDS]
        ax.bar(xs, heights, width=width * 0.92, color=VCOL[v], label=v, zorder=3)

    ax.set_xticks(group_x)
    ax.set_xticklabels([f"{SPEED_LABEL[s]} m/s" for s in SPEEDS])
    ax.set_ylabel("falls / 8 seeds")
    ax.set_ylim(0, 11.0)  # headroom above the tallest bar (8) reserved for the inset
    ax.set_yticks(range(0, 9, 2))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", lw=0.5, color="#d9d9d9", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, ncol=2, columnspacing=1.0,
             handlelength=1.2, handletextpad=0.5)

    # --- inset: mean +/- SD distance-to-fall (BL), fallers only, pooled over speed ---
    # Placed inside the headroom above the tallest bar (ylim goes to 11, data tops
    # out at 8) so it can never overlap a bar regardless of which group is tallest.
    inset = ax.inset_axes([0.60, 0.80, 0.39, 0.19])
    xs = np.arange(len(VARIANTS))
    means = [np.mean(dist_fall_bl[v]) if dist_fall_bl[v] else np.nan for v in VARIANTS]
    stds = [np.std(dist_fall_bl[v]) if dist_fall_bl[v] else 0.0 for v in VARIANTS]
    for i, v in enumerate(VARIANTS):
        inset.errorbar(i, means[i], yerr=stds[i], fmt="o", ms=3.5, color=VCOL[v],
                       capsize=2, lw=1.0, mec="none")
    inset.set_xticks(xs)
    inset.set_xticklabels([v[0].upper() for v in VARIANTS], fontsize=6.5)
    inset.set_ylabel("dist. to fall (BL)", fontsize=6.0, labelpad=1.5)
    inset.tick_params(labelsize=6.2, length=2)
    for s in ("top", "right"):
        inset.spines[s].set_visible(False)
    inset.set_facecolor("white")

    fig.tight_layout(pad=0.4)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}  (n_fell: " +
         ", ".join(f"{v}={[fell[(v, s)] for s in SPEEDS]}" for v in VARIANTS) + ")")


if __name__ == "__main__":
    main()
