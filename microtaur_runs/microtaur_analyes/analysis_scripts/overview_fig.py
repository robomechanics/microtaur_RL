"""One-glance results figure: the headline metric of each test against speed.

    python scripts/overview_fig.py   -> rollouts/tidy/overview_{light,dark}.png

Reads rollouts/tidy/tidy_rollouts.csv (first episode only, yaw at measured mass).
Four panels, one per test, each showing the metric that test was built to
measure -- not a shared metric forced onto all four:

  flat    CoT+ (open loop)
  curb    share of the run with two feet on the lip (open loop)
  steps   progress down the course in 15 s (waypoint-steered)
  weave   correlation between the robot's track and the slalom path

Points are means over 8 seeds; bars are +/- 1 sd.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

V = ["rigid", "pitch", "yaw", "roll"]
TIDY = "rollouts/tidy/tidy_rollouts.csv"
OUT = "rollouts/tidy"
COL = {"light": {"rigid": "#33688f", "pitch": "#bd6f1c", "yaw": "#2f8a61", "roll": "#8f4bbf"},
       "dark":  {"rigid": "#5da0c8", "pitch": "#e0954c", "yaw": "#45b585", "roll": "#b78fd6"}}
TH = {"light": dict(bg="#ffffff", ink="#131d25", ink2="#586a76", grid="#d9e0e5"),
      "dark":  dict(bg="#141d21", ink="#e3ebef", ink2="#90a3ad", grid="#26343b")}
MARK = {"rigid": "o", "pitch": "s", "yaw": "D", "roll": "^"}   # secondary encoding

PANELS = [
    ("flat",  "openloop", "CoT_pos",    "CoT⁺",                     "Flat ground: energy cost",       None),
    ("curb",  "openloop", "straddle",   "share of run on the lip",       "Curb: holding the lip",          (0, 1.08)),
    ("steps", "waypoint", "progress_m", "progress down the course [m]",  "Step field: distance gained",    None),
    ("weave", "openloop", "weave_corr", "track-to-path correlation",     "Weave: tracking the slalom",     (0.70, 1.01)),
]


def main():
    d = pd.read_csv(TIDY)
    for mode, t in TH.items():
        plt.rcParams.update({"text.color": t["ink"], "axes.labelcolor": t["ink2"],
                             "xtick.color": t["ink2"], "ytick.color": t["ink2"], "font.size": 9.5})
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.3), facecolor=t["bg"])
        for ax, (terr, md, col, ylab, title, ylim) in zip(axes, PANELS):
            ax.set_facecolor(t["bg"])
            q = d[(d.terrain == terr) & (d["mode"] == md)]
            for i, v in enumerate(V):
                g = q[q.variant == v].groupby("cmd")[col]
                mu, sd = g.mean(), g.std()
                x = mu.index.to_numpy() + (i - 1.5) * 0.0028   # dodge so bars don't overlap
                ax.errorbar(x, mu.to_numpy(), yerr=sd.to_numpy(), marker=MARK[v], ms=6,
                            lw=2.0, capsize=3, elinewidth=1.2, color=COL[mode][v], label=v,
                            mec=t["bg"], mew=0.9, zorder=3)
            ax.set_title(title, fontsize=11, color=t["ink"], loc="left")
            ax.set_ylabel(ylab)
            ax.set_xlabel("commanded speed  [m/s]")
            ax.set_xticks([0.08, 0.14, 0.20])
            if ylim:
                ax.set_ylim(*ylim)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(t["grid"])
            ax.grid(color=t["grid"], lw=0.6, alpha=0.8)
            ax.set_axisbelow(True)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=10,
                   labelcolor=t["ink2"], bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(w_pad=2.2, rect=(0, 0, 1, 0.91))
        p = f"{OUT}/overview_{mode}.png"
        fig.savefig(p, dpi=130, facecolor=t["bg"])
        plt.close(fig)
        print("wrote", p)


if __name__ == "__main__":
    main()
