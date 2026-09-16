"""Figure 4: spine torque-angle work loops, curb vs steps, one panel per
spine-bearing variant (pitch, yaw, roll). Rigid has no spine joint and is
not shown.

    python scripts/paper_figures/fig4_spine_loops.py
      -> rollouts/paper_figures/fig4_spine_loops.pdf

Source (no re-derivation): rollouts/spine_function/loops.npz, the existing
stride-phase-averaged (angle, torque, power, angular velocity) loops built
by spine_function.py/spine_figs.py. This figure only re-plots channels 0
(angle) and 1 (torque) at the 0.20 m/s condition -- no new fitting.

Double column, 7.16 in wide (3 panels need the width to stay readable).
No title -- caption carries it:
  "Stride-averaged spine torque vs. angle at 0.20 m/s, curb (solid) vs.
  steps (dashed). Roll traces the same closed, damper-like loop on both
  terrains; pitch's loop changes shape between terrains, consistent with
  switching mechanical role rather than acting as a fixed spring/damper."

Note: yaw/curb at 0.20 m/s pools only 4 runs (of 8 seeds) -- the other 4
did not yield a full stride in the on-lip window at that speed/terrain
(see spine_function.py's stride-extraction criteria). Shown as-is, flagged
rather than hidden.
"""
from pathlib import Path

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "rollouts/spine_function/loops.npz"
OUT = ROOT / "rollouts/paper_figures/fig4_spine_loops.pdf"

VARIANTS = ["pitch", "yaw", "roll"]  # rigid excluded: no spine joint
TERRAINS = ["curb", "steps"]
SPEED = "0.20"
VCOL = {"rigid": "#0072B2", "pitch": "#E69F00", "yaw": "#009E73", "roll": "#CC79A7"}
TERRAIN_STYLE = {"curb": dict(ls="-", lw=1.4), "steps": dict(ls="--", lw=1.4)}


def main():
    L = dict(np.load(SRC))
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 7.5})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))

    n_runs = {}
    for ax, v in zip(axes, VARIANTS):
        for t in TERRAINS:
            k = f"{v}|{t}|{SPEED}"
            if k not in L:
                continue
            arr = L[k]                      # (n_runs, 4, NPH)
            n_runs[(v, t)] = arr.shape[0]
            a = arr.mean(0)                 # (4, NPH): angle_deg, tau_mNm, P_mW, qd_dps
            qx = np.r_[a[0], a[0][0]]
            tq = np.r_[a[1], a[1][0]]
            ax.plot(qx, tq, color=VCOL[v], label=t, **TERRAIN_STYLE[t], zorder=3)
        ax.axhline(0, color="#999999", lw=0.6, zorder=1)
        ax.axvline(0, color="#999999", lw=0.6, zorder=1)
        ax.set_xlabel("spine angle about mean (deg)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(lw=0.5, color="#e6e6e6", zorder=0)
        ax.set_axisbelow(True)
        # panel identifier (which variant), not a figure title -- placed as
        # in-axes text rather than ax.set_title to keep every panel title-free.
        ax.text(0.03, 0.96, v, transform=ax.transAxes, ha="left", va="top",
               fontsize=8.5, color=VCOL[v], fontweight="bold")
    axes[0].set_ylabel("spine torque (mN*m)")
    axes[0].legend(loc="best", frameon=False, handlelength=1.6)

    fig.tight_layout(pad=0.4, w_pad=1.1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}  n_runs={n_runs}")


if __name__ == "__main__":
    main()
