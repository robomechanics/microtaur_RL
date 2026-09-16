"""Body attitude phase portraits: which robot design is the most stable?

    python scripts/body_phase_portraits.py
      -> rollouts/body_stability/{roll,pitch}_portrait_{light,dark}.png

Same idea as the spine phase portraits, applied to the body itself: roll rate
vs roll angle, and pitch rate vs pitch angle (body-frame wx/wy stand in for
roll-rate/pitch-rate). Each stride is resampled to a common phase (leg-2
touchdown to touchdown, as in spine_function.strides) and averaged first
within a run, then across seeds -- one clean closed loop per speed, not a
smear of every raw sample. Loop tightness/shape is what indicates stability;
the inset RMS/peak numbers are still computed from the raw (unaveraged)
signal so they're not flattered by the averaging.

Uses the same runs, same first-episode/settle/on-course selection as
spine_function.py (see that file for why), so results are apples-to-apples
with the spine figures -- just add "rigid" back in, since this is about the
whole robot, not the spine joint.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from spine_function import NPH, SETTLE, SPEEDS, first_episode, resample, strides, test_mask  # noqa: E402
from spine_function import runs as spine_runs  # noqa: E402

OUT = "rollouts/body_stability"
VARIANTS = ["rigid", "pitch", "yaw", "roll"]
T = ["flat", "curb", "steps", "weave"]
TNAME = {"flat": "flat", "curb": "curb (on the lip)", "steps": "step field (steered)", "weave": "weave"}
VCOL = {"light": {"rigid": "#3d5a80", "pitch": "#bd6f1c", "yaw": "#2f8a61", "roll": "#8f4bbf"},
        "dark":  {"rigid": "#7ea6d6", "pitch": "#e0954c", "yaw": "#45b585", "roll": "#b78fd6"}}
TH = {"light": dict(bg="#ffffff", ink="#131d25", ink2="#586a76", grid="#d9e0e5"),
      "dark":  dict(bg="#141d21", ink="#e3ebef", ink2="#90a3ad", grid="#26343b")}
SHADE = {0.08: 0.38, 0.14: 0.68, 0.20: 1.0}


def style(ax, t):
    ax.set_facecolor(t["bg"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["grid"])
    ax.grid(color=t["grid"], lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def mix(c, bg, a):
    c = np.array(matplotlib.colors.to_rgb(c))
    b = np.array(matplotlib.colors.to_rgb(bg))
    return tuple(b + (c - b) * a)


def gather():
    """Returns (loops, stats).

    loops[(variant, terrain, cmd)] = list of per-run (4, NPH) arrays:
        [roll_deg, wx_dps, pitch_deg, wy_dps], each one run's stride-average.
    stats[(variant, terrain, cmd)] = list of per-run (roll_deg, wx_dps,
        pitch_deg, wy_dps) RAW (unaveraged, full settled window) arrays, for
        the RMS/peak annotations.
    """
    loops, stats = {}, {}
    for (v, t, cmd, seed), (csv, meta) in sorted(spine_runs(VARIANTS).items()):
        df = first_episode(pd.read_csv(csv))
        ss = df[(df.t >= SETTLE) & (df.done == 0)]
        mask = test_mask(ss, t)
        if mask.sum() < 50:
            continue
        roll = np.degrees(ss.roll.to_numpy())
        wx = np.degrees(ss.wx.to_numpy())
        pitch = np.degrees(ss.pitch.to_numpy())
        wy = np.degrees(ss.wy.to_numpy())
        key = (v, t, cmd)
        stats.setdefault(key, []).append((roll[mask], wx[mask], pitch[mask], wy[mask]))

        st = strides(ss, mask)
        if not st:
            continue
        per = []
        for a, b in st:
            per.append(np.stack([resample(roll, a, b), resample(wx, a, b),
                                 resample(pitch, a, b), resample(wy, a, b)]))
        loops.setdefault(key, []).append(np.stack(per).mean(0))   # (4, NPH), this run
    return loops, stats


def portrait(loops, stats, mode, t, which):
    """which: 'roll' -> (roll, wx); 'pitch' -> (pitch, wy)."""
    idx = 0 if which == "roll" else 2
    fig, axes = plt.subplots(4, 4, figsize=(16.5, 14.5), facecolor=t["bg"])
    for r, v in enumerate(VARIANTS):
        for cidx, terr in enumerate(T):
            ax = axes[r, cidx]
            style(ax, t)
            notes = []
            for s in SPEEDS:
                key = (v, terr, s)
                if key not in loops:
                    continue
                a = np.stack(loops[key]).mean(0)              # mean over seeds: (4, NPH)
                qx, vy = np.r_[a[idx], a[idx][:1]], np.r_[a[idx + 1], a[idx + 1][:1]]
                col = mix(VCOL[mode][v], t["bg"], SHADE[s])
                ax.plot(qx, vy, color=col, lw=2.2 if s == 0.20 else 1.6, zorder=3)
                j = len(a[idx]) // 4
                ax.add_patch(FancyArrowPatch((qx[j], vy[j]), (qx[j + 1], vy[j + 1]), arrowstyle="-|>",
                                             mutation_scale=13, color=col, lw=0, zorder=4))
                ang_cat = np.concatenate([d[idx] for d in stats[key]])
                rate_cat = np.concatenate([d[idx + 1] for d in stats[key]])
                notes.append(f"{s:.2f}: {ang_cat.std():4.1f} deg rms, {np.abs(rate_cat).max():3.0f} deg/s pk")
            ax.axhline(0, color=t["ink2"], lw=0.8)
            ax.axvline(0, color=t["ink2"], lw=0.8)
            ax.text(0.02, 0.98, "\n".join(notes), transform=ax.transAxes, va="top", ha="left",
                    fontsize=7.2, color=t["ink2"], family="monospace", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.22", fc=t["bg"], ec=t["grid"], alpha=0.85))
            if r == 0:
                ax.set_title(TNAME[terr], fontsize=11, color=t["ink"])
            if cidx == 0:
                ax.set_ylabel(f"{v}\nbody {which} rate  [deg/s]", color=VCOL[mode][v], fontsize=10.0)
            if r == len(VARIANTS) - 1:
                ax.set_xlabel(f"body {which}  [deg]")
    fig.suptitle(f"Stride-averaged body {which} phase portrait: {which} rate vs {which} angle (inset: raw, "
                 "un-averaged RMS angle and peak rate at each speed; light -> dark = 0.08 -> 0.20 m/s).",
                 fontsize=11, color=t["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def main():
    os.makedirs(OUT, exist_ok=True)
    loops, stats = gather()
    print(f"[body_phase_portraits] {len(loops)} cells with stride loops, {len(stats)} with raw stats")
    missing = [(v, terr, s) for v in VARIANTS for terr in T for s in SPEEDS if (v, terr, s) not in loops]
    if missing:
        print(f"[body_phase_portraits] no stride loop for {len(missing)} cells: {missing}")
    for mode, t in TH.items():
        plt.rcParams.update({"text.color": t["ink"], "axes.labelcolor": t["ink2"],
                             "xtick.color": t["ink2"], "ytick.color": t["ink2"], "font.size": 9.5})
        for which in ("roll", "pitch"):
            fig = portrait(loops, stats, mode, t, which)
            p = f"{OUT}/{which}_portrait_{mode}.png"
            fig.savefig(p, dpi=125, facecolor=t["bg"])
            plt.close(fig)
            print("wrote", p)


if __name__ == "__main__":
    main()
