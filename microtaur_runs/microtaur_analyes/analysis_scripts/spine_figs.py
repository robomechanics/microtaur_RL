"""Figures for the spine-function section.

    python scripts/spine_figs.py    (after spine_function.py)
      -> rollouts/spine_function/{budget,loops,phase}_{light,dark}.png

budget  where the spine's energy goes, per terrain and speed: put in by the
        policy (right), dissipated by servo damping / absorbed by the policy /
        lost to joint friction (left), net as a diamond, role on the right.
loops   stride-averaged work loops, torque vs spine angle. Counter-clockwise
        absorbs energy, clockwise delivers it, a closed-up line is a spring.
phase   spine power across the stride (0 = leg-2 touchdown).
portrait  state-space phase portrait, spine angular velocity vs spine angle
        (both about the stride mean). A closed orbit is periodic motion; its
        area is |angle| x |velocity| swept per stride, not energy (that's the
        torque loop above). CCW here is the sign convention of a standard
        (q, qdot) portrait, not an energy direction.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

D = "rollouts/spine_function"
V = ["pitch", "yaw", "roll"]
T = ["flat", "curb", "steps", "weave"]
TNAME = {"flat": "flat", "curb": "curb (on the lip)", "steps": "step field (steered)", "weave": "weave"}
S = [0.08, 0.14, 0.20]
VCOL = {"light": {"pitch": "#bd6f1c", "yaw": "#2f8a61", "roll": "#8f4bbf"},
        "dark":  {"pitch": "#e0954c", "yaw": "#45b585", "roll": "#b78fd6"}}
TH = {"light": dict(bg="#ffffff", ink="#131d25", ink2="#586a76", grid="#d9e0e5",
                    cin="#b5542b", cdamp="#5b8db8", cabs="#1f4e79", cfric="#a7b1b9"),
      "dark":  dict(bg="#141d21", ink="#e3ebef", ink2="#90a3ad", grid="#26343b",
                    cin="#e07b4f", cdamp="#6fa3cf", cabs="#a9cdf0", cfric="#5d6b73")}
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
    c = np.array(matplotlib.colors.to_rgb(c)); b = np.array(matplotlib.colors.to_rgb(bg))
    return tuple(b + (c - b) * a)


def role_label(votes):
    """Majority role across seeds; 'a/b' when fewer than 3 in 4 runs agree."""
    kv = sorted(((int(n), r) for r, n in (x.split(":") for x in votes.split(","))), reverse=True)
    tot = sum(n for n, _ in kv)
    if kv[0][0] / tot >= 0.75 or len(kv) == 1:
        return kv[0][1]
    return f"{kv[0][1]}/{kv[1][1]}"


def budget(cells, mode, t):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.4), sharey=True, facecolor=t["bg"])
    ylab, ypos = [], []
    y = 0
    for terr in T:
        for s in S:
            ylab.append(f"{TNAME[terr].split(' (')[0]}  {s:.2f}")
            ypos.append(y); y -= 1
        y -= 0.6
    for ax, v in zip(axes, V):
        style(ax, t)
        i = 0
        for terr in T:
            for s in S:
                yy = ypos[i]; i += 1
                if (v, terr, s) not in cells.index:
                    continue
                c = cells.loc[(v, terr, s)]
                pin = max(c.policy, 0) * 1e3
                damp = -c.damper * 1e3
                pab = max(-c.policy, 0) * 1e3
                fr = c.friction * 1e3
                ax.barh(yy, pin, color=t["cin"], height=0.72)
                ax.barh(yy, -damp, color=t["cdamp"], height=0.72)
                ax.barh(yy, -pab, left=-damp, color=t["cabs"], height=0.72)
                ax.barh(yy, -fr, left=-damp - pab, color=t["cfric"], height=0.72)
                ax.plot(c.net * 1e3, yy, "D", ms=6.5, color=t["ink"], mec=t["bg"], mew=1.0, zorder=5)
                lab = role_label(c.role_votes)
                ax.text(1.01, yy, f"{lab}  {c.returned * 100:.0f}%", transform=ax.get_yaxis_transform(),
                        ha="left", va="center", fontsize=8.6,
                        color=t["ink"] if "/" in lab else VCOL[mode][v],
                        fontweight="normal" if "/" in lab else "bold")
        ax.axvline(0, color=t["ink2"], lw=0.9)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, max(hi, 0.12 * (hi - lo)))
        ax.set_title(v, fontsize=12, color=VCOL[mode][v], loc="left", fontweight="bold")
        ax.set_xlabel("spine mechanical power  [mW]      ← absorbed      delivered →")
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(ylab, fontsize=8.8)
    hs = [plt.Rectangle((0, 0), 1, 1, color=t[k]) for k in ("cin", "cdamp", "cabs", "cfric")]
    hs.append(plt.Line2D([], [], marker="D", ls="", color=t["ink"], ms=6.5))
    fig.legend(hs, ["put in by the policy", "servo damping", "absorbed by the policy",
                    "joint friction (outside the servo)", "servo net"], loc="upper center", ncol=5, frameon=False,
               fontsize=9.5, labelcolor=t["ink2"], bbox_to_anchor=(0.5, 1.0))
    fig.text(0.995, 0.935, "role · share returned", ha="right", fontsize=8.6, color=t["ink2"])
    fig.subplots_adjust(left=0.085, right=0.915, top=0.86, bottom=0.09, wspace=0.42)
    return fig


def loops(L, cells, mode, t):
    fig, axes = plt.subplots(3, 4, figsize=(16.5, 11), facecolor=t["bg"])
    for r, v in enumerate(V):
        for cidx, terr in enumerate(T):
            ax = axes[r, cidx]
            style(ax, t)
            notes = []
            for s in S:
                k = f"{v}|{terr}|{s:.2f}"
                if k not in L:
                    continue
                a = L[k].mean(0)                    # (3, NPH): q deg, tau mN m, P mW
                qx, tq = np.r_[a[0], a[0][:1]], np.r_[a[1], a[1][:1]]
                col = mix(VCOL[mode][v], t["bg"], SHADE[s])
                ax.plot(qx, tq, color=col, lw=2.0 if s == 0.20 else 1.5, zorder=3)
                j = len(a[0]) // 4
                ax.add_patch(FancyArrowPatch((qx[j], tq[j]), (qx[j + 1], tq[j + 1]), arrowstyle="-|>",
                                             mutation_scale=13, color=col, lw=0, zorder=4))
                w = cells.loc[(v, terr, s)].stride_net_mJ   # exact, not from the averaged loop
                notes.append(f"{s:.2f}: {w:+.2f} mJ")
            ax.text(0.02, 0.98, "\n".join(notes), transform=ax.transAxes, va="top", ha="left",
                    fontsize=7.8, color=t["ink2"], family="monospace", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.25", fc=t["bg"], ec=t["grid"], alpha=0.85))
            if r == 0:
                ax.set_title(TNAME[terr], fontsize=11, color=t["ink"])
            if cidx == 0:
                ax.set_ylabel(f"{v}\nspine torque  [mN m]", color=VCOL[mode][v], fontsize=10.5)
            if r == 2:
                ax.set_xlabel("spine angle about its mean  [deg]")
    fig.suptitle("Stride-averaged spine work loops (inset: exact net work per stride at each speed; "
                 "light → dark = 0.08 → 0.20 m/s).   Counter-clockwise absorbs, clockwise delivers.",
                 fontsize=11, color=t["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def portrait(L, mode, t):
    fig, axes = plt.subplots(3, 4, figsize=(16.5, 11), facecolor=t["bg"])
    for r, v in enumerate(V):
        for cidx, terr in enumerate(T):
            ax = axes[r, cidx]
            style(ax, t)
            notes = []
            for s in S:
                k = f"{v}|{terr}|{s:.2f}"
                if k not in L:
                    continue
                a = L[k].mean(0)                    # (4, NPH): q deg, tau mN m, P mW, qd deg/s
                qx, vy = np.r_[a[0], a[0][:1]], np.r_[a[3], a[3][:1]]
                col = mix(VCOL[mode][v], t["bg"], SHADE[s])
                ax.plot(qx, vy, color=col, lw=2.0 if s == 0.20 else 1.5, zorder=3)
                j = len(a[0]) // 4
                ax.add_patch(FancyArrowPatch((qx[j], vy[j]), (qx[j + 1], vy[j + 1]), arrowstyle="-|>",
                                             mutation_scale=13, color=col, lw=0, zorder=4))
                notes.append(f"{s:.2f}: {np.abs(a[3]).max():.0f} deg/s pk")
            ax.axhline(0, color=t["ink2"], lw=0.8)
            ax.axvline(0, color=t["ink2"], lw=0.8)
            ax.text(0.02, 0.98, "\n".join(notes), transform=ax.transAxes, va="top", ha="left",
                    fontsize=7.8, color=t["ink2"], family="monospace", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.25", fc=t["bg"], ec=t["grid"], alpha=0.85))
            if r == 0:
                ax.set_title(TNAME[terr], fontsize=11, color=t["ink"])
            if cidx == 0:
                ax.set_ylabel(f"{v}\nspine angular velocity  [deg/s]", color=VCOL[mode][v], fontsize=10.5)
            if r == 2:
                ax.set_xlabel("spine angle about its mean  [deg]")
    fig.suptitle("Stride-averaged spine phase portraits: angular velocity vs angle (inset: peak angular "
                 "velocity at each speed; light -> dark = 0.08 -> 0.20 m/s).", fontsize=11, color=t["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def phase(L, mode, t):
    fig, axes = plt.subplots(3, 4, figsize=(16.5, 9.5), sharex=True, facecolor=t["bg"])
    for r, v in enumerate(V):
        for cidx, terr in enumerate(T):
            ax = axes[r, cidx]
            style(ax, t)
            for s in S:
                k = f"{v}|{terr}|{s:.2f}"
                if k not in L:
                    continue
                p = L[k].mean(0)[2]
                ph = np.linspace(0, 1, len(p), endpoint=False)
                ax.plot(ph, p, color=mix(VCOL[mode][v], t["bg"], SHADE[s]), lw=1.8, label=f"{s:.2f} m/s")
            ax.axhline(0, color=t["ink2"], lw=0.8)
            ax.axvline(0.5, color=t["grid"], lw=1.0, ls="--")
            if r == 0:
                ax.set_title(TNAME[terr], fontsize=11, color=t["ink"])
            if cidx == 0:
                ax.set_ylabel(f"{v}\nspine power  [mW]", color=VCOL[mode][v], fontsize=10.5)
            if r == 2:
                ax.set_xlabel("stride phase")
    h = [plt.Line2D([], [], lw=1.8, color=mix(t["ink"], t["bg"], SHADE[s])) for s in S]
    l = [f"{s:.2f} m/s" for s in S]
    fig.legend(h, l, loc="upper right", ncol=3, frameon=False, fontsize=9, labelcolor=t["ink2"],
               bbox_to_anchor=(0.99, 0.975))
    fig.suptitle("Spine power over the stride. 0 = leg-2 touchdown; dashed line ≈ the other diagonal "
                 "lands. Above 0 the spine delivers energy, below 0 it absorbs.", fontsize=11,
                 color=t["ink"], x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def main():
    cells = pd.read_csv(f"{D}/cells.csv").set_index(["variant", "terrain", "cmd"])
    L = dict(np.load(f"{D}/loops.npz"))
    for mode, t in TH.items():
        plt.rcParams.update({"text.color": t["ink"], "axes.labelcolor": t["ink2"],
                             "xtick.color": t["ink2"], "ytick.color": t["ink2"], "font.size": 9.5})
        for name, fn in (("budget", lambda: budget(cells, mode, t)),
                         ("loops", lambda: loops(L, cells, mode, t)),
                         ("portrait", lambda: portrait(L, mode, t)),
                         ("phase", lambda: phase(L, mode, t))):
            fig = fn()
            p = f"{D}/{name}_{mode}.png"
            fig.savefig(p, dpi=125, facecolor=t["bg"])
            plt.close(fig)
            print("wrote", p)


if __name__ == "__main__":
    main()
