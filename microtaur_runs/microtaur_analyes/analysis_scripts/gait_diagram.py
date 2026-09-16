"""Gait diagrams (stance bars per leg) + foot-XY snapshots, per variant.

Reads the stability-probe CSVs (which have foot world XY + contact). Writes
rollouts/stability_figs/g1_gait_diagram.png and g2_foot_tracks.png.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VARS = ["rigid", "pitch", "yaw", "roll"]
COL = {"rigid": "#33688f", "pitch": "#c9761f", "yaw": "#2f8a61", "roll": "#8f4bbf"}
SRC = Path("rollouts/stability")
OD = Path("rollouts/stability_figs"); OD.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def layout(d):
  xs = [d[f"foot{j}_x"].mean() - d.com_x.mean() for j in range(4)]
  ys = [d[f"foot{j}_y"].mean() - d.com_y.mean() for j in range(4)]
  xm = np.median(xs)
  return {j: ("F" if xs[j] > xm else "H") + ("L" if ys[j] > 0 else "R") for j in range(4)}


# ---- g1: gait diagram ---- #
def classify(d):
  lab = layout(d)
  inv = {n: j for j, n in lab.items()}
  C = {n: d[f"foot{inv[n]}_contact"].to_numpy().astype(float) for n in lab.values()}

  def cr(a, b):
    a = C[a] - C[a].mean(); b = C[b] - C[b].mean()
    return float((a * b).mean() / (a.std() * b.std() + 1e-9))
  diag = np.mean([cr("FL", "HR"), cr("FR", "HL")])
  lat = np.mean([cr("FL", "HL"), cr("FR", "HR")])
  bnd = np.mean([cr("FL", "FR"), cr("HL", "HR")])
  return ["trot", "pace", "bound"][int(np.argmax([diag, lat, bnd]))], (diag, lat, bnd)


PAIR = {"trot": ({"FL", "HR"}, {"FR", "HL"}), "pace": ({"FL", "HL"}, {"FR", "HR"}),
        "bound": ({"FL", "FR"}, {"HL", "HR"})}
fig, axes = plt.subplots(len(VARS), 1, figsize=(10, 6.8), sharex=True)
WIN = (2.0, 6.0)
for ax, v in zip(axes, VARS):
  d = pd.read_csv(SRC / f"{v}.csv")
  dd = d[(d.t >= WIN[0]) & (d.t <= WIN[1])]
  lab = layout(d)
  gait, (di, la, bn) = classify(d[d.t >= 1.0])
  p0, p1 = PAIR[gait]
  order = sorted(range(4), key=lambda j: {"FL": 0, "FR": 1, "HL": 2, "HR": 3}[lab[j]])
  for row, j in enumerate(order):
    hue = COL[v] if lab[j] in p0 else "#9aa7b1"
    c = dd[f"foot{j}_contact"].to_numpy(); t = dd.t.to_numpy()
    s = None
    for i in range(len(c)):
      if c[i] and s is None:
        s = t[i]
      elif not c[i] and s is not None:
        ax.barh(row, t[i] - s, left=s, height=0.62, color=hue); s = None
    if s is not None:
      ax.barh(row, t[-1] - s, left=s, height=0.62, color=hue)
  ax.set_yticks(range(4)); ax.set_yticklabels([lab[j] for j in order], fontsize=8.5)
  ax.set_ylim(-0.6, 3.6); ax.invert_yaxis()
  ax.set_ylabel(f"{v}\n{gait.upper()}", color=COL[v], fontweight="bold", fontsize=9.5)
  ax.text(0.995, 0.5, f"diag {di:+.2f} · lat {la:+.2f}", transform=ax.transAxes,
          ha="right", va="center", fontsize=7.5, color="#666")
  ax.grid(axis="x", alpha=.3)
axes[-1].set_xlabel("t  [s]")
fig.suptitle("Gait diagram — bar = foot in stance.  Coloured rows = one synchronised pair, grey = the other.\n"
             "trot = diagonal pairs (FL+HR / FR+HL);  pace = lateral pairs (FL+HL / FR+HR).", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OD / "g1_gait_diagram.png", dpi=130); plt.close(fig)

# ---- g2: foot-XY tracks over ~1.5 gait cycles, CoM-centred ---- #
fig, axes = plt.subplots(1, len(VARS), figsize=(14, 3.6))
for ax, v in zip(axes, VARS):
  d = pd.read_csv(SRC / f"{v}.csv")
  d = d[(d.t >= 3.0) & (d.t <= 4.2)].reset_index(drop=True)
  lab = layout(d)
  for j in range(4):
    fx = d[f"foot{j}_x"] - d.com_x
    fy = d[f"foot{j}_y"] - d.com_y
    con = d[f"foot{j}_contact"].to_numpy().astype(bool)
    ax.plot(fx, fy, "-", color=COL[v], lw=.6, alpha=.4)
    ax.scatter(fx[con], fy[con], s=10, color=COL[v])
    ax.scatter(fx[~con], fy[~con], s=8, facecolors="none", edgecolors=COL[v], lw=.5)
    ax.annotate(lab[j], (fx.iloc[0], fy.iloc[0]), fontsize=7, color="#333")
  ax.scatter(0, 0, marker="+", s=80, color="#c0392b")
  ax.set_title(v, color=COL[v]); ax.set_aspect("equal")
  ax.set_xlabel("fore–aft rel CoM [m]"); ax.set_xlim(-0.28, 0.20); ax.set_ylim(-0.16, 0.16)
axes[0].set_ylabel("lateral rel CoM [m]  (+ = left)")
fig.suptitle("Foot positions over ~1.5 gait cycles (filled = contact, hollow = swing); + = CoM", fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OD / "g2_foot_tracks.png", dpi=130); plt.close(fig)

# ---- text classification with duty + phase ---- #
print(f"{'variant':8s} {'gait':6s}  diag/lat/bound contact-corr   duty(FL,FR,HL,HR)   phase vs FL")
for v in VARS:
  d = pd.read_csv(SRC / f"{v}.csv"); d = d[d.t >= 1.0].reset_index(drop=True)
  lab = layout(d)
  inv = {name: j for j, name in lab.items()}
  C = {name: d[f"foot{inv[name]}_contact"].to_numpy().astype(float) for name in lab.values()}

  def corr(a, b):
    a = C[a] - C[a].mean(); b = C[b] - C[b].mean()
    return float((a * b).mean() / (a.std() * b.std() + 1e-9))

  diag = np.mean([corr("FL", "HR"), corr("FR", "HL")])
  lat = np.mean([corr("FL", "HL"), corr("FR", "HR")])
  bnd = np.mean([corr("FL", "FR"), corr("HL", "HR")])
  g = ["TROT", "PACE", "BOUND"][int(np.argmax([diag, lat, bnd]))]

  def ph(name):
    s = (C[name] > 0.5).astype(int)
    r = np.where((s[1:] == 1) & (s[:-1] == 0))[0] + 1
    return r

  rFL = ph("FL")
  per = np.median(np.diff(rFL)) if len(rFL) > 2 else np.nan
  phs = {}
  for name in ("FL", "FR", "HL", "HR"):
    r = ph(name)
    phs[name] = ((r[0] - rFL[0]) / per) % 1 if len(r) and not np.isnan(per) else np.nan
  duty = {name: C[name].mean() for name in ("FL", "FR", "HL", "HR")}
  print(f"{v:8s} {g:6s}  d={diag:+.2f} l={lat:+.2f} b={bnd:+.2f}   "
        f"({duty['FL']:.2f},{duty['FR']:.2f},{duty['HL']:.2f},{duty['HR']:.2f})   "
        f"FL={phs['FL']:.2f} FR={phs['FR']:.2f} HL={phs['HL']:.2f} HR={phs['HR']:.2f}")
