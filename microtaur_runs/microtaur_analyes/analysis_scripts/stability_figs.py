"""Support-polygon / static-stability figures from the stability_probe CSVs.

Produces rollouts/stability_figs/:
  s1_polygons.png   support polygon + CoM ground projection through one gait cycle, per variant
  s2_margin.png     CoM-to-support signed distance vs time, and CoM lateral track
  s3_summary.png    lateral-CoM RMS, %cycle with >=3 feet, CoM-to-support-line distance, polygon area
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VARS = ["rigid", "pitch", "yaw", "roll"]
COL = {"rigid": "#33688f", "pitch": "#c9761f", "yaw": "#2f8a61", "roll": "#8f4bbf"}
INK, MUT, LINE = "#16212a", "#5a6b78", "#d0d7dd"
IN = "#2f8a61"    # CoM inside support
OUT_ = "#c0392b"  # CoM outside / on a line
SRC = Path("rollouts/stability")
OD = Path("rollouts/stability_figs"); OD.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUT,
                     "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})


def hull(pts):
  pts = np.asarray(pts, float)
  if len(pts) < 3:
    return pts
  c = pts.mean(0)
  return pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]


def seg_dist(p, a, b):
  ab = b - a
  t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
  return np.linalg.norm(p - (a + t * ab))


def point_in_poly(p, poly):
  x, y = p
  inside = False
  n = len(poly)
  for i in range(n):
    x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
    if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
      inside = not inside
  return inside


def margin(com, feet, contact):
  """signed CoM-to-support distance [m]: >0 inside a >=3-foot polygon, else negative."""
  pts = feet[contact.astype(bool)]
  k = len(pts)
  if k < 2:
    return np.nan, k
  if k == 2:
    return -seg_dist(com, pts[0], pts[1]), 2
  poly = hull(pts)
  edges = [seg_dist(com, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]
  d = min(edges)
  return (d if point_in_poly(com, poly) else -d), k


def poly_area(pts):
  if len(pts) < 3:
    return 0.0
  p = hull(pts)
  return 0.5 * abs(sum(p[i, 0] * p[(i + 1) % len(p), 1] - p[(i + 1) % len(p), 0] * p[i, 1]
                       for i in range(len(p))))


D = {v: pd.read_csv(SRC / f"{v}.csv") for v in VARS}
for v in VARS:
  d = D[v]
  d = d[d.t >= 1.0].reset_index(drop=True)
  feet = np.stack([d[[f"foot{j}_x", f"foot{j}_y"]].to_numpy() for j in range(4)], axis=1)  # (T,4,2)
  con = d[[f"foot{j}_contact" for j in range(4)]].to_numpy()
  com = d[["com_x", "com_y"]].to_numpy()
  m = np.array([margin(com[i], feet[i], con[i]) for i in range(len(d))], dtype=object)
  d["margin_m"] = np.array([x[0] for x in m], float)
  d["ncontact"] = np.array([x[1] for x in m], int)
  d["poly_area"] = [poly_area(feet[i][con[i].astype(bool)]) for i in range(len(d))]
  D[v] = d


# ---------- s1: polygons through one gait cycle ---------- #
def one_cycle_idx(d):
  c0 = d["foot0_contact"].to_numpy()
  rises = np.where((c0[1:] == 1) & (c0[:-1] == 0))[0] + 1
  if len(rises) >= 3:
    return rises[1], rises[2]
  return 0, min(len(d), 40)


NPH = 8
fig, axes = plt.subplots(len(VARS), NPH, figsize=(1.55 * NPH, 1.7 * len(VARS)))
for row, v in enumerate(VARS):
  d = D[v].reset_index(drop=True)
  i0, i1 = one_cycle_idx(d)
  idx = np.linspace(i0, i1 - 1, NPH).round().astype(int)
  feet = np.stack([d[[f"foot{j}_x", f"foot{j}_y"]].to_numpy() for j in range(4)], axis=1)
  con = d[[f"foot{j}_contact" for j in range(4)]].to_numpy()
  com = d[["com_x", "com_y"]].to_numpy()
  for col, i in enumerate(idx):
    ax = axes[row, col]
    c = com[i]
    fx = feet[i] - c
    down = con[i].astype(bool)
    if down.sum() >= 3:
      poly = hull(fx[down])
      ax.fill(poly[:, 0], poly[:, 1], color=COL[v], alpha=.18)
      ax.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]), color=COL[v], lw=1)
    elif down.sum() == 2:
      pp = fx[down]
      ax.plot(pp[:, 0], pp[:, 1], color=COL[v], lw=1.4)
    ax.scatter(fx[down, 0], fx[down, 1], s=26, color=COL[v], zorder=3)
    ax.scatter(fx[~down, 0], fx[~down, 1], s=20, facecolors="none", edgecolors=MUT, zorder=3)
    mg, k = margin(com[i], feet[i], con[i])
    ax.scatter(0, 0, marker="+", s=90, lw=2.0, color=(IN if (mg is not None and mg > 0) else OUT_), zorder=4)
    ax.set_xlim(-0.20, 0.20); ax.set_ylim(-0.13, 0.13); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
      s.set_color(LINE)
    if col == 0:
      ax.set_ylabel(v, color=COL[v], fontsize=10, fontweight="bold", rotation=0, ha="right", va="center")
    if row == 0:
      ax.set_title(f"φ {col / NPH:.2f}", fontsize=8, color=MUT)
fig.suptitle("Support polygon (hull of stance feet) and CoM ground projection (+) through one gait cycle"
             "\ngreen + = CoM inside a ≥3-foot polygon · red + = 2-foot support or CoM outside · "
             "frame centred on CoM, +x = forward, +y = left · axis span ±0.20 m × ±0.13 m",
             fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OD / "s1_polygons.png", dpi=130); plt.close(fig)

# ---------- s2: margin + lateral CoM ---------- #
fig, ax = plt.subplots(2, 1, figsize=(9.4, 5.2), sharex=True)
for v in VARS:
  d = D[v]
  ax[0].plot(d.t, d.margin_m * 1000, lw=.9, color=COL[v], label=v)
  yy = d.com_y - np.polyval(np.polyfit(d.t, d.com_y, 1), d.t)
  ax[1].plot(d.t, yy * 1000, lw=.9, color=COL[v], label=v)
ax[0].axhline(0, color=INK, lw=1)
ax[0].axhspan(0, 60, color=IN, alpha=.06)
ax[0].set_ylabel("CoM → support  [mm]\n(+ inside ≥3-foot polygon)")
ax[0].set_title("Static-stability margin  (mostly 2-foot support → negative by construction)")
ax[0].legend(frameon=False, fontsize=8, ncol=4)
ax[1].axhline(0, color=INK, lw=1)
ax[1].set_ylabel("CoM lateral offset  [mm]\n(heading-detrended)")
ax[1].set_xlabel("t  [s]"); ax[1].set_title("CoM side-to-side excursion")
fig.tight_layout(); fig.savefig(OD / "s2_margin.png", dpi=130); plt.close(fig)

# ---------- s3: summary bars ---------- #
rows = []
for v in VARS:
  d = D[v]
  yy = (d.com_y - np.polyval(np.polyfit(d.t, d.com_y, 1), d.t)) * 1000
  mg = d.margin_m.dropna().to_numpy() * 1000
  # slow lateral wander = RMS of a ~0.6 s moving average (removes stride ripple)
  w = max(3, int(round(0.6 / (d.t.iloc[1] - d.t.iloc[0]))))
  slow = pd.Series(yy).rolling(w, center=True, min_periods=1).mean()
  rows.append(dict(v=v,
                   lat_rms_mm=yy.std(),
                   lat_peak_mm=yy.abs().max(),
                   slow_wander_mm=slow.std(),
                   worst_margin_p1_mm=np.percentile(mg, 1),
                   z_rms_mm=d.com_z.std() * 1000,
                   frac_ge3=float((d.ncontact >= 3).mean())))
S = pd.DataFrame(rows)
S.to_csv(OD / "stability_summary.csv", index=False)
fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.4))
specs = [("lat_rms_mm", "CoM lateral RMS [mm]\n(total side-to-side sway, lower = steadier)"),
         ("slow_wander_mm", "CoM slow lateral wander [mm]\n(0.6 s-smoothed, i.e. the body drift)"),
         ("worst_margin_p1_mm", "worst static margin, 1st pctile [mm]\n(depth of near-topples)"),
         ("frac_ge3", "fraction of cycle with ≥3 feet down")]
for k, (col, title) in enumerate(specs):
  ax[k].bar(S.v, S[col], color=[COL[v] for v in S.v])
  ax[k].set_title(title, fontsize=8.3)
  ax[k].axhline(0, color=INK, lw=.8)
  for i, val in enumerate(S[col]):
    ax[k].text(i, val, f"{val:.1f}" if abs(val) >= 1 else f"{val:.2f}",
               ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
fig.tight_layout(); fig.savefig(OD / "s3_summary.png", dpi=130); plt.close(fig)

print("wrote", *(p.name for p in sorted(OD.glob("*.png"))))
print(S.round(2).to_string(index=False))
