"""Per-robot CoM attitude (RPY) and CoM position (XYZ) traces, all treatments.

    python scripts/com_figs.py                      # all four variants, all 3 speeds, seed 0
    python scripts/com_figs.py --vx 0.14 0.20        # just these speeds

One figure per robot: 6 rows (roll/pitch/yaw, CoM x/y/z) x 4 columns (terrain),
each panel overlaying all commanded speeds (light -> dark = 0.08 -> 0.20 m/s).
Rendered twice -- light and dark -- so the artifact can swap them with the
viewer's theme instead of showing a white slab on a dark page.

Colour is the robot's own accent (matching the spine/body phase-portrait
figures elsewhere in this report); terrain is a column facet and speed is
shade, not colour, so all three treatments per terrain are visible at once
instead of needing one figure per speed.

Palette: dataviz reference instance, categorical slots 1-4 in fixed order.
"""
from __future__ import annotations

import argparse, glob, json, os, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TERRAINS = ["flat", "curb", "steps", "weave"]
TNAME = {"flat": "flat", "curb": "curb (on the lip)", "steps": "step field (steered)", "weave": "weave"}
# lane_keep required per terrain -- steps is waypoint-steered (open-loop step-field
# runs are drift + reset confounded, see CORRECTION.md); the others are open-loop,
# matching spine_function.py's MODE so every figure in this report agrees on which
# runs are "the" steps test.
MODE = {"flat": False, "curb": False, "weave": False, "steps": True}
VARIANTS = ["rigid", "pitch", "yaw", "roll"]
SPEEDS = [0.08, 0.14, 0.20]
SHADE = {0.08: 0.38, 0.14: 0.68, 0.20: 1.0}
OUT = Path("rollouts/com_figs")

# dataviz reference palette, categorical slots 1-4 (fixed order, not cycled) --
# reused here as one accent per VARIANT, matching spine_figs.py / body_phase_portraits.py
VCOL_LIGHT = {"rigid": "#2a78d6", "pitch": "#eb6834", "yaw": "#1baf7a", "roll": "#8f4bbf"}
VCOL_DARK  = {"rigid": "#3987e5", "pitch": "#d95926", "yaw": "#199e70", "roll": "#b78fd6"}

THEME = {
  "light": dict(vcol=VCOL_LIGHT, surface="#fcfcfb", ink="#0b0b0b",
                ink2="#52514e", grid="#d9dbd6"),
  "dark":  dict(vcol=VCOL_DARK,  surface="#1a1a19", ink="#ffffff",
                ink2="#c3c2b7", grid="#3a3a38"),
}


def mix(c, bg, a):
  c = np.array(matplotlib.colors.to_rgb(c))
  b = np.array(matplotlib.colors.to_rgb(bg))
  return tuple(b + (c - b) * a)


def savefig_retry(fig, path, **kw):
  """This machine's disk (OneDrive-synced Downloads) intermittently raises
  OSError/errno 22 on a fresh file write, seemingly from a sync/AV process
  transiently holding the path -- retry a few times before giving up."""
  for attempt in range(5):
    try:
      fig.savefig(path, **kw)
      return
    except OSError:
      if attempt == 4:
        raise
      time.sleep(0.5 * (attempt + 1))


def discover(variant):
  """ALL raw-action sweep dirs per terrain, newest last.

  Extra-seed runs land in their own timestamped directory, so a newest-only
  lookup finds the top-up (seeds 3-7) and misses seed 0. Pre-fix sweeps are
  excluded by the `raw_action` flag they lack.
  """
  found = {}
  for d in sorted(glob.glob(f"rollouts/{variant}_*"), key=os.path.getmtime):
    p = Path(d)
    if not (p / "summary.json").exists():
      continue
    metas = sorted(glob.glob(str(p / "rollout_*.meta.json")))
    if not metas:
      continue
    try:
      m0 = json.loads(Path(metas[0]).read_text())
      rows = json.loads((p / "summary.json").read_text())
    except Exception:  # noqa: BLE001
      continue
    if not rows or not m0.get("raw_action", False):
      continue
    terr = rows[0].get("terrain", "flat")
    if bool(m0.get("lane_keep", False)) != MODE.get(terr, False):
      continue
    if terr == "weave" and (m0.get("terrain_kw") or {}).get("spacing") != 0.55:
      continue
    found.setdefault(terr, []).append(p)
  return found


def origin_z(meta) -> float:
  """Height of the terrain's spawn origin above the road datum, metres.

  `com_z` in the CSV is relative to the env origin, and the curb spawns the robot
  at kerb-top height -- so without this its CoM reads ~15 mm low against the other
  terrains, which is a frame artifact rather than the robot riding lower.
  """
  if meta.get("terrain") == "curb":
    return float((meta.get("terrain_kw") or {}).get("curb_height", 0.015))
  return 0.0


def trace(dirs, vx: float, seed: int):
  for d in dirs:
   for mf in sorted(glob.glob(str(Path(d) / "rollout_*.meta.json"))):
    m = json.loads(Path(mf).read_text())
    if abs(m["cmd_vx"] - vx) < 1e-6 and m["seed"] == seed:
      csv = mf.replace(".meta.json", ".csv")
      if Path(csv).exists():
        df = pd.read_csv(csv)
        df["com_z_road"] = df.com_z + origin_z(m)   # common datum
        return df
   # fall through to the next dir
  return None


def render(variant, data, speeds, seed, mode):
  """data: {(terrain, vx): df or None}."""
  t = THEME[mode]
  vcol = t["vcol"][variant]
  plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "text.color": t["ink"], "axes.labelcolor": t["ink2"],
    "xtick.color": t["ink2"], "ytick.color": t["ink2"],
  })
  panels = [
    ("roll",  lambda df: np.degrees(df.roll),  "roll  [deg]"),
    ("pitch", lambda df: np.degrees(df.pitch), "pitch  [deg]"),
    ("yaw",   lambda df: np.degrees(np.unwrap(df.yaw.to_numpy())), "yaw  [deg]"),
    ("com_x", lambda df: df.com_x,             "CoM x  [m]   (travel)"),
    ("com_y", lambda df: df.com_y * 1000.0,    "CoM y  [mm]  (lateral)"),
    ("com_z", lambda df: df.com_z_road * 1000.0, "CoM z  [mm]  (above road)"),
  ]
  nrow, ncol = len(panels), len(TERRAINS)
  fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 2.15 * nrow), facecolor=t["surface"])

  for r, (key, fn, ylab) in enumerate(panels):
    for c, terr in enumerate(TERRAINS):
      ax = axes[r][c]
      ax.set_facecolor(t["surface"])
      ends = []
      for s in speeds:
        df = data.get((terr, s))
        if df is None:
          continue
        y = fn(df)
        col = mix(vcol, t["surface"], SHADE[s])
        ax.plot(df.t, y, lw=1.8 if s == max(speeds) else 1.3, color=col,
                solid_capstyle="round", zorder=3)
        yv = float(y.iloc[-1] if hasattr(y, "iloc") else y[-1])
        ends.append([float(df.t.iloc[-1]), yv, f"{s:.2f}", col])
      if ends:
        lo, hi = ax.get_ylim()
        gap = (hi - lo) * 0.09
        ends.sort(key=lambda e: e[1])
        for j in range(1, len(ends)):
          if ends[j][1] - ends[j - 1][1] < gap:
            ends[j][1] = ends[j - 1][1] + gap
        for xe, ye, lab, col in ends:
          ax.annotate(lab, xy=(xe, ye), xytext=(3, 0), textcoords="offset points",
                      fontsize=6.5, color=col, va="center", zorder=6,
                      annotation_clip=False)
      ax.grid(True, color=t["grid"], lw=0.6, alpha=0.7)
      ax.set_axisbelow(True)
      for s in ("top", "right"):
        ax.spines[s].set_visible(False)
      for s in ("left", "bottom"):
        ax.spines[s].set_color(t["grid"])
      if c == 0:
        ax.set_ylabel(ylab)
      if r == 0:
        ax.set_title(TNAME[terr], fontsize=10, color=t["ink"])
      if r == nrow - 1:
        ax.set_xlabel("t  [s]")
      ax.margins(x=0.09)

  handles = [plt.Line2D([], [], color=mix(vcol, t["surface"], SHADE[s]),
                        lw=1.8 if s == max(speeds) else 1.3, label=f"{s:.2f} m/s")
             for s in speeds]
  fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False,
             bbox_to_anchor=(0.995, 0.995), fontsize=8.5,
             labelcolor=t["ink2"])
  fig.suptitle(f"{variant} — trunk attitude and CoM position, all commanded speeds   (seed {seed})",
               fontsize=11.5, color=t["ink"], x=0.008, ha="left", y=0.995)
  fig.tight_layout(rect=(0, 0, 1, 0.96))
  OUT.mkdir(parents=True, exist_ok=True)
  p = OUT / f"com_{variant}_{mode}.png"
  savefig_retry(fig, p, dpi=135, facecolor=t["surface"])
  plt.close(fig)
  return p


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--vx", type=float, nargs="+", default=SPEEDS)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--variants", nargs="+", default=VARIANTS)
  a = ap.parse_args()
  for v in a.variants:
    got = discover(v)
    data = {(terr, s): trace(dirs, s, a.seed) for terr, dirs in got.items() for s in a.vx}
    have = sorted({terr for (terr, s), x in data.items() if x is not None})
    if not have:
      print(f"{v}: no traces at vx={a.vx} seed={a.seed}"); continue
    for mode in ("light", "dark"):
      p = render(v, data, a.vx, a.seed, mode)
    print(f"{v}: terrains {have}, speeds {a.vx} -> {OUT}/com_{v}_{{light,dark}}.png")


if __name__ == "__main__":
  main()
