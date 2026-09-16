"""Cross-variant comparison over the task terrains.

    python scripts/compare_variants_terrain.py
    python scripts/compare_variants_terrain.py --variants rigid pitch yaw roll

Reads each variant's four terrain sweeps (flat / curb / steps / weave) and
writes rollouts/terrain_compare/all/ with:

  * ``VARIANT_TERRAIN.md``  -- absolute numbers, and the penalty relative to
    each variant's OWN flat baseline;
  * ``v1_absolute.png``     -- metric vs speed, one panel per terrain;
  * ``v2_penalty.png``      -- CoT and speed-loss penalty vs flat;
  * ``rollouts.csv``.

The penalty view matters because the variants differ in baseline efficiency and
mass (rigid 0.465 kg, yaw 0.724 kg). Absolute CoT on a terrain conflates "this
morphology is efficient" with "this morphology handles terrain well"; dividing
by the variant's own flat CoT isolates the second.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TERRAINS = ["flat", "curb", "steps", "weave"]
VARIANTS = ["rigid", "pitch", "yaw", "roll"]
VCOL = {"rigid": "#33688f", "pitch": "#bd6f1c", "yaw": "#2f8a61", "roll": "#8f4bbf"}
VCOL_DARK = {"rigid": "#5da0c8", "pitch": "#e0954c", "yaw": "#45b585", "roll": "#b78fd6"}
OUT = Path("rollouts/terrain_compare/all")

# Rendered in both themes so the report can swap them with the viewer's theme
# instead of showing a white slab on a dark page.
THEME = {
  "light": dict(col=VCOL, surface="#ffffff", ink="#131d25", ink2="#586a76", grid="#d9e0e5"),
  "dark":  dict(col=VCOL_DARK, surface="#141d21", ink="#e3ebef", ink2="#90a3ad", grid="#26343b"),
}

# Below this achieved speed the robot is effectively stationary; CoT divides by
# max(|v|, 0.02) so it stops being a meaningful efficiency number.
STALL_V = 0.03


def _is_raw_action(d: Path) -> bool:
  """True only for sweeps run with the unclipped previous action.

  Runs logged before 2026-09-09 fed the policy a clipped `last_action`, which
  changed the trajectories and inflated CoT. They carry no `raw_action` key, so
  a missing key means clipped and the sweep is excluded rather than pooled in.
  """
  metas = sorted(glob.glob(str(d / "rollout_*.meta.json")))
  if not metas:
    return False
  try:
    return bool(json.loads(Path(metas[0]).read_text()).get("raw_action", False))
  except Exception:  # noqa: BLE001
    return False


def discover(variant: str) -> dict[str, list[Path]]:
  """ALL sweep dirs per terrain for this variant, not just the newest.

  Extra-seed runs land in their own timestamped directory, so a newest-only
  lookup would silently report the top-up instead of the pooled set. Rows are
  de-duplicated on (terrain, cmd_vx, seed) downstream.
  """
  found: dict[str, list[Path]] = {}
  for d in sorted(glob.glob(f"rollouts/{variant}_*"), key=os.path.getmtime):
    p = Path(d)
    if not (p / "summary.json").exists():
      continue
    try:
      rows = json.loads((p / "summary.json").read_text())
    except Exception:  # noqa: BLE001
      continue
    if not rows:
      continue
    if not _is_raw_action(p):
      continue          # pre-fix sweep: prev-action was clipped, not comparable
    if json.loads(Path(sorted(glob.glob(str(p / "rollout_*.meta.json")))[0]).read_text()).get("lane_keep"):
      continue          # waypoint-steered run: separate experiment, see waypoint_analysis.py
    found.setdefault(rows[0].get("terrain", "flat"), []).append(p)
  return found


def load(variants: list[str]) -> pd.DataFrame:
  recs = []
  for v in variants:
    got = discover(v)
    missing = [t for t in TERRAINS if t not in got]
    if missing:
      print(f"  {v}: MISSING {missing}")
    for t, dirs in got.items():
      for d in dirs:
        for r in json.loads((d / "summary.json").read_text()):
          r["variant"] = v
          r["_dir"] = d.name
          recs.append(r)
  if not recs:
    raise SystemExit("no sweeps found")
  df = pd.DataFrame(recs)
  # later dirs win, so an intentional re-run supersedes an earlier one
  df = df.drop_duplicates(subset=["variant", "terrain", "cmd_vx", "seed"], keep="last")
  for src, dst in (("straight_dev_final_m", "abs_dev_final_m"),
                   ("straight_dev_per_m", "abs_dev_per_m"),
                   ("heading_drift_deg", "abs_heading_drift_deg")):
    if src in df:
      df[dst] = df[src].abs()
  df["stalled"] = df["mean_v_body_x"] < STALL_V
  df["speed_ratio"] = df["mean_v_body_x"] / df["cmd_vx"]
  return df


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--variants", nargs="+", default=VARIANTS)
  ap.add_argument("--speeds", nargs="+", type=float, default=[0.08, 0.14, 0.20],
                  help="commands to include; ad-hoc development runs at other "
                       "speeds are excluded so the grid stays square")
  args = ap.parse_args()

  print("discovering sweeps:")
  df = load(args.variants)
  df = df[df.cmd_vx.isin(args.speeds)].copy()
  OUT.mkdir(parents=True, exist_ok=True)
  df.to_csv(OUT / "rollouts.csv", index=False)

  vs = [v for v in args.variants if v in set(df.variant)]
  ts = [t for t in TERRAINS if t in set(df.terrain)]
  speeds = sorted(df.cmd_vx.unique())

  def cell(v, t, s, col):
    q = df[(df.variant == v) & (df.terrain == t) & (df.cmd_vx == s)]
    if not len(q) or col not in q:
      return np.nan, np.nan, False
    return float(q[col].mean()), float(q[col].std()), bool(q["stalled"].all())

  # flat baseline per variant/speed, for the penalty view
  base = {(v, s): cell(v, "flat", s, "mean_CoT_pos")[0] for v in vs for s in speeds}
  basev = {(v, s): cell(v, "flat", s, "mean_v_body_x")[0] for v in vs for s in speeds}

  L = ["# Terrain comparison across spine variants", "",
       f"{len(df)} rollouts, {int(df.any_terminated.sum())} terminations. "
       f"Speeds {speeds} m/s, seeds {sorted(df.seed.unique())}.", "",
       "`stall` marks rollouts whose achieved speed is under "
       f"{STALL_V} m/s -- the robot is effectively stationary and CoT "
       "(which divides by max(|v|, 0.02)) is not a meaningful efficiency "
       "number there.", ""]

  L += ["## Absolute", ""]
  for t in ts:
    L += [f"### {t}", "",
          "| variant | cmd vx | achieved | CoT⁺ | roll° | |dev|/m | tortuosity |",
          "|---|---|---|---|---|---|---|"]
    for v in vs:
      for s in speeds:
        a_v, s_v, stall = cell(v, t, s, "mean_v_body_x")
        if np.isnan(a_v):
          continue
        def f(c, k=2):
          a, sd, _ = cell(v, t, s, c)
          return "—" if np.isnan(a) else (f"{a:.{k}f} ± {sd:.{k}f}"
                                          if not np.isnan(sd) else f"{a:.{k}f}")
        tag = " **stall**" if stall else ""
        L.append(f"| {v}{tag} | {s:.2f} | {f('mean_v_body_x',3)} | {f('mean_CoT_pos')} | "
                 f"{f('mean_abs_roll_deg',1)} | {f('abs_dev_per_m',3)} | "
                 f"{f('path_tortuosity',3)} |")
    L.append("")

  L += ["## Penalty relative to each variant's own flat baseline", "",
        "CoT ratio = terrain CoT⁺ / that variant's flat CoT⁺ at the same speed. "
        "Speed ratio = achieved / commanded.", "",
        "| variant | terrain | " + " | ".join(f"CoT×@{s:.2f}" for s in speeds) +
        " | " + " | ".join(f"v/cmd@{s:.2f}" for s in speeds) + " |",
        "|---|---|" + "---|" * (2 * len(speeds))]
  for v in vs:
    for t in ts:
      if t == "flat":
        continue
      cots, spds = [], []
      for s in speeds:
        a, _, stall = cell(v, t, s, "mean_CoT_pos")
        b = base.get((v, s))
        cots.append("stall" if stall else
                    ("—" if (np.isnan(a) or not b) else f"{a / b:.2f}"))
        av, _, _ = cell(v, t, s, "mean_v_body_x")
        spds.append("—" if np.isnan(av) else f"{av / s:.2f}")
      L.append(f"| {v} | {t} | " + " | ".join(cots) + " | " + " | ".join(spds) + " |")
  (OUT / "VARIANT_TERRAIN.md").write_text("\n".join(L) + "\n", encoding="utf8")

  # ---------------- figures ---------------- #
  metrics = [("mean_CoT_pos", "CoT⁺"), ("mean_v_body_x", "achieved v [m/s]"),
             ("mean_abs_roll_deg", "mean |roll| [deg]"),
             ("abs_dev_per_m", "|dev| per m [m/m]")]

  def style(ax, T):
    ax.set_facecolor(T["surface"])
    for sp in ("top", "right"):
      ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
      ax.spines[sp].set_color(T["grid"])
    ax.grid(color=T["grid"], lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)

  for mode, T in THEME.items():
    VC = T["col"]
    plt.rcParams.update({"text.color": T["ink"], "axes.labelcolor": T["ink2"],
                         "xtick.color": T["ink2"], "ytick.color": T["ink2"],
                         "font.size": 9})

    fig, axes = plt.subplots(len(metrics), len(ts),
                             figsize=(4.3 * len(ts), 3.1 * len(metrics)),
                             squeeze=False, facecolor=T["surface"])
    for r, (col, ylab) in enumerate(metrics):
      for c, t in enumerate(ts):
        ax = axes[r][c]
        style(ax, T)
        for v in vs:
          xs, ys, es = [], [], []
          for sp in speeds:
            a, sd, _ = cell(v, t, sp, col)
            if not np.isnan(a):
              xs.append(sp); ys.append(a); es.append(0.0 if np.isnan(sd) else sd)
          if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=4, capsize=2.5, lw=1.6,
                        color=VC.get(v, "#888"), label=v)
        if col == "mean_v_body_x":
          ax.plot(speeds, speeds, ls=":", color=T["ink2"], lw=1.0)
        if r == 0:
          ax.set_title(t, fontsize=12, color=T["ink"])
        if c == 0:
          ax.set_ylabel(ylab)
        if r == len(metrics) - 1:
          ax.set_xlabel("commanded v_x [m/s]")
    axes[0][0].legend(fontsize=8, frameon=False, labelcolor=T["ink2"])
    fig.suptitle("Spine variants across task terrains (mean ± sd over seeds)",
                 fontsize=12.5, color=T["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / f"v1_absolute_{mode}.png", dpi=125, facecolor=T["surface"])
    plt.close(fig)

    others = [x for x in ts if x != "flat"]
    fig, axes = plt.subplots(1, len(others), figsize=(5.0 * len(others), 4.0),
                             squeeze=False, facecolor=T["surface"])
    for c, t in enumerate(others):
      ax = axes[0][c]
      style(ax, T)
      for v in vs:
        xs, ys = [], []
        for sp in speeds:
          a, _, stall = cell(v, t, sp, "mean_CoT_pos")
          b = base.get((v, sp))
          if not np.isnan(a) and b and not stall:
            xs.append(sp); ys.append(a / b)
        if xs:
          ax.plot(xs, ys, marker="o", ms=5, lw=1.8, color=VC.get(v, "#888"), label=v)
      ax.axhline(1.0, ls="--", color=T["ink2"], lw=1.0)
      ax.set_title(f"{t}: CoT⁺ ÷ own flat CoT⁺", fontsize=11, color=T["ink"])
      ax.set_xlabel("commanded v_x [m/s]")
      if c == 0:
        ax.set_ylabel("energy penalty vs own flat  [×]")
    axes[0][0].legend(fontsize=9, frameon=False, labelcolor=T["ink2"])
    fig.suptitle("Terrain penalty, normalised by each variant's own flat baseline "
                 "(1.0 = terrain costs nothing extra)", fontsize=12, color=T["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / f"v2_penalty_{mode}.png", dpi=125, facecolor=T["surface"])
    plt.close(fig)

  print("\n".join(L))
  print(f"\nwrote {OUT}/VARIANT_TERRAIN.md, v1_absolute_{{light,dark}}.png, "
        f"v2_penalty_{{light,dark}}.png, rollouts.csv")


if __name__ == "__main__":
  main()
