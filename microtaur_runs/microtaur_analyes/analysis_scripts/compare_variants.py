"""Overlay CoT / tracking / attitude / spine metrics across morphology variants.

    python scripts/compare_variants.py \
        rollouts/rigid_<ts> rollouts/pitch_<ts> rollouts/yaw_<ts> rollouts/roll_<ts>

With no args it auto-picks the newest sweep dir for each of rigid/pitch/yaw/roll.
Writes rollouts/compare/<ts>/ : PNGs, compare_by_speed.csv, compare.md.
"""

from __future__ import annotations

import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ROLL = REPO / "rollouts"
ORDER = ["rigid", "pitch", "yaw", "roll"]
COLOR = {"rigid": "#33688f", "pitch": "#bd6f1c", "yaw": "#2f8a61", "roll": "#9c4da0"}


def discover():
  out = []
  for v in ORDER:
    # newest dir for this variant that is a *forward* sweep (>= 14 rollouts,
    # multiple commanded speeds) — not a single-speed yaw-command sweep.
    cand = []
    for d in sorted(glob.glob(str(ROLL / f"{v}_*"))):
      csvs = glob.glob(f"{d}/rollout_*.csv")
      speeds = {c.split("rollout_vx")[-1][:4] for c in csvs}
      if len(csvs) >= 14 and len(speeds) >= 4:
        cand.append(d)
    if cand:
      out.append(cand[-1])
  return out


def load_dir(d):
  d = Path(d)
  sj = d / "summary.json"
  if not sj.exists():
    return None
  s = pd.DataFrame(json.loads(sj.read_text()))
  metas = sorted(d.glob("rollout_*.meta.json"))
  meta = json.loads(metas[0].read_text()) if metas else {}
  s["variant"] = s.get("variant", meta.get("variant", d.name.split("_")[0]))
  frames = {p.stem: pd.read_csv(p) for p in sorted(d.glob("rollout_*.csv"))}
  return {"dir": d, "summary": s, "meta": meta, "frames": frames}


def agg_by_speed(s):
  g = s.groupby("cmd_vx")
  cols = dict(
    v_body_x=("mean_v_body_x", "mean"), track_err=("tracking_err_vx", "mean"),
    cot_pos=("mean_CoT_pos", "mean"), cot_pos_sd=("mean_CoT_pos", "std"),
    cot_abs=("mean_CoT_abs", "mean"), cot_abs_sd=("mean_CoT_abs", "std"),
    tilt=("mean_tilt_deg", "mean"), roll=("mean_abs_roll_deg", "mean"),
    pitch=("mean_abs_pitch_deg", "mean"), yaw_drift=("yaw_drift_rate_rad_s", "mean"),
    duty=("contact_duty_factor", "mean"), term=("any_terminated", "max"),
  )
  if "mean_abs_spine_deg" in s:
    cols["spine_deg"] = ("mean_abs_spine_deg", "mean")
    cols["spine_range_deg"] = ("spine_range_deg", "mean")
  return g.agg(**cols).reset_index()


def main():
  dirs = sys.argv[1:] or discover()
  if not dirs:
    sys.exit("no variant sweep dirs found; pass them explicitly")
  data = [d for d in (load_dir(x) for x in dirs) if d is not None]
  data.sort(key=lambda d: ORDER.index(d["summary"]["variant"].iloc[0])
            if d["summary"]["variant"].iloc[0] in ORDER else 99)

  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  out = ROLL / "compare" / stamp
  out.mkdir(parents=True, exist_ok=True)

  aggs = {}
  for d in data:
    v = d["summary"]["variant"].iloc[0]
    a = agg_by_speed(d["summary"])
    a.insert(0, "variant", v)
    a.insert(2, "mass_kg", round(float(d["summary"]["mass_kg"].iloc[0]), 4))
    a.insert(3, "revision", d["summary"].get("env_cfg_revision", pd.Series([d["meta"].get("env_cfg_revision")])).iloc[0])
    aggs[v] = a
  comp = pd.concat(aggs.values(), ignore_index=True)
  comp.round(4).to_csv(out / "compare_by_speed.csv", index=False)

  labels = list(aggs)
  # reference speed = the one closest to 0.12 that every variant actually ran
  common = set.intersection(*(set(a.cmd_vx.round(3)) for a in aggs.values())) or \
           set(next(iter(aggs.values())).cmd_vx.round(3))
  REF = min(common, key=lambda x: abs(x - 0.12))
  print(f"[compare] variants: {labels}  ref speed {REF}  ->  {out}")

  # ---------- fig A: CoT + tracking ---------- #
  fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
  for v, a in aggs.items():
    c = COLOR.get(v, None)
    ax[0].plot(a.cmd_vx, a.cot_pos, "o-", color=c, label=v)
    ax[0].fill_between(a.cmd_vx, a.cot_pos - a.cot_pos_sd.fillna(0), a.cot_pos + a.cot_pos_sd.fillna(0), color=c, alpha=.12)
    ax[1].plot(a.cmd_vx, a.cot_abs, "s--", color=c, label=v)
    ax[2].plot(a.cmd_vx, a.v_body_x - a.cmd_vx, "o-", color=c, label=v)
  ax[0].set_title("CoT⁺ vs speed"); ax[1].set_title("CoT_abs vs speed")
  ax[2].set_title("tracking error  (v_body_x − cmd)"); ax[2].axhline(0, color="#888", lw=.7)
  for k in (0, 1, 2):
    ax[k].set_xlabel("commanded vₓ [m/s]"); ax[k].grid(alpha=.3); ax[k].legend(frameon=False)
  ax[0].set_ylabel("CoT⁺"); ax[1].set_ylabel("CoT_abs"); ax[2].set_ylabel("Δv [m/s]")
  fig.tight_layout(); fig.savefig(out / "A_cot_tracking.png", dpi=125); plt.close(fig)

  # ---------- fig B: attitude + duty + yaw drift ---------- #
  fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
  for v, a in aggs.items():
    c = COLOR.get(v, None)
    ax[0].plot(a.cmd_vx, a.tilt, "o-", color=c, label=f"{v} tilt")
    ax[0].plot(a.cmd_vx, a.roll, "^:", color=c, alpha=.7)
    ax[1].plot(a.cmd_vx, a.duty, "o-", color=c, label=v)
    ax[2].plot(a.cmd_vx, a.yaw_drift, "o-", color=c, label=v)
  ax[0].set_title("trunk tilt (o) & |roll| (△) [deg]"); ax[0].set_ylabel("deg")
  ax[1].set_title("contact duty factor"); ax[1].set_ylabel("fraction")
  ax[2].set_title("yaw drift rate [rad/s]"); ax[2].axhline(0, color="#888", lw=.7); ax[2].set_ylabel("rad/s")
  for k in (0, 1, 2):
    ax[k].set_xlabel("commanded vₓ [m/s]"); ax[k].grid(alpha=.3); ax[k].legend(frameon=False)
  fig.tight_layout(); fig.savefig(out / "B_attitude_duty.png", dpi=125); plt.close(fig)

  # ---------- fig C: spine usage (variants that have it) ---------- #
  spine_variants = [v for v, a in aggs.items() if "spine_deg" in a]
  if spine_variants:
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
    for v in spine_variants:
      a = aggs[v]; c = COLOR.get(v, None)
      ax[0].plot(a.cmd_vx, a.spine_deg, "o-", color=c, label=v)
      ax[1].plot(a.cmd_vx, a.spine_range_deg, "o-", color=c, label=v)
    ax[0].set_title("mean |spine angle| [deg]"); ax[1].set_title("spine peak-to-peak [deg]")
    for k in (0, 1):
      ax[k].set_xlabel("commanded vₓ [m/s]"); ax[k].grid(alpha=.3); ax[k].legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "C_spine.png", dpi=125); plt.close(fig)

  # ---------- fig D: v_body_x traces, seed 0, reference speed ---------- #
  fig, ax = plt.subplots(2, 1, figsize=(9.5, 5.4), sharex=True)
  for d in data:
    v = d["summary"]["variant"].iloc[0]; c = COLOR.get(v, None)
    key = next((k for k in d["frames"] if f"vx{REF:.2f}" in k and "seed0" in k),
               next((k for k in d["frames"] if "seed0" in k), None))
    if key is None:
      continue
    df = d["frames"][key]
    ax[0].plot(df.t, df.v_body_x, lw=.8, color=c, label=v)
    ax[1].plot(df.t, np.degrees(df.tilt), lw=.8, color=c)
  ax[0].axhline(REF, color="#888", ls=":", lw=1)
  ax[0].set_ylabel(f"v_body_x [m/s]  (cmd {REF})"); ax[1].set_ylabel("tilt [deg]")
  ax[1].set_xlabel("t [s]"); ax[0].legend(frameon=False); ax[0].set_title(f"seed-0 rollout @ {REF} m/s")
  for k in (0, 1):
    ax[k].grid(alpha=.3)
  fig.tight_layout(); fig.savefig(out / "D_traces_ref.png", dpi=125); plt.close(fig)

  # ---------- headline table ---------- #
  head_rows = []
  for v, a in aggs.items():
    ref = a.iloc[(a.cmd_vx - REF).abs().argmin()]
    head_rows.append({
      "variant": v, "mass_kg": a.mass_kg.iloc[0],
      "CoT+_mean": round(a.cot_pos.mean(), 3), f"CoT+_@{REF:g}": round(float(ref.cot_pos), 3),
      "CoT_abs_mean": round(a.cot_abs.mean(), 3),
      "track_err_mean": round(a.track_err.mean(), 4),
      "tilt_mean_deg": round(a.tilt.mean(), 2),
      "duty_mean": round(a.duty.mean(), 3),
      "yaw_drift_mean": round(a.yaw_drift.mean(), 4),
      f"spine_deg_@{REF:g}": round(float(ref.spine_deg), 2) if "spine_deg" in a else None,
      "terminations": int(a.term.max()),
    })
  head = pd.DataFrame(head_rows)
  head.to_csv(out / "compare_headline.csv", index=False)

  md = [f"# Microtaur variant comparison", "",
        f"- variants: {', '.join(labels)}",
        f"- sweep dirs: " + ", ".join(str(d['dir'].name) for d in data),
        f"- reference speed for @ref columns: {REF} m/s", "",
        "## headline (mean over the 0.08–0.20 sweep unless noted)", "",
        "```", head.to_string(index=False), "```", "",
        "## by speed", "", "```", comp.round(4).to_string(index=False), "```"]
  (out / "compare.md").write_text("\n".join(md), encoding="utf-8")
  print("\n".join(md[:14]))
  print(f"\n[compare] wrote {out}/  (A_cot_tracking, B_attitude_duty"
        + (", C_spine" if spine_variants else "") + ", D_traces_ref .png; compare.md; compare_by_speed.csv)")


if __name__ == "__main__":
  main()
