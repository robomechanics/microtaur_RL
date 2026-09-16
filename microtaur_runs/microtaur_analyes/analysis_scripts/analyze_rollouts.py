"""Consolidate rollout sweeps into a report + plots and run the verification checks.

    python scripts/analyze_rollouts.py rollouts/sweep_fwd/<ts> [rollouts/sweep_yaw/<ts> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TB = REPO.parent / "rigid_flat_fixed_m077" / "events.out.tfevents.1788712412.ARINA.4800.0"


def tb_finals():
  try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  except Exception:  # noqa: BLE001
    return {}
  ea = EventAccumulator(str(TB), size_guidance={"scalars": 0})
  ea.Reload()
  out = {}
  for tag in ("Metrics/microtaur/forward_velocity_m_s", "Metrics/microtaur/tilt_deg",
              "Metrics/microtaur/root_height_m"):
    if tag in ea.Tags()["scalars"]:
      s = ea.Scalars(tag)
      out[tag] = float(np.mean([x.value for x in s[-20:]]))
  return out


def load(dirs):
  frames, summaries = {}, []
  for d in dirs:
    d = Path(d)
    for c in sorted(d.glob("rollout_*.csv")):
      frames[f"{d.name}/{c.stem}"] = pd.read_csv(c)
    sj = d / "summary.json"
    if sj.exists():
      summaries.extend(json.loads(sj.read_text()))
  return frames, pd.DataFrame(summaries)


def main():
  dirs = sys.argv[1:] or [str(sorted((REPO / "rollouts").glob("sweep_fwd/*"))[-1])]
  frames, sdf = load(dirs)
  outdir = Path(dirs[0])
  print(f"[analyze] {len(frames)} rollouts from {len(dirs)} dir(s) -> {outdir}")

  # ---- aggregate CoT / tracking vs speed ----
  g = sdf.groupby("cmd_vx")
  agg = g.agg(
    v_body_x=("mean_v_body_x", "mean"),
    v_body_x_sd=("mean_v_body_x", "std"),
    track_err=("tracking_err_vx", "mean"),
    cot_pos=("mean_CoT_pos", "mean"), cot_pos_sd=("mean_CoT_pos", "std"),
    cot_abs=("mean_CoT_abs", "mean"), cot_abs_sd=("mean_CoT_abs", "std"),
    cot_pos_p95=("p95_CoT_pos", "mean"),
    tilt=("mean_tilt_deg", "mean"),
    roll=("mean_abs_roll_deg", "mean"), pitch=("mean_abs_pitch_deg", "mean"),
    yaw_drift=("yaw_drift_rate_rad_s", "mean"),
    duty=("contact_duty_factor", "mean"),
    term=("any_terminated", "max"),
  ).reset_index()
  agg.to_csv(outdir / "aggregate_by_speed.csv", index=False)
  print("\n", agg.to_string(index=False))

  fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
  ax[0].errorbar(agg.cmd_vx, agg.cot_pos, yerr=agg.cot_pos_sd, fmt="o-", capsize=3, label="CoT+  (Σmax(τq̇,0))")
  ax[0].errorbar(agg.cmd_vx, agg.cot_abs, yerr=agg.cot_abs_sd, fmt="s--", capsize=3, label="CoT_abs (Σ|τq̇|)")
  ax[0].plot(agg.cmd_vx, agg.cot_pos_p95, "^:", color="gray", label="CoT+ p95")
  ax[0].set_xlabel("commanded vₓ [m/s]"); ax[0].set_ylabel("cost of transport"); ax[0].legend(); ax[0].grid(alpha=.3)
  ax[0].set_title("Mechanical CoT vs speed")

  ax[1].plot([agg.cmd_vx.min(), agg.cmd_vx.max()], [agg.cmd_vx.min(), agg.cmd_vx.max()], "k:", label="ideal")
  ax[1].errorbar(agg.cmd_vx, agg.v_body_x, yerr=agg.v_body_x_sd, fmt="o-", capsize=3, label="achieved v_body_x")
  ax[1].set_xlabel("commanded vₓ [m/s]"); ax[1].set_ylabel("mean v_body_x [m/s]"); ax[1].legend(); ax[1].grid(alpha=.3)
  ax[1].set_title("Forward-velocity tracking")

  ax[2].plot(agg.cmd_vx, agg.roll, "o-", label="|roll|")
  ax[2].plot(agg.cmd_vx, agg.pitch, "s-", label="|pitch|")
  ax[2].plot(agg.cmd_vx, agg.tilt, "^-", label="tilt (proj-g)")
  ax[2].set_xlabel("commanded vₓ [m/s]"); ax[2].set_ylabel("deg"); ax[2].legend(); ax[2].grid(alpha=.3)
  ax[2].set_title("Trunk attitude vs speed")
  fig.tight_layout(); fig.savefig(outdir / "summary_by_speed.png", dpi=130); plt.close(fig)

  # ---- attitude / velocity traces (seed 0, all speeds) ----
  fig, ax = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
  for key, df in sorted(frames.items()):
    if "seed0" not in key:
      continue
    lbl = key.split("/")[-1].replace("rollout_", "")
    ax[0].plot(df.t, np.degrees(df.roll), lw=.7, label=lbl)
    ax[1].plot(df.t, np.degrees(df.pitch), lw=.7)
    ax[2].plot(df.t, np.degrees(np.unwrap(df.yaw)), lw=.7)
    ax[3].plot(df.t, df.v_body_x, lw=.7)
  ax[0].set_ylabel("roll [deg]"); ax[1].set_ylabel("pitch [deg]")
  ax[2].set_ylabel("yaw (unwrap) [deg]"); ax[3].set_ylabel("v_body_x [m/s]")
  ax[3].set_xlabel("t [s]"); ax[0].legend(fontsize=6, ncol=4)
  fig.tight_layout(); fig.savefig(outdir / "traces_seed0.png", dpi=130); plt.close(fig)

  # ---- verification checklist ----
  allrows = pd.concat(frames.values(), ignore_index=True)
  tb = tb_finals()
  checks = []
  checks.append(("CoT_pos <= CoT_abs (all rows)", bool((allrows.CoT_pos <= allrows.CoT_abs + 1e-9).all())))
  checks.append(("P_abs >= 0 (all rows)", bool((allrows.P_abs >= 0).all())))
  fin = np.isfinite(allrows[["CoT_pos", "CoT_abs", "CoT_net"]].to_numpy())
  checks.append(("CoT finite where v>=v_eps",
                 bool(fin[allrows.v_below_eps.to_numpy() == 0].all())))
  checks.append(("no terminations in any rollout", bool(sdf.any_terminated.max() == 0)))
  checks.append(("tracking err < FORWARD_SIGMA (0.04) at every speed",
                 bool((agg.track_err < 0.04).all())))
  ss = allrows[(allrows.t >= 1.0) & (allrows.done == 0)]
  mv = float(ss.v_body_x.mean())
  mt = float(ss.tilt_deg.mean())
  variant = str(sdf["variant"].iloc[0]) if "variant" in sdf and sdf["variant"].notna().any() else "rigid"
  # The ARINA TensorBoard log is the rigid training run only.
  if tb and variant == "rigid":
    checks.append((f"rollout mean v_body_x {mv:.3f} vs TB {tb.get('Metrics/microtaur/forward_velocity_m_s', float('nan')):.3f} (<0.03)",
                   abs(mv - tb.get("Metrics/microtaur/forward_velocity_m_s", 0)) < 0.03))
    checks.append((f"rollout mean tilt {mt:.2f}deg vs TB {tb.get('Metrics/microtaur/tilt_deg', float('nan')):.2f}deg (<1.0)",
                   abs(mt - tb.get("Metrics/microtaur/tilt_deg", 0)) < 1.0))

  lines = [f"# Microtaur forward-rollout report ({variant})", "",
           f"- rollouts: {len(frames)}  |  dirs: {', '.join(dirs)}",
           f"- steady-state window: t >= 1.0 s, done == 0",
           f"- mass = {sdf.mass_kg.iloc[0]:.4f} kg, g = 9.81, policy_dt = {sdf.policy_dt.iloc[0]} s", ""]
  if tb and variant == "rigid":
    lines += ["## training TensorBoard finals (last-20 mean @ step 4499)"]
    lines += [f"- `{k}` = {v:.4f}" for k, v in tb.items()] + [""]
  lines += ["## verification checklist"]
  for name, ok in checks:
    lines.append(f"- [{'x' if ok else ' '}] {name}")
  a = agg.round(4)
  lines += ["", "## aggregate by commanded speed", "", "```", a.to_string(index=False), "```"]
  (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
  print("\n".join(lines[-len(checks) - 4:]))
  print(f"\n[analyze] wrote {outdir}/report.md, summary_by_speed.png, traces_seed0.png, aggregate_by_speed.csv")


if __name__ == "__main__":
  main()
