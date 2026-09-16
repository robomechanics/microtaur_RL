"""Score an open-loop spine sweep and write the chosen spine into a board gait file.

Inputs: a harmonic-mode sweep from openloop_spine_cpg.py (--h-spine-amp-deg, --h-spine-phase-deg,
--repeats), the replay of the same gait with its RL-fitted spine (the baseline), and that gait
JSON. Each sweep setting replaced the spine wave by offset + amp * sin(2 pi f t + phase) on the
gait clock, where leg1_a's first harmonic peaks at t = 0; "spine peak" below is when that sine
peaks, in % of the stride after the peak of leg1_a's first-harmonic (fundamental) wave. That is not the
peak of leg1_a's full command, which comes about 20 % of a stride after it.

A setting qualifies when no robot fell, it walks roughly straight (|heading drift| <= --max-drift-deg), the spine really swings at least --min-swing-deg
(measured joint angle, 5-95 % half range) and it walks at least --min-speed-frac of the
baseline speed. The fastest qualifier is chosen.

  python spine_sweep_report.py pitch_match/sweep_d1_k3.npz pitch_match/h_d1_k3.npz \
      pitch_match/gait_d1_k3.json pitch_match/spine_d1
"""

import argparse
import copy
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# Reference sequential blue ramp, steps 100 -> 650 (light -> dark = slow -> fast).
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]
SURFACE, INK, INK_2, MUTED, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("sweep")
  parser.add_argument("baseline")
  parser.add_argument("gait")
  parser.add_argument("out")
  parser.add_argument("--min-swing-deg", type=float, default=10.0)
  parser.add_argument("--min-speed-frac", type=float, default=0.95)
  parser.add_argument("--max-drift-deg", type=float, default=45.0,
                      help="max |heading drift| over the run; body speed alone would favour settings that walk in circles")
  args = parser.parse_args()

  sweeps = [np.load(path) for path in args.sweep.split(",")]  # comma-separated sweeps are merged
  names = [str(n) for n in sweeps[0]["param_names"]]
  params = np.concatenate([z["params"] for z in sweeps])
  amp = params[:, names.index("spine_amp_deg")]
  phase = params[:, names.index("spine_phase_deg")]
  metrics = {k[2:]: np.concatenate([z[k] for z in sweeps]) for k in sweeps[0].files if k.startswith("m_")}
  mass_kg = json.loads(str(sweeps[0]["meta"]))["mass_kg"]
  # Walking speed and cost of transport along the body axis: a curving gait is not "slower" or "less efficient".
  metrics["cot"] = metrics["power_w"] / (mass_kg * 9.81 * np.maximum(metrics["body_vx_mean_m_s"], 1e-3))
  amps, phases = np.unique(amp), np.unique(phase)
  peak = np.remainder(90.0 - phases, 360.0) / 3.6  # % of stride where the spine sine peaks
  cols = np.argsort(peak)
  phases, peak = phases[cols], peak[cols]

  def grid(metric, reduce=np.mean):
    out = np.empty((len(amps), len(phases)))
    for i, a in enumerate(amps):
      for j, p in enumerate(phases):
        out[i, j] = reduce(metrics[metric][(amp == a) & (phase == p)])
    return out

  speed, falls, swing = grid("body_vx_mean_m_s"), grid("falls", np.sum), grid("spine_amp_deg")
  drift = grid("heading_drift_deg")
  tilt, cot, spine_tau = grid("imu_tilt_max_deg"), grid("cot"), grid("spine_torque_peak_nm", np.max)

  base = np.load(args.baseline)
  ok = base["m_falls"] == 0
  base_row = {k: float(base[f"m_{k}"][ok].mean()) for k in ("body_vx_mean_m_s", "spine_amp_deg", "imu_tilt_max_deg", "heading_drift_deg")}
  base_row["cot"] = float((base["m_power_w"][ok] / (mass_kg * 9.81 * np.maximum(base["m_body_vx_mean_m_s"][ok], 1e-3))).mean())
  base_row["forward_m_s"] = base_row["body_vx_mean_m_s"]
  straight = np.abs(drift) <= args.max_drift_deg
  qualifies = (falls == 0) & straight & (swing >= args.min_swing_deg) & (speed >= args.min_speed_frac * base_row["forward_m_s"])
  chosen = None
  if qualifies.any():
    chosen = np.unravel_index(np.argmax(np.where(qualifies, speed, -np.inf)), speed.shape)

  # --- heatmap: speed per spine amplitude x timing; x = fell, outline = qualifies ---------------
  fig, ax = plt.subplots(figsize=(8.5, 3.6), facecolor=SURFACE)
  ax.set_facecolor(SURFACE)
  image = ax.imshow(speed, origin="lower", aspect="auto", cmap=LinearSegmentedColormap.from_list("blue", BLUE))
  ax.set_xticks(range(len(peak)), [f"{p:.0f}" for p in peak])
  ax.set_yticks(range(len(amps)), [f"±{a:g}" for a in amps])
  ax.set_xlabel("spine peak (% of stride after leg1_a first-harmonic peak)", fontsize=8, color=INK_2)
  ax.set_ylabel("spine amplitude (deg)", fontsize=8, color=INK_2)
  ax.tick_params(colors=MUTED, labelsize=7, length=0)
  for side in ax.spines.values():
    side.set_visible(False)
  for i, j in zip(*np.nonzero(falls > 0)):
    ax.text(j, i, "×", ha="center", va="center", fontsize=9, color=INK)
  for i, j in zip(*np.nonzero(qualifies)):
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.2))
  if chosen is not None:
    ax.annotate(f"{speed[chosen]:.3f}", chosen[::-1], ha="center", va="center", fontsize=7, color=SURFACE,
                fontweight="bold")
  bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
  bar.set_label("body speed (m/s)", fontsize=8, color=INK_2)
  bar.ax.tick_params(colors=MUTED, labelsize=7)
  bar.outline.set_visible(False)
  ax.set_title(f"Open-loop spine sweep: speed (baseline with RL-fitted spine {base_row['forward_m_s']:.3f} m/s). "
               f"× = fell, box = no falls, |drift| ≤ {args.max_drift_deg:g}°, swing ≥ {args.min_swing_deg:g}°, speed ≥ {args.min_speed_frac:.0%}",
               fontsize=8.5, color=INK, loc="left")
  fig.tight_layout()
  fig.savefig(f"{args.out}_sweep.png", dpi=150, facecolor=SURFACE)
  plt.close(fig)

  # --- table ------------------------------------------------------------------------------------
  def row(label, s, dr, sw, ti, co, fa="0", tau="-"):
    return f"| {label} | {s:.3f} | {dr:+.0f} | ±{sw:.1f} | {ti:.1f} | {co:.2f} | {fa} | {tau} |"

  lines = ["| setting | body speed (m/s) | heading drift over the run (deg) | spine swing (deg) | IMU tilt max (deg) | cost of transport | falls | spine torque peak (N m) |",
           "|---|---|---|---|---|---|---|---|",
           row("RL-fitted spine (baseline)", base_row["forward_m_s"], base_row["heading_drift_deg"], base_row["spine_amp_deg"],
               base_row["imu_tilt_max_deg"], base_row["cot"])]
  for i, a in enumerate(amps):  # best timing per amplitude, among settings with no falls that walk straight
    safe = np.where((falls[i] == 0) & straight[i], speed[i], -np.inf)
    j = int(np.argmax(safe))
    if np.isfinite(safe[j]):
      lines.append(row(f"±{a:g}°, peak at {peak[j]:.0f}% (best timing)", speed[i, j], drift[i, j], swing[i, j], tilt[i, j], cot[i, j],
                       f"{int(falls[i].sum())} of {len(phases) * int((amp == a).sum() / len(phases))} fall events at other timings" if falls[i].sum() else "0",
                       f"{spine_tau[i, j]:.3f}"))
  if chosen is not None:
    i, j = chosen
    lines.append(row(f"**chosen: ±{amps[i]:g}°, peak at {peak[j]:.0f}%**", speed[i, j], drift[i, j], swing[i, j], tilt[i, j], cot[i, j],
                     "0", f"{spine_tau[i, j]:.3f}"))
  text = "\n".join(lines) + "\n"
  with open(f"{args.out}_sweep.md", "w") as handle:
    handle.write(text)
  print(text)

  if chosen is None:
    print("[sweep] no setting qualifies; relax --min-swing-deg or --min-speed-frac")
    return
  i, j = chosen
  a_rad, p_rad = np.radians(amps[i]), np.radians(phases[j])
  gait = json.load(open(args.gait))
  out = copy.deepcopy(gait)
  s = gait["joint_order"].index("spine_joint_act")
  k = len(gait["cos_rad"][s])
  # a sin(wt + p) = a cos(p) sin(wt) + a sin(p) cos(wt) = a cos(wt + p - 90 deg)
  out["cos_rad"][s] = [float(a_rad * np.sin(p_rad))] + [0.0] * (k - 1)
  out["sin_rad"][s] = [float(a_rad * np.cos(p_rad))] + [0.0] * (k - 1)
  out["amplitude_deg"][s] = [round(float(amps[i]), 2)] + [0.0] * (k - 1)
  out["phase_deg"][s] = [round(float(phases[j] - 90.0), 1)] + [0.0] * (k - 1)
  out["spine_sweep"] = {"sweep": args.sweep, "amplitude_deg": float(amps[i]), "sine_phase_deg": float(phases[j]),
                        "peak_pct_after_leg1a_fundamental": float(peak[j]), "speed_m_s": float(speed[i, j]),
                        "spine_swing_deg": float(swing[i, j]), "imu_tilt_max_deg": float(tilt[i, j]),
                        "cot": float(cot[i, j]), "baseline_speed_m_s": base_row["forward_m_s"]}
  with open(f"{args.out}_gait.json", "w") as handle:
    json.dump(out, handle, indent=1)
  print(f"[sweep] wrote {args.out}_sweep.png, {args.out}_sweep.md, {args.out}_gait.json")


if __name__ == "__main__":
  main()
