"""Copy the data behind results/ into results/data/ and export every plotted curve as CSV.

  python export_results_data.py

Writes results/data/<robot>/ (simulation logs, gait files, tables; real copies with descriptive names) and
results/data/plot_curves/ (the stride curves, stride rates and sweep grid exactly as plotted).
"""

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from compare_rl_cpg import BINS, stride_curves

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "data"

# destination name <- source path (relative to cpg_rl/)
COPIES = {
  "pitch": {
    "rl_24s.npz": "pitch_match/rl_d1.npz",
    "rl_24s_40ms_delay.npz": "pitch_match/rl_d2.npz",
    "rl_64s.npz": "pitch_match/long_rl_d1.npz",
    "rl_fresh_64s.npz": "pitch_match/rl_fresh.npz",
    "old_cpg_24s.npz": "pitch_match/old_cpg.npz",
    "final_cpg_40s.npz": "pitch_match/final_40s.npz",
    "final_cpg_64s.npz": "pitch_match/long_final.npz",
    "cpg_1sine_24s.npz": "pitch_match/h_d1_k1.npz",
    "cpg_3sine_24s.npz": "pitch_match/h_d1_k3.npz",
    "board_start_24s.npz": "pitch_match/board_start.npz",
    "spine_sweep_part1.npz": "pitch_match/sweep_d1_k3.npz",
    "spine_sweep_part2.npz": "pitch_match/sweep2_d1_k3.npz",
    "spine15_trim_64s.npz": "pitch_match/long_spine15_trim.npz",
    "spine175_trim_64s.npz": "pitch_match/long_spine175_trim.npz",
    "robust_3sine_25hz.npz": "pitch_match/robust/k3_25hz.npz",
    "robust_3sine_smoothed.npz": "pitch_match/robust/k3_lp10.npz",
    "robust_4sine_25hz.npz": "pitch_match/robust/k4_25hz.npz",
    "robust_4sine_smoothed.npz": "pitch_match/robust/k4_lp10.npz",
    "robust_6sine_25hz.npz": "pitch_match/robust/k6_25hz.npz",
    "robust_6sine_smoothed.npz": "pitch_match/robust/k6_lp10.npz",
    "gait_pitch_final.json": "pitch_match/gait_pitch_final.json",
    "gait_4sine_backup.json": "pitch_match/gait_pitch_final_k4_backup.json",
    "gait_1sine.json": "pitch_match/gait_d1_k1.json",
    "gait_3sine.json": "pitch_match/gait_d1_k3.json",
    "gait_3sine_40ms_delay.json": "pitch_match/gait_d2_k3.json",
    "gait_spine15_trim.json": "pitch_match/spine15_trim_gait.json",
    "gait_spine175_trim.json": "pitch_match/spine175_trim_gait.json",
    "old_cpg_fit.json": "fit_pitch.json",
    "compare_final_table.md": "pitch_match/compare_final_table.md",
    "compare_sines_table.md": "pitch_match/compare_sines_table.md",
    "compare_spine_versions_table.md": "pitch_match/compare_spine_final_table.md",
    "spine_sweep_table.md": "pitch_match/spine_sweep_d1_sweep.md",
    "fair_scores.txt": "pitch_match/fair_scores.txt",
  },
  "yaw": {
    "rl_24s.npz": "yaw_match/rl_d1.npz",
    "rl_64s.npz": "yaw_match/long_rl_d1.npz",
    "rl_fresh_64s.npz": "yaw_match/rl_fresh.npz",
    "old_cpg_24s.npz": "yaw_match/old_cpg.npz",
    "final_cpg_64s.npz": "yaw_match/long_final.npz",
    "final_cpg_before_correction_64s.npz": "yaw_match/long_final_before_ilc.npz",
    "board_start_24s.npz": "yaw_match/board_start.npz",
    "gait_yaw_final.json": "yaw_match/gait_yaw_final.json",
    "gait_yaw_before_correction.json": "yaw_match/gait_yaw_final_before_ilc.json",
    "old_cpg_fit.json": "fit_yaw.json",
    "compare_final_table.md": "yaw_match/compare_final_table.md",
    "compare_correction_table.md": "yaw_match/compare_correction_table.md",
    "fair_scores.txt": "yaw_match/fair_scores.txt",
  },
}
for robot, fit in (("roll", "fit_roll.json"), ("rigid", "fit_rigid.json")):
  COPIES[robot] = {
    "rl_24s.npz": f"{robot}_match/rl_d1.npz",
    "rl_64s.npz": f"{robot}_match/long_rl_d1.npz",
    "rl_fresh_64s.npz": f"{robot}_match/rl_fresh.npz",
    "old_cpg_24s.npz": f"{robot}_match/old_cpg.npz",
    "final_cpg_40s.npz": f"{robot}_match/final_40s.npz",
    "final_cpg_64s.npz": f"{robot}_match/long_final.npz",
    "board_start_24s.npz": f"{robot}_match/board_start.npz",
    f"gait_{robot}_final.json": f"{robot}_match/gait_{robot}_final.json",
    "old_cpg_fit.json": fit,
    "compare_final_table.md": f"{robot}_match/compare_final_table.md",
    "fair_scores.txt": f"{robot}_match/fair_scores.txt",
  }

# plot number -> runs as plotted (label, rollout[@window]); curves are exported for every signal
PLOTS = {
  "01_02_pitch_rl_vs_final_vs_old_cpg": [("RL", "pitch_match/rl_d1.npz"), ("Old CPG", "pitch_match/old_cpg.npz"),
                                         ("Final CPG", "pitch_match/final_40s.npz@0:20")],
  "03_04_yaw_rl_vs_final_vs_old_cpg": [("RL", "yaw_match/rl_d1.npz"), ("Old CPG", "yaw_match/old_cpg.npz"),
                                       ("Final CPG", "yaw_match/long_final.npz@0:20")],
  "05_06_roll_rl_vs_final_vs_old_cpg": [("RL", "roll_match/rl_d1.npz"), ("Old CPG", "roll_match/old_cpg.npz"),
                                        ("Final CPG", "roll_match/final_40s.npz@0:20")],
  "07_08_rigid_rl_vs_final_vs_old_cpg": [("RL", "rigid_match/rl_d1.npz"), ("Old CPG", "rigid_match/old_cpg.npz"),
                                         ("Final CPG", "rigid_match/final_40s.npz@0:20")],
  "09_10_pitch_1_vs_3_vs_6_sines": [("RL", "pitch_match/rl_d1.npz"), ("1 sine", "pitch_match/h_d1_k1.npz"),
                                    ("3 sines", "pitch_match/h_d1_k3.npz"), ("6 sines", "pitch_match/final_40s.npz@0:20")],
  "13_14_pitch_spine_versions": [("RL", "pitch_match/rl_d1.npz"), ("Final CPG", "pitch_match/long_final.npz@0:20"),
                                 ("Spine 15 trim", "pitch_match/long_spine15_trim.npz@0:20"),
                                 ("Spine 17.5 trim", "pitch_match/long_spine175_trim.npz@0:20")],
  "15_16_yaw_before_vs_after_correction": [("RL", "yaw_match/rl_d1.npz"),
                                           ("Before correction", "yaw_match/long_final_before_ilc.npz@0:20"),
                                           ("After correction", "yaw_match/long_final.npz@0:20")],
}
IMU_CHANNELS = ["IMU pitch (deg)", "IMU roll (deg)", "pitch rate (deg/s)", "roll rate (deg/s)", "yaw rate (deg/s)"]


def export_curves(name: str, runs) -> None:
  header, columns = ["stride_pct"], [(np.arange(BINS) + 0.5) / BINS * 100]
  for label, spec in runs:
    run = stride_curves(str(HERE / spec.split("@")[0]) + ("@" + spec.split("@")[1] if "@" in spec else ""))
    names = run["names"]
    for i, motor in enumerate(names):
      header.append(f"{label} | {motor} angle (deg)")
      columns.append(run["curves"]["angle"][:, i])
    for i, channel in enumerate(IMU_CHANNELS):
      header.append(f"{label} | {channel}")
      columns.append(run["curves"]["imu"][:, i])
    for i, motor in enumerate(names):
      header.append(f"{label} | {motor} torque (N m)")
      columns.append(run["curves"]["torque"][:, i])
  with open(OUT / "plot_curves" / f"{name}.csv", "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(header)
    writer.writerows(np.round(np.stack(columns, 1), 5).tolist())


def export_stride_rates() -> None:
  rows = []
  for delay_ms, gait in ((20, "pitch_match/gait_d1_k3.json"), (40, "pitch_match/gait_d2_k3.json")):
    rates = json.load(open(HERE / gait))["fit"]["stride_rates_hz"]
    for group, values in rates.items():
      rows += [(delay_ms, robot, value) for robot, value in enumerate(values)]
  with open(OUT / "plot_curves" / "11_pitch_rl_stride_rate_by_motor_delay.csv", "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["motor_delay_ms", "robot_index", "stride_rate_hz"])
    writer.writerows(rows)


def export_spine_sweep() -> None:
  sweeps = [np.load(HERE / p) for p in ("pitch_match/sweep_d1_k3.npz", "pitch_match/sweep2_d1_k3.npz")]
  names = [str(n) for n in sweeps[0]["param_names"]]
  params = np.concatenate([z["params"] for z in sweeps])
  metrics = {k: np.concatenate([z[k] for z in sweeps]) for k in sweeps[0].files if k.startswith("m_")}
  mass = json.loads(str(sweeps[0]["meta"]))["mass_kg"]
  amp, phase = params[:, names.index("spine_amp_deg")], params[:, names.index("spine_phase_deg")]
  with open(OUT / "plot_curves" / "12_pitch_spine_sweep_grid.csv", "w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["spine_amp_cmd_deg", "spine_sine_phase_deg", "spine_peak_pct_after_leg1a_fundamental", "robots",
                     "falls", "body_speed_m_s", "heading_drift_deg_per_20s", "spine_swing_real_deg", "imu_tilt_max_deg",
                     "cost_of_transport"])
    for a in np.unique(amp):
      for p in np.unique(phase):
        sel = (amp == a) & (phase == p)
        vx = metrics["m_body_vx_mean_m_s"][sel]
        writer.writerow([a, p, round(float(np.remainder(90.0 - p, 360.0) / 3.6), 1), int(sel.sum()),
                         int(metrics["m_falls"][sel].sum()), round(float(vx.mean()), 4),
                         round(float(metrics["m_heading_drift_deg"][sel].mean()), 1),
                         round(float(metrics["m_spine_amp_deg"][sel].mean()), 2),
                         round(float(metrics["m_imu_tilt_max_deg"][sel].mean()), 2),
                         round(float((metrics["m_power_w"][sel] / (mass * 9.81 * np.maximum(vx, 1e-3))).mean()), 3)])


def main() -> None:
  (OUT / "plot_curves").mkdir(parents=True, exist_ok=True)
  total = 0
  for robot, files in COPIES.items():
    (OUT / robot).mkdir(parents=True, exist_ok=True)
    for dest, src in files.items():
      shutil.copy2(HERE / src, OUT / robot / dest)
      total += (OUT / robot / dest).stat().st_size
  print(f"[export] copied {sum(len(f) for f in COPIES.values())} files, {total / 1e6:.0f} MB")
  for name, runs in PLOTS.items():
    export_curves(name, runs)
    print(f"[export] plot_curves/{name}.csv")
  export_stride_rates()
  export_spine_sweep()
  print("[export] plot_curves/11_pitch_rl_stride_rate_by_motor_delay.csv, 12_pitch_spine_sweep_grid.csv")


if __name__ == "__main__":
  main()
