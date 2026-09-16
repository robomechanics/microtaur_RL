"""Overlay RL and CPG rollouts stride by stride: 9 motor angles and torques, and the IMU.

Every rollout comes from openloop_spine_cpg.py --series. Each robot's time axis becomes stride
phase, using its own stride rate and the phase of its leg1_a joint angle (0 % = leg1_a at its
peak), so RL and CPG are compared per stride even if their clocks differ. Robots that fell are
left out; the rest are averaged. Only robot-measurable signals are plotted.

  python compare_rl_cpg.py pitch_match/compare "RL=pitch_match/rl_d1.npz" "Matched CPG=pitch_match/h_k3.npz"

Writes <out>_angles_imu.png, <out>_torque.png and <out>_table.md. The first rollout is the
reference the others are scored against (R^2 of the stride curves: 1 = identical).
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fit_harmonic_gait import design, stride_rate  # noqa: E402

BINS = 48
# Reference categorical palette, slots in fixed order (light chart surface).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def split_window(spec: str) -> tuple[str, tuple[float, float] | None]:
  """"path@t0:t1" -> (path, (t0, t1)) in seconds of the logged series (after settle); plain path -> whole run."""
  if "@" not in spec:
    return spec, None
  path, window = spec.rsplit("@", 1)
  t0, t1 = (float(x) for x in window.split(":"))
  return path, (t0, t1)


def stride_curves(spec: str) -> dict:
  path, window = split_window(spec)
  z = np.load(path)
  dt = float(z["dt"])
  part = slice(None) if window is None else slice(int(round(window[0] / dt)), int(round(window[1] / dt)))
  q = z["s_q"][part].astype(float)
  t = np.arange(q.shape[0]) * dt
  ok = np.flatnonzero(z["m_falls"] == 0)
  grav, gyro = z["s_imu_gravity"][part].astype(float), z["s_imu_gyro"][part].astype(float)
  signals = {
    "angle": np.degrees(q),
    "torque": z["s_tau"][part].astype(float),
    "imu": np.stack([
      np.degrees(np.arctan2(grav[..., 0], -grav[..., 2])),  # IMU pitch
      np.degrees(np.arctan2(-grav[..., 1], -grav[..., 2])),  # IMU roll (+ about body +x)
      np.degrees(gyro[..., 1]),  # pitch rate
      np.degrees(gyro[..., 0]),  # roll rate
      np.degrees(gyro[..., 2]),  # yaw rate
    ], -1),
  }
  curves = {k: [] for k in signals}
  periodic = {"legs": [], "imu": [], "torque": []}  # share of each raw signal that is the repeating stride
  rates = []
  for r in ok:
    f = stride_rate(q[:, r, :8], t)
    # Stride shape = all harmonics below Nyquist, fitted over the whole run. Phase binning left
    # empty bins when a stride is a whole number of half-samples (4.00 Hz = 12.5 samples).
    k = int((0.5 / dt - 0.3) / f)
    x = design(t, f, k)
    lead, *_ = np.linalg.lstsq(design(t, f, 1), q[:, r, 0], rcond=None)
    peak = np.arctan2(lead[2], lead[1])  # leg1_a's fundamental peaks at 2 pi f t = peak
    t_stride = (2 * np.pi * (np.arange(BINS) + 0.5) / BINS + peak) / (2 * np.pi * f)
    for key, s in signals.items():
      coef, *_ = np.linalg.lstsq(x, s[:, r, :], rcond=None)
      curves[key].append(design(t_stride, f, k) @ coef)
      raw = s[:, r, :]
      share = 1 - ((raw - x @ coef) ** 2).sum(0) / np.maximum(((raw - raw.mean(0)) ** 2).sum(0), 1e-12)
      if key == "angle":
        periodic["legs"].append(np.median(share[:8]))
      elif key == "imu":
        periodic["imu"].append(np.median(share[:2]))
      else:
        periodic["torque"].append(np.median(share[:8]))
    rates.append(f)
  m = {k[2:]: z[k] for k in z.files if k.startswith("m_")}
  meta = json.loads(str(z["meta"]))
  body_vx = z["s_truth_vel_b"][part][..., 0].astype(float).mean(0)
  # Cost of transport per metre walked along the body axis (recomputed here; older logs divided by start-heading progress).
  power = np.clip(z["s_tau"][part].astype(float) * z["s_qd"][part].astype(float), 0.0, None).sum(-1).mean(0)
  cot = power / (float(meta["mass_kg"]) * 9.81 * np.maximum(body_vx, 1e-3))
  table = {
    "robots": f"{len(ok)}/{q.shape[1]}",
    "stride rate (Hz)": f"{np.median(rates):.2f}" + (f" ({min(rates):.2f}-{max(rates):.2f})" if max(rates) - min(rates) > 0.05 else ""),
    "body speed (m/s)": f"{body_vx[ok].mean():.3f}",
    "speed along start direction (m/s)": f"{m['forward_m_s'][ok].mean():.3f}",
    "falls": f"{int((m['falls'] > 0).sum())}",
    "IMU tilt max (deg)": f"{m['imu_tilt_max_deg'][ok].mean():.1f}",
    "cost of transport": f"{cot[ok].mean():.2f}",
    "leg commands clipped at stand ±30° (share / max)": (
      f"{100 * m['leg_clip_share'][ok].mean():.0f}% / {m['leg_clip_max_deg'][ok].max():.1f}°" if "leg_clip_share" in m else "-"),
    "spine swing (deg)": f"±{m['spine_amp_deg'][ok].mean():.1f}" if "spine_amp_deg" in m else "-",
    "peak leg target speed (rad/s)": f"{m['leg_target_speed_peak_rad_s'][ok].mean():.0f}",
    "safety filter correction mean / max (deg)": (
      f"{m['leg_filter_correction_deg'][ok].mean():.2f} / {m['leg_filter_correction_max_deg'][ok].max():.1f}"
      if "leg_filter_correction_deg" in m else "-"),
  }
  names = [str(n).replace("_joint_act", "") for n in z["motor_names"]]
  return {"curves": {k: np.mean(v, 0) for k, v in curves.items()}, "table": table, "names": names, "meta": meta,
          "periodic": {k: float(np.median(v)) for k, v in periodic.items()}, "body_vx": float(body_vx[ok].mean())}


def r2_curve(ref: np.ndarray, other: np.ndarray) -> np.ndarray:
  return 1 - ((other - ref) ** 2).sum(0) / np.maximum(((ref - ref.mean(0)) ** 2).sum(0), 1e-12)


def style(ax, title):
  ax.set_facecolor(SURFACE)
  ax.set_title(title, fontsize=8.5, color=INK, loc="left")
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color(AXIS)
    ax.spines[side].set_linewidth(0.8)
  ax.tick_params(colors=MUTED, labelsize=7, length=2)
  ax.grid(True, color=GRID, linewidth=0.6)
  ax.set_axisbelow(True)
  ax.set_xlim(0, 100)
  ax.set_xticks([0, 25, 50, 75, 100])


def panel_figure(runs, key, panels, ylabel, path, suptitle):
  n = len(panels)
  cols = 3
  rows = int(np.ceil(n / cols))
  fig, axes = plt.subplots(rows, cols, figsize=(10, 2.1 * rows + 0.6), facecolor=SURFACE, squeeze=False)
  x = (np.arange(BINS) + 0.5) / BINS * 100
  for i, (title, idx, unit) in enumerate(panels):
    ax = axes.flat[i]
    style(ax, title)
    for s, (label, run) in enumerate(runs):
      y = run["curves"][key][:, idx]
      ax.plot(np.append(x, x[0] + 100), np.append(y, y[0]), color=SERIES[s], lw=1.6,
              solid_capstyle="round", solid_joinstyle="round", label=label)
    ax.set_ylabel(unit, fontsize=7, color=INK_2)
    if i >= n - cols:
      ax.set_xlabel("stride (%)", fontsize=7, color=INK_2)
  for ax in list(axes.flat)[n:]:
    ax.set_visible(False)
  handles, labels = axes.flat[0].get_legend_handles_labels()
  fig.suptitle(suptitle, x=0.01, y=0.995, ha="left", fontsize=10, color=INK)
  fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 1.0), ncol=len(runs), frameon=False,
             fontsize=8, labelcolor=INK)
  fig.tight_layout(rect=(0, 0, 1, 0.975))
  fig.savefig(path, dpi=150, facecolor=SURFACE)
  plt.close(fig)


def main() -> None:
  out = sys.argv[1]
  runs = []
  for arg in sys.argv[2:]:
    label, path = arg.split("=", 1)
    runs.append((label, stride_curves(path)))
  names = runs[0][1]["names"]
  motors = len(names)

  angle_panels = [(n, i, "deg") for i, n in enumerate(names)]
  angle_panels += [("IMU pitch", 0, "deg"), ("IMU roll", 1, "deg"), ("IMU pitch rate", 2, "deg/s")]
  # IMU channels live in their own array; plot them through a merged view.
  for _, run in runs:
    run["curves"]["angle_imu"] = np.concatenate([run["curves"]["angle"], run["curves"]["imu"]], 1)
  angle_panels = [(t, i if i < motors and u == "deg" and t in names else motors + i, u)
                  for t, i, u in angle_panels]
  title = f"{runs[0][1]['meta']['variant']}: joint angles and IMU over one stride (robots averaged)"
  panel_figure(runs, "angle_imu", angle_panels, "deg", f"{out}_angles_imu.png", title)
  panel_figure(runs, "torque", [(n, i, "N m") for i, n in enumerate(names)], "N m", f"{out}_torque.png",
               f"{runs[0][1]['meta']['variant']}: motor torque over one stride (robots averaged)")

  ref = runs[0][1]["curves"]
  lines = ["| | " + " | ".join(label for label, _ in runs) + " |", "|---" * (len(runs) + 1) + "|"]
  for row in runs[0][1]["table"]:
    lines.append(f"| {row} | " + " | ".join(run["table"][row] for _, run in runs) + " |")
  scores = {
    f"match to {runs[0][0]}: leg angles (R²)": lambda c: r2_curve(ref["angle"][:, :8], c["angle"][:, :8]).mean(),
    f"match to {runs[0][0]}: spine angle (R²)": lambda c: r2_curve(ref["angle"][:, 8:], c["angle"][:, 8:]).mean(),
    f"match to {runs[0][0]}: IMU pitch + roll (R²)": lambda c: r2_curve(ref["imu"][:, :2], c["imu"][:, :2]).mean(),
    f"match to {runs[0][0]}: gyro pitch/roll/yaw rate (R²)": lambda c: r2_curve(ref["imu"][:, 2:5], c["imu"][:, 2:5]).mean(),
    f"match to {runs[0][0]}: leg torque (R²)": lambda c: r2_curve(ref["torque"][:, :8], c["torque"][:, :8]).mean(),
  }
  for row, fn in scores.items():
    if "spine" in row and motors < 9:
      continue
    lines.append(f"| {row} | " + " | ".join(f"{fn(run['curves']):.2f}" for _, run in runs) + " |")
  per_motor = [f"| {n} angle (R²) | " + " | ".join(
    f"{r2_curve(ref['angle'][:, i:i + 1], run['curves']['angle'][:, i:i + 1])[0]:.2f}" for _, run in runs) + " |"
    for i, n in enumerate(names)]
  text = "\n".join(lines + per_motor) + "\n"
  with open(f"{out}_table.md", "w") as handle:
    handle.write(text)
  print(text)
  print(f"[compare] wrote {out}_angles_imu.png, {out}_torque.png, {out}_table.md")


if __name__ == "__main__":
  main()
