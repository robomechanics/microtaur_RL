"""Plot traces written by isaac_render_policy.py --trace (one figure, one row per trace).

  python scripts/plot_turn_trace.py --traces a.npz b.npz --labels "wz 0.25" "wz 0.50" --out fig.png [--skip 5]

Per trace: the path (x, y), the spine angle, the measured yaw rate against the command (1 s moving average and
raw), and the foot contact pattern of the last 2 s. Printed per trace, after --skip seconds: mean yaw rate and
forward speed, turn radius = speed / yaw rate, spine angle mean / std, and the spine angle's oscillation at the
stride frequency (amplitude of the strongest spectral line above 1 Hz).
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--traces", nargs="+", required=True)
ap.add_argument("--labels", nargs="+", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--skip", type=float, default=5.0, help="seconds left out of the statistics (start-up)")
args = ap.parse_args()

fig, axs = plt.subplots(len(args.traces), 4, figsize=(20, 4.4 * len(args.traces)), squeeze=False)
for row, (path, label) in enumerate(zip(args.traces, args.labels)):
  d = np.load(path)
  t, dt = d["t"], float(d["t"][1] - d["t"][0])
  m = t >= args.skip
  wz, vx, sp = d["wz"][m].mean(), d["vx"][m].mean(), np.degrees(d["spine"][m])
  spec = np.abs(np.fft.rfft(sp - sp.mean())) * 2 / sp.size
  f = np.fft.rfftfreq(sp.size, dt)
  k = np.argmax(np.where(f > 1.0, spec, 0))
  print(f"{label}: cmd vx {d['cmd_vx'][m].mean():.2f} wz {d['cmd_wz'][m].mean():+.2f} | measured yaw rate {wz:+.3f} rad/s "
        f"({wz / d['cmd_wz'][m].mean():.2f} of cmd), vx {vx:.3f} m/s, radius {vx / wz if abs(wz) > 1e-3 else float('inf'):.2f} m "
        f"(cmd {d['cmd_vx'][m].mean() / d['cmd_wz'][m].mean():.2f} m) | spine {sp.mean():+.2f} +- {sp.std():.2f} deg, "
        f"strongest oscillation {spec[k]:.2f} deg at {f[k]:.1f} Hz")
  a = axs[row]
  a[0].plot(d["x"], d["y"], lw=1)
  a[0].plot(d["x"][0], d["y"][0], "go")
  a[0].set_aspect("equal")
  a[0].set_title(f"{label}: path (start = green), radius {vx / wz:.2f} m")
  a[0].set_xlabel("x m")
  a[0].set_ylabel("y m")
  a[1].plot(t, np.degrees(d["spine"]), lw=0.8)
  a[1].axhline(0, color="k", lw=0.5)
  a[1].set_title(f"spine angle (deg): {sp.mean():+.1f} +- {sp.std():.1f}; joint range +-20")
  a[1].set_xlabel("t s")
  w = max(1, round(1.0 / dt))
  a[2].plot(t, d["wz"], lw=0.4, alpha=0.4, label="yaw rate, per step")
  a[2].plot(t, np.convolve(d["wz"], np.ones(w) / w, mode="same"), lw=1.5, label="yaw rate, 1 s mean")
  a[2].plot(t, d["cmd_wz"], "k--", lw=1, label="command")
  a[2].set_title("yaw rate (rad/s)")
  a[2].legend(fontsize=8)
  a[2].set_xlabel("t s")
  c = d["contact"][-round(2.0 / dt):]
  a[3].imshow(c.T, aspect="auto", cmap="Greys", interpolation="nearest", extent=(t[-len(c)], t[-1], 3.5, -0.5))
  a[3].set_yticks(range(4), ["RR (1)", "RL (2)", "FL (3)", "FR (4)"])
  a[3].set_title("foot contact, last 2 s (black = stance)")
  a[3].set_xlabel("t s")
fig.tight_layout()
fig.savefig(args.out, dpi=110)
print("PLOT_OK", args.out)
