"""Render the figures used in the rollout report (compact, for embedding)."""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_fwd_default = sorted(glob.glob("rollouts/sweep_fwd/*")) or sorted(glob.glob("rollouts/rigid_*"))
FWD = sys.argv[1] if len(sys.argv) > 1 else _fwd_default[-1]
_yaw_default = sorted(glob.glob("rollouts/sweep_yaw/*"))
YAW = sys.argv[2] if len(sys.argv) > 2 else (_yaw_default[-1] if _yaw_default else None)
OUT = Path("rollouts/report_figs")
OUT.mkdir(parents=True, exist_ok=True)

INK, MUT, LINE = "#16212a", "#5a6b78", "#c9d2d9"
SIGNAL, PROBE, OK = "#d9822b", "#3a7ca5", "#2f9e6e"
plt.rcParams.update({
  "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUT,
  "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
  "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.6, "figure.facecolor": "white",
  "axes.facecolor": "white", "savefig.facecolor": "white",
})


def load(d):
  return {Path(f).stem.replace("rollout_", ""): pd.read_csv(f)
          for f in sorted(glob.glob(f"{d}/rollout_*.csv"))}


fwd = load(FWD)
yaw = load(YAW) if YAW else {}
agg = pd.read_csv(f"{FWD}/aggregate_by_speed.csv")

_metas = sorted(Path(FWD).glob("rollout_*.meta.json"))
META = json.loads(_metas[0].read_text()) if _metas else {}
JOINTS = META.get("joint_order", ["leg1_a", "leg1_e", "leg2_a", "leg2_e",
                                  "leg3_a", "leg3_e", "leg4_a", "leg4_e"])
VARIANT = META.get("variant", "rigid")


# ---- fig 1: CoT + tracking vs speed --------------------------------------- #
fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.3))
ax[0].fill_between(agg.cmd_vx, agg.cot_pos - agg.cot_pos_sd, agg.cot_pos + agg.cot_pos_sd, color=SIGNAL, alpha=.13)
ax[0].plot(agg.cmd_vx, agg.cot_pos, "o-", color=SIGNAL, lw=1.8, label=r"CoT$^+$  $\Sigma\max(\tau\dot q,0)$")
ax[0].plot(agg.cmd_vx, agg.cot_abs, "s--", color=INK, lw=1.4, label=r"CoT$_{abs}$  $\Sigma|\tau\dot q|$")
ax[0].plot(agg.cmd_vx, agg.cot_pos_p95, "^:", color=MUT, lw=1, label=r"CoT$^+$ p95")
ax[0].set_xlabel("commanded $v_x$  [m/s]"); ax[0].set_ylabel("mechanical cost of transport")
ax[0].set_ylim(0, None); ax[0].legend(frameon=False, fontsize=7.5)

ax[1].plot([0.07, 0.21], [0.07, 0.21], ":", color=MUT, lw=1, label="ideal $v=v_{cmd}$")
ax[1].fill_between(agg.cmd_vx, agg.v_body_x - agg.v_body_x_sd, agg.v_body_x + agg.v_body_x_sd, color=PROBE, alpha=.15)
ax[1].plot(agg.cmd_vx, agg.v_body_x, "o-", color=PROBE, lw=1.8, label="achieved $v_{body,x}$")
ax[1].set_xlabel("commanded $v_x$  [m/s]"); ax[1].set_ylabel("mean body-frame $v_x$  [m/s]")
ax[1].legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(OUT / "fig1_cot_tracking.png", dpi=110); plt.close(fig)


# ---- fig 2: attitude + velocity traces (seed 0) ------------------------- #
fig, ax = plt.subplots(3, 1, figsize=(8.4, 5.2), sharex=True)
speeds = sorted({k for k in fwd if "seed0" in k})
cmap = plt.cm.viridis(np.linspace(0, .85, len(speeds)))
for c, key in zip(cmap, speeds):
  d = fwd[key]; v = key.split("_")[0].replace("vx", "")
  ax[0].plot(d.t, np.degrees(d.roll), lw=.7, color=c, label=v)
  ax[1].plot(d.t, np.degrees(d.pitch), lw=.7, color=c)
  ax[2].plot(d.t, d.v_body_x, lw=.7, color=c)
ax[0].set_ylabel("roll  [deg]"); ax[1].set_ylabel("pitch  [deg]"); ax[2].set_ylabel("$v_{body,x}$  [m/s]")
ax[2].set_xlabel("t  [s]")
ax[0].legend(frameon=False, ncol=7, fontsize=7, title="commanded $v_x$ [m/s]", title_fontsize=7,
             loc="lower center", bbox_to_anchor=(.5, 1.02))
fig.tight_layout(); fig.savefig(OUT / "fig2_traces.png", dpi=110); plt.close(fig)


# ---- fig 3: per-joint torque & power, one rollout ---------------------- #
_k3 = next((k for k in fwd if "vx0.14" in k and "seed0" in k),
           next((k for k in sorted(fwd) if "seed0" in k), sorted(fwd)[0]))
d = fwd[_k3]
_t1 = float(d.t.min()) + 3.0
w = d[(d.t >= _t1) & (d.t <= _t1 + 4.0)]
joints = [j for j in JOINTS if f"tau_{j}" in w.columns]
fig, ax = plt.subplots(2, 1, figsize=(8.4, 4.2), sharex=True)
cj = plt.cm.tab10(np.linspace(0, 1, 10))
for i, j in enumerate(joints):
  style = dict(lw=1.4, color="#111") if j == "spine" else dict(lw=.8, color=cj[i % 10])
  ax[0].plot(w.t, w[f"tau_{j}"] * 1000, label=j, **style)
  ax[1].plot(w.t, w[f"P_{j}"], **style)
ax[0].axhline(129, color=MUT, ls=":", lw=.8); ax[0].axhline(-129, color=MUT, ls=":", lw=.8)
ax[0].set_ylabel(r"$\tau$  [mN$\cdot$m]"); ax[1].set_ylabel(r"$\tau\dot q$  [W]")
ax[1].axhline(0, color=MUT, lw=.6); ax[1].set_xlabel("t  [s]")
ax[0].legend(frameon=False, ncol=8, fontsize=6.5, loc="lower center", bbox_to_anchor=(.5, 1.02))
ax[0].text(w.t.iloc[0], 133, "XL330 effort_limit 0.129 N·m", fontsize=6.5, color=MUT)
fig.tight_layout(); fig.savefig(OUT / "fig3_torque_power.png", dpi=110); plt.close(fig)


# ---- fig 4: yaw-command rollouts (skipped if no yaw sweep) ----------- #
def _fig4_yaw():
  fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2))
  for key, d in sorted(yaw.items()):
    if "seed0" not in key:
      continue
    cy = float(d.cmd_yaw.iloc[0]); col = SIGNAL if cy > 0 else PROBE
    ax[0].plot(d.t, d.wz, lw=.8, color=col)
    ax[0].axhline(cy, color=col, ls=":", lw=1)
    ax[1].plot(d.t, np.degrees(np.unwrap(d.yaw)), lw=1.2, color=col, label=f"cmd yaw {cy:+.1f} rad/s")
  ax[0].set_xlabel("t  [s]"); ax[0].set_ylabel(r"$\omega_z$  [rad/s]"); ax[0].set_title("yaw-rate tracking", fontsize=9)
  ax[1].set_xlabel("t  [s]"); ax[1].set_ylabel("heading (unwrapped)  [deg]")
  ax[1].legend(frameon=False, fontsize=7.5); ax[1].set_title("heading drift", fontsize=9)
  fig.tight_layout(); fig.savefig(OUT / "fig4_yaw.png", dpi=110); plt.close(fig)


if yaw:
  _fig4_yaw()

print("wrote", *(p.name for p in sorted(OUT.glob("*.png"))))
for p in sorted(OUT.glob("*.png")):
  print(f"  {p}  {p.stat().st_size/1024:.0f} KB")
