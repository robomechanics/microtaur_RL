"""Report-quality figures for the 4-variant comparison (for the HTML artifact)."""

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

OUT = Path("rollouts/compare_figs")
OUT.mkdir(parents=True, exist_ok=True)

INK, MUT, LINE = "#16212a", "#5a6b78", "#d0d7dd"
COL = {"rigid": "#33688f", "pitch": "#c9761f", "yaw": "#2f8a61", "roll": "#8f4bbf"}
plt.rcParams.update({
  "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUT,
  "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUT, "ytick.color": MUT,
  "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.6,
  "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})

# forward-sweep dirs (>=14 rollouts, multi-speed) — newest per variant
FWD = {}
for v in ("rigid", "pitch", "yaw", "roll"):
  for d in sorted(glob.glob(f"rollouts/{v}_*")):
    cs = glob.glob(f"{d}/rollout_*.csv")
    if len(cs) >= 14 and len({c.split("rollout_vx")[-1][:4] for c in cs}) >= 4:
      FWD[v] = d
YAWD = {}
for v in ("rigid", "pitch", "yaw", "roll"):
  for d in sorted(glob.glob(f"rollouts/{v}_*")):
    cs = glob.glob(f"{d}/rollout_*.csv")
    if 6 <= len(cs) < 14:
      YAWD[v] = d


def agg(d):
  rows = []
  for f in sorted(glob.glob(f"{d}/rollout_*.csv")):
    x = pd.read_csv(f)
    ss = x[(x.t >= 1.0) & (x.done == 0)]
    if len(ss) == 0:
      ss = x
    rows.append(dict(
      cmd=float(x.cmd_vx.iloc[0]),
      v=ss.v_body_x.mean(), terr=(ss.v_body_x - ss.cmd_vx).abs().mean(),
      cotp=ss.CoT_pos.mean(), cota=ss.CoT_abs.mean(),
      tilt=ss.tilt_deg.mean(), roll=np.degrees(ss.roll.abs().mean()),
      duty=(x[[f"foot_contact_{i}" for i in range(4)]].to_numpy() > 0.5).mean(),
      spine=(np.degrees(ss.q_spine.abs().mean()) if "q_spine" in ss else np.nan),
      spinerange=(np.degrees(ss.q_spine.max() - ss.q_spine.min()) if "q_spine" in ss else np.nan),
      frozen=ss.v_below_eps.mean(),
    ))
  return pd.DataFrame(rows).groupby("cmd").mean().reset_index()


A = {v: agg(d) for v, d in FWD.items()}

# ---- fig 1: CoT+ and tracking ---- #
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
for v, a in A.items():
  m = a.frozen < 0.5  # hide frozen points from the line, mark them hollow
  ax[0].plot(a.cmd[m], a.cotp[m], "o-", color=COL[v], lw=1.8, label=v)
  if (~m).any():
    ax[0].plot(a.cmd[~m], a.cotp[~m], "o", mfc="white", mec=COL[v], ms=5)
  ax[1].plot(a.cmd, a.v - a.cmd, "o-", color=COL[v], lw=1.6, label=v)
ax[0].set_ylabel(r"CoT$^+$  $=\Sigma\max(\tau\dot q,0)/(mgv)$"); ax[0].set_ylim(0, None)
ax[0].set_title("Mechanical cost of transport"); ax[0].set_xlabel("commanded $v_x$  [m/s]")
ax[0].legend(frameon=False, fontsize=8)
ax[1].axhline(0, color=MUT, lw=.7); ax[1].axhspan(-0.04, 0.04, color=MUT, alpha=.08)
ax[1].set_ylabel(r"$v_{body,x}-v_{cmd}$  [m/s]"); ax[1].set_title("Forward tracking error")
ax[1].set_xlabel("commanded $v_x$  [m/s]"); ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "c1_cot_tracking.png", dpi=120); plt.close(fig)

# ---- fig 2: attitude + spine ---- #
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
for v, a in A.items():
  ax[0].plot(a.cmd, a.tilt, "o-", color=COL[v], lw=1.7, label=v)
for v, a in A.items():
  if a.spine.notna().any():
    ax[1].plot(a.cmd, a.spinerange, "o-", color=COL[v], lw=1.7, label=f"{v} (p-p)")
    ax[1].plot(a.cmd, a.spine, "^:", color=COL[v], lw=1, alpha=.7)
ax[0].set_ylabel("trunk tilt  [deg]"); ax[0].set_title("Trunk tilt vs speed")
ax[0].set_xlabel("commanded $v_x$  [m/s]"); ax[0].legend(frameon=False, fontsize=8)
ax[1].set_ylabel("spine angle  [deg]"); ax[1].set_title("Spine use — peak-to-peak (○) & mean |·| (△)")
ax[1].set_xlabel("commanded $v_x$  [m/s]"); ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "c2_attitude_spine.png", dpi=120); plt.close(fig)

# ---- fig 3: v_body_x traces at 0.16 m/s, seed 0 ---- #
fig, ax = plt.subplots(2, 1, figsize=(9.2, 4.8), sharex=True)
for v, d in FWD.items():
  k = next((p for p in sorted(glob.glob(f"{d}/rollout_*.csv")) if "vx0.16" in p and "seed0" in p), None)
  if not k:
    continue
  x = pd.read_csv(k)
  ax[0].plot(x.t, x.v_body_x, lw=.8, color=COL[v], label=v)
  ax[1].plot(x.t, np.degrees(x.tilt), lw=.8, color=COL[v])
ax[0].axhline(0.16, color=MUT, ls=":", lw=1)
ax[0].set_ylabel("$v_{body,x}$  [m/s]  (cmd 0.16)"); ax[0].legend(frameon=False, fontsize=8, ncol=4)
ax[0].set_title("Seed-0 rollout at 0.16 m/s")
ax[1].set_ylabel("trunk tilt  [deg]"); ax[1].set_xlabel("t  [s]")
fig.tight_layout(); fig.savefig(OUT / "c3_traces.png", dpi=120); plt.close(fig)

# ---- fig 4: turning — realised yaw rate ---- #
fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
order = ["rigid", "pitch", "yaw", "roll"]
xpos = np.arange(len(order))
for j, sign in enumerate((+0.2, -0.2)):
  fr = []
  for v in order:
    d = YAWD.get(v)
    vals = []
    if d:
      for f in glob.glob(f"{d}/rollout_*.csv"):
        x = pd.read_csv(f)
        ss = x[(x.t >= 1.0) & (x.done == 0)]
        if abs(ss.cmd_yaw.iloc[0] - sign) < 1e-6:
          vals.append(ss.wz.mean() / sign)
    fr.append(np.mean(vals) if vals else np.nan)
  ax[0].bar(xpos + (j - 0.5) * 0.38, fr, 0.36,
           color=["#8aa9bd", "#e0a668", "#77b79a", "#b78fd6"][:1] * 0 + [COL[v] for v in order],
           alpha=0.65 if j else 1.0, label=f"cmd {sign:+.1f} rad/s")
ax[0].axhline(1.0, color=MUT, ls=":", lw=1)
ax[0].set_xticks(xpos); ax[0].set_xticklabels(order)
ax[0].set_ylabel("realised / commanded yaw rate"); ax[0].set_ylim(0, 1.15)
ax[0].set_title("Yaw-rate tracking  (1.0 = perfect)"); ax[0].legend(frameon=False, fontsize=8)

for v in order:
  d = YAWD.get(v)
  if not d:
    continue
  for f in sorted(glob.glob(f"{d}/rollout_*.csv")):
    x = pd.read_csv(f)
    if "seed0" not in f:
      continue
    ax[1].plot(x.t, np.degrees(np.unwrap(x.yaw)), lw=1.1, color=COL[v],
              ls="-" if x.cmd_yaw.iloc[0] > 0 else "--",
              label=f"{v}" if x.cmd_yaw.iloc[0] > 0 else None)
ax[1].set_ylabel("heading (unwrapped)  [deg]"); ax[1].set_xlabel("t  [s]")
ax[1].set_title("Heading, seed 0  (solid +0.2, dashed −0.2)")
ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "c4_turning.png", dpi=120); plt.close(fig)

# ---- fig 5: spine power — motor or brake? ---- #
LEGS = ["leg1_a", "leg1_e", "leg2_a", "leg2_e", "leg3_a", "leg3_e", "leg4_a", "leg4_e"]


def spine_power(d):
  rows = []
  for f in sorted(glob.glob(f"{d}/rollout_*.csv")):
    x = pd.read_csv(f)
    ss = x[(x.t >= 1.0) & (x.done == 0)]
    if len(ss) < 50 or "P_spine" not in ss:
      continue
    P = ss["P_spine"].to_numpy()
    tot = np.abs(P).sum()
    legneg = np.abs(np.clip(ss[[f"P_{j}" for j in LEGS]].to_numpy(), None, 0)).sum()
    rows.append(dict(
      cmd=round(float(x.cmd_vx.iloc[0]), 2),
      pos_mW=np.clip(P, 0, None).mean() * 1e3,
      neg_mW=np.clip(P, None, 0).mean() * 1e3,
      net_mW=P.mean() * 1e3,
      neg_share=(np.abs(np.clip(P, None, 0)).sum() / tot) if tot > 0 else np.nan,
      spine_share_robot_negwork=(np.abs(np.clip(P, None, 0)).sum()
                                 / (np.abs(np.clip(P, None, 0)).sum() + legneg)),
      frozen=ss.v_below_eps.mean(),
    ))
  if not rows:
    return pd.DataFrame()
  return pd.DataFrame(rows).groupby("cmd").mean().reset_index()


SP = {v: spine_power(d) for v, d in FWD.items()}
SP = {v: s for v, s in SP.items() if len(s)}

fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
for v, s in SP.items():
  m = s.frozen < 0.5
  ax[0].plot(s.cmd[m], s.pos_mW[m], "o-", color=COL[v], lw=1.8, label=v)
  ax[0].plot(s.cmd[m], s.neg_mW[m], "s--", color=COL[v], lw=1.5)
ax[0].axhline(0, color=INK, lw=1)
ax[0].set_ylabel("spine actuator power  [mW / step]")
ax[0].set_xlabel("commanded $v_x$  [m/s]")
ax[0].set_title("Spine work rate", pad=16)
ax[0].text(0.5, 1.015, "solid = motoring  (W$^+$)        dashed = braking  (W$^-$)",
           transform=ax[0].transAxes, ha="center", fontsize=7.5, color=MUT)
ax[0].legend(frameon=False, fontsize=8, loc="lower left")

for v, s in SP.items():
  m = s.frozen < 0.5
  ax[1].plot(s.cmd[m], s.neg_share[m], "o-", color=COL[v], lw=2.0, label=v)
ax[1].axhspan(0.92, 1.0, color=MUT, alpha=.1)
ax[1].axhline(0.5, color=MUT, ls=":", lw=1)
ax[1].set_ylim(0, 1.02)
ax[1].set_ylabel(r"negative-work share   $|W^-|\,/\,(|W^+|+|W^-|)$")
ax[1].set_xlabel("commanded $v_x$  [m/s]")
ax[1].set_title("Brake-like?   1 = pure damper, 0 = pure motor", pad=16)
ax[1].text(0.03, 0.955, "pure damper", transform=ax[1].transAxes,
           fontsize=7.5, color=MUT, va="center")
ax[1].text(0.03, 0.53, "spring / balanced", transform=ax[1].transAxes,
           fontsize=7.5, color=MUT, va="center")
ax[1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.0, 0.32))
fig.tight_layout(); fig.savefig(OUT / "c5_spine_power.png", dpi=120); plt.close(fig)

# ---- fig 6: CoT normalised per motor ---- #
import json as _json
NMOT = {}
for v, d in FWD.items():
  mj = sorted(glob.glob(f"{d}/rollout_*.meta.json"))
  NMOT[v] = _json.loads(Path(mj[0]).read_text())["action_dim"] if mj else 8


def cot_norm(d):
  rows = []
  for f in sorted(glob.glob(f"{d}/rollout_*.csv")):
    x = pd.read_csv(f); ss = x[(x.t >= 1.0) & (x.done == 0)]
    if len(ss) == 0:
      ss = x
    rows.append(dict(cmd=round(float(x.cmd_vx.iloc[0]), 2),
                     cotp=ss.CoT_pos.mean(), legs_only=ss.CoT_pos_legs_only.mean(),
                     frozen=ss.v_below_eps.mean()))
  return pd.DataFrame(rows).groupby("cmd").mean().reset_index()


CN = {v: cot_norm(d) for v, d in FWD.items()}
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.6))
for v, c in CN.items():
  m = c.frozen < 0.5
  ax[0].plot(c.cmd[m], c.cotp[m] / NMOT[v], "o-", color=COL[v], lw=1.8,
             label=f"{v}  (÷{NMOT[v]})")
  ax[1].plot(c.cmd[m], c.legs_only[m] / 8.0, "o-", color=COL[v], lw=1.8, label=v)
ax[0].set_ylabel("CoT⁺ per actuated motor"); ax[0].set_ylim(0, None)
ax[0].set_title("CoT⁺ ÷ number of motors  (8 rigid · 9 spine variants)")
ax[0].set_xlabel("commanded $v_x$  [m/s]"); ax[0].legend(frameon=False, fontsize=8)
ax[1].set_ylabel("leg-only CoT⁺ ÷ 8"); ax[1].set_ylim(0, None)
ax[1].set_title("Leg motors only (8 for every variant) — spine work excluded")
ax[1].set_xlabel("commanded $v_x$  [m/s]"); ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "c6_cot_per_motor.png", dpi=120); plt.close(fig)

# ---- fig 7: CoM roll / pitch / yaw ---- #
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
for v, d in FWD.items():
  a = agg(d)  # reuse: has tilt/roll but we want rpy directly -> recompute
  rows = []
  for f in sorted(glob.glob(f"{d}/rollout_*.csv")):
    x = pd.read_csv(f); ss = x[(x.t >= 1.0) & (x.done == 0)]
    if len(ss) == 0 or ss.v_below_eps.mean() > 0.5:
      continue
    rows.append(dict(cmd=round(float(x.cmd_vx.iloc[0]), 2),
                     roll_amp=np.degrees(ss.roll.std()), pitch_mean=np.degrees(ss.pitch.mean()),
                     pitch_amp=np.degrees(ss.pitch.std()), yaw_drift=np.degrees(
                       (np.unwrap(ss.yaw.to_numpy())[-1] - np.unwrap(ss.yaw.to_numpy())[0])
                       / (ss.t.to_numpy()[-1] - ss.t.to_numpy()[0]))))
  g = pd.DataFrame(rows).groupby("cmd").mean().reset_index()
  ax[0].plot(g.cmd, g.roll_amp, "o-", color=COL[v], lw=1.8, label=v)
  ax[1].plot(g.cmd, g.pitch_mean, "o-", color=COL[v], lw=1.8, label=v)
  ax[1].plot(g.cmd, g.pitch_amp, "^:", color=COL[v], lw=1, alpha=.6)
  ax[2].plot(g.cmd, g.yaw_drift, "o-", color=COL[v], lw=1.8, label=v)
ax[0].set_title("CoM roll — oscillation amplitude (1σ)"); ax[0].set_ylabel("deg")
ax[1].set_title("CoM pitch — mean (○) and amplitude (△)"); ax[1].set_ylabel("deg")
ax[1].axhline(0, color=MUT, lw=.7)
ax[2].set_title("CoM yaw — heading drift rate"); ax[2].set_ylabel("deg/s"); ax[2].axhline(0, color=MUT, lw=.7)
for k in (0, 1, 2):
  ax[k].set_xlabel("commanded $v_x$  [m/s]"); ax[k].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "c7_com_rpy.png", dpi=120); plt.close(fig)

# ---- fig 8: CoM roll/pitch/yaw time traces, seed 0 @ 0.16 ---- #
fig, ax = plt.subplots(3, 1, figsize=(9.4, 6.0), sharex=True)
for v, d in FWD.items():
  k = next((p for p in sorted(glob.glob(f"{d}/rollout_*.csv")) if "vx0.16" in p and "seed0" in p), None)
  if not k:
    continue
  x = pd.read_csv(k)
  ax[0].plot(x.t, np.degrees(x.roll), lw=.8, color=COL[v], label=v)
  ax[1].plot(x.t, np.degrees(x.pitch), lw=.8, color=COL[v])
  ax[2].plot(x.t, np.degrees(np.unwrap(x.yaw)), lw=.8, color=COL[v])
ax[0].set_ylabel("roll [deg]"); ax[1].set_ylabel("pitch [deg]"); ax[2].set_ylabel("yaw (unwrap) [deg]")
ax[2].set_xlabel("t [s]"); ax[0].legend(frameon=False, fontsize=8, ncol=4)
ax[0].set_title("CoM orientation, seed-0 rollout at 0.16 m/s")
for k in (0, 1, 2):
  ax[k].grid(alpha=.3); ax[k].axhline(0, color=MUT, lw=.6)
fig.tight_layout(); fig.savefig(OUT / "c8_com_rpy_traces.png", dpi=120); plt.close(fig)

print("wrote:", *(p.name for p in sorted(OUT.glob("*.png"))))
for p in sorted(OUT.glob("*.png")):
  print(f"  {p} {p.stat().st_size/1024:.0f} KB")
print("FWD dirs:", {k: Path(v).name for k, v in FWD.items()})
print("YAW dirs:", {k: Path(v).name for k, v in YAWD.items()})
