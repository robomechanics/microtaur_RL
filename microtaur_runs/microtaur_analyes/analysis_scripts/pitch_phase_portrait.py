from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"c:\Users\AMC\Downloads\rigid_flat_fixed_m077\Microtaur_RL")
ROLLDIR = ROOT / "rollouts" / "pitch_2026-09-11_10-10-03"
SPEEDS = [0.08, 0.14, 0.20]

fig, axes = plt.subplots(1, len(SPEEDS), figsize=(15, 4.5), sharex=True, sharey=True)
fig.suptitle("Pitch spine phase portrait", fontsize=13)

for ax, vx in zip(axes, SPEEDS):
    files = sorted(ROLLDIR.glob(f"rollout_vx{vx:.2f}_yaw+0.00_seed*.substep.npz"))
    for f in files:
        sub = np.load(f)
        q = np.degrees(sub["q"][:, 8])
        qd = np.degrees(sub["qd"][:, 8])
        ax.plot(q, qd, lw=1.0, alpha=0.35, color="tab:blue")

    ax.set_title(f"cmd {vx:g} m/s")
    ax.set_xlabel("spine angle [deg]")
    if ax is axes[0]:
        ax.set_ylabel("spine rate [deg/s]")
    ax.grid(alpha=0.25)

for ax in axes:
    ax.set_aspect("auto")

out = ROLLDIR / "pitch_phase_portrait.png"
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out, dpi=180)
print(f"Saved {out}")
