"""One iteration of learning control: nudge a gait so the joints *actually* move like the RL's in sim.

A gait fitted to the RL's motor commands does not always produce the RL's joint motion: the robot turns, loads
and saturates differently. This compares the realized joint angles of a CPG rollout with those of an RL rollout,
stride by stride, and adds the difference (times --gain) to the gait's commands:

    command_k  <-  command_k + gain * (RL joint angle_k - CPG joint angle_k)     for every motor and harmonic k

The result is still offset + K sines per motor on one clock (same frequency, same harmonic count), so it stays
open loop and 50 Hz-compatible; the ±30° leg clip applies as before. The corrections are learned against the
simulator's motor model.

  python refine_gait_ilc.py yaw_match/rl_d1.npz yaw_match/ilc/h_it0.npz yaw_match/gait_yaw_final.json yaw_match/ilc/gait_it1.json

The CPG rollout must be harmonic mode with --series of the given gait (its logged gait phase is the time base).
"""

import argparse
import json

import numpy as np

from fit_harmonic_gait import design, stride_rate


def harmonics_of(y: np.ndarray, t: np.ndarray, f: float, k: int):
  """offset (M,), complex C (k, M) with y = offset + Re(C_k e^{i k 2 pi f t})."""
  coef, *_ = np.linalg.lstsq(design(t, f, k), y, rcond=None)
  return coef[0], coef[1::2] - 1j * coef[2::2]


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("rl")
  parser.add_argument("cpg")
  parser.add_argument("gait")
  parser.add_argument("out")
  parser.add_argument("--gain", type=float, default=0.7)
  args = parser.parse_args()

  gait = json.load(open(args.gait))
  f, k = float(gait["frequency_hz"]), len(gait["cos_rad"][0])
  lead = gait["joint_order"].index("leg1_a_joint_act")

  # CPG joint motion in the gait's own command clock (logged gait phase -> time).
  z = np.load(args.cpg)
  dt = float(z["dt"])
  offsets, coefs = [], []
  for r in np.flatnonzero(z["m_falls"] == 0):
    t_gait = np.unwrap(z["s_phase_leg1"][:, r].astype(float)) / (2 * np.pi * f)
    off, c = harmonics_of(z["s_q"][:, r, :].astype(float), t_gait, f, k)
    offsets.append(off)
    coefs.append(c)
  cpg_off, cpg_c = np.mean(offsets, 0), np.mean(coefs, 0)
  cpg_lead_phase = np.angle(cpg_c[0, lead])

  # RL joint motion: each robot on its own stride clock, rotated so leg1_a's realized fundamental lines up with the CPG's.
  rl = np.load(args.rl)
  t = np.arange(rl["s_q"].shape[0]) * float(rl["dt"])
  offsets, coefs = [], []
  for r in np.flatnonzero(rl["m_falls"] == 0):
    q = rl["s_q"][:, r, :].astype(float)
    off, c = harmonics_of(q, t, stride_rate(q[:, :8], t), k)
    c = c * np.exp(1j * np.arange(1, k + 1)[:, None] * (cpg_lead_phase - np.angle(c[0, lead])))
    offsets.append(off)
    coefs.append(c)
  rl_off, rl_c = np.mean(offsets, 0), np.mean(coefs, 0)

  err_off, err_c = rl_off - cpg_off, rl_c - cpg_c
  cmd_c = np.array(gait["cos_rad"]).T - 1j * np.array(gait["sin_rad"]).T  # (k, M), same convention
  cmd_c = cmd_c + args.gain * err_c
  out = dict(gait)
  out["offset_rad"] = (np.array(gait["offset_rad"]) + args.gain * err_off).tolist()
  out["cos_rad"] = np.real(cmd_c).T.tolist()
  out["sin_rad"] = (-np.imag(cmd_c)).T.tolist()
  out["amplitude_deg"] = np.degrees(np.abs(cmd_c)).T.round(2).tolist()
  out["phase_deg"] = np.degrees(np.angle(cmd_c)).T.round(1).tolist()
  out["ilc_iterations"] = int(gait.get("ilc_iterations", 0)) + 1
  out["ilc_gain"] = args.gain
  with open(args.out, "w") as handle:
    json.dump(out, handle, indent=1)

  names = [n.replace("_joint_act", "") for n in gait["joint_order"]]
  rms = lambda c, o: np.degrees(np.sqrt(np.abs(o) ** 2 + 0.5 * (np.abs(c) ** 2).sum(0)))  # RMS of a harmonic signal
  print(f"[ilc] {args.out}: iteration {out['ilc_iterations']}, gain {args.gain}")
  print("  realized-angle error RMS per motor (deg): "
        + ", ".join(f"{n} {e:.2f}" for n, e in zip(names, rms(err_c, err_off))))


if __name__ == "__main__":
  main()
