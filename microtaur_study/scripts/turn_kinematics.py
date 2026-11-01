"""Turning limits of a trotting Microtaur from its kinematics (no simulator, no dynamics).

  python scripts/turn_kinematics.py --out <dir> [--stride-hz 4.8] [--duty 0.5]

The legs are planar five-bars without abduction: a stance foot moves along its half-body's x axis (and z), never
sideways. In a trot one foot of each diagonal pair stands on each half (rigid robot: on the one body).
  1. Leg workspace (five-bar FK) inside the walk-action limits: stand +- 0.75 rad per joint, |common|, |diff| <= 0.52,
     |common| + |diff| <= 0.70 in paired coordinates; the fore-aft foot range at stance heights.
  2. No-slip turning. A stance foot cannot slip sideways, so the body point above it has no lateral velocity in its
     half's frame: the instantaneous centre of rotation lies on the line through each stance foot perpendicular to
     its half. Rigid body (phi = 0): the two lines are parallel, so it cannot turn without slipping. Spine bent by
     phi (fixed): the lines cross at R = d / sin(phi / 2) from the spine joint (d = fore-aft distance of each stance
     foot from the spine joint; the stance sweep moves both feet back together, so d_f + d_r stays constant).
  3. Lateral foot scrub needed per stance for a turn of radius R at speed v: legs only (phi = 0) and with the spine
     at its action limit; the body takes the lateral velocity that splits the slip evenly between the two feet.
Writes turn_kinematics.txt / .json / .png.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from microtaur_common.kinematics import five_bar_for  # noqa: E402
from microtaur_common.robot_constants import JOINT_HALF_RANGE_RAD, LEG_SIGNS, STAND_A, STAND_E  # noqa: E402
from microtaur_common.walk_action import WalkActionParams  # noqa: E402

HIP_X_M = (0.0841, 0.0825)  # rear / front hip-pair midpoints from the spine axis (robot_2imu.xml; rigid: 84.5 / 83.6 mm)
SPINE_ACTION_LIMIT_RAD = 0.33  # MicrotaurWalkActionCfg.spine_limit_rad
HEIGHTS_M = (0.075, 0.085, 0.095)  # foot below the motor-pivot midpoint: stand pose 74.7 mm, y3 / r5 stance ~95 mm


def leg_workspace(leg: int, n: int = 241) -> np.ndarray:
  """Foot (x, z) of every valid pose inside the walk-action limits (leg 1..4)."""
  p = WalkActionParams()
  kin = five_bar_for(True)
  c, d = np.meshgrid(np.linspace(-p.common_limit_rad, p.common_limit_rad, n),
                     np.linspace(-p.diff_limit_rad, p.diff_limit_rad, n))
  keep = np.abs(c) + np.abs(d) <= p.diamond_limit_rad
  s = LEG_SIGNS[leg - 1]
  da, de = s * (c[keep] - d[keep]), s * (c[keep] + d[keep])
  keep2 = (np.abs(da) <= JOINT_HALF_RANGE_RAD) & (np.abs(de) <= JOINT_HALF_RANGE_RAD)
  out = []
  for qa, qe in zip(STAND_A[leg - 1] + da[keep2], STAND_E[leg - 1] + de[keep2]):
    sol = kin.forward_numpy(float(qa), float(qe), leg_index=leg)
    if sol.valid:
      out.append((sol.foot_x, sol.foot_z))
  return np.asarray(out)


def fore_aft_range(ws: np.ndarray, h: float, tol: float = 0.001) -> tuple[float, float]:
  x = ws[np.abs(ws[:, 1] + h) < tol, 0]
  return (float(x.min()), float(x.max())) if x.size else (math.nan, math.nan)


def noslip_radius(phi: float, d: float) -> float:
  """Radius of the spine joint's path for a fixed bend phi (inf for phi = 0)."""
  return math.inf if abs(phi) < 1e-9 else d / math.sin(abs(phi) / 2)


def scrub_per_stance(v: float, radius: float, phi: float, d: float, t_stance: float) -> float:
  """Lateral slip of each stance foot over one stance (m) for the body twist (v, v / radius) with the spine at phi."""
  w = v / radius
  c, s = math.cos(phi / 2), math.sin(phi / 2)
  # body frame = the mean heading; spine joint at 0 moving (v, u); front foot at d (c, s), rear at -d (c, -s);
  # lateral axes of the halves (-s, c) and (s, c)
  feet = ((np.array([d * c, d * s]), np.array([-s, c])), (np.array([-d * c, d * s]), np.array([s, c])))

  def slips(u: float) -> list[float]:
    return [float(lat @ (np.array([v, u]) + w * np.array([-p[1], p[0]]))) for p, lat in feet]

  a0, a1 = slips(0.0), slips(1.0)
  # choose u with slip_front = -slip_rear (equal magnitude), linear in u
  k = [a1[i] - a0[i] for i in range(2)]
  u = -(a0[0] + a0[1]) / (k[0] + k[1]) if abs(k[0] + k[1]) > 1e-12 else 0.0
  return abs(slips(u)[0]) * t_stance


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--stride-hz", type=float, default=4.8, help="trot frequency of the trained policies")
  ap.add_argument("--duty", type=float, default=0.5, help="stance fraction of each leg")
  a = ap.parse_args()
  a.out.mkdir(parents=True, exist_ok=True)
  t_st = a.duty / a.stride_hz
  d = sum(HIP_X_M) / 2
  ws = {leg: leg_workspace(leg) for leg in (1, 3)}
  res = {"t_stance_s": t_st, "d_m": d, "heights": [], "bend": [], "scrub": []}
  lines = [f"trot {a.stride_hz} Hz, duty {a.duty}: stance {1e3 * t_st:.0f} ms; stance feet {1e3 * d:.1f} mm fore/aft of the spine axis",
           "", "1. leg workspace inside the walk-action limits, fore-aft foot range at stance heights (leg-plane x, + forward)"]
  for h in HEIGHTS_M:
    r1, r3 = fore_aft_range(ws[1], h), fore_aft_range(ws[3], h)
    span = min(r1[1] - r1[0], r3[1] - r3[0])
    res["heights"].append({"h_m": h, "rear_x_m": r1, "front_x_m": r3, "span_m": span, "v_max_m_s": span / t_st})
    lines.append(f"  foot {1e3 * h:.0f} mm below the hip: rear leg x {1e3 * r1[0]:+.0f}..{1e3 * r1[1]:+.0f} mm, front leg "
                 f"{1e3 * r3[0]:+.0f}..{1e3 * r3[1]:+.0f} mm -> stance sweep up to {1e3 * span:.0f} mm = {span / t_st:.2f} m/s")
  lines += ["", "2. no-slip turning (spine joint's path radius); rigid robot / spine at 0: none (straight only)"]
  for deg in (2.7, 5.4, 6.2, 10.0, 15.0, math.degrees(SPINE_ACTION_LIMIT_RAD), 20.0):
    rr = noslip_radius(math.radians(deg), d)
    res["bend"].append({"phi_deg": deg, "radius_m": rr})
    lines.append(f"  spine bend {deg:5.1f} deg -> R = {rr:5.2f} m")
  lines += ["", "3. lateral scrub of each stance foot per stance (mm) to hold radius R; legs only (spine 0) vs spine at its "
                f"{math.degrees(SPINE_ACTION_LIMIT_RAD):.1f} deg limit (where the bend alone suffices: the bend for R, scrub 0)",
            "   R m  |  v 0.10 m/s: legs only  spine+legs |  v 0.20 m/s: legs only  spine+legs"]
  for rad in (0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0):
    row = {"radius_m": rad}
    txt = f"  {rad:4.2f} |"
    for v in (0.10, 0.20):
      phi_need = 2 * math.asin(min(1.0, d / rad))
      phi = min(phi_need, SPINE_ACTION_LIMIT_RAD)
      legs, both = scrub_per_stance(v, rad, 0.0, d, t_st), scrub_per_stance(v, rad, phi, d, t_st)
      row[f"v{v:.2f}"] = {"legs_only_mm": 1e3 * legs, "spine_and_legs_mm": 1e3 * both, "phi_deg": math.degrees(phi)}
      txt += f"              {1e3 * legs:5.1f}       {1e3 * both:5.1f} |"
    res["scrub"].append(row)
    lines.append(txt)
  (a.out / "turn_kinematics.txt").write_text("\n".join(lines) + "\n")
  (a.out / "turn_kinematics.json").write_text(json.dumps(res, indent=1))
  figure(a.out, ws, res, d, t_st)
  print("\n".join(lines))


def figure(out: Path, ws: dict, res: dict, d: float, t_st: float):
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, ax = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
  w = ws[3]
  ax[0].plot(1e3 * w[:, 0], 1e3 * w[:, 1], ".", ms=1, color="#9ec5f4")
  for h in res["heights"]:
    x0, x1 = h["front_x_m"]
    ax[0].plot([1e3 * x0, 1e3 * x1], [-1e3 * h["h_m"]] * 2, color="#1c5cab", lw=2)
    ax[0].text(1e3 * x1 + 3, -1e3 * h["h_m"], f"{1e3 * h['span_m']:.0f} mm", va="center", fontsize=8)
  ax[0].plot(0, 0, "k+", ms=10)
  ax[0].set_aspect("equal")
  ax[0].set_title("front leg: reachable foot positions inside the action limits\n(lines: fore-aft range at 75 / 85 / 95 mm)",
                  fontsize=10)
  ax[0].set_xlabel("x mm (+ forward)")
  ax[0].set_ylabel("z mm (from the motor-pivot midpoint)")
  phi = np.radians(np.linspace(1, 20, 200))
  ax[1].plot(np.degrees(phi), [noslip_radius(p, d) for p in phi], color="#2a78d6", lw=2)
  for deg, lab in ((6.2, "y3 at a 0.5 rad/s command"), (math.degrees(SPINE_ACTION_LIMIT_RAD), "action limit")):
    rr = noslip_radius(math.radians(deg), d)
    ax[1].plot(deg, rr, "o", color="#eb6834")
    ax[1].annotate(f"{lab}\n{deg:.1f} deg -> {rr:.2f} m", (deg, rr), textcoords="offset points", xytext=(8, 8), fontsize=8)
  ax[1].set_yscale("log")
  ax[1].set_xlabel("spine bend (deg)")
  ax[1].set_ylabel("no-slip turn radius (m)")
  ax[1].set_title("trot with a fixed spine bend: radius without foot slip\n(rigid / spine 0: no turning without slip)",
                  fontsize=10)
  rads = [r["radius_m"] for r in res["scrub"]]
  for v, col in (("v0.10", "#2a78d6"), ("v0.20", "#1baf7a")):
    ax[2].plot(rads, [r[v]["legs_only_mm"] for r in res["scrub"]], "-o", color=col, label=f"legs only, {v[1:]} m/s")
    ax[2].plot(rads, [r[v]["spine_and_legs_mm"] for r in res["scrub"]], "--o", color=col, label=f"spine at limit + legs, {v[1:]} m/s")
  ax[2].set_xscale("log")
  ax[2].set_xlabel("turn radius (m)")
  ax[2].set_ylabel("lateral foot scrub per stance (mm)")
  ax[2].set_title(f"sideways slip each stance foot needs ({1e3 * t_st:.0f} ms stance)", fontsize=10)
  ax[2].legend(fontsize=8, frameon=False)
  for x in ax:
    x.grid(color="#e6e6e6", lw=0.6)
  fig.savefig(out / "turn_kinematics.png", dpi=100)


if __name__ == "__main__":
  main()
