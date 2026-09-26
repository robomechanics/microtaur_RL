"""Pass/fail of an isaac_eval_gait.py summary.json against the tuning-loop criteria.

  python scripts/isaac_eval_criteria.py <eval dir>      (prints a table, writes criteria.json)

Straight commands (yaw 0):  0.8 <= speed_ratio <= 1.2, gait "trot", stride 2-5 Hz,
                            swing clearance p95 >= 5 mm on every leg, no resets.
Turning commands:           0.7 <= yaw_ratio <= 1.3, speed_ratio >= 0.7, no resets.
Everything passes -> ALL PASS (the loop stops).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SPEED = (0.8, 1.2)
YAW = (0.7, 1.3)
TURN_MIN_SPEED = 0.7
STRIDE_HZ = (2.0, 5.0)
CLEARANCE_MM = 5.0

d = Path(sys.argv[1])
summary = json.loads((d / "summary.json").read_text())
rows, all_ok = [], True
for label, s in summary["commands"].items():
  checks = {}
  if s["cmd_yaw_rad_s"] == 0.0:
    checks["speed"] = SPEED[0] <= s["speed_ratio"] <= SPEED[1]
    checks["trot"] = s["gait"] == "trot"
    checks["stride"] = STRIDE_HZ[0] <= s["stride_hz"] <= STRIDE_HZ[1]
    checks["clearance"] = min(s["swing_clearance_p95_mm"]) >= CLEARANCE_MM
  else:
    checks["yaw"] = YAW[0] <= s["yaw_ratio"] <= YAW[1]
    checks["speed"] = s["speed_ratio"] >= TURN_MIN_SPEED
  checks["no_resets"] = s["resets_during_measure"] == 0
  ok = all(checks.values())
  all_ok &= ok
  rows.append({"command": label, "ok": ok, "checks": checks,
               "speed_ratio": round(s["speed_ratio"], 3), "yaw_ratio": None if s["yaw_ratio"] is None else round(s["yaw_ratio"], 3),
               "gait": s["gait"], "stride_hz": round(s["stride_hz"], 2), "clearance_mm": s["swing_clearance_p95_mm"],
               "duty": s["duty_factor"], "torque_sat": round(s["torque_saturated_frac"], 3), "cot": round(s["electrical_cot"], 1),
               "resets": s["resets_during_measure"]})

(d / "criteria.json").write_text(json.dumps({"all_pass": all_ok, "rows": rows}, indent=1))
print(f"{'command':14s} {'ok':4s} {'v ratio':>8s} {'w ratio':>8s} {'stride':>7s} {'clear mm (min)':>14s}  gait / failed checks")
for r in rows:
  failed = [k for k, v in r["checks"].items() if not v]
  yaw = "" if r["yaw_ratio"] is None else f"{r['yaw_ratio']:.3f}"
  print(f"{r['command']:14s} {'PASS' if r['ok'] else 'FAIL':4s} {r['speed_ratio']:8.3f} {yaw:>8s} "
        f"{r['stride_hz']:7.2f} {min(r['clearance_mm']):14.1f}  {r['gait']} / {','.join(failed) or '-'}")
print("\nleg space (five-bar FK, legs RR RL FL FR): extension std / swing arc std / swing retraction, mm; mean angle deg")
for label, s in summary["commands"].items():
  ls = s.get("leg_space")
  if ls:
    print(f"{label:14s} ext {ls['r_std_mm']}  arc {ls['arc_std_mm']}  retract {ls['swing_retraction_mm']}  angle {ls['angle_mean_deg']}")
print("ALL PASS" if all_ok else "NOT PASSED")
