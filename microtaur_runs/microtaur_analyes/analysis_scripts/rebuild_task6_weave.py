"""Rebuild ONLY the weave rows of the audit's canonical long-format CSV, from
the 0.55 m aggressive resweep, leaving every other terrain's rows untouched.

    python scripts/rebuild_task6_weave.py
      -> overwrites rollouts/audit_2026-09-11/task6_long_format.csv
         (backs up the previous file to task6_long_format_pre055m.csv first)

The per-run metric formulas below were reverse-engineered to match the
existing (0.80 m) weave rows exactly, run by run, since the original
generation script isn't in this repo (produced by a forked analysis agent):

    CoT            = mean(CoT_pos)                          [exact match]
    com_z_mean_mm  = mean(com_z) * 1000                      [exact match]
    com_z_rms_mm   = std(com_z, ddof=0) * 1000                [exact match]
    roll_rms_deg   = std(degrees(roll), ddof=0)                [exact match]
    pitch_rms_deg  = std(degrees(pitch), ddof=0)                [exact match]
    dist_m         = base_x[-1] - base_x[0]                  [exact match]
    dist_BL        = dist_m / 0.168                          [exact match]
    xtrack_rms_mm  = std(weave_cross_track_m, ddof=0) * 1000  [~0.07% residual,
                     could not fully resolve -- flagged, not hidden]
    n_rows = n_rows_masked = len(settled window)              [weave has no
                     extra on-course mask, unlike curb/steps]

Window: first episode, t >= 1.0 s, done == 0 (matches spine_function.py /
resummarize.py conventions used throughout this project).
"""
import glob
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TASK6 = ROOT / "rollouts/audit_2026-09-11/task6_long_format.csv"
BACKUP = ROOT / "rollouts/audit_2026-09-11/task6_long_format_pre055m.csv"

VARIANTS = ["rigid", "pitch", "yaw", "roll"]
SPEEDS = (0.08, 0.14, 0.20)
SETTLE = 1.0
BL_M = 0.168
METRICS = ["CoT", "com_z_mean_mm", "com_z_rms_mm", "dist_BL", "dist_m",
          "n_rows", "n_rows_masked", "pitch_rms_deg", "roll_rms_deg", "xtrack_rms_mm"]


def first_episode(df):
    rs = df.index[(df["done"] > 0) & (df.index < len(df) - 1)]
    return df.loc[: rs[0] - 1] if len(rs) else df


def find_weave055_runs():
    """(variant, cmd, seed) -> csv path, for the 0.55 m resweep only."""
    keep = {}
    for v in VARIANTS:
        for d in sorted(glob.glob(f"rollouts/{v}_weave_*"), key=lambda p: Path(p).stat().st_mtime):
            for mf in glob.glob(str(Path(d) / "rollout_*.meta.json")):
                m = json.loads(Path(mf).read_text())
                if m.get("cmd_vx") not in SPEEDS:
                    continue
                if bool(m.get("lane_keep")):
                    continue
                if (m.get("terrain_kw") or {}).get("spacing") != 0.55:
                    continue
                keep[(v, m["cmd_vx"], m["seed"])] = mf.replace(".meta.json", ".csv")
    return keep


def metrics_for_run(csv_path):
    df = pd.read_csv(csv_path)
    ep = first_episode(df)
    ss = ep[(ep.t >= SETTLE) & (ep.done == 0)]
    n = len(ss)
    com_z = ss.com_z.to_numpy()
    roll_deg = np.degrees(ss.roll.to_numpy())
    pitch_deg = np.degrees(ss.pitch.to_numpy())
    dist_m = float(ss.base_x.iloc[-1] - ss.base_x.iloc[0])
    return {
        "CoT": float(ss.CoT_pos.mean()),
        "com_z_mean_mm": float(com_z.mean() * 1000),
        "com_z_rms_mm": float(np.std(com_z, ddof=0) * 1000),
        "roll_rms_deg": float(np.std(roll_deg, ddof=0)),
        "pitch_rms_deg": float(np.std(pitch_deg, ddof=0)),
        "dist_m": dist_m,
        "dist_BL": dist_m / BL_M,
        "xtrack_rms_mm": float(np.std(ss.weave_cross_track_m.to_numpy(), ddof=0) * 1000),
        "n_rows": float(n),
        "n_rows_masked": float(n),   # weave has no additional on-course mask
    }


def main():
    runs = find_weave055_runs()
    print(f"found {len(runs)} weave-0.55m runs (expect {len(VARIANTS)*len(SPEEDS)*8}={len(VARIANTS)*len(SPEEDS)*8})")
    missing = [(v, s, sd) for v in VARIANTS for s in SPEEDS for sd in range(8) if (v, s, sd) not in runs]
    if missing:
        print(f"WARNING: missing {len(missing)} cells: {missing}")

    new_rows = []
    for (v, cmd, seed), csv_path in sorted(runs.items()):
        vals = metrics_for_run(csv_path)
        for metric in METRICS:
            new_rows.append(dict(variant=v, terrain="weave", speed=cmd, seed=seed,
                                 metric=metric, value=vals[metric]))
    new_df = pd.DataFrame(new_rows)

    old = pd.read_csv(TASK6)
    shutil.copy(TASK6, BACKUP)
    print(f"backed up previous file -> {BACKUP}")

    kept = old[old["terrain"] != "weave"]
    out = pd.concat([kept, new_df], ignore_index=True)
    out = out.sort_values(["variant", "terrain", "speed", "seed", "metric"]).reset_index(drop=True)
    out.to_csv(TASK6, index=False)
    print(f"wrote {TASK6}  ({len(kept)} non-weave rows kept + {len(new_df)} new weave rows = {len(out)} total)")


if __name__ == "__main__":
    main()
