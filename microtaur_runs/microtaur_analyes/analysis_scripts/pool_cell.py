"""Pool every raw-action sweep for one (variant, terrain) cell and summarise it.

    python scripts/pool_cell.py yaw steps
    python scripts/pool_cell.py --all-done          # every complete cell

Extra-seed runs land in their own timestamped directory, so a newest-dir lookup
reports the top-up instead of the pooled set. Pre-fix sweeps (clipped
previous-action) are excluded by the `raw_action` flag they lack.

Tortuosity is reported as a median: it is path length over net displacement, so a
run that ends near where it started sends the mean to absurd values.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np, pandas as pd

SPEEDS = [0.08, 0.14, 0.20]


def _raw(d):
    m = sorted(glob.glob(os.path.join(d, "rollout_*.meta.json")))
    if not m:
        return False
    try:
        return bool(json.loads(open(m[0]).read()).get("raw_action", False))
    except Exception:
        return False


def _lane_keep(d):
    m = sorted(glob.glob(os.path.join(d, "rollout_*.meta.json")))
    try:
        return bool(json.loads(open(m[0]).read()).get("lane_keep", False)) if m else False
    except Exception:
        return False


def pool(variant, terrain, lane_keep=False):
    """Open-loop runs by default. Lane-kept (waypoint-steered) runs share
    terrain="steps" but are a different experiment -- and being newer, they would
    otherwise REPLACE the open-loop rows in the (cmd, seed) de-duplication."""
    rows = []
    for d in glob.glob(f"rollouts/{variant}_*"):
        sj = os.path.join(d, "summary.json")
        if not os.path.exists(sj) or not _raw(d) or _lane_keep(d) != lane_keep:
            continue
        try:
            for r in json.loads(open(sj).read()):
                if r.get("terrain") == terrain:
                    rows.append(r)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[df.cmd_vx.isin(SPEEDS)]
    return df.drop_duplicates(subset=["cmd_vx", "seed"], keep="last")


def flat_base(variant):
    f = pool(variant, "flat")
    return f.groupby("cmd_vx").mean_CoT_pos.mean() if len(f) else None


def report(variant, terrain):
    df = pool(variant, terrain)
    if df.empty:
        print(f"{variant}/{terrain}: no data"); return
    base = flat_base(variant)
    seeds = sorted(int(s) for s in df.seed.unique())
    print(f"\n=== {variant} / {terrain} — {len(df)} rollouts, seeds {seeds} ===")
    out = []
    for s in SPEEDS:
        q = df[df.cmd_vx == s]
        if not len(q):
            continue
        row = dict(cmd=s, n=len(q), v=q.mean_v_body_x.mean(),
                   v_cmd=q.mean_v_body_x.mean() / s,
                   CoT=q.mean_CoT_pos.mean(), sd=q.mean_CoT_pos.std())
        if base is not None and s in base.index and base[s]:
            row["CoTx"] = row["CoT"] / base[s]
        if "path_tortuosity" in q:
            row["tort_med"] = q.path_tortuosity.median()
        if terrain == "curb" and "curb_straddle_frac" in q:
            row["straddle"] = q.curb_straddle_frac.mean()
        out.append(row)
    print(pd.DataFrame(out).set_index("cmd").round(3).to_string())
    key = "curb_straddle_frac" if terrain == "curb" else "mean_CoT_pos"
    print(f"\nper-seed {key}:")
    print(df.pivot_table(index="seed", columns="cmd_vx", values=key).round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", nargs="?")
    ap.add_argument("terrain", nargs="?")
    ap.add_argument("--all-done", action="store_true")
    a = ap.parse_args()
    if a.all_done:
        for v in ["rigid", "pitch", "yaw", "roll"]:
            for t in ["flat", "curb", "steps", "weave"]:
                d = pool(v, t)
                if len(d) >= 24:
                    report(v, t)
    else:
        report(a.variant, a.terrain)
