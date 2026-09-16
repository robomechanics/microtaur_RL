"""Waypoint test on the step field: how far down the course, how straight, and
at what energy per metre actually gained.

    python scripts/waypoint_analysis.py

Reads every raw-action, lane-kept steps rollout. For each, FIRST EPISODE ONLY
(a mid-run `done` is a reset that teleports the robot to spawn), after a 1 s
settle:

  progress_m       furthest point reached along the line to the waypoint
  progress_frac    progress / (commanded speed x time) -- 1.0 = kept pace
  xtrack_rms_mm    RMS lateral deviation from the waypoint line
  xtrack_max_mm    worst lateral deviation
  on_field         share of time on the tiles
  t_reach_{d}      seconds to first cover d metres (NaN if never)
  CoT_progress     sum(P_pos dt) / (m g * net progress)   <- the headline

CoT_progress differs from the logger's CoT in its denominator: net distance
gained toward the goal rather than instantaneous body-frame speed. A robot that
wobbles, crabs sideways or stalls is charged for every joule of it.
"""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from pool_cell import _raw

X0, X1, Y_HALF = 0.175, 3.775, 0.60
G, SETTLE, DT = 9.81, 1.0, 0.02
MARKS = (0.5, 1.0, 1.5, 2.0)
OUT = "rollouts/waypoint"


def runs(variant):
    seen = {}
    for d in sorted(glob.glob(f"rollouts/{variant}_steps_*"), key=os.path.getmtime):
        if not _raw(d):
            continue
        for mf in glob.glob(os.path.join(d, "rollout_*.meta.json")):
            m = json.loads(open(mf).read())
            if not m.get("lane_keep") or m["cmd_vx"] not in (0.08, 0.14, 0.20):
                continue
            seen[(m["cmd_vx"], m["seed"])] = (mf.replace(".meta.json", ".csv"), m)
    return seen


def one(csv, meta):
    df = pd.read_csv(csv)
    rs = df.index[(df.done > 0) & (df.index < len(df) - 1)]
    ep = df.loc[: rs[0] - 1] if len(rs) else df
    ss = ep[ep.t >= SETTLE]
    if len(ss) < 10:
        return None
    x, y, t = ss.base_x.to_numpy(), ss.base_y.to_numpy(), ss.t.to_numpy()
    prog = float(np.maximum.accumulate(x)[-1] - x[0])
    dur = float(t[-1] - t[0])
    on = (x >= X0) & (x <= X1) & (np.abs(y) <= Y_HALF)
    energy = float(ss.P_pos.sum() * DT)
    net = float(x[-1] - x[0])
    mass = float(meta.get("mass_kg", np.nan))
    r = dict(progress_m=prog,
             progress_frac=prog / (meta["cmd_vx"] * dur) if dur > 0 else np.nan,
             xtrack_rms_mm=float(np.sqrt(np.mean(y ** 2)) * 1000),
             xtrack_max_mm=float(np.abs(y).max() * 1000),
             on_field=float(on.mean()), reset=bool(len(rs)),
             energy_J=energy,
             CoT_progress=energy / (mass * G * net) if net > 0.05 else np.nan,
             CoT_body=float(ss.CoT_pos.mean()), mass_kg=mass)
    for d in MARKS:
        hit = np.where(x - x[0] >= d)[0]
        r[f"t_reach_{d}"] = float(t[hit[0]] - t[0]) if len(hit) else np.nan
    return r


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for v in ["rigid", "pitch", "yaw", "roll"]:
        for (cmd, seed), (csv, meta) in sorted(runs(v).items()):
            r = one(csv, meta)
            if r:
                r.update(variant=v, cmd=cmd, seed=seed); rows.append(r)
    if not rows:
        raise SystemExit("no lane-kept steps rollouts yet")
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/waypoint_rollouts.csv", index=False)
    order = {"rigid": 0, "pitch": 1, "yaw": 2, "roll": 3}
    g = (df.groupby(["cmd", "variant"])
           .agg(n=("seed", "count"), resets=("reset", "sum"), on_field=("on_field", "mean"),
                progress_m=("progress_m", "mean"), progress_sd=("progress_m", "std"),
                pace=("progress_frac", "mean"),
                xtrack_rms_mm=("xtrack_rms_mm", "mean"), xtrack_max_mm=("xtrack_max_mm", "max"),
                CoT_prog=("CoT_progress", "median"), CoT_prog_sd=("CoT_progress", "std"),
                t_1m=("t_reach_1.0", "median"))
           .reset_index())
    g = g.sort_values(["cmd", "variant"],
                      key=lambda s: s.map(order) if s.name == "variant" else s)
    g.to_csv(f"{OUT}/waypoint_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(g.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
