"""How much of each step-field rollout is actually spent ON the tiles?

The step field is 4.0 m x 1.2 m. The robots walk open-loop and drift, so a run
can wander off the side onto the flat border -- where every second it spends
lowers its CoT and raises its speed, flattering the variant that drifts most.
This measures on-field time per rollout and recomputes CoT over on-field samples
only.

Tile region, in the logger's origin-relative frame (spawn at patch-local
x=0.175, y=0.6; tiles span local x 0.35-3.95, y 0-1.2):
    base_x in [0.175, 3.775],  base_y in [-0.60, 0.60]
"""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from pool_cell import _raw

X0, X1, Y_HALF = 0.175, 3.775, 0.60
SETTLE = 1.0


def rollouts(variant, lane_keep=False):
    seen = {}
    for d in sorted(glob.glob(f"rollouts/{variant}_steps_*"), key=os.path.getmtime):
        if not _raw(d):
            continue
        for mf in glob.glob(os.path.join(d, "rollout_*.meta.json")):
            m = json.loads(open(mf).read())
            if m["cmd_vx"] not in (0.08, 0.14, 0.20):
                continue
            if bool(m.get("lane_keep", False)) != lane_keep:
                continue
            seen[(m["cmd_vx"], m["seed"])] = mf.replace(".meta.json", ".csv")
    return seen


def analyse(csv):
    df = pd.read_csv(csv)
    ss = df[(df.t >= SETTLE) & (df.done == 0)]
    on = ((ss.base_x >= X0) & (ss.base_x <= X1) & (ss.base_y.abs() <= Y_HALF)).to_numpy()
    off_lat = (ss.base_y.abs() > Y_HALF).to_numpy()
    first_off = ss.t.to_numpy()[np.argmax(~on)] if (~on).any() else np.nan
    return dict(
        frac_on=float(on.mean()),
        left_side=bool(off_lat.any()),
        t_first_off=float(first_off),
        max_abs_y=float(ss.base_y.abs().max()),
        cot_all=float(ss.CoT_pos.mean()),
        cot_on=float(ss.CoT_pos[on].mean()) if on.any() else np.nan,
        v_all=float(ss.v_body_x.mean()),
        v_on=float(ss.v_body_x[on].mean()) if on.any() else np.nan,
        x_end=float(ss.base_x.iloc[-1]),
    )


def main():
    rows = []
    for v in ["rigid", "pitch", "yaw", "roll"]:
        for (cmd, seed), csv in sorted(rollouts(v).items()):
            r = analyse(csv); r.update(variant=v, cmd=cmd, seed=seed); rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv("rollouts/steps_onfield.csv", index=False)
    g = df.groupby(["variant", "cmd"]).agg(
        n=("seed", "count"), frac_on=("frac_on", "mean"),
        runs_left_side=("left_side", "sum"), max_y=("max_abs_y", "mean"),
        CoT_all=("cot_all", "mean"), CoT_on=("cot_on", "mean"),
        v_all=("v_all", "mean"), v_on=("v_on", "mean"))
    order = {"rigid": 0, "pitch": 1, "yaw": 2, "roll": 3}
    g = g.reset_index().sort_values(["cmd", "variant"], key=lambda s: s.map(order) if s.name == "variant" else s)
    print(g.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
