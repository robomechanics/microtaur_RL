"""Check the exact-work re-run against the original rollouts, cell by cell.

    python scripts/exact_vs_sampled.py    -> rollouts/exact_work/{pairs.csv, cells.csv}

For every rollout re-run with --substep-log, find the original (same variant,
terrain, steering mode, speed, seed; raw previous action; no substep log) and
report

  * reproduction: max |difference| in trunk x, y and every joint angle over the
    whole run -- 0 means the re-run is the same trajectory, so every non-energy
    result already published stands;
  * CoT+ from the 50 Hz tau*qd sample vs from exact substep work, same mass,
    same first-episode window.
"""
import glob, json, os
import numpy as np, pandas as pd

SPEEDS = (0.08, 0.14, 0.20)
MEASURED_MASS_KG = {"yaw": 0.4577}
OUT = "rollouts/exact_work"


def index(substep):
    out = {}
    for d in sorted(glob.glob("rollouts/*_20*"), key=os.path.getmtime):
        v = os.path.basename(d).split("_")[0]
        for mf in glob.glob(os.path.join(d, "rollout_*.meta.json")):
            m = json.loads(open(mf).read())
            if not m.get("raw_action") or m["cmd_vx"] not in SPEEDS or bool(m.get("substep_log")) != substep:
                continue
            t = m.get("terrain", "flat")
            if t == "weave" and (m.get("terrain_kw") or {}).get("spacing") != 0.80:
                continue
            if t == "curb" and m.get("lane_keep"):
                continue
            out[(v, t, bool(m.get("lane_keep")), m["cmd_vx"], m["seed"])] = (mf.replace(".meta.json", ".csv"), m)
    return out


def first_episode(df):
    rs = df.index[(df["done"] > 0) & (df.index < len(df) - 1)]
    return df.loc[: rs[0] - 1] if len(rs) else df


def main():
    os.makedirs(OUT, exist_ok=True)
    new, old = index(True), index(False)
    rows = []
    for key, (csv, m) in sorted(new.items()):
        if key not in old:
            continue
        a, b = pd.read_csv(old[key][0]), pd.read_csv(csv)
        n = min(len(a), len(b))
        qcols = [c for c in a.columns if c.startswith("q_")]
        dev = max(float(np.abs(a[c][:n] - b[c][:n]).max()) for c in ["base_x", "base_y"] + qcols)
        ep = first_episode(b)
        ss = ep[(ep.t >= 1.0) & (ep.done == 0)]
        m_use = MEASURED_MASS_KG.get(key[0], float(m["mass_kg"]))
        den = m_use * 9.81 * np.maximum(ss.v_body_x.abs().to_numpy(), 0.02)
        rows.append(dict(variant=key[0], terrain=key[1], mode="waypoint" if key[2] else "openloop",
                         cmd=key[3], seed=key[4], max_traj_diff=dev,
                         CoT_50hz=float(np.mean(ss.P_pos.to_numpy() / den)),
                         CoT_exact=float(np.mean(ss.Wxpos_total.to_numpy() / 0.02 / den))))
    df = pd.DataFrame(rows)
    df["ratio"] = df.CoT_exact / df.CoT_50hz
    df.to_csv(f"{OUT}/pairs.csv", index=False)
    print(f"{len(df)} paired rollouts; identical trajectories: {(df.max_traj_diff == 0).sum()}  "
          f"max diff {df.max_traj_diff.max():.3g}")
    c = df.groupby(["variant", "terrain", "mode", "cmd"])[["CoT_50hz", "CoT_exact", "ratio"]].mean()
    c.to_csv(f"{OUT}/cells.csv")
    pd.set_option("display.width", 200)
    print(c.round(3).to_string())
    print("\nexact / 50 Hz by variant:", df.groupby("variant").ratio.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
