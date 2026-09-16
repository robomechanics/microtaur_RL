"""Rebuild every per-rollout metric from the raw CSVs, in one tidy table.

    python scripts/resummarize.py            -> rollouts/tidy/tidy_rollouts.csv

Why not reuse summary.json: those were written before two fixes -- they average
across mid-run resets (an out-of-bounds time-out teleports the robot to spawn),
and they use each model's simulated mass. This recomputes from the CSVs with:

  * FIRST EPISODE ONLY, after a 1 s settle;
  * measured mass where one exists (MEASURED_MASS_KG). Energy metrics scale as
    1/m, so CoT is rescaled by m_sim / m_measured. The simulation itself still
    ran at the simulated mass; this re-expresses its mechanical power per unit
    of the real robot's weight;
  * EXACT joint work where the run was logged with --substep-log (positive work
    summed at the 5 ms physics step). The 50 Hz tau*qd sample under-reads
    positive power by ~40 % and is kept as CoT_pos_50hz for comparison only.

One row per rollout; open-loop and waypoint-steered runs are both included and
tagged by `mode`.
"""
import glob, json, math, os, sys
import numpy as np, pandas as pd

MEASURED_MASS_KG = {"yaw": 0.4577}        # weighed 2026-09-10
SPEEDS = (0.08, 0.14, 0.20)
SETTLE, DT, G = 1.0, 0.02, 9.81
X0, X1, Y_HALF = 0.175, 3.775, 0.60        # step-field tiles, origin-relative
OUT = "rollouts/tidy"


def meta0(d):
    m = sorted(glob.glob(os.path.join(d, "rollout_*.meta.json")))
    return json.loads(open(m[0]).read()) if m else {}


def first_episode(df):
    rs = df.index[(df["done"] > 0) & (df.index < len(df) - 1)]
    return (df.loc[: rs[0] - 1] if len(rs) else df), int(len(rs))


def weave_track(ss):
    y = ss.base_y.to_numpy() - ss.base_y.mean()
    tg = ss.weave_target_y.to_numpy() - ss.weave_target_y.mean()
    if y.std() < 1e-9 or tg.std() < 1e-9:
        return np.nan, np.nan
    c = np.correlate(y, tg, mode="full") / (len(y) * y.std() * tg.std())
    return float(c.max()), float(y.std() / tg.std())


def summarise(csv, m, variant):
    df = pd.read_csv(csv)
    ep, n_resets = first_episode(df)
    ss = ep[(ep.t >= SETTLE) & (ep.done == 0)]
    if len(ss) < 20:
        return None
    m_sim = float(m.get("mass_kg", np.nan))
    m_use = MEASURED_MASS_KG.get(variant, m_sim)
    k = m_sim / m_use
    t = m.get("terrain", "flat")
    mode = "waypoint" if m.get("lane_keep") else "openloop"
    x, y = ss.base_x.to_numpy(), ss.base_y.to_numpy()
    exact = "Wxpos_total" in ss
    # Positive power per policy step. Exact (--substep-log): positive joint work
    # summed at the 5 ms physics step. Otherwise the 50 Hz tau*qd sample, which
    # reads low (see run_exact_work_sweep.sh) and is kept only for comparison.
    denom = m_use * G * np.maximum(ss.v_body_x.abs().to_numpy(), 0.02)
    if exact:
        wx = [c for c in ss.columns if c.startswith("Wxpos_") and c != "Wxpos_total"]
        wn = [c.replace("Wxpos_", "Wxneg_") for c in wx]
        p_pos = ss.Wxpos_total.to_numpy() / DT
        p_abs = (ss[wx].to_numpy().sum(1) + ss[wn].to_numpy().sum(1)) / DT
        # same work, but positive/negative split per 20 ms step instead of per
        # 5 ms substep: how much CoT+ depends on the time resolution chosen
        p_pos20 = np.clip(ss[[c.replace("Wxpos_", "Wx_") for c in wx]].to_numpy(), 0, None).sum(1) / DT
    else:
        p_pos, p_abs = ss.P_pos.to_numpy(), ss.P_abs.to_numpy()
        p_pos20 = np.full(len(ss), np.nan)
    r = dict(variant=variant, terrain=t, mode=mode, cmd=m["cmd_vx"], seed=m["seed"],
             n_resets=n_resets, mass_sim=m_sim, mass_used=m_use, exact_work=exact,
             CoT_pos=float(np.mean(p_pos / denom)), CoT_abs=float(np.mean(p_abs / denom)),
             CoT_pos_50hz=float(np.mean(ss.P_pos.to_numpy() / denom)),
             CoT_pos_20ms=float(np.mean(p_pos20 / denom)),
             P_pos_W=float(p_pos.mean()),
             v=float(ss.v_body_x.mean()), roll_deg=float(np.degrees(ss.roll.abs().mean())))
    if "straight_dev_m" in ss and mode == "openloop" and t != "weave":
        dev = ss.straight_dev_m.to_numpy(); plen = ss.path_len_m.to_numpy()
        trav = plen[-1] - plen[0]
        r["dev_per_m"] = float(abs(dev[-1] - dev[0]) / trav) if trav > 1e-6 else np.nan
    if t == "curb":
        strad = np.abs(y) < 0.04
        r["straddle"] = float(strad.mean())
        r["roll_straddling_deg"] = float(np.degrees(ss.roll.abs()[strad].mean())) if strad.any() else np.nan
    if t == "weave" and "weave_target_y" in ss:
        r["weave_corr"], r["weave_amp_ratio"] = weave_track(ss)
        r["weave_xtrack_mm"] = float(ss.weave_cross_track_m.abs().mean() * 1000)
    if t == "steps":
        on = (x >= X0) & (x <= X1) & (np.abs(y) <= Y_HALF)
        r["on_field"] = float(on.mean())
        prog = float(np.maximum.accumulate(x)[-1] - x[0])
        dur = float(ss.t.iloc[-1] - ss.t.iloc[0])
        net = float(x[-1] - x[0])
        energy = float(p_pos.sum() * DT)
        r.update(progress_m=prog, pace=prog / (m["cmd_vx"] * dur) if dur > 0 else np.nan,
                 xtrack_rms_mm=float(np.sqrt(np.mean(y ** 2)) * 1000),
                 xtrack_max_mm=float(np.abs(y).max() * 1000),
                 CoT_progress=energy / (m_use * G * net) if net > 0.05 else np.nan)
    return r


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, keep = [], {}
    for v in ["rigid", "pitch", "yaw", "roll"]:
        for d in sorted(glob.glob(f"rollouts/{v}_*"), key=os.path.getmtime):
            m0 = meta0(d)
            if not m0 or not m0.get("raw_action", False):
                continue                           # pre-fix clipped-action sweep
            for mf in glob.glob(os.path.join(d, "rollout_*.meta.json")):
                m = json.loads(open(mf).read())
                if m["cmd_vx"] not in SPEEDS:
                    continue
                if m.get("terrain") == "weave" and (m.get("terrain_kw") or {}).get("spacing") != 0.80:
                    continue                       # development runs at other pitches
                if m.get("terrain") == "curb" and m.get("lane_keep"):
                    continue                       # early curb lane-keep tests
                key = (v, m.get("terrain", "flat"), bool(m.get("lane_keep")), m["cmd_vx"], m["seed"])
                keep[key] = (mf.replace(".meta.json", ".csv"), m, v)   # later dirs win
    for csv, m, v in keep.values():
        r = summarise(csv, m, v)
        if r:
            rows.append(r)
    df = pd.DataFrame(rows).sort_values(["variant", "terrain", "mode", "cmd", "seed"])
    df.to_csv(f"{OUT}/tidy_rollouts.csv", index=False)
    print(f"{len(df)} rollouts -> {OUT}/tidy_rollouts.csv")
    print(df.groupby(["mode", "terrain", "variant"]).size().unstack().to_string())


if __name__ == "__main__":
    main()
