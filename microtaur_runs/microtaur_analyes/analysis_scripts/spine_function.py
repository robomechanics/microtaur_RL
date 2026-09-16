"""What is each spine doing, terrain by terrain: spring, damper, brake or motor?

    python scripts/spine_function.py     -> rollouts/spine_function/{per_run.csv, cells.csv, loops.npz}

Needs rollouts logged with --substep-log (exact joint work at the 5 ms physics
step; see run_exact_work_sweep.sh for why the 50 Hz tau*qd sample is not used).

The actuator
------------
Every spine is the same actuator: a PD position servo (kp = 1.0 N m/rad,
kd = 0.045 N m s/rad) chasing a target the policy sets once per 20 ms. At the
physics step its torque is, to R^2 = 0.9999,

    tau = kp (tgt - q) - kd qd            (q, qd at the start of the substep)

so the work of each substep, tau * dq, splits exactly into

    spring  -kp (q - q0) dq      stored and given back (q0 = mean spine angle)
    damper  -kd qd dq            always <= 0: dissipated by the servo's damping
    policy   kp (tgt - q0) dq    work done by the policy moving the target:
                                 > 0 drives the joint, < 0 absorbs
    clip    remainder            torque-speed saturation

Joint Coulomb friction (0.010 N m, outside the actuator) adds 0.010 |dq|.

Role (at the stride scale, from exact work per 20 ms control step)
------------------------------------------------------------------
    W+ / W-  positive / negative work of the spine, per second
    returned = min(W+, W-) / max(W+, W-)   share of the energy flow that comes back

    spring   returned >= 0.5    it gives back at least half of what it takes
    motor    net > 0            otherwise, and the spine puts energy in
    damper   net < 0            and the torque is a passive spring-damper law
                                (impedance fit R^2 >= 0.5, damping > 0)
    brake    net < 0            but not a passive law: the policy shapes the
                                absorption (fit R^2 < 0.5)

Samples: first episode only, t >= 1 s, and only while the robot is doing the
test -- straddling the lip (curb), over the tiles (steps, waypoint-steered).
"""
import glob, json, os
import numpy as np, pandas as pd

KP, KD, FRICTION = 1.0, 0.045, 0.010
EFFORT = {"pitch": 0.312, "yaw": 0.129, "roll": 0.129}          # N m, safe effort limit
HARD_STOP = {"pitch": np.radians(30.0), "yaw": 0.35, "roll": 0.35}
MEASURED_MASS_KG = {"yaw": 0.4577}
SPEEDS = (0.08, 0.14, 0.20)
SETTLE, DT, G = 1.0, 0.02, 9.81
X0, X1, Y_HALF = 0.175, 3.775, 0.60
NPH = 48                                                         # phase bins per stride
OUT = "rollouts/spine_function"
MODE = {"flat": False, "curb": False, "weave": False, "steps": True}   # lane_keep per terrain
VARIANTS = ["pitch", "yaw", "roll"]


def first_episode(df):
    rs = df.index[(df["done"] > 0) & (df.index < len(df) - 1)]
    return df.loc[: rs[0] - 1] if len(rs) else df


def runs(variants=VARIANTS):
    keep = {}
    for v in variants:
        for d in sorted(glob.glob(f"rollouts/{v}_*"), key=os.path.getmtime):
            for mf in glob.glob(os.path.join(d, "rollout_*.meta.json")):
                m = json.loads(open(mf).read())
                t = m.get("terrain", "flat")
                if not m.get("substep_log") or m["cmd_vx"] not in SPEEDS or t not in MODE:
                    continue
                if bool(m.get("lane_keep")) != MODE[t]:
                    continue
                if t == "weave" and (m.get("terrain_kw") or {}).get("spacing") != 0.55:
                    continue
                keep[(v, t, m["cmd_vx"], m["seed"])] = (mf.replace(".meta.json", ".csv"), m)
    return keep


def test_mask(ss, terrain):
    if terrain == "curb":
        return np.abs(ss.base_y.to_numpy()) < 0.04
    if terrain == "steps":
        x, y = ss.base_x.to_numpy(), ss.base_y.to_numpy()
        return (x >= X0) & (x <= X1) & (np.abs(y) <= Y_HALF)
    return np.ones(len(ss), bool)


def strides(ss, mask):
    """Leg-2 touchdowns (after >= 2 airborne samples) bounding fully-masked strides."""
    c = ss.foot_contact_1.to_numpy() > 0.5
    td = [i for i in range(2, len(c)) if c[i] and not c[i - 1] and not c[i - 2]]
    return [(a, b) for a, b in zip(td[:-1], td[1:]) if 8 <= b - a <= 25 and mask[a:b + 1].all()]


def resample(x, a, b):
    return np.interp(np.linspace(a, b, NPH, endpoint=False), np.arange(a, b + 1), x[a:b + 1])


def analyse(v, terrain, cmd, seed, csv, meta):
    df = first_episode(pd.read_csv(csv))
    ss = df[(df.t >= SETTLE) & (df.done == 0)]
    mask = test_mask(ss, terrain)
    if mask.sum() < 50:
        return None, None
    S = np.load(csv.replace(".csv", ".substep.npz"))
    dt = float(S["physics_dt"])
    nsub = int(round(DT / dt))
    k = S["k"]
    # substeps belonging to the analysed (masked) policy steps
    steps_ok = ss.step.to_numpy()[mask]
    sm = np.isin(k, steps_ok)
    q, qd, tau, dq, tgt = S["q"][:, 8], S["qd"][:, 8], S["tau"][:, 8], S["dq"][:, 8], S["tgt"][:, 8]
    q_pre = q - dq
    qd_pre = np.r_[qd[0], qd[:-1]]
    q0 = q[sm].mean()
    w = tau * dq                                        # exact work per substep [J]
    w_spring = -KP * (q_pre - q0) * dq
    w_damper = -KD * qd_pre * dq
    w_policy = KP * (tgt - q0) * dq
    w_clip = w - (w_spring + w_damper + w_policy)
    T = sm.sum() * dt
    # stride-scale: exact work per 20 ms control step
    Wx = ss.Wx_spine.to_numpy()[mask]
    Wp20, Wn20 = np.clip(Wx, 0, None).sum() / T, np.clip(-Wx, 0, None).sum() / T
    # impedance fit at the physics step (passive spring-damper about q0)
    X = np.c_[-(q_pre[sm] - q0), -qd_pre[sm], np.ones(sm.sum())]
    coef, *_ = np.linalg.lstsq(X, tau[sm], rcond=None)
    r2 = lambda a, b: 1 - np.var(a - b) / np.var(a)
    # whole-robot exact positive work -> CoT+ (measured mass where one exists)
    m_sim = float(meta["mass_kg"]); m_use = MEASURED_MASS_KG.get(v, m_sim)
    Pp_robot = ss.Wxpos_total.to_numpy()[mask].sum() / T
    Pabs_legs = sum(ss[f"Wxpos_{j}"].to_numpy()[mask].sum() + ss[f"Wxneg_{j}"].to_numpy()[mask].sum()
                    for j in [c[6:] for c in ss.columns if c.startswith("Wxpos_leg")]) / T
    vbar = max(ss.v_body_x.to_numpy()[mask].mean(), 0.02)
    st = strides(ss, mask)
    Wst = np.array([ss.Wx_spine.to_numpy()[a:b].sum() for a, b in st])
    row = dict(variant=v, terrain=terrain, cmd=cmd, seed=seed, n_steps=int(mask.sum()),
               n_strides=len(st), stride_s=float(np.median([b - a for a, b in st]) * DT) if st else np.nan,
               net=w[sm].sum() / T, W_pos20=Wp20, W_neg20=Wn20,
               W_pos_sub=np.clip(w[sm], 0, None).sum() / T, W_neg_sub=np.clip(-w[sm], 0, None).sum() / T,
               spring=w_spring[sm].sum() / T, damper=w_damper[sm].sum() / T,
               policy=w_policy[sm].sum() / T, clip=w_clip[sm].sum() / T,
               friction=FRICTION * np.abs(dq[sm]).sum() / T,
               k_eff=coef[0], c_eff=coef[1], imp_r2=r2(tau[sm], X @ coef),
               q_mean_deg=np.degrees(q0),
               q_p2p_deg=np.degrees(np.percentile(q[sm], 95) - np.percentile(q[sm], 5)),
               tau_mean=tau[sm].mean(), tau_rms=np.sqrt(np.mean(tau[sm] ** 2)),
               sat_frac=float((np.abs(tau[sm]) >= 0.98 * EFFORT[v]).mean()),
               stop_frac=float((np.abs(q[sm]) >= HARD_STOP[v] - np.radians(1.0)).mean()),
               stride_net_mJ=float(Wst.mean() * 1e3) if len(Wst) else np.nan,
               robot_Ppos=Pp_robot, legs_Pabs=Pabs_legs, v=vbar,
               CoT_exact=Pp_robot / (m_use * G * vbar),
               CoT_50hz=ss.P_pos.to_numpy()[mask].mean() / (m_use * G * vbar))
    loop = None
    if st:
        # phase-averaged angle, torque, power, velocity. Angle, torque and velocity
        # are boxcar-averaged over one control step (nsub substeps): the policy
        # moves the target in 20 ms jumps, which leaves a sawtooth in the torque
        # (and in qd) that would otherwise hide the stride-scale loop. (The
        # per-stride work quoted is exact.)
        box = np.ones(nsub) / nsub
        qs = np.convolve(q, box, mode="same")
        ts = np.convolve(tau, box, mode="same")
        vs = np.convolve(qd, box, mode="same")
        per = []
        for a, b in st:
            s0, s1 = np.searchsorted(k, ss.step.iloc[a]), np.searchsorted(k, ss.step.iloc[b])
            qq, tt = np.degrees(qs[s0:s1 + 1] - q0), ts[s0:s1 + 1] * 1e3
            vv = np.degrees(vs[s0:s1 + 1])
            pp = ss.Wx_spine.to_numpy()[a:b + 1] / DT * 1e3
            ph = np.linspace(0, 1, NPH, endpoint=False)
            src = np.linspace(0, 1, len(qq))
            per.append(np.stack([np.interp(ph, src, qq),
                                 np.interp(ph, src, tt),
                                 np.interp(ph, np.linspace(0, 1, len(pp)), pp),
                                 np.interp(ph, src, vv)]))
        loop = np.stack(per).mean(0)                     # (4, NPH): q deg, tau mN m, P mW, qd deg/s
    return row, loop


def role(c):
    ret = min(c.W_pos20, c.W_neg20) / max(c.W_pos20, c.W_neg20)
    if ret >= 0.5:
        return "spring"
    if c.net > 0:
        return "motor"
    return "damper" if (c.imp_r2 >= 0.5 and c.c_eff > 0) else "brake"


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, loops = [], {}
    for (v, t, cmd, seed), (csv, meta) in sorted(runs().items()):
        r, lp = analyse(v, t, cmd, seed, csv, meta)
        if r is None:
            continue
        rows.append(r)
        if lp is not None:
            loops.setdefault(f"{v}|{t}|{cmd:.2f}", []).append(lp)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/per_run.csv", index=False)
    np.savez(f"{OUT}/loops.npz", **{k: np.stack(x) for k, x in loops.items()})
    g = df.groupby(["variant", "terrain", "cmd"])
    cells = g.mean(numeric_only=True).drop(columns="seed")
    cells["n_runs"] = g.size()
    cells["returned"] = (np.minimum(cells.W_pos20, cells.W_neg20) / np.maximum(cells.W_pos20, cells.W_neg20))
    cells["role"] = [role(c) for c in cells.itertuples()]
    cells["role_votes"] = [
        ",".join(f"{k}:{n}" for k, n in df[(df.variant == v) & (df.terrain == t) & (df.cmd == s)]
                 .apply(role, axis=1).value_counts().items())
        for v, t, s in cells.index]
    cells.to_csv(f"{OUT}/cells.csv")
    mw = ["net", "W_pos20", "W_neg20", "W_pos_sub", "W_neg_sub", "spring", "damper", "policy", "clip", "friction"]
    show = cells.copy()
    show[mw] = show[mw] * 1e3
    pd.set_option("display.width", 260)
    print(show[mw + ["returned", "k_eff", "c_eff", "imp_r2", "q_mean_deg", "q_p2p_deg", "sat_frac",
                     "stride_net_mJ", "CoT_exact", "CoT_50hz", "n_runs", "role", "role_votes"]].round(3).to_string())


if __name__ == "__main__":
    main()
