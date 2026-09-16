"""Part 3: spine function on flat (trained equilibrium) + target-interpretability
regression (flat, weave) + spine-angle PSD (yaw, flat vs weave).

    python scripts/spine_part3.py
      -> rollouts/flat_weave/spine_function_flat.csv
      -> rollouts/flat_weave/target_regression.csv
      -> rollouts/flat_weave/roll_damper_lag.csv
      -> rollouts/flat_weave/fig_spine_psd_{light,dark}.png

Reuses scripts/spine_function.py's analyse()/strides()/first_episode() (exact
5ms-substep energy decomposition, kp=1.0/kd=0.045 spring+damper+policy+clip
split) rather than reimplementing it. Deviates from spine_function.runs() in
ONE place: weave spacing is 0.80 here (canonical per this project's other
scripts and the coordinator's explicit correction), not spine_function.py's
own hardcoded 0.55.
"""
import glob, json, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import spine_function as sf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

OUT = "rollouts/flat_weave"
os.makedirs(OUT, exist_ok=True)
VARIANTS = ["pitch", "yaw", "roll"]
SPEEDS = (0.08, 0.14, 0.20)
WEAVE_SPACING = 0.80
STEER_GAIN, STEER_SIGN = 0.80, 1.0  # yaw's architected bias term, see env_cfgs.py


def build_index(weave_spacing=WEAVE_SPACING):
    """Like spine_function.runs() but flat+weave only, weave spacing=0.80."""
    keep = {}
    for v in VARIANTS:
        for d in sorted(glob.glob(f"rollouts/{v}_*"), key=os.path.getmtime):
            for mf in glob.glob(os.path.join(d, "rollout_*.meta.json")):
                m = json.loads(open(mf).read())
                t = m.get("terrain", "flat")
                if t not in ("flat", "weave") or not m.get("substep_log") or m["cmd_vx"] not in SPEEDS:
                    continue
                if bool(m.get("lane_keep")):
                    continue
                if t == "weave" and (m.get("terrain_kw") or {}).get("spacing") != weave_spacing:
                    continue
                if not m.get("raw_action", False):
                    continue
                keep[(v, t, m["cmd_vx"], m["seed"])] = (mf.replace(".meta.json", ".csv"), m)
    return keep


# --------------------------------------------------------------------------- #
# Energy / impedance table, flat only
# --------------------------------------------------------------------------- #

def spine_function_flat(idx):
    rows = []
    for v in VARIANTS:
        for cmd in SPEEDS:
            per_seed, raw_stats = [], []
            for seed in range(8):
                key = (v, "flat", cmd, seed)
                if key not in idx:
                    continue
                csv, meta = idx[key]
                r, _loop = sf.analyse(v, "flat", cmd, seed, csv, meta)
                if r is None:
                    continue
                per_seed.append(r)
                df_raw = sf.first_episode(pd.read_csv(csv))
                ss = df_raw[(df_raw.t >= sf.SETTLE) & (df_raw.done == 0)]
                q_deg = np.degrees(ss.q_spine.to_numpy())
                qd_deg = np.degrees(ss.qd_spine.to_numpy())
                raw_stats.append((q_deg.max() - q_deg.min(), np.sqrt(np.mean((q_deg - q_deg.mean()) ** 2)),
                                  np.abs(qd_deg).max()))
            if not per_seed:
                print(f"[part3] WARNING no flat data for {v} {cmd}")
                continue
            df = pd.DataFrame(per_seed)
            rom, rms, peak = np.array(raw_stats).T
            sum_abs_w = (df.spring.abs() + df.damper.abs() + df.policy.abs() + df["clip"].abs())
            returned = np.minimum(df.W_pos20, df.W_neg20) / np.maximum(df.W_pos20, df.W_neg20)
            # denominator for "spine work as % of total mechanical work", stated
            # explicitly per the brief: robot's total positive mechanical power
            # (robot_Ppos, exact-work, same denominator for every variant/speed).
            spine_work_pct_of_robot = (df.W_pos20 + df.W_neg20) / df.robot_Ppos * 100
            rows.append(dict(
                variant=v, speed=cmd, n_seeds=len(per_seed),
                returned=float(returned.mean()),
                net_work=float(df.net.mean()),
                policy_work_frac=float((df.policy.abs() / sum_abs_w).mean()),
                clip_fraction=float((df["clip"].abs() / sum_abs_w).mean()),
                k_eff=float(df.k_eff.mean()), c_eff=float(df.c_eff.mean()), R2=float(df.imp_r2.mean()),
                ROM_deg=float(rom.mean()), rms_deg=float(rms.mean()), peak_rate_dps=float(peak.mean()),
                spine_work_pct_of_robot=float(spine_work_pct_of_robot.mean()),
            ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Target regression
# --------------------------------------------------------------------------- #

def load_episode_features(csv, meta, variant, terrain):
    df = sf.first_episode(pd.read_csv(csv))
    ss = df[(df.t >= sf.SETTLE) & (df.done == 0)]
    if len(ss) < 50:
        return None
    mask = sf.test_mask(ss, terrain)
    if mask.sum() < 50:
        return None
    st = sf.strides(ss, mask)
    if not st:
        return None
    n = len(ss)
    phase = np.full(n, np.nan)
    idx_arr = np.arange(n)
    for a, b in st:
        seg = idx_arr[a:b + 1]
        phase[seg] = np.linspace(0, 1, len(seg), endpoint=False)
    contact_cols = [c for c in ss.columns if c.startswith("foot_contact_")]
    stance = ss[contact_cols].sum(axis=1).to_numpy() if contact_cols else np.zeros(n)
    out = pd.DataFrame({
        "phase_sin": np.sin(2 * np.pi * phase), "phase_cos": np.cos(2 * np.pi * phase),
        "cmd_yaw": ss.cmd_yaw.to_numpy() if "cmd_yaw" in ss else np.zeros(n),
        "cmd_vx": ss.cmd_vx.to_numpy() if "cmd_vx" in ss else np.full(n, meta["cmd_vx"]),
        "roll": ss.roll.to_numpy(), "pitch": ss.pitch.to_numpy(),
        "wx": ss.wx.to_numpy(), "wy": ss.wy.to_numpy(),
        "stance": stance,
        "q_spine": ss.q_spine.to_numpy(), "qd_spine": ss.qd_spine.to_numpy(),
        "target": ss.applied_target_spine.to_numpy(),
        "t": ss.t.to_numpy(),
    }, index=ss.index).dropna()
    return out


PREDICTORS = ["phase_sin", "phase_cos", "cmd_yaw", "cmd_vx", "roll", "pitch", "wx", "wy",
             "stance", "q_spine", "qd_spine"]


def standardized_regression(pooled, seeds, predictors=PREDICTORS, outcome="target", n_boot=500, rng=None):
    rng = rng or np.random.default_rng(0)
    X = pooled[predictors].to_numpy()
    y = pooled[outcome].to_numpy()
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd
    ys = (y - y.mean()) / (y.std() if y.std() > 1e-9 else 1.0)
    Xd = np.column_stack([np.ones(len(Xs)), Xs])
    coef, *_ = np.linalg.lstsq(Xd, ys, rcond=None)
    pred = Xd @ coef
    r2 = 1 - np.sum((ys - pred) ** 2) / np.sum((ys - ys.mean()) ** 2)

    seed_ids = pooled["_seed"].to_numpy()
    uniq = np.unique(seed_ids)
    boots = np.empty((n_boot, len(predictors)))
    for i in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([np.where(seed_ids == s)[0] for s in pick])
        b, *_ = np.linalg.lstsq(Xd[rows], ys[rows], rcond=None)
        boots[i] = b[1:]
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return dict(model_R2=float(r2), std_coef=coef[1:], ci_lo=ci_lo, ci_hi=ci_hi, n_rows=len(Xs), n_seeds=len(uniq))


def roll_damper_lag_test(pooled, rng=None):
    """tgt ~ a + b*q + c*qdot, UNSTANDARDIZED (physical units), bootstrap by seed."""
    rng = rng or np.random.default_rng(1)
    X = np.column_stack([np.ones(len(pooled)), pooled.q_spine.to_numpy(), pooled.qd_spine.to_numpy()])
    y = pooled.target.to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    seed_ids = pooled["_seed"].to_numpy(); uniq = np.unique(seed_ids)
    boots = np.empty((500, 3))
    for i in range(500):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([np.where(seed_ids == s)[0] for s in pick])
        b, *_ = np.linalg.lstsq(X[rows], y[rows], rcond=None)
        boots[i] = b
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return dict(a=coef[0], b=coef[1], c=coef[2], a_ci=(lo[0], hi[0]), b_ci=(lo[1], hi[1]), c_ci=(lo[2], hi[2]), R2=r2)


def run_regressions(idx):
    reg_rows, lag_rows = [], []
    for v in VARIANTS:
        for terrain in ("flat", "weave"):
            frames = []
            for cmd in SPEEDS:
                for seed in range(8):
                    key = (v, terrain, cmd, seed)
                    if key not in idx:
                        continue
                    csv, meta = idx[key]
                    feat = load_episode_features(csv, meta, v, terrain)
                    if feat is None:
                        continue
                    feat["_seed"] = seed
                    frames.append(feat)
            if not frames:
                print(f"[part3] WARNING no regression data for {v}/{terrain}")
                continue
            pooled = pd.concat(frames, ignore_index=True)
            res = standardized_regression(pooled, None)
            for p, c, lo, hi in zip(PREDICTORS, res["std_coef"], res["ci_lo"], res["ci_hi"]):
                reg_rows.append(dict(variant=v, terrain=terrain, predictor=p, std_coef=c,
                                     ci_low=lo, ci_high=hi, model_R2=res["model_R2"],
                                     n_rows=res["n_rows"], n_seeds=res["n_seeds"]))

            # yaw architecture: also regress the RESIDUAL (target minus the known
            # hard-coded bias = steer_sign*steer_gain*cmd_yaw) so the policy's own
            # contribution isn't confounded with the architected feedforward term.
            if v == "yaw":
                resid = pooled.copy()
                resid["target"] = resid["target"] - STEER_SIGN * STEER_GAIN * resid["cmd_yaw"]
                res_r = standardized_regression(resid, None)
                for p, c, lo, hi in zip(PREDICTORS, res_r["std_coef"], res_r["ci_lo"], res_r["ci_hi"]):
                    reg_rows.append(dict(variant="yaw_residual_only", terrain=terrain, predictor=p,
                                         std_coef=c, ci_low=lo, ci_high=hi, model_R2=res_r["model_R2"],
                                         n_rows=res_r["n_rows"], n_seeds=res_r["n_seeds"]))

            lag = roll_damper_lag_test(pooled)
            lag_rows.append(dict(variant=v, terrain=terrain, a=lag["a"], b=lag["b"], c=lag["c"],
                                 a_ci_lo=lag["a_ci"][0], a_ci_hi=lag["a_ci"][1],
                                 b_ci_lo=lag["b_ci"][0], b_ci_hi=lag["b_ci"][1],
                                 c_ci_lo=lag["c_ci"][0], c_ci_hi=lag["c_ci"][1], R2=lag["R2"],
                                 n_rows=len(pooled)))
    return pd.DataFrame(reg_rows), pd.DataFrame(lag_rows)


# --------------------------------------------------------------------------- #
# PSD figure
# --------------------------------------------------------------------------- #

VCOL = {"light": {"pitch": "#bd6f1c", "yaw": "#2f8a61", "roll": "#8f4bbf"},
        "dark":  {"pitch": "#e0954c", "yaw": "#45b585", "roll": "#b78fd6"}}
TH = {"light": dict(bg="#ffffff", ink="#131d25", ink2="#586a76", grid="#d9e0e5"),
      "dark":  dict(bg="#141d21", ink="#e3ebef", ink2="#90a3ad", grid="#26343b")}


def psd_figure(idx):
    fs = 50.0
    for mode, t in TH.items():
        fig, axes = plt.subplots(1, len(VARIANTS), figsize=(15, 4.2), facecolor=t["bg"])
        for ax, v in zip(axes, VARIANTS):
            ax.set_facecolor(t["bg"])
            for terr, ls in (("flat", "-"), ("weave", "--")):
                key = (v, terr, 0.20, 0)
                if key not in idx:
                    continue
                csv, meta = idx[key]
                df = sf.first_episode(pd.read_csv(csv))
                ss = df[(df.t >= sf.SETTLE) & (df.done == 0)]
                q = ss.q_spine.to_numpy()
                if len(q) < 64:
                    continue
                f, pxx = signal.welch(q - q.mean(), fs=fs, nperseg=min(256, len(q)))
                ax.semilogy(f, pxx, ls, color=VCOL[mode][v], lw=1.6, label=terr)
                st = sf.strides(ss, np.ones(len(ss), bool))
                if st:
                    stride_s = np.median([b - a for a, b in st]) * sf.DT
                    ax.axvline(1 / stride_s, color=t["ink2"], lw=0.7, ls=":", alpha=0.7)
                if terr == "weave":
                    slalom_hz = 0.20 / (2 * WEAVE_SPACING)
                    ax.axvline(slalom_hz, color=VCOL[mode][v], lw=1.0, ls="-.", alpha=0.8)
            ax.set_title(v, color=t["ink"]); ax.set_xlim(0, 8)
            ax.set_xlabel("Hz", color=t["ink2"]); ax.tick_params(colors=t["ink2"])
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ax.spines.values():
                s.set_color(t["grid"])
            ax.legend(fontsize=7, frameon=False)
        axes[0].set_ylabel("spine angle PSD [rad^2/Hz]", color=t["ink"])
        fig.suptitle("Spine-angle PSD, flat vs weave, vx=0.20 m/s, seed 0 "
                     "(dotted = gait freq, dash-dot = slalom freq, weave spacing=0.80)",
                     color=t["ink"], fontsize=10.5)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        p = f"{OUT}/fig_spine_psd_{mode}.png"
        fig.savefig(p, dpi=130, facecolor=t["bg"])
        plt.close(fig)
        print("wrote", p)


def main():
    idx = build_index()
    print(f"[part3] index: {len(idx)} files "
          f"(flat={sum(1 for k in idx if k[1]=='flat')}, weave@0.80={sum(1 for k in idx if k[1]=='weave')})")

    sff = spine_function_flat(idx)
    sff.to_csv(f"{OUT}/spine_function_flat.csv", index=False)
    print(f"wrote {OUT}/spine_function_flat.csv ({len(sff)} rows)")
    print(sff.round(4).to_string(index=False))

    reg, lag = run_regressions(idx)
    reg.to_csv(f"{OUT}/target_regression.csv", index=False)
    lag.to_csv(f"{OUT}/roll_damper_lag.csv", index=False)
    print(f"\nwrote {OUT}/target_regression.csv ({len(reg)} rows)")
    print(f"wrote {OUT}/roll_damper_lag.csv ({len(lag)} rows)")
    print("\n-- damper-lag test (tgt ~ a + b*q + c*qdot), unstandardized --")
    print(lag.round(4).to_string(index=False))

    psd_figure(idx)


if __name__ == "__main__":
    main()
