"""Flat (equivalence) + weave (in-distribution capability) analysis for the
morphology paper. Single entry point:

    python scripts/paper_flat_weave_analysis.py

Outputs -> rollouts/paper_flat_weave/{*.csv, fig_*_{light,dark}.png}
"""
import glob, json, os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(0)
VARIANTS = ["rigid", "pitch", "yaw", "roll"]
SPINE_VARIANTS = ["pitch", "yaw", "roll"]
SPEEDS = (0.08, 0.14, 0.20)
SETTLE = 1.0
BL_M = 0.168
G = 9.81
KP, KD = 1.0, 0.045
MEASURED_MASS_KG = {"yaw": 0.4577}
OUT = "rollouts/paper_flat_weave"
VCOL = {"rigid": "#2a78d6", "pitch": "#eb6834", "yaw": "#1baf7a", "roll": "#8f4bbf"}
VCOL_DARK = {"rigid": "#3987e5", "pitch": "#d95926", "yaw": "#199e70", "roll": "#b78fd6"}
THEME = {"light": dict(vcol=VCOL, surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#d9dbd6"),
         "dark": dict(vcol=VCOL_DARK, surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#3a3a38")}

FOOT_CHANNELS = json.load(open("rollouts/foot_channels.json")) if os.path.exists("rollouts/foot_channels.json") else {}


# --------------------------------------------------------------------------- #
# canonical index -- mirrors resummarize.py / poincare_stability.py exactly
# --------------------------------------------------------------------------- #
def build_index(terrain):
    keep = {}
    for v in VARIANTS:
        for d in sorted(glob.glob(f"rollouts/{v}_{terrain}_*"), key=os.path.getmtime):
            mfs = sorted(glob.glob(os.path.join(d, "rollout_*.meta.json")))
            if not mfs:
                continue
            m0 = json.loads(open(mfs[0]).read())
            if not m0.get("raw_action", False):
                continue
            for mf in mfs:
                m = json.loads(open(mf).read())
                if m["cmd_vx"] not in SPEEDS:
                    continue
                if terrain == "weave" and (m.get("terrain_kw") or {}).get("spacing") != 0.80:
                    continue
                key = (v, m["cmd_vx"], m["seed"])
                keep[key] = (mf.replace(".meta.json", ".csv"), m)
    return keep


def first_episode(df):
    rs = df.index[(df["done"] > 0) & (df.index < len(df) - 1)]
    n_resets = int(len(rs))
    return (df.loc[: rs[0] - 1] if len(rs) else df), n_resets


def fl_channel(variant):
    return (FOOT_CHANNELS.get(variant, {}) or {}).get("FL", 2 if variant != "roll" else 0)


# --------------------------------------------------------------------------- #
# per-run metrics
# --------------------------------------------------------------------------- #
POLE_SPACING, N_POLES, LEAD_IN, COURSE_LX = 0.80, 5, 0.40, 4.0


def pole_positions():
    course_len = (N_POLES - 1) * POLE_SPACING
    x_spawn = min(0.25, max(0.0, COURSE_LX - course_len - LEAD_IN))
    x0 = x_spawn + LEAD_IN
    return [x0 + k * POLE_SPACING for k in range(N_POLES)]


def per_run(variant, terrain, cmd, seed, csv, meta):
    df = pd.read_csv(csv)
    ep, n_resets = first_episode(df)
    ss = ep[ep.t >= SETTLE]
    if len(ss) < 10:
        return None
    m_sim = float(meta.get("mass_kg", np.nan))
    m_use = MEASURED_MASS_KG.get(variant, m_sim)
    exact = "Wxpos_total" in ss.columns
    v = ss.v_body_x.to_numpy()
    denom = m_use * G * np.maximum(np.abs(v), 0.02)
    if exact:
        p_pos = ss.Wxpos_total.to_numpy() / 0.02
    else:
        p_pos = ss.P_pos.to_numpy()
    cot = float(np.mean(p_pos / denom))
    fell = bool(ep.done.iloc[-1] > 0) if len(ep) else False
    row = dict(variant=variant, terrain=terrain, cmd=cmd, seed=seed, n=len(ss),
               n_resets=n_resets, fell=fell, mass_used=m_use, exact_work=exact,
               CoT=cot, v_achieved=float(v.mean()),
               v_shortfall_pct=float(100 * (cmd - v.mean()) / cmd),
               v_track_err=float(np.mean(np.abs(v - cmd))),
               roll_rms_deg=float(np.degrees(np.sqrt(np.mean(ss.roll.to_numpy() ** 2)))),
               pitch_rms_deg=float(np.degrees(np.sqrt(np.mean(ss.pitch.to_numpy() ** 2)))),
               roll_peak_rate_deg_s=float(np.degrees(np.abs(ss.wx.to_numpy()).max())),
               pitch_peak_rate_deg_s=float(np.degrees(np.abs(ss.wy.to_numpy()).max())))
    if terrain == "weave" and "weave_cross_track_m" in ss.columns:
        xt_mm = ss.weave_cross_track_m.to_numpy() * 1000
        row.update(xtrack_median_mm=float(np.median(np.abs(xt_mm))),
                   xtrack_q1_mm=float(np.percentile(np.abs(xt_mm), 25)),
                   xtrack_q3_mm=float(np.percentile(np.abs(xt_mm), 75)),
                   xtrack_iqr_mm=float(np.percentile(np.abs(xt_mm), 75) - np.percentile(np.abs(xt_mm), 25)),
                   xtrack_rms_mm=float(np.sqrt(np.mean(xt_mm ** 2))))
        yaw_err = np.abs(ss.wz.to_numpy() - ep.cmd_yaw.reindex(ss.index).to_numpy())
        row["yaw_rate_track_err"] = float(np.mean(yaw_err))
        if cmd == 0.20:
            x = ss.base_x.to_numpy()
            poles = pole_positions()
            row["poles_cleared"] = int(sum(1 for p in poles if x[-1] >= p))
    if variant in SPINE_VARIANTS and "q_spine" in ss.columns:
        q, qd = ss.q_spine.to_numpy(), ss.qd_spine.to_numpy()
        tgt = ss.applied_target_spine.to_numpy() if "applied_target_spine" in ss.columns else ss.action_raw_spine.to_numpy()
        q0 = q.mean()
        if exact and "Wxpos_spine" in ss.columns and "Wx_spine" in ss.columns:
            wpos = ss.Wxpos_spine.to_numpy().sum()
            wneg = wpos - ss.Wx_spine.to_numpy().sum()
        else:
            w = (ss.tau_spine.to_numpy() * qd) if "tau_spine" in ss.columns else np.zeros_like(q)
            wpos, wneg = np.clip(w, 0, None).sum(), np.clip(-w, 0, None).sum()
        returned = min(wpos, wneg) / max(wpos, wneg, 1e-12)
        net_work = wpos - wneg
        X = np.c_[-(q - q0), -qd, np.ones_like(q)]
        tau = ss.tau_spine.to_numpy() if "tau_spine" in ss.columns else np.zeros_like(q)
        coef, *_ = np.linalg.lstsq(X, tau, rcond=None)
        pred = X @ coef
        r2 = 1 - np.var(tau - pred) / max(np.var(tau), 1e-12)
        total_pos_work = ss.Wxpos_total.to_numpy().sum() if exact else np.clip(p_pos, 0, None).sum() * 0.02
        row.update(spine_returned=float(returned), spine_net_work=float(net_work),
                   spine_k_eff=float(coef[0]), spine_c_eff=float(coef[1]), spine_imp_r2=float(r2),
                   spine_ROM=float(q.max() - q.min()), spine_rms=float(np.sqrt(np.mean(q ** 2))),
                   spine_peak_rate=float(np.abs(qd).max()),
                   spine_work_pct_of_total=float(100 * abs(wpos - wneg) / max(total_pos_work, 1e-12)),
                   q0=float(q0))
        row["_reg"] = dict(t=ss.t.to_numpy(), q=q, qd=qd, tgt=tgt,
                           roll=ss.roll.to_numpy(), pitch=ss.pitch.to_numpy(),
                           cmd_yaw=ep.cmd_yaw.reindex(ss.index).to_numpy(),
                           wx=ss.wx.to_numpy(), wy=ss.wy.to_numpy(),
                           fc=ss[f"foot_contact_{fl_channel(variant)}"].to_numpy(),
                           v=v)
    return row


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def paired_bootstrap_ci(diffs, n_boot=10000):
    diffs = np.asarray(diffs)
    boots = [np.mean(RNG.choice(diffs, size=len(diffs), replace=True)) for _ in range(n_boot)]
    return float(np.mean(diffs)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def holm(pvals):
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, i in enumerate(idx):
        val = min(1.0, (m - rank) * pvals[i])
        running_max = max(running_max, val)
        adj[i] = running_max
    return adj


def perm_pvalue(diffs, n_perm=10000):
    diffs = np.asarray(diffs)
    obs = np.abs(np.mean(diffs))
    signs = RNG.choice([-1, 1], size=(n_perm, len(diffs)))
    boot = np.abs((signs * diffs).mean(axis=1))
    return float((boot >= obs).mean())


def main():
    os.makedirs(OUT, exist_ok=True)
    all_rows = {"flat": [], "weave": []}
    reg_data = {"flat": {}, "weave": {}}
    for terrain in ("flat", "weave"):
        index = build_index(terrain)
        print(f"[{terrain}] canonical files: {len(index)} (expect {len(VARIANTS)*len(SPEEDS)*8}={len(VARIANTS)*3*8})")
        for (v, cmd, seed), (csv, meta) in sorted(index.items()):
            r = per_run(v, terrain, cmd, seed, csv, meta)
            if r is None:
                continue
            reg = r.pop("_reg", None)
            if reg is not None:
                reg_data[terrain].setdefault((v, cmd), []).append(reg)
            all_rows[terrain].append(r)

    flat_df = pd.DataFrame(all_rows["flat"])
    weave_df = pd.DataFrame(all_rows["weave"])
    flat_df.to_csv(f"{OUT}/flat_metrics.csv", index=False)
    weave_df.to_csv(f"{OUT}/weave_metrics.csv", index=False)
    print(f"flat n={len(flat_df)}, weave n={len(weave_df)}")

    # ---------------- PART 1: flat equivalence ----------------
    eq_rows = []
    metric_families = {
        "energetics": ["CoT"],
        "tracking": ["v_achieved", "v_track_err"],
        "stability": ["roll_rms_deg", "pitch_rms_deg", "roll_peak_rate_deg_s", "pitch_peak_rate_deg_s"],
    }
    for fam, metrics in metric_families.items():
        for metric in metrics:
            for cmd in SPEEDS:
                base = flat_df[(flat_df.variant == "rigid") & (flat_df.cmd == cmd)].sort_values("seed")
                pvals_this_metric_cmd = []
                tmp_rows = []
                for v in SPINE_VARIANTS:
                    sub = flat_df[(flat_df.variant == v) & (flat_df.cmd == cmd)].sort_values("seed")
                    common = sorted(set(base.seed) & set(sub.seed))
                    b = base.set_index("seed").loc[common, metric].to_numpy()
                    s = sub.set_index("seed").loc[common, metric].to_numpy()
                    diffs = s - b
                    mean_diff, lo, hi = paired_bootstrap_ci(diffs)
                    rigid_mean = float(b.mean())
                    pct = lambda x: 100 * x / rigid_mean if rigid_mean != 0 else np.nan
                    delta = cliffs_delta(s, b)
                    p = perm_pvalue(diffs)
                    pvals_this_metric_cmd.append(p)
                    tmp_rows.append(dict(variant=v, speed=cmd, metric_family=fam, metric=metric,
                                         mean_diff_vs_rigid=mean_diff, ci_low=lo, ci_high=hi,
                                         ci_low_pct_of_rigid=pct(lo), ci_high_pct_of_rigid=pct(hi),
                                         mean_diff_pct_of_rigid=pct(mean_diff),
                                         cliffs_delta=delta, p_raw=p, n=len(common),
                                         rigid_fall_rate=float(base.fell.mean()),
                                         variant_fall_rate=float(sub.fell.mean())))
                adj = holm(np.array(pvals_this_metric_cmd))
                for r, a in zip(tmp_rows, adj):
                    r["p_holm"] = a
                    eq_rows.append(r)
    eq_df = pd.DataFrame(eq_rows)
    eq_df.to_csv(f"{OUT}/equivalence_summary.csv", index=False)

    # roll peak-rate speed-gating check specifically
    print("\n=== Roll peak roll-rate advantage vs rigid, by speed (verification) ===")
    speedgate_rows = []
    for cmd in SPEEDS:
        r = eq_df[(eq_df.variant == "roll") & (eq_df.metric == "roll_peak_rate_deg_s") & (eq_df.speed == cmd)]
        roll_mean = flat_df[(flat_df.variant == "roll") & (flat_df.cmd == cmd)].roll_peak_rate_deg_s.mean()
        rigid_mean = flat_df[(flat_df.variant == "rigid") & (flat_df.cmd == cmd)].roll_peak_rate_deg_s.mean()
        print(f"  vx={cmd:.2f}: roll={roll_mean:.1f} deg/s, rigid={rigid_mean:.1f} deg/s, "
              f"diff CI=[{r.ci_low.iloc[0]:.1f},{r.ci_high.iloc[0]:.1f}]")
        speedgate_rows.append(dict(speed=cmd, roll_mean=roll_mean, rigid_mean=rigid_mean,
                                   advantage_deg_s=rigid_mean - roll_mean))
    sg = pd.DataFrame(speedgate_rows)
    # interaction test: is the (rigid-roll) advantage shrinking with speed? Spearman-style monotonic check
    adv = sg.advantage_deg_s.to_numpy()
    print(f"  advantage sequence (0.08,0.14,0.20): {adv.round(1).tolist()}  "
          f"monotonic decreasing 0.14->0.20: {adv[1] > adv[2]}")
    sg.to_csv(f"{OUT}/roll_speedgating.csv", index=False)

    # ---------------- PART 2: weave capability ----------------
    print("\n=== Weave cross-track median/IQR at vx=0.20 (verification) ===")
    weave_summary_rows = []
    for cmd in SPEEDS:
        for v in VARIANTS:
            sub = weave_df[(weave_df.variant == v) & (weave_df.cmd == cmd)]
            if len(sub) == 0 or "xtrack_median_mm" not in sub.columns:
                continue
            med = sub.xtrack_median_mm.median()
            q1, q3 = sub.xtrack_median_mm.quantile([0.25, 0.75])
            iqr = q3 - q1
            row = dict(variant=v, speed=cmd, n=len(sub), xtrack_median_mm=med,
                      xtrack_iqr_mm=iqr, xtrack_q1=q1, xtrack_q3=q3)
            if v == "yaw":
                rigid_sub = weave_df[(weave_df.variant == "rigid") & (weave_df.cmd == cmd)]
                r_iqr = rigid_sub.xtrack_median_mm.quantile(0.75) - rigid_sub.xtrack_median_mm.quantile(0.25)
                row["iqr_ratio_rigid_over_yaw"] = r_iqr / iqr if iqr > 0 else np.nan
            weave_summary_rows.append(row)
            if cmd == 0.20:
                print(f"  {v:6s} vx=0.20: median={med:.2f}mm IQR={iqr:.2f}mm [{q1:.2f},{q3:.2f}]")
    pd.DataFrame(weave_summary_rows).to_csv(f"{OUT}/weave_xtrack_summary.csv", index=False)

    print("\n=== Weave v_shortfall_pct (expected ~0%) ===")
    print(weave_df.groupby(["variant", "cmd"]).v_shortfall_pct.mean().round(2))

    print("\n=== Poles cleared @ vx=0.20 (of 5), pole x-positions:", [round(p, 2) for p in pole_positions()], "===")
    if "poles_cleared" in weave_df.columns:
        print(weave_df[weave_df.cmd == 0.20].groupby("variant").poles_cleared.agg(["mean", "min", "max"]))

    # axis-specificity CIs for xtrack at 0.20 (weave equivalence-style table)
    axis_rows = []
    base20 = weave_df[(weave_df.variant == "rigid") & (weave_df.cmd == 0.20)].sort_values("seed")
    for v in SPINE_VARIANTS:
        sub = weave_df[(weave_df.variant == v) & (weave_df.cmd == 0.20)].sort_values("seed")
        common = sorted(set(base20.seed) & set(sub.seed))
        b = base20.set_index("seed").loc[common, "xtrack_median_mm"].to_numpy()
        s = sub.set_index("seed").loc[common, "xtrack_median_mm"].to_numpy()
        diffs = s - b
        mean_diff, lo, hi = paired_bootstrap_ci(diffs)
        axis_rows.append(dict(variant=v, speed=0.20, metric="xtrack_median_mm",
                              mean_diff_vs_rigid=mean_diff, ci_low=lo, ci_high=hi,
                              cliffs_delta=cliffs_delta(s, b), n=len(common)))
    pd.DataFrame(axis_rows).to_csv(f"{OUT}/weave_axis_specificity.csv", index=False)
    print("\n=== Axis specificity (xtrack vs rigid @0.20) ===")
    print(pd.DataFrame(axis_rows).round(2))

    # pole contact check (poles are non-colliding by construction, see microtaur_terrains.py)
    print("\n=== Pole contact check: WeavePoleTerrainCfg sets contype=conaffinity=0 on poles "
          "(microtaur_terrains.py) -- poles cannot register a physics contact event by "
          "construction. No contact-with-pole channel exists in the logged data; "
          "reachability is assessed geometrically (base_x vs pole x-position) not via contact. ===")

    # command trajectory check
    print("\n=== Command trajectory check (item a) ===")
    print("Training resampling_time_range = (6.0, 10.0) s (piecewise-constant), from env_cfgs.py")
    for cmd in SPEEDS:
        period = 2 * POLE_SPACING / cmd
        print(f"  vx={cmd:.2f}: slalom period (2*spacing/v) = {period:.2f}s")
    print("  Weave's actual cmd_yaw is a CONTINUOUSLY-VARYING pure-pursuit+curvature-feedforward "
         "signal updated every policy step (20ms) -- not a piecewise-constant resample. So "
         "command TYPE (yaw-rate) matches training, but command TRAJECTORY SHAPE does not "
         "(continuous tracking curve vs step function) regardless of the period-vs-resample-"
         "interval comparison above.")

    # ---------------- PART 3: spine function on flat ----------------
    spine_rows = []
    for v in SPINE_VARIANTS:
        for cmd in SPEEDS:
            sub = flat_df[(flat_df.variant == v) & (flat_df.cmd == cmd)]
            if len(sub) == 0:
                continue
            spine_rows.append(dict(
                variant=v, speed=cmd, n=len(sub),
                returned=sub.spine_returned.mean(), net_work=sub.spine_net_work.mean(),
                k_eff=sub.spine_k_eff.mean(), c_eff=sub.spine_c_eff.mean(),
                imp_r2=sub.spine_imp_r2.mean(), ROM=sub.spine_ROM.mean(),
                rms=sub.spine_rms.mean(), peak_rate=sub.spine_peak_rate.mean(),
                work_pct_of_total=sub.spine_work_pct_of_total.mean()))
    pd.DataFrame(spine_rows).to_csv(f"{OUT}/spine_function_flat.csv", index=False)

    # target regression: tgt ~ q, qdot (roll damper hypothesis) + broader predictor set
    from numpy.linalg import lstsq
    reg_rows = []
    for terrain in ("flat", "weave"):
        for v in SPINE_VARIANTS:
            for cmd in SPEEDS:
                runs = reg_data[terrain].get((v, cmd))
                if not runs:
                    continue
                tgt = np.concatenate([r["tgt"] for r in runs])
                q = np.concatenate([r["q"] for r in runs])
                qd = np.concatenate([r["qd"] for r in runs])
                cmd_yaw = np.concatenate([r["cmd_yaw"] for r in runs])
                roll = np.concatenate([r["roll"] for r in runs])
                pitch = np.concatenate([r["pitch"] for r in runs])
                wx = np.concatenate([r["wx"] for r in runs])
                wy = np.concatenate([r["wy"] for r in runs])
                fc = np.concatenate([r["fc"] for r in runs])
                v_x = np.concatenate([r["v"] for r in runs])
                preds = dict(q=q, qdot=qd, cmd_yaw=cmd_yaw, roll=roll, pitch=pitch,
                            roll_rate=wx, pitch_rate=wy, foot_contact=fc, v_body=v_x)
                names = list(preds.keys())
                Z = np.column_stack([preds[n] for n in names])
                Zs = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
                X = np.column_stack([Zs, np.ones(len(tgt))])
                coef, *_ = lstsq(X, tgt, rcond=None)
                pred = X @ coef
                r2 = 1 - np.var(tgt - pred) / max(np.var(tgt), 1e-12)
                # bootstrap CI on standardized coefs
                idx = np.arange(len(tgt))
                boots = np.zeros((300, len(names)))
                for bi in range(300):
                    s_idx = RNG.choice(idx, size=len(idx), replace=True)
                    c, *_ = lstsq(X[s_idx], tgt[s_idx], rcond=None)
                    boots[bi] = c[:-1]
                for i, name in enumerate(names):
                    lo, hi = np.percentile(boots[:, i], [2.5, 97.5])
                    reg_rows.append(dict(variant=v, terrain=terrain, speed=cmd, predictor=name,
                                         std_coef=coef[i], ci_low=lo, ci_high=hi, model_R2=r2, n=len(tgt)))
                # roll damper hypothesis: tgt ~ a + b*q + c*qdot (raw, unstandardized, for
                # interpretable b (dimensionless) and c (seconds))
                Xr = np.column_stack([q, qd, np.ones_like(q)])
                cr, *_ = lstsq(Xr, tgt, rcond=None)
                predr = Xr @ cr
                r2r = 1 - np.var(tgt - predr) / max(np.var(tgt), 1e-12)
                boots_r = np.zeros((300, 2))
                for bi in range(300):
                    s_idx = RNG.choice(idx, size=len(idx), replace=True)
                    c, *_ = lstsq(Xr[s_idx], tgt[s_idx], rcond=None)
                    boots_r[bi] = c[:2]
                b_ci = np.percentile(boots_r[:, 0], [2.5, 97.5])
                c_ci = np.percentile(boots_r[:, 1], [2.5, 97.5])
                if v == "roll":
                    print(f"[roll damper check] {terrain} vx={cmd:.2f}: tgt = {cr[2]:.4f} + "
                          f"{cr[0]:.3f}*q + {cr[1]:.4f}*qdot  (b CI=[{b_ci[0]:.3f},{b_ci[1]:.3f}], "
                          f"c CI=[{c_ci[0]:.4f},{c_ci[1]:.4f}] s, R2={r2r:.3f})")
    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(f"{OUT}/target_regression.csv", index=False)

    # ---------------- diagnostics: bimodal / seed spread ----------------
    print("\n=== Diagnostics: between-seed vs within-seed spread flags ===")
    for terrain, df in (("flat", flat_df), ("weave", weave_df)):
        for v in VARIANTS:
            for cmd in SPEEDS:
                sub = df[(df.variant == v) & (df.cmd == cmd)]
                if len(sub) < 4:
                    continue
                cot_std = sub.CoT.std()
                if cot_std > 0.5 * sub.CoT.mean():
                    print(f"  FLAG high CoT spread: {terrain} {v} vx={cmd}: mean={sub.CoT.mean():.2f} std={cot_std:.2f}")

    # ---------------- figures ----------------
    make_equivalence_forest(eq_df)
    make_roll_speedgate_fig(sg)
    make_weave_box_fig(weave_df)
    make_psd_fig(reg_data)

    print("\nDONE. Outputs in", OUT)


def make_equivalence_forest(eq_df):
    metrics = eq_df.metric.unique().tolist()
    for theme_name, t in THEME.items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 8), facecolor=t["surface"], sharey=True)
        for ax, cmd in zip(axes, SPEEDS):
            ax.set_facecolor(t["surface"])
            sub = eq_df[eq_df.speed == cmd]
            ys, labels = [], []
            y = 0
            for metric in metrics:
                for v in SPINE_VARIANTS:
                    r = sub[(sub.metric == metric) & (sub.variant == v)]
                    if len(r) == 0:
                        continue
                    r = r.iloc[0]
                    ax.plot([r.ci_low, r.ci_high], [y, y], color=t["vcol"][v], lw=2)
                    ax.scatter([r.mean_diff_vs_rigid], [y], color=t["vcol"][v], s=20, zorder=3)
                    labels.append(f"{metric} ({v})")
                    ys.append(y)
                    y += 1
                y += 0.5
            ax.axvline(0, ls="--", color=t["ink2"], lw=0.8)
            ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=6, color=t["ink"])
            ax.set_title(f"vx={cmd:.2f} m/s", color=t["ink"], fontsize=10)
            ax.tick_params(colors=t["ink2"])
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        fig.suptitle("Equivalence: 95% CI on (variant - rigid), flat ground", color=t["ink"], fontsize=12)
        fig.tight_layout()
        fig.savefig(f"{OUT}/fig_equivalence_forest_{theme_name}.png", dpi=130, facecolor=t["surface"])
        plt.close(fig)


def make_roll_speedgate_fig(sg):
    for theme_name, t in THEME.items():
        fig, ax = plt.subplots(figsize=(5, 4), facecolor=t["surface"])
        ax.set_facecolor(t["surface"])
        ax.plot(sg.speed, sg.advantage_deg_s, "o-", color=t["vcol"]["roll"])
        ax.axhline(0, ls="--", color=t["ink2"], lw=0.8)
        ax.set_xlabel("commanded vx (m/s)", color=t["ink"])
        ax.set_ylabel("rigid - roll peak roll-rate (deg/s)", color=t["ink"])
        ax.set_title("Roll's trunk-stabilization advantage vs speed", color=t["ink"], fontsize=10)
        ax.tick_params(colors=t["ink2"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(f"{OUT}/fig_roll_speedgate_{theme_name}.png", dpi=130, facecolor=t["surface"])
        plt.close(fig)


def make_weave_box_fig(weave_df):
    if "xtrack_median_mm" not in weave_df.columns:
        return
    for theme_name, t in THEME.items():
        fig, ax = plt.subplots(figsize=(7, 5), facecolor=t["surface"])
        ax.set_facecolor(t["surface"])
        width = 0.19
        for i, v in enumerate(VARIANTS):
            for j, cmd in enumerate(SPEEDS):
                sub = weave_df[(weave_df.variant == v) & (weave_df.cmd == cmd)]
                if len(sub) == 0:
                    continue
                x = j + (i - 1.5) * width
                ax.boxplot([sub.xtrack_median_mm.values], positions=[x], widths=width * 0.8,
                          patch_artist=True, boxprops=dict(facecolor=t["vcol"][v], alpha=0.7),
                          medianprops=dict(color=t["ink"]))
        ax.set_yscale("log")
        ax.set_xticks(range(len(SPEEDS))); ax.set_xticklabels([f"{c:.2f}" for c in SPEEDS])
        ax.set_xlabel("commanded vx (m/s)", color=t["ink"])
        ax.set_ylabel("cross-track median |error| (mm, log scale)", color=t["ink"])
        ax.set_title("Weave cross-track error by variant x speed", color=t["ink"], fontsize=10)
        ax.tick_params(colors=t["ink2"])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        fig.savefig(f"{OUT}/fig_weave_xtrack_box_{theme_name}.png", dpi=130, facecolor=t["surface"])
        plt.close(fig)


def make_psd_fig(reg_data):
    fs = 50.0
    for theme_name, t in THEME.items():
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=t["surface"], sharey=True)
        for ax, v in zip(axes, SPINE_VARIANTS):
            ax.set_facecolor(t["surface"])
            for terr, ls in (("flat", "-"), ("weave", "--")):
                runs = reg_data[terr].get((v, 0.20))
                if not runs:
                    continue
                q = runs[0]["q"]
                q = q - q.mean()
                n = len(q)
                freqs = np.fft.rfftfreq(n, d=1 / fs)
                psd = np.abs(np.fft.rfft(q * np.hanning(n))) ** 2
                ax.semilogy(freqs, psd, ls, color=t["vcol"][v], label=terr)
            if v == "yaw":
                slalom_f = 0.20 / (2 * POLE_SPACING)
                ax.axvline(slalom_f, color=t["ink2"], ls=":", lw=1, label=f"slalom {slalom_f:.2f}Hz")
            ax.set_xlim(0, 5)
            ax.set_title(v, color=t["ink"], fontsize=10)
            ax.set_xlabel("Hz", color=t["ink"])
            ax.legend(fontsize=6, frameon=False)
            ax.tick_params(colors=t["ink2"])
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[0].set_ylabel("spine angle PSD", color=t["ink"])
        fig.suptitle("Spine-angle PSD, flat vs weave, vx=0.20 m/s", color=t["ink"], fontsize=11)
        fig.tight_layout()
        fig.savefig(f"{OUT}/fig_spine_psd_{theme_name}.png", dpi=130, facecolor=t["surface"])
        plt.close(fig)


if __name__ == "__main__":
    main()
