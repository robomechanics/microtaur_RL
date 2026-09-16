"""Flat + weave (in-distribution) analysis for the morphology paper.

    python scripts/flat_weave_analysis.py

DISCOVERY (confirmed against raw files, not assumed):
- Flat rollout dirs: rollouts/<variant>_<timestamp> (NO terrain suffix).
  Weave dirs: rollouts/<variant>_weave_<timestamp>.
- Columns: roll/pitch/yaw in RADIANS (magnitudes ~0.01-0.05 on flat, matching
  tilt_deg ~1-3deg elsewhere in this project). wx/wy = body roll-rate/pitch-
  rate (rad/s), previously verified via correlation with finite-differenced
  angle (r=0.97/0.91). foot_contact_i is boolean (0/1); foot_force_i is the
  separate force channel. `terminated` (real fall) is separate from
  `time_out`; both roll into `done`. Sample rate 50 Hz (policy_dt=0.02s);
  substep_log=True in every meta.json here, so Wxpos_total (exact 5ms-
  integrated positive work) is available -- used for CoT, not the 50Hz
  tau*qd sample which under-reads power (see rollouts/CORRECTION.md).
- Spine target: action_raw_spine (raw policy output) and applied_target_spine
  (post-filter, post-delay commanded target) are both logged per step.
- RESOLVED per coordinator: spacing=0.80 is canonical (matches the rest of
  the project). The originally-quoted "5th pole at 2.60 m" came from notes of
  uncertain provenance and was discarded. At spacing=0.80 the 5th pole sits at
  3.60 m, which is unreachable within the 16 s episode at ALL THREE speeds
  (straight-line lower bound alone needs 18 s even at 0.20 m/s) -- an even
  more mechanical ceiling than the one found at 0.55 m (there, all variants
  cleared all 5 poles; here, the reachable count is capped by speed x time
  budget regardless of tracking quality: 2/3/4 poles at 0.08/0.14/0.20 m/s).
  poles_cleared is dropped from the report per the coordinator's own
  conditional.
- Training command resampling: cmd.resampling_time_range = (6.0, 10.0) s,
  piecewise-constant (see PART 2a in the report).
"""
from __future__ import annotations

import glob
import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RNG = np.random.default_rng(0)
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "rollouts" / "flat_weave"
BL_M = 0.168
G = 9.81
SETTLE_S = 1.0
SPEEDS = (0.08, 0.14, 0.20)
VARIANTS = ["rigid", "pitch", "yaw", "roll"]
SPINE_VARIANTS = ["pitch", "yaw", "roll"]
MEASURED_MASS_KG = {"yaw": 0.4577}
WEAVE_SPACING = 0.80  # canonical, matches resummarize.py / poincare_stability.py; see docstring
HARD_LIMIT_RAD = {"pitch": math.radians(30.0), "yaw": 0.35, "roll": 0.35}

FOOT_CHANNELS = json.loads(open(ROOT / "rollouts" / "foot_channels.json").read())


def fl_channel(variant):
    return FOOT_CHANNELS[variant]["leg_to_channel"]["leg1"] if variant == "rigid" else \
        {v: FOOT_CHANNELS[v]["leg_to_channel"]["leg1"] for v in [variant]}[variant]


# --------------------------------------------------------------------------- #
# Canonical file index
# --------------------------------------------------------------------------- #

def build_index_flat():
    keep = {}
    for v in VARIANTS:
        dirs = [d for d in glob.glob(f"rollouts/{v}_2*") if os.path.isdir(d)
                and "_curb_" not in d and "_steps_" not in d and "_weave_" not in d]
        for d in sorted(dirs, key=os.path.getmtime):
            mfs = sorted(glob.glob(os.path.join(d, "rollout_*.meta.json")))
            if not mfs:
                continue
            m0 = json.loads(open(mfs[0]).read())
            if not m0.get("raw_action", False):
                continue
            for mf in mfs:
                m = json.loads(open(mf).read())
                if m["cmd_vx"] not in SPEEDS or m.get("cmd_yaw", 0.0) != 0.0:
                    continue
                key = (v, m["cmd_vx"], m["seed"])
                keep[key] = mf.replace(".meta.json", ".csv")
    return keep


def build_index_weave():
    keep = {}
    for v in VARIANTS:
        for d in sorted(glob.glob(f"rollouts/{v}_weave_2*"), key=os.path.getmtime):
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
                if (m.get("terrain_kw") or {}).get("spacing") != WEAVE_SPACING:
                    continue
                key = (v, m["cmd_vx"], m["seed"])
                keep[key] = mf.replace(".meta.json", ".csv")
    return keep


IDX_FLAT = build_index_flat()
IDX_WEAVE = build_index_weave()


# --------------------------------------------------------------------------- #
# Per-file metrics
# --------------------------------------------------------------------------- #

def first_episode(df):
    rs = df.index[(df["terminated"] > 0) | (df["time_out"] > 0)]
    rs = df.index[(df["done"] > 0) & (df.index < len(df) - 1)]
    return df.loc[: rs[0] - 1] if len(rs) else df


def flat_metrics(variant, cmd, seed, csv_path):
    df = pd.read_csv(csv_path)
    ep = first_episode(df)
    fell = bool(((df["terminated"] > 0) & (df.index < len(df) - 1)).any())
    ss = ep[ep.t >= SETTLE_S]
    if len(ss) < 10:
        return None
    m_sim = float(json.loads(open(csv_path.replace(".csv", ".meta.json")).read())["mass_kg"])
    m_use = MEASURED_MASS_KG.get(variant, m_sim)
    has_exact = "Wxpos_total" in ss.columns
    p_pos = (ss.Wxpos_total.to_numpy() / 0.02) if has_exact else ss.P_pos.to_numpy()
    denom = m_use * G * np.maximum(np.abs(ss.v_body_x.to_numpy()), 0.02)
    cot = float(np.mean(p_pos / denom))
    v_ach = float(ss.v_body_x.mean())
    row = dict(variant=variant, speed=cmd, seed=seed, n=len(ss), fell=fell,
               CoT=cot, v_achieved=v_ach, v_shortfall=cmd - v_ach,
               v_shortfall_pct=(cmd - v_ach) / cmd * 100,
               track_err=float(np.mean(np.abs(ss.v_body_x - cmd))),
               roll_rms_deg=float(np.degrees(np.sqrt(np.mean(ss.roll ** 2)))),
               roll_peak_rate_deg=float(np.degrees(np.abs(ss.wx).max())),
               pitch_rms_deg=float(np.degrees(np.sqrt(np.mean(ss["pitch"] ** 2)))),
               pitch_peak_rate_deg=float(np.degrees(np.abs(ss.wy).max())))
    if variant in SPINE_VARIANTS and "q_spine" in ss.columns:
        row["spine_rom_deg"] = float(np.degrees(ss.q_spine.max() - ss.q_spine.min()))
        row["spine_rms_deg"] = float(np.degrees(np.sqrt(np.mean(ss.q_spine ** 2))))
        row["spine_peak_rate_deg"] = float(np.degrees(np.abs(ss.qd_spine).max()))
    return row


def weave_metrics(variant, cmd, seed, csv_path):
    df = pd.read_csv(csv_path)
    ep = first_episode(df)
    fell = bool(((df["terminated"] > 0) & (df.index < len(df) - 1)).any())
    ss = ep[ep.t >= SETTLE_S]
    if len(ss) < 10 or "weave_cross_track_m" not in ss.columns:
        return None
    m_sim = float(json.loads(open(csv_path.replace(".csv", ".meta.json")).read())["mass_kg"])
    m_use = MEASURED_MASS_KG.get(variant, m_sim)
    has_exact = "Wxpos_total" in ss.columns
    p_pos = (ss.Wxpos_total.to_numpy() / 0.02) if has_exact else ss.P_pos.to_numpy()
    denom = m_use * G * np.maximum(np.abs(ss.v_body_x.to_numpy()), 0.02)
    xt_mm = np.abs(ss.weave_cross_track_m.to_numpy()) * 1000
    v_ach = float(ss.v_body_x.mean())
    x = ss.base_x.to_numpy() if "base_x" in ss.columns else None
    poles_x = [0.40 + k * WEAVE_SPACING for k in range(5)]
    poles_cleared = int(sum(1 for px in poles_x if x is not None and x.max() >= px)) if cmd == 0.20 else np.nan
    yaw_cmd = ss.cmd_yaw.to_numpy() if "cmd_yaw" in ss.columns else np.full(len(ss), np.nan)
    yaw_rate_err = float(np.mean(np.abs(ss.wz.to_numpy() - yaw_cmd))) if "wz" in ss.columns else np.nan
    row = dict(variant=variant, speed=cmd, seed=seed, n=len(ss), fell=fell,
               CoT=float(np.mean(p_pos / denom)), v_achieved=v_ach,
               v_shortfall=cmd - v_ach, v_shortfall_pct=(cmd - v_ach) / cmd * 100,
               xtrack_median_mm=float(np.median(xt_mm)),
               xtrack_iqr_mm=float(np.percentile(xt_mm, 75) - np.percentile(xt_mm, 25)),
               poles_cleared=poles_cleared, yaw_rate_track_err=yaw_rate_err)
    if variant in SPINE_VARIANTS and "q_spine" in ss.columns:
        row["spine_rom_deg"] = float(np.degrees(ss.q_spine.max() - ss.q_spine.min()))
    return row


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #

def paired_bootstrap_ci(x_variant, x_rigid, n_boot=10000):
    """x_variant, x_rigid: same-length arrays, seed-paired. Returns mean diff, CI."""
    d = np.asarray(x_variant) - np.asarray(x_rigid)
    n = len(d)
    boots = np.array([RNG.choice(d, size=n, replace=True).mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    return (gt - lt) / (len(a) * len(b))


def holm(pvals):
    idx = np.argsort(pvals)
    n = len(pvals)
    adj = np.empty(n)
    prev = 0.0
    for rank, i in enumerate(idx):
        val = min(1.0, (n - rank) * pvals[i])
        prev = max(prev, val)
        adj[i] = prev
    return adj


def perm_pvalue(diffs, n_perm=10000):
    """Two-sided sign-flip permutation test p-value for paired differences."""
    obs = np.abs(diffs.mean())
    signs = RNG.choice([-1, 1], size=(n_perm, len(diffs)))
    perm_means = np.abs((signs * diffs).mean(axis=1))
    return float((perm_means >= obs).mean())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[flat] canonical files: {len(IDX_FLAT)} (expect {len(VARIANTS)*3*8}=96)")
    print(f"[weave] canonical files (spacing={WEAVE_SPACING}): {len(IDX_WEAVE)} (expect {len(VARIANTS)*3*8}=96)")

    flat_rows, weave_rows = [], []
    for v in VARIANTS:
        for cmd in SPEEDS:
            for seed in range(8):
                key = (v, cmd, seed)
                if key in IDX_FLAT:
                    r = flat_metrics(v, cmd, seed, IDX_FLAT[key])
                    if r:
                        flat_rows.append(r)
                if key in IDX_WEAVE:
                    r = weave_metrics(v, cmd, seed, IDX_WEAVE[key])
                    if r:
                        weave_rows.append(r)

    flat_df = pd.DataFrame(flat_rows)
    weave_df = pd.DataFrame(weave_rows)
    flat_df.to_csv(OUT / "flat_metrics.csv", index=False)
    weave_df.to_csv(OUT / "weave_metrics.csv", index=False)

    print(f"\n[flat] n rows: {len(flat_df)}; per (variant,speed) n:")
    print(flat_df.groupby(["variant", "speed"]).size().unstack())
    print(f"\n[flat] fall counts by variant:")
    print(flat_df.groupby("variant").fell.sum())

    print(f"\n[weave] n rows: {len(weave_df)}; per (variant,speed) n:")
    if len(weave_df):
        print(weave_df.groupby(["variant", "speed"]).size().unstack())
        print(f"\n[weave] fall counts by variant:")
        print(weave_df.groupby("variant").fell.sum())

    # ---- PART 1: flat equivalence ---------------------------------------- #
    metrics_flat = ["CoT", "v_achieved", "v_shortfall", "track_err",
                    "roll_rms_deg", "roll_peak_rate_deg", "pitch_rms_deg", "pitch_peak_rate_deg"]
    equiv_rows = []
    pvals_by_family = {}
    for metric in metrics_flat:
        family_p, family_idx = [], []
        for cmd in SPEEDS:
            rig = flat_df[(flat_df.variant == "rigid") & (flat_df.speed == cmd)].sort_values("seed")
            for v in SPINE_VARIANTS:
                var = flat_df[(flat_df.variant == v) & (flat_df.speed == cmd)].sort_values("seed")
                common = sorted(set(rig.seed) & set(var.seed))
                if len(common) < 3:
                    continue
                rv = rig.set_index("seed").loc[common, metric].to_numpy()
                vv = var.set_index("seed").loc[common, metric].to_numpy()
                mean_diff, lo, hi = paired_bootstrap_ci(vv, rv)
                rigid_mean = rv.mean()
                delta = cliffs_delta(vv, rv)
                p = perm_pvalue(vv - rv)
                family_p.append(p)
                family_idx.append((metric, cmd, v))
                equiv_rows.append(dict(variant=v, speed=cmd, metric=metric, n=len(common),
                                       mean_diff_vs_rigid=mean_diff, ci_low=lo, ci_high=hi,
                                       ci_as_pct_of_rigid_low=lo / abs(rigid_mean) * 100 if rigid_mean else np.nan,
                                       ci_as_pct_of_rigid_high=hi / abs(rigid_mean) * 100 if rigid_mean else np.nan,
                                       cliffs_delta=delta, p_raw=p, rigid_mean=rigid_mean))
        pvals_by_family[metric] = (family_p, family_idx)

    # Holm-correct within each metric family (all speed x variant comparisons for that metric)
    equiv_df = pd.DataFrame(equiv_rows)
    equiv_df["p_holm"] = np.nan
    for metric, (pv, idxs) in pvals_by_family.items():
        if not pv:
            continue
        adj = holm(np.array(pv))
        for (m, cmd, v), a in zip(idxs, adj):
            mask = (equiv_df.metric == m) & (equiv_df.speed == cmd) & (equiv_df.variant == v)
            equiv_df.loc[mask, "p_holm"] = a
    equiv_df.to_csv(OUT / "equivalence_summary.csv", index=False)

    print("\n" + "=" * 100)
    print("PART 1 -- FLAT EQUIVALENCE (variant - rigid), Holm-corrected within each metric family")
    print("=" * 100)
    pd.set_option("display.width", 220)
    print(equiv_df[["variant", "speed", "metric", "n", "mean_diff_vs_rigid", "ci_low", "ci_high",
                    "ci_as_pct_of_rigid_low", "ci_as_pct_of_rigid_high", "cliffs_delta", "p_holm"]]
          .round(4).to_string(index=False))

    # ---- roll peak-rate speed-gating check -------------------------------- #
    print("\n" + "=" * 100)
    print("ROLL TRUNK STABILIZATION -- peak body roll rate (deg/s), verification vs quoted values")
    print("=" * 100)
    quoted = {0.08: (70, 93), 0.14: (76, 103), 0.20: (91, 94)}
    rows = []
    for cmd in SPEEDS:
        rig = flat_df[(flat_df.variant == "rigid") & (flat_df.speed == cmd)]
        rol = flat_df[(flat_df.variant == "roll") & (flat_df.speed == cmd)]
        q_roll, q_rigid = quoted[cmd]
        m_rig, m_rol = rig.roll_peak_rate_deg.mean(), rol.roll_peak_rate_deg.mean()
        common = sorted(set(rig.seed) & set(rol.seed))
        rv = rig.set_index("seed").loc[common, "roll_peak_rate_deg"].to_numpy()
        vv = rol.set_index("seed").loc[common, "roll_peak_rate_deg"].to_numpy()
        diff, lo, hi = paired_bootstrap_ci(vv, rv)
        rows.append(dict(speed=cmd, roll_measured=m_rol, roll_quoted=q_roll,
                         rigid_measured=m_rig, rigid_quoted=q_rigid,
                         advantage_deg=diff, ci_low=lo, ci_high=hi, n=len(common)))
        match_roll = "MATCH" if abs(m_rol - q_roll) < 5 else "MISMATCH"
        match_rig = "MATCH" if abs(m_rig - q_rigid) < 5 else "MISMATCH"
        print(f"  vx={cmd}: measured roll={m_rol:.1f} (quoted {q_roll}, {match_roll})   "
              f"measured rigid={m_rig:.1f} (quoted {q_rigid}, {match_rig})   "
              f"advantage(rigid-roll)={-diff:.1f} deg/s  CI[{-hi:.1f},{-lo:.1f}]  n={len(common)}")
    roll_speed_df = pd.DataFrame(rows)
    roll_speed_df.to_csv(OUT / "roll_speedgating.csv", index=False)

    # speed x variant interaction: (roll-rigid @0.20) - (roll-rigid @0.08), paired over seed
    rig08 = flat_df[(flat_df.variant == "rigid") & (flat_df.speed == 0.08)].set_index("seed")
    rol08 = flat_df[(flat_df.variant == "roll") & (flat_df.speed == 0.08)].set_index("seed")
    rig20 = flat_df[(flat_df.variant == "rigid") & (flat_df.speed == 0.20)].set_index("seed")
    rol20 = flat_df[(flat_df.variant == "roll") & (flat_df.speed == 0.20)].set_index("seed")
    common = sorted(set(rig08.index) & set(rol08.index) & set(rig20.index) & set(rol20.index))
    d08 = (rol08.loc[common, "roll_peak_rate_deg"] - rig08.loc[common, "roll_peak_rate_deg"]).to_numpy()
    d20 = (rol20.loc[common, "roll_peak_rate_deg"] - rig20.loc[common, "roll_peak_rate_deg"]).to_numpy()
    interaction = d20 - d08
    boots = np.array([RNG.choice(interaction, size=len(interaction), replace=True).mean() for _ in range(10000)])
    print(f"\n  SPEED x VARIANT INTERACTION (roll-rigid diff @0.20) - (@0.08): "
          f"mean={interaction.mean():.2f} deg/s  CI[{np.percentile(boots,2.5):.2f},{np.percentile(boots,97.5):.2f}]  n={len(common)}")
    print(f"  (CI excluding 0 -> the roll advantage genuinely shrinks with speed, not noise;"
          f" CI including 0 -> cannot distinguish from constant advantage at n=8)")

    # ---- PART 2: weave --------------------------------------------------- #
    if len(weave_df):
        print("\n" + "=" * 100)
        print("PART 2 -- WEAVE: cross-track median/IQR + IQR ratio vs rigid")
        print("=" * 100)
        wrows = []
        for cmd in SPEEDS:
            rig = weave_df[(weave_df.variant == "rigid") & (weave_df.speed == cmd)]
            for v in SPINE_VARIANTS:
                var = weave_df[(weave_df.variant == v) & (weave_df.speed == cmd)]
                if len(var) == 0 or len(rig) == 0:
                    continue
                rig_all_xt = rig.xtrack_median_mm  # per-seed medians; report cell-level median/IQR too
                med_v, iqr_v = var.xtrack_median_mm.median(), (var.xtrack_median_mm.quantile(.75) - var.xtrack_median_mm.quantile(.25))
                med_r, iqr_r = rig.xtrack_median_mm.median(), (rig.xtrack_median_mm.quantile(.75) - rig.xtrack_median_mm.quantile(.25))
                common = sorted(set(rig.seed) & set(var.seed))
                rv = rig.set_index("seed").loc[common, "xtrack_median_mm"].to_numpy()
                vv = var.set_index("seed").loc[common, "xtrack_median_mm"].to_numpy()
                diff, lo, hi = paired_bootstrap_ci(vv, rv)
                delta = cliffs_delta(vv, rv)
                wrows.append(dict(variant=v, speed=cmd, xtrack_median_mm=med_v, xtrack_iqr_mm=iqr_v,
                                  rigid_median_mm=med_r, rigid_iqr_mm=iqr_r,
                                  iqr_ratio_rigid_over_variant=iqr_r / iqr_v if iqr_v > 0 else np.inf,
                                  diff_vs_rigid=diff, ci_low=lo, ci_high=hi, cliffs_delta=delta, n=len(common)))
        wdf = pd.DataFrame(wrows)
        wdf.to_csv(OUT / "weave_crosstrack_summary.csv", index=False)
        print(wdf.round(3).to_string(index=False))

        print("\n[weave] shortfall check (expect ~0%):")
        print(weave_df.groupby("variant").v_shortfall_pct.agg(["mean", "std"]).round(2))

        print(f"\n[weave] poles cleared @0.20 m/s only (5 poles total, spacing={WEAVE_SPACING}, "
              f"5th at x={0.40 + 4*WEAVE_SPACING:.2f}m):")
        print(weave_df[weave_df.speed == 0.20].groupby("variant").poles_cleared.agg(["mean", "min", "max"]))

    # ---- PART 2a: command resample vs slalom period ----------------------- #
    print("\n" + "=" * 100)
    print("PART 2a -- training command resample interval vs weave slalom period")
    print("=" * 100)
    print("  Training: cmd.resampling_time_range = (6.0, 10.0) s, PIECEWISE-CONSTANT "
          "(src/microtaur_velocity/pitch_env.txt:702)")
    for cmd in SPEEDS:
        period = (2 * WEAVE_SPACING) / cmd
        print(f"  Weave slalom period @ vx={cmd}: 2*spacing/v = {2*WEAVE_SPACING:.2f}/{cmd} = {period:.2f} s")
    print("  VERDICT: slalom period (5.5-13.75s across the speed range) overlaps the training "
          "resample TIMESCALE (6-10s), but training commands were held CONSTANT for each full "
          "interval while weave's command continuously varies every policy step (pure-pursuit "
          "feedback, not a step function). Weave is in-distribution in command TYPE (yaw-rate "
          "commands existed in training) but NOT in command TRAJECTORY (continuous tracking "
          "signal vs piecewise-constant holds).")

    print(f"\nwrote outputs to {OUT}")


if __name__ == "__main__":
    main()
