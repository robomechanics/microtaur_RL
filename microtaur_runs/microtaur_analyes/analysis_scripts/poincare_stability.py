"""Poincare return maps and Floquet-style orbital stability of the SPINE joint's
own oscillation, for the 4-variant x 4-terrain x 3-speed x 8-seed terrain study.

    python scripts/poincare_stability.py
      -> rollouts/poincare/return_map_univariate.csv
      -> rollouts/poincare/return_map_multivariate.csv
      -> rollouts/poincare/fig_return_map_{light,dark}.png
      -> rollouts/poincare/fig_spectral_radius_{light,dark}.png

State variable: spine angle (`q_spine`, rad) and, as its natural phase-plane
companion, spine angular rate (`qd_spine`, rad/s) -- a 1-DoF oscillator's state
is (angle, rate), and the multivariate step needs both to produce a meaningful
2x2 Jacobian/eigenvalue estimate. `rigid` has no spine joint at all (8 leg
motors, no spine actuator) and is excluded from every fit with an explicit
reason -- never silently dropped -- rather than fit against a nonexistent
column.

METHOD (see printed summary for what was actually found / excluded)
---------------------------------------------------------------------------
Section (reset event): touchdown of the front-left (FL) foot, resolved
per-variant from rollouts/foot_channels.json (contact channel order is
permuted for pitch and roll -- see CORRECTION.md -- so "channel 0" is NOT the
same physical foot across variants). FL chosen arbitrarily but held fixed
across every cell -- unchanged from the original body-attitude version of this
analysis; only the state variable being sectioned changed.
Rising edge with >=2 preceding airborne samples (same robustness criterion
scripts/spine_function.py uses for stride segmentation), and 8-25 samples
(0.16-0.5 s at 50 Hz) between consecutive touchdowns to reject contact-sensor
chatter, also matching spine_function.py's `strides()`.

Data selection matches scripts/resummarize.py's canonical-run picker (raw
post-fix rollouts only, weave spacing 0.80, later dirs win on duplicate keys)
with one change: steps uses lane_keep=True (waypoint-steered), matching
scripts/com_figs.py / spine_function.py's MODE convention -- open-loop
step-field runs are "drift + reset confounded" per CORRECTION.md and are
NOT what the rest of this report calls "the" step-field test.

Discovery (see report): `terminated` never fires in any of the 384 canonical
files -- every episode runs its full fixed 800-step (16 s) length regardless
of a "fall". Curb "falls" (see rollouts/audit_2026-09-11/task3_curb_falloff_
per_run.csv) are a derived drift-off-the-lip event, not a sim termination, so
they are handled entirely by the on-lip mask below, not by trimming before a
termination flag. Only 2/384 files have a genuine mid-episode reset
(`done>0`, an out-of-bounds teleport): rigid/steps/0.20/seed7 and
yaw/curb/0.20/seed4. Per spec, those are truncated to the first episode and
additionally have their last 0.5 s trimmed.

wx/wy verified (not assumed) as body-frame roll-rate/pitch-rate: correlation
with d(roll)/dt, d(pitch)/dt via finite difference is 0.97 / 0.91 on a sample
run; roll/pitch/yaw columns are radians (magnitudes ~0.01-0.05 rad, matching
tilt_deg ~1-3 deg elsewhere in this report); foot_contact_i is a 0/1 float
(boolean), not force (that's the separate foot_force_i column).
"""
from __future__ import annotations

import glob
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def safe_savefig(fig, path, **kw):
    """fig.savefig with a couple retries: this machine's Downloads folder is
    synced (OneDrive) and transiently locks a just-written file, which makes
    the very next sequential savefig() in the same directory fail with
    OSError/Errno 22 -- not a bug in the figure itself."""
    import time
    last = None
    for attempt in range(5):
        try:
            fig.savefig(path, **kw)
            return
        except OSError as e:
            last = e
            time.sleep(0.5 * (attempt + 1))
    raise last

RNG = np.random.default_rng(0)
SPEEDS = (0.08, 0.14, 0.20)
VARIANTS = ["rigid", "pitch", "yaw", "roll"]
TERRAINS = ["flat", "curb", "steps", "weave"]
MODE = {"flat": False, "curb": False, "weave": False, "steps": True}  # lane_keep required
SETTLE_S = 1.0
POST_RESET_TRIM_S = 0.5
OUT = Path("rollouts/poincare")

# FL contact channel per variant, from rollouts/foot_channels.json
# channel_to_corner: rigid {2:FL}, pitch {3:FL}, yaw {2:FL}, roll {0:FL}
FL_CHANNEL = {"rigid": 2, "pitch": 3, "yaw": 2, "roll": 0}

SPINE_VARIANTS = ["pitch", "yaw", "roll"]   # rigid has no spine joint -- excluded, not dropped
COL_MAP = {"spine_angle": "q_spine", "spine_rate": "qd_spine"}
HARD_STOP = {"pitch": np.radians(30.0), "yaw": 0.35, "roll": 0.35}  # rad; from spine_function.py
OBS_UNI = ["spine_angle", "spine_rate"]
OBS_MV = ["spine_angle", "spine_rate"]  # multivariate state z = (angle, rate)
STATE_DIM = len(OBS_MV)

VCOL_LIGHT = {"rigid": "#2a78d6", "pitch": "#eb6834", "yaw": "#1baf7a", "roll": "#8f4bbf"}
VCOL_DARK = {"rigid": "#3987e5", "pitch": "#d95926", "yaw": "#199e70", "roll": "#b78fd6"}
THEME = {
    "light": dict(vcol=VCOL_LIGHT, surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#d9dbd6"),
    "dark": dict(vcol=VCOL_DARK, surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#3a3a38"),
}


# --------------------------------------------------------------------------- #
# Canonical run index (mirrors scripts/resummarize.py exactly, steps->waypoint)
# --------------------------------------------------------------------------- #

def build_index():
    keep = {}
    for v in VARIANTS:
        for d in sorted(glob.glob(f"rollouts/{v}_*"), key=os.path.getmtime):
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
                t = m.get("terrain", "flat")
                if t not in MODE or bool(m.get("lane_keep")) != MODE[t]:
                    continue
                if t == "weave" and (m.get("terrain_kw") or {}).get("spacing") != 0.80:
                    continue
                key = (v, t, m["cmd_vx"], m["seed"])
                keep[key] = mf.replace(".meta.json", ".csv")
    return keep


# --------------------------------------------------------------------------- #
# Per-file: touchdown runs -> consecutive-pair observations
# --------------------------------------------------------------------------- #

def touchdown_runs(contact, mask):
    """Rising-edge touchdowns (>=2 airborne samples before), split into maximal
    runs of consecutive touchdowns whose gap is 8-25 samples AND fully inside
    `mask`. Mirrors spine_function.py's strides() validity rule."""
    td = [i for i in range(2, len(contact)) if contact[i] and not contact[i - 1] and not contact[i - 2]]
    if len(td) < 2:
        return []
    runs, cur = [], [td[0]]
    for a, b in zip(td[:-1], td[1:]):
        valid = (8 <= b - a <= 25) and mask[a:b + 1].all()
        if valid:
            cur.append(b)
        else:
            if len(cur) >= 2:
                runs.append(cur)
            cur = [b]
    if len(cur) >= 2:
        runs.append(cur)
    return runs


def load_masked_episode(csv_path, variant, terrain):
    """Shared loader: episode truncated at any genuine mid-episode reset, masked
    to settle/on-lip/post-reset-trim, same rule used everywhere in this script.
    Returns (t, mask, contact, series, had_reset) where `series` maps each
    OBS_MV name to its full (unmasked-length) column array."""
    df = pd.read_csv(csv_path)
    n = len(df)
    rs = df.index[(df["done"].to_numpy() > 0) & (df.index < n - 1)]
    had_reset = len(rs) > 0
    end = int(rs[0]) if had_reset else n
    ep = df.iloc[:end]

    t = ep["t"].to_numpy()
    mask = t >= SETTLE_S
    if had_reset:
        mask &= t <= (t[-1] - POST_RESET_TRIM_S)
    if terrain == "curb":
        mask &= np.abs(ep["base_y"].to_numpy()) < 0.04

    contact = ep[f"foot_contact_{FL_CHANNEL[variant]}"].to_numpy() > 0.5
    series = {o: ep[COL_MAP[o]].to_numpy() for o in OBS_MV}
    return t, mask, contact, series, had_reset


def process_file(csv_path, variant, terrain, seed):
    t, mask, contact, series, had_reset = load_masked_episode(csv_path, variant, terrain)
    runs = touchdown_runs(contact, mask)

    state_pairs = {o: [] for o in OBS_MV}
    mv_pairs = []  # (z_n (state_dim,), z_np1 (state_dim,))

    for run in runs:
        for i, j in zip(run[:-1], run[1:]):
            zn = np.array([series[o][i] for o in OBS_MV])
            znp1 = np.array([series[o][j] for o in OBS_MV])
            for o in OBS_MV:
                # 4th element is episode time at the section crossing (t_i), carried
                # only so the scatter figure can color points by time -- unused by
                # any fit (seed_disagreement/fit_ar1 only read the first 3 fields).
                state_pairs[o].append((series[o][i], series[o][j], seed, float(t[i])))
            mv_pairs.append((zn, znp1, seed))

    return state_pairs, mv_pairs, had_reset


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #

def bimodality_coef(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 10 or x.std() < 1e-12:
        return np.nan
    z = (x - x.mean()) / x.std()
    skew = np.mean(z ** 3)
    kurt = np.mean(z ** 4)  # Pearson (non-excess); normal ~ 3
    return (skew ** 2 + 1) / kurt


def fit_ar1(xn, xnp1, n_boot=2000):
    xn, xnp1 = np.asarray(xn, float), np.asarray(xnp1, float)
    n = len(xn)
    reason = ""
    if n < 6:
        return dict(n=n, x_star=np.nan, lam=np.nan, lam_raw=np.nan, ci=(np.nan, np.nan),
                    r2=np.nan, reason="too few strides (n<6), not fit")
    if xn.std() < 1e-9:
        return dict(n=n, x_star=np.nan, lam=np.nan, lam_raw=np.nan, ci=(np.nan, np.nan),
                    r2=np.nan, reason="degenerate: no variance in x_n")
    b, a = np.polyfit(xn, xnp1, 1)
    x_star = a / (1 - b) if abs(1 - b) > 1e-8 else np.nan
    pred = a + b * xn
    ss_res = np.sum((xnp1 - pred) ** 2)
    ss_tot = np.sum((xnp1 - xnp1.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else np.nan

    boot = np.empty(n_boot)
    idx = np.arange(n)
    for k in range(n_boot):
        s = RNG.choice(idx, size=n, replace=True)
        if xn[s].std() < 1e-9:
            boot[k] = np.nan
            continue
        bb, _ = np.polyfit(xn[s], xnp1[s], 1)
        boot[k] = bb
    ci = tuple(np.nanpercentile(boot, [2.5, 97.5]))
    return dict(n=n, x_star=x_star, lam=b, lam_raw=b, ci=ci, r2=r2, reason=reason)


def seed_disagreement(pairs):
    """Between-seed spread of per-seed mean vs. pooled within-seed spread."""
    by_seed = {}
    for a, b, s, *_ in pairs:
        by_seed.setdefault(s, []).extend([a, b])
    seed_means = np.array([np.mean(v) for v in by_seed.values() if len(v) > 0])
    resid = []
    for v in by_seed.values():
        v = np.array(v)
        resid.append(v - v.mean())
    resid = np.concatenate(resid) if resid else np.array([])
    between = seed_means.std() if len(seed_means) > 1 else 0.0
    within = resid.std() if len(resid) > 1 else np.nan
    flag = bool(within > 0 and not np.isnan(within) and between > within)
    return flag, between, within


def fit_multivariate(mv_pairs, state_dim=STATE_DIM):
    n = len(mv_pairs)
    reason = ""
    if n < state_dim + 1:
        return dict(n=n, spectral_radius=np.nan, eig=[np.nan] * state_dim,
                    cond=np.nan, underdet=True, reason=f"too few strides (n<{state_dim+1}), not fit")
    Zn = np.array([p[0] for p in mv_pairs])
    Znp1 = np.array([p[1] for p in mv_pairs])
    X = np.column_stack([np.ones(n), Zn])
    coef, *_ = np.linalg.lstsq(X, Znp1, rcond=None)  # (state_dim+1, state_dim)
    a = coef[0, :]
    J = coef[1:, :].T
    try:
        z_star = np.linalg.solve(np.eye(state_dim) - J, a)
    except np.linalg.LinAlgError:
        z_star = np.full(state_dim, np.nan)
    eig = np.linalg.eigvals(J)
    spectral_radius = float(np.max(np.abs(eig)))
    cond = float(np.linalg.cond(J))
    underdet = n < 5 * state_dim
    if underdet:
        reason = f"underdetermined: n_strides={n} < 5*state_dim={5*state_dim}"
    return dict(n=n, spectral_radius=spectral_radius, eig=eig.tolist(), cond=cond,
                underdet=underdet, reason=reason, z_star=z_star)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = build_index()
    print(f"[poincare] canonical files: {len(index)} (expect {len(VARIANTS)*len(TERRAINS)*len(SPEEDS)*8})")

    uni_rows, mv_rows = [], []
    n_reset_files = 0
    scatter_cache = {}  # (variant, terrain) at vx=0.20 -> (spine_angle_n, spine_angle_np1, lam, x_star)

    for v in VARIANTS:
        for terr in TERRAINS:
            for cmd in SPEEDS:
                if v not in SPINE_VARIANTS:
                    # rigid has no spine actuator -- excluded explicitly, never silently
                    # dropped, and never fed a nonexistent q_spine/qd_spine column.
                    for obs in OBS_UNI:
                        uni_rows.append(dict(variant=v, terrain=terr, speed=cmd, observable=obs,
                                             n_strides=0, x_star=np.nan, lam=np.nan, ci_low=np.nan,
                                             ci_high=np.nan, r2=np.nan, dev_lo=np.nan, dev_hi=np.nan,
                                             bimodal=np.nan, seed_disagree=np.nan, lam_raw=np.nan,
                                             reason="rigid has no spine joint (8 leg motors only, "
                                                    "no spine actuator) -- not applicable"))
                    eig_cols = {f"eig{i+1}": np.nan for i in range(STATE_DIM)}
                    mv_rows.append(dict(variant=v, terrain=terr, speed=cmd, n_strides=0,
                                        spectral_radius=np.nan, spectral_radius_raw=np.nan,
                                        **eig_cols, condition_number=np.nan, underdetermined=True,
                                        reason="rigid has no spine joint -- not applicable"))
                    continue

                pooled_state = {o: [] for o in OBS_MV}
                pooled_mv = []
                n_files = 0
                for seed in range(8):
                    key = (v, terr, cmd, seed)
                    if key not in index:
                        continue
                    n_files += 1
                    sp, mvp, had_reset = process_file(index[key], v, terr, seed)
                    n_reset_files += int(had_reset)
                    for o in OBS_MV:
                        pooled_state[o].extend(sp[o])
                    pooled_mv.extend(mvp)

                for obs in OBS_UNI:
                    pairs = pooled_state[obs]
                    n_strides = len(pairs)
                    if n_strides == 0:
                        uni_rows.append(dict(variant=v, terrain=terr, speed=cmd, observable=obs,
                                             n_strides=0, x_star=np.nan, lam=np.nan, ci_low=np.nan,
                                             ci_high=np.nan, r2=np.nan, dev_lo=np.nan, dev_hi=np.nan,
                                             bimodal=np.nan, seed_disagree=np.nan, lam_raw=np.nan,
                                             reason=f"no valid touchdown pairs ({n_files} files found)"))
                        continue
                    xn = np.array([p[0] for p in pairs]); xnp1 = np.array([p[1] for p in pairs])
                    fit = fit_ar1(xn, xnp1)
                    dis_flag, between, within = seed_disagreement(pairs)
                    dev = np.concatenate([xn, xnp1]) - (fit["x_star"] if not np.isnan(fit["x_star"]) else np.mean(np.concatenate([xn, xnp1])))
                    bc = bimodality_coef(dev)
                    bimodal_flag = bool(not np.isnan(bc) and bc > 0.555)

                    reason_parts = [fit["reason"]] if fit["reason"] else []
                    suppress = bool(fit["reason"]) or dis_flag or bimodal_flag
                    if dis_flag:
                        reason_parts.append(f"seed disagreement (between={between:.4g} > within={within:.4g})")
                    if bimodal_flag:
                        reason_parts.append(f"bimodal deviations (BC={bc:.3f})")
                    report_lam = np.nan if suppress else fit["lam"]
                    report_ci = (np.nan, np.nan) if suppress else fit["ci"]
                    # x* presupposes a single shared fixed point across seeds; if seeds
                    # disagree that premise is false, so null it too (not just lambda).
                    report_xstar = np.nan if dis_flag else fit["x_star"]

                    uni_rows.append(dict(variant=v, terrain=terr, speed=cmd, observable=obs,
                                         n_strides=n_strides, x_star=report_xstar, lam=report_lam,
                                         ci_low=report_ci[0], ci_high=report_ci[1], r2=fit["r2"],
                                         dev_lo=float(dev.min()), dev_hi=float(dev.max()),
                                         bimodal=bimodal_flag, seed_disagree=dis_flag,
                                         lam_raw=fit["lam_raw"],
                                         reason="; ".join(reason_parts)))

                    if terr in TERRAINS and cmd == 0.20 and obs == "spine_angle":
                        # cache the SAME (possibly-suppressed) lambda the CSV reports,
                        # never the raw fit -- the figure must not show a number the
                        # tidy CSV declined to report. Also cache each pair's episode
                        # time (t_i) so the scatter can be colored by time, not variant.
                        times = np.array([p[3] for p in pairs])
                        scatter_cache[(v, terr)] = (xn, xnp1, times, report_lam, report_xstar,
                                                    "; ".join(reason_parts))

                mv = fit_multivariate(pooled_mv)
                eig = mv["eig"] if isinstance(mv["eig"], list) else [np.nan] * STATE_DIM
                eig = eig + [np.nan] * (STATE_DIM - len(eig))
                dis_flags = [seed_disagreement(pooled_state[o])[0] for o in OBS_MV if len(pooled_state[o])]
                mv_seed_disagree = bool(any(dis_flags))
                suppress_mv = bool(mv["underdet"]) or mv_seed_disagree
                report_sr = np.nan if suppress_mv else mv["spectral_radius"]
                reason = mv["reason"]
                if mv_seed_disagree and not mv["underdet"]:
                    reason = (reason + "; " if reason else "") + "seed disagreement in >=1 state component"
                eig_cols = {f"eig{i+1}": eig[i] for i in range(STATE_DIM)}
                mv_rows.append(dict(variant=v, terrain=terr, speed=cmd, n_strides=mv["n"],
                                    spectral_radius=report_sr, spectral_radius_raw=mv["spectral_radius"],
                                    **eig_cols,
                                    condition_number=mv["cond"], underdetermined=mv["underdet"],
                                    reason=reason))

    uni_df = pd.DataFrame(uni_rows)
    mv_df = pd.DataFrame(mv_rows)
    uni_df.to_csv(OUT / "return_map_univariate.csv", index=False)
    mv_df.to_csv(OUT / "return_map_multivariate.csv", index=False)

    # --------------------------- figures ---------------------------------- #
    make_scatter_fig(scatter_cache)
    make_spectral_bar_fig(mv_df)

    best = pick_best_trial(index)
    make_best_trial_fig(best)
    make_classified_scatter_fig(best)
    make_walkcycle_fig(best, index)
    make_phase_space_fig(best, index)
    print("\n[poincare] single best trial per (variant, terrain) at vx=0.20 "
          "(best = most valid FL-touchdown strides in the analysis window):")
    for v in SPINE_VARIANTS:
        for terr in TERRAINS:
            top = best.get((v, terr))
            if top is None:
                print(f"    {v:6s} {terr:6s}  -- no canonical file at cmd=0.20")
            else:
                n, seed, _sp = top
                flag = "  ** too few strides to fit (n<6)" if 0 < n < 6 else ""
                print(f"    {v:6s} {terr:6s}  seed={seed}  n_strides={n}{flag}")

    # --------------------------- summary ----------------------------------- #
    n_spine_files = len(SPINE_VARIANTS) * len(TERRAINS) * len(SPEEDS) * 8
    print(f"\n[poincare] state variable: spine angle (q_spine) + spine rate (qd_spine); "
          f"rigid excluded (no spine joint) -- {n_spine_files} spine-bearing files scanned "
          f"({len(SPINE_VARIANTS)} variants x {len(TERRAINS)} terrains x {len(SPEEDS)} speeds x 8 seeds).")
    print(f"[poincare] {n_reset_files} of those had a genuine mid-episode reset (done>0); "
          f"those had their tail truncated + last 0.5s trimmed. "
          f"0 files had terminated>0 (see module docstring).")
    n_cells_uni = len(uni_df)
    n_na_variant = int((uni_df.reason.str.contains("no spine joint", na=False)).sum())
    n_suppressed = int((uni_df.lam.isna() & (uni_df.n_strides > 0)).sum())
    n_zero = int((uni_df.n_strides == 0).sum())
    print(f"[poincare] univariate: {n_cells_uni} (variant,terrain,speed,observable) rows "
          f"({n_na_variant} are rigid, not applicable).")
    print(f"           {n_zero} had zero valid touchdown pairs (rigid rows included in this count).")
    print(f"           {n_suppressed} had pairs but lambda suppressed (too few strides / "
          f"seed disagreement / bimodal / degenerate) -- see `reason` column.")
    stable = uni_df[uni_df.lam.notna()]
    print(f"           {len(stable)} rows report a usable lambda; "
          f"|lambda|<1 (stable) in {(stable.lam.abs()<1).sum()}/{len(stable)} of those.")

    n_mv = len(mv_df)
    n_mv_under = int(mv_df.underdetermined.sum())
    n_mv_report = int(mv_df.spectral_radius.notna().sum())
    print(f"\n[poincare] multivariate: {n_mv} cells, {n_mv_under} flagged underdetermined "
          f"(n_strides < 5*state_dim={5*STATE_DIM}, rigid's {len(TERRAINS)*len(SPEEDS)} "
          f"not-applicable cells counted as underdetermined), {n_mv_report} report a usable "
          f"spectral radius.")
    print("\n[poincare] cells with zero valid touchdown pairs (variant, terrain, speed):")
    for _, r in uni_df[(uni_df.observable == "spine_angle") & (uni_df.n_strides == 0)].iterrows():
        print(f"    {r.variant:6s} {r.terrain:6s} {r.speed:.2f}  -- {r.reason}")
    print("\n[poincare] cells with a suppressed (not-reported) lambda, observable=spine_angle:")
    for _, r in uni_df[(uni_df.observable == "spine_angle") & (uni_df.n_strides > 0) & (uni_df.lam.isna())].iterrows():
        print(f"    {r.variant:6s} {r.terrain:6s} {r.speed:.2f}  n={r.n_strides:3d}  -- {r.reason}")
    print(f"\nwrote {OUT/'return_map_univariate.csv'} ({len(uni_df)} rows)")
    print(f"wrote {OUT/'return_map_multivariate.csv'} ({len(mv_df)} rows)")


def pick_best_trial(index, cmd=0.20):
    """Best trial per (variant, terrain) at the highest tested speed.

    "Best" = the seed with the most valid FL-touchdown pairs for spine_angle
    inside this script's own analysis window (settle / on-lip / reset masking
    -- identical to every other number in this report) -- i.e. the run that
    sustained the longest clean, steady-state gait record under that terrain's
    specific challenge. Ties broken by lowest seed number for determinism.
    rigid is excluded (no spine joint), same as everywhere else.

    Returns {(variant, terrain): (n_strides, seed, state_pairs) | None}.
    """
    best = {}
    for v in SPINE_VARIANTS:
        for terr in TERRAINS:
            top = None
            for seed in range(8):
                key = (v, terr, cmd, seed)
                if key not in index:
                    continue
                sp, _mvp, _had_reset = process_file(index[key], v, terr, seed)
                n = len(sp["spine_angle"])
                if top is None or n > top[0]:
                    top = (n, seed, sp)
            best[(v, terr)] = top
    return best


def make_best_trial_fig(best, cmd=0.20):
    all_times = [p[3] for top in best.values() if top and top[0] > 0
                 for p in top[2]["spine_angle"]]
    tmin = min(all_times) if all_times else SETTLE_S
    norm = matplotlib.colors.Normalize(vmin=tmin, vmax=16.0)
    cmap = matplotlib.colormaps["viridis"]

    for theme_name, t in THEME.items():
        fig, axes = plt.subplots(len(SPINE_VARIANTS), len(TERRAINS), figsize=(14, 10.5),
                                 facecolor=t["surface"])
        for r, v in enumerate(SPINE_VARIANTS):
            for c, terr in enumerate(TERRAINS):
                ax = axes[r, c]
                ax.set_facecolor(t["surface"])
                top = best.get((v, terr))
                if top is None or top[0] == 0:
                    msg = "no canonical file" if top is None else "no valid strides"
                    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=8,
                           color=t["ink2"], transform=ax.transAxes, style="italic")
                    ax.set_xticks([]); ax.set_yticks([])
                else:
                    n, seed, sp = top
                    pairs = sp["spine_angle"]
                    xn = np.array([p[0] for p in pairs])
                    xnp1 = np.array([p[1] for p in pairs])
                    times = np.array([p[3] for p in pairs])
                    order = np.argsort(times)
                    xn, xnp1, times = xn[order], xnp1[order], times[order]

                    # Stride-order path first (under the dots): this is ONE continuous
                    # trial, unlike the pooled figure, so connecting consecutive strides
                    # is meaningful -- it traces the actual orbit toward (or away from)
                    # the fixed point.
                    ax.plot(xn, xnp1, "-", lw=0.6, color=t["grid"], alpha=0.9, zorder=1)
                    ax.scatter(xn, xnp1, s=16, c=times, cmap=cmap, norm=norm,
                              edgecolors="none", zorder=2)

                    lo, hi = min(xn.min(), xnp1.min()), max(xn.max(), xnp1.max())
                    pad = 0.05 * (hi - lo + 1e-9)
                    lo, hi = lo - pad, hi + pad
                    ax.plot([lo, hi], [lo, hi], "--", color=t["ink2"], lw=0.8, label="identity")
                    if n >= 6:
                        fit = fit_ar1(xn, xnp1)
                        if not np.isnan(fit["lam"]):
                            xx = np.linspace(lo, hi, 2)
                            b, a = fit["lam"], (xnp1.mean() - fit["lam"] * xn.mean())
                            ax.plot(xx, a + b * xx, "-", color=t["ink"], lw=1.2,
                                    label=f"lambda={fit['lam']:.2f}")
                    ax.legend(fontsize=6, frameon=False, loc="upper left")
                    ax.text(0.98, 0.02, f"seed {seed}, n={n}", fontsize=6, color=t["ink2"],
                           ha="right", va="bottom", transform=ax.transAxes, style="italic")
                    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
                if r == 0:
                    ax.set_title(terr, color=t["ink"], fontsize=11)
                if c == 0:
                    ax.set_ylabel(f"{v}\nspine angle$_{{n+1}}$ (rad)", color=t["ink"], fontsize=9)
                if r == len(SPINE_VARIANTS) - 1:
                    ax.set_xlabel("spine angle$_n$ (rad)", color=t["ink"], fontsize=9)
                ax.tick_params(colors=t["ink2"], labelsize=7)
                for s in ax.spines.values():
                    s.set_color(t["grid"])
        fig.suptitle(f"Single best trial per variant x terrain, cmd vx={cmd:.2f} m/s "
                     "(best = most valid strides; rigid has no spine joint; "
                     "color = time within episode, line = stride order)",
                     color=t["ink"], fontsize=11)
        fig.tight_layout(rect=[0, 0, 0.93, 0.95])
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cax = fig.add_axes([0.94, 0.12, 0.015, 0.76])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("time in episode (s)", color=t["ink"], fontsize=9)
        cb.ax.tick_params(colors=t["ink2"], labelsize=7)
        cb.outline.set_edgecolor(t["grid"])
        p = OUT / f"fig_return_map_best_trial_{theme_name}.png"
        safe_savefig(fig, p, dpi=130, facecolor=t["surface"])
        plt.close(fig)
        print(f"wrote {p}")


# --------------------------------------------------------------------------- #
# Reproducing Poincare Map (Chen & Zhang, Ch.4 "Control of Motion and
# Compliance"): Fig 4.6.3 (Poincare section marked on the walking trajectory),
# Fig 4.6.4-left (leg angle vs time for one step), Fig 4.6.4-right (phase
# portrait for one step), and the Sect. 4.1.4.1 / Table 4.6.1 classification of
# Poincare-section points as feasible/stable/unstable vs. unfeasible.
# --------------------------------------------------------------------------- #

CLASS_COLORS_LIGHT = {"stable": "#1b8a5a", "unstable end": "#c0392b",
                      "feasible": "#8a8f98", "unfeasible": "#000000"}
CLASS_COLORS_DARK = {"stable": "#3fd68a", "unstable end": "#ff6b5b",
                     "feasible": "#aeb4bd", "unfeasible": "#ffffff"}


def classify_pairs(xn, xnp1, x_star, variant):
    """Operationalizes the book's basin-of-attraction classification (Sect.
    4.1.4.1: sink cells are "unfeasible or represent a fixed point"; cells in
    the basin converge to the fixed point, cells outside it diverge toward a
    fall) onto individual observed (x_n, x_n+1) touchdown pairs, since we have
    real strides rather than a systematically-sampled grid of initial
    conditions:

      unfeasible   |x_n| or |x_n+1| within 1 deg of the joint's hard stop
                   (HARD_STOP) -- a saturated, not freely-oscillating, state.
      stable       not unfeasible, and the step moved clearly closer to the
                   fixed point: |x_n+1 - x*| <= 0.95 * |x_n - x*|.
      unstable end not unfeasible, and the step moved clearly farther from the
                   fixed point: |x_n+1 - x*| >= 1.05 * |x_n - x*|.
      feasible     not unfeasible, and neither clearly convergent nor
                   divergent (a valid state, just not confidently classified
                   as heading toward or away from the fixed point).

    This is our own operational rule for applying the book's concept to
    stride-level data, not a value computed by the book -- stated explicitly
    so it isn't mistaken for a literal reproduction of Table 4.6.1's method
    (which classifies a single fixed point via Jacobian eigenvalues, not
    individual strides).
    """
    stop = HARD_STOP[variant] - np.radians(1.0)
    unfeasible = (np.abs(xn) >= stop) | (np.abs(xnp1) >= stop)
    dn = np.abs(xn - x_star)
    dnp1 = np.abs(xnp1 - x_star)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(dn > 1e-12, dnp1 / dn, 1.0)
    labels = np.full(len(xn), "feasible", dtype=object)
    labels[(~unfeasible) & (ratio <= 0.95)] = "stable"
    labels[(~unfeasible) & (ratio >= 1.05)] = "unstable end"
    labels[unfeasible] = "unfeasible"
    return labels


def make_classified_scatter_fig(best):
    """Fig 4.6.3-style Poincare map: the best-trial return map (same data as
    make_best_trial_fig), but colored by the feasible/stable/unstable-end/
    unfeasible classification instead of time."""
    for theme_name, t in THEME.items():
        cc = CLASS_COLORS_LIGHT if theme_name == "light" else CLASS_COLORS_DARK
        fig, axes = plt.subplots(len(SPINE_VARIANTS), len(TERRAINS), figsize=(14, 10.5),
                                 facecolor=t["surface"])
        for r, v in enumerate(SPINE_VARIANTS):
            for c, terr in enumerate(TERRAINS):
                ax = axes[r, c]
                ax.set_facecolor(t["surface"])
                top = best.get((v, terr))
                if top is None or top[0] == 0:
                    msg = "no canonical file" if top is None else "no valid strides"
                    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=8,
                           color=t["ink2"], transform=ax.transAxes, style="italic")
                    ax.set_xticks([]); ax.set_yticks([])
                else:
                    n, seed, sp = top
                    pairs = sp["spine_angle"]
                    xn = np.array([p[0] for p in pairs])
                    xnp1 = np.array([p[1] for p in pairs])
                    fit = fit_ar1(xn, xnp1) if n >= 6 else None
                    x_star = fit["x_star"] if fit and not np.isnan(fit["x_star"]) else \
                        float(np.mean(np.concatenate([xn, xnp1])))
                    labels = classify_pairs(xn, xnp1, x_star, v)
                    for cat in ["feasible", "stable", "unstable end", "unfeasible"]:
                        m = labels == cat
                        if m.any():
                            ax.scatter(xn[m], xnp1[m], s=16, color=cc[cat], label=cat,
                                      edgecolors="none", alpha=0.85, zorder=3 if cat != "feasible" else 2)
                    lo, hi = min(xn.min(), xnp1.min()), max(xn.max(), xnp1.max())
                    pad = 0.05 * (hi - lo + 1e-9)
                    lo, hi = lo - pad, hi + pad
                    ax.plot([lo, hi], [lo, hi], "--", color=t["ink2"], lw=0.8, zorder=1)
                    ax.axvline(x_star, ls=":", color=t["ink2"], lw=0.7, zorder=1)
                    ax.axhline(x_star, ls=":", color=t["ink2"], lw=0.7, zorder=1)
                    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
                    ax.text(0.98, 0.02, f"seed {seed}, n={n}", fontsize=6, color=t["ink2"],
                           ha="right", va="bottom", transform=ax.transAxes, style="italic")
                if r == 0:
                    ax.set_title(terr, color=t["ink"], fontsize=11)
                if c == 0:
                    ax.set_ylabel(f"{v}\nspine angle$_{{n+1}}$ (rad)", color=t["ink"], fontsize=9)
                if r == len(SPINE_VARIANTS) - 1:
                    ax.set_xlabel("spine angle$_n$ (rad)", color=t["ink"], fontsize=9)
                ax.tick_params(colors=t["ink2"], labelsize=7)
                for s in ax.spines.values():
                    s.set_color(t["grid"])
        handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=cc[cat],
                              markersize=6, label=cat)
                  for cat in ["stable", "unstable end", "feasible", "unfeasible"]]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.97),
                  fontsize=8, frameon=False, ncol=1)
        fig.suptitle("Poincare map classified per Sect. 4.1.4.1's basin-of-attraction concept "
                     "(single best trial, vx=0.20 m/s; dotted lines mark the fitted fixed point)",
                     color=t["ink"], fontsize=11)
        fig.tight_layout(rect=[0, 0, 0.90, 0.95])
        p = OUT / f"fig_return_map_classified_{theme_name}.png"
        safe_savefig(fig, p, dpi=130, facecolor=t["surface"])
        plt.close(fig)
        print(f"wrote {p}")


def make_walkcycle_fig(best, index, cmd=0.20):
    """Fig 4.6.4-left analog: spine angle vs time over the best trial, with FL
    touchdowns marked as gray vertical lines -- Fig 4.6.3's "gray region is the
    Poincare section" translated to a time-series (a marked instant, since our
    section is a touchdown event, not a continuous phase band). No time
    gradient here: time already IS the x-axis, so re-encoding it as color would
    be redundant -- the gradient is used only in the two plots below where time
    is not already an axis."""
    for theme_name, t in THEME.items():
        fig, axes = plt.subplots(len(SPINE_VARIANTS), len(TERRAINS), figsize=(15, 8.5),
                                 facecolor=t["surface"], sharey=False)
        for r, v in enumerate(SPINE_VARIANTS):
            for c, terr in enumerate(TERRAINS):
                ax = axes[r, c]
                ax.set_facecolor(t["surface"])
                top = best.get((v, terr))
                if top is None or top[0] == 0:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=8,
                           color=t["ink2"], transform=ax.transAxes, style="italic")
                    ax.set_xticks([]); ax.set_yticks([])
                    continue
                n, seed, sp = top
                key = (v, terr, cmd, seed)
                tt, mask, contact, series, _ = load_masked_episode(index[key], v, terr)
                runs = touchdown_runs(contact, mask)
                td_idx = sorted({i for run in runs for i in run})
                tm, qm = tt[mask], series["spine_angle"][mask]
                ax.plot(tm, qm, "-", lw=1.0, color=t["vcol"][v])
                for i in td_idx:
                    ax.axvline(tt[i], color=t["grid"], lw=0.6, alpha=0.8, zorder=0)
                if r == 0:
                    ax.set_title(terr, color=t["ink"], fontsize=11)
                if c == 0:
                    ax.set_ylabel(f"{v}\nspine angle (rad)", color=t["ink"], fontsize=9)
                if r == len(SPINE_VARIANTS) - 1:
                    ax.set_xlabel("time (s)", color=t["ink"], fontsize=9)
                ax.tick_params(colors=t["ink2"], labelsize=7)
                for s in ax.spines.values():
                    s.set_color(t["grid"])
        fig.suptitle(f"Spine angle over the walk cycle, single best trial, vx={cmd:.2f} m/s "
                     "(gray lines = FL-touchdown Poincare-section crossings, cf. Fig 4.6.3)",
                     color=t["ink"], fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        p = OUT / f"fig_walkcycle_{theme_name}.png"
        safe_savefig(fig, p, dpi=130, facecolor=t["surface"])
        plt.close(fig)
        print(f"wrote {p}")


def make_phase_space_fig(best, index, cmd=0.20):
    """Fig 4.6.4-right analog: continuous (spine angle, spine rate) trajectory
    for the best trial, colored by time (meaningful here since time is not an
    axis), connected in time order."""
    norm = matplotlib.colors.Normalize(vmin=SETTLE_S, vmax=16.0)
    cmap = matplotlib.colormaps["viridis"]

    for theme_name, t in THEME.items():
        fig, axes = plt.subplots(len(SPINE_VARIANTS), len(TERRAINS), figsize=(14, 10.5),
                                 facecolor=t["surface"])
        for r, v in enumerate(SPINE_VARIANTS):
            for c, terr in enumerate(TERRAINS):
                ax = axes[r, c]
                ax.set_facecolor(t["surface"])
                top = best.get((v, terr))
                if top is None or top[0] == 0:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=8,
                           color=t["ink2"], transform=ax.transAxes, style="italic")
                    ax.set_xticks([]); ax.set_yticks([])
                    continue
                n, seed, sp = top
                key = (v, terr, cmd, seed)
                tt, mask, contact, series, _ = load_masked_episode(index[key], v, terr)
                tm = tt[mask]
                qm, qdm = series["spine_angle"][mask], series["spine_rate"][mask]
                ax.plot(qm, qdm, "-", lw=0.5, color=t["grid"], alpha=0.8, zorder=1)
                ax.scatter(qm, qdm, s=4, c=tm, cmap=cmap, norm=norm, edgecolors="none", zorder=2)
                ax.text(0.98, 0.02, f"seed {seed}", fontsize=6, color=t["ink2"],
                       ha="right", va="bottom", transform=ax.transAxes, style="italic")
                if r == 0:
                    ax.set_title(terr, color=t["ink"], fontsize=11)
                if c == 0:
                    ax.set_ylabel(f"{v}\nspine rate (rad/s)", color=t["ink"], fontsize=9)
                if r == len(SPINE_VARIANTS) - 1:
                    ax.set_xlabel("spine angle (rad)", color=t["ink"], fontsize=9)
                ax.tick_params(colors=t["ink2"], labelsize=7)
                for s in ax.spines.values():
                    s.set_color(t["grid"])
        fig.suptitle("Phase portrait, single best trial, vx=0.20 m/s (cf. Fig 4.6.4-right; "
                     "color = time within episode)", color=t["ink"], fontsize=11)
        fig.tight_layout(rect=[0, 0, 0.93, 0.95])
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cax = fig.add_axes([0.94, 0.12, 0.015, 0.76])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("time in episode (s)", color=t["ink"], fontsize=9)
        cb.ax.tick_params(colors=t["ink2"], labelsize=7)
        cb.outline.set_edgecolor(t["grid"])
        p = OUT / f"fig_phase_space_{theme_name}.png"
        safe_savefig(fig, p, dpi=130, facecolor=t["surface"])
        plt.close(fig)
        print(f"wrote {p}")


def make_scatter_fig(scatter_cache):
    # Global time normalization (not per-panel) so a given color means the same
    # absolute episode time everywhere -- e.g. a curb panel only reaching the
    # early part of the colormap is itself informative (falls cut the run short).
    tmin = min(float(v[2].min()) for v in scatter_cache.values())
    tmax = 16.0  # fixed episode length
    norm = matplotlib.colors.Normalize(vmin=tmin, vmax=tmax)
    cmap = matplotlib.colormaps["viridis"]

    for theme_name, t in THEME.items():
        fig, axes = plt.subplots(4, 4, figsize=(14, 13), facecolor=t["surface"])
        for r, v in enumerate(VARIANTS):
            for c, terr in enumerate(TERRAINS):
                ax = axes[r, c]
                ax.set_facecolor(t["surface"])
                key = (v, terr)
                if key in scatter_cache:
                    xn, xnp1, times, lam, x_star, reason = scatter_cache[key]
                    ax.scatter(xn, xnp1, s=10, alpha=0.75, c=times, cmap=cmap, norm=norm,
                              edgecolors="none")
                    lo, hi = min(xn.min(), xnp1.min()), max(xn.max(), xnp1.max())
                    pad = 0.05 * (hi - lo + 1e-9)
                    lo, hi = lo - pad, hi + pad
                    ax.plot([lo, hi], [lo, hi], "--", color=t["ink2"], lw=0.8, label="identity")
                    if not np.isnan(lam):
                        xx = np.linspace(lo, hi, 2)
                        b, a = lam, (xnp1.mean() - lam * xn.mean())
                        ax.plot(xx, a + b * xx, "-", color=t["ink"], lw=1.2,
                                label=f"lambda={lam:.2f}")
                        ax.legend(fontsize=6, frameon=False, loc="upper left")
                    else:
                        # Same suppression as the CSV: never print a lambda the tidy
                        # table declined to report -- show the raw scatter only.
                        ax.legend(fontsize=6, frameon=False, loc="upper left")
                        ax.text(0.5, 0.03, f"not reported: {reason}", ha="center", va="bottom",
                               fontsize=6, color=t["ink2"], transform=ax.transAxes, style="italic")
                    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
                else:
                    msg = "no spine joint" if v not in SPINE_VARIANTS else "no data"
                    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=8,
                           color=t["ink2"], transform=ax.transAxes, style="italic")
                    ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(terr, color=t["ink"], fontsize=11)
                if c == 0:
                    ax.set_ylabel(f"{v}\nspine angle$_{{n+1}}$ (rad)", color=t["ink"], fontsize=9)
                if r == 3:
                    ax.set_xlabel("spine angle$_n$ (rad)", color=t["ink"], fontsize=9)
                ax.tick_params(colors=t["ink2"], labelsize=7)
                for s in ax.spines.values():
                    s.set_color(t["grid"])
        fig.suptitle("FL-touchdown return map, spine angle, cmd vx=0.20 m/s "
                     "(rows=variant, cols=terrain; rigid has no spine joint; "
                     "color = time within episode)",
                     color=t["ink"], fontsize=12)
        fig.tight_layout(rect=[0, 0, 0.93, 0.97])
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cax = fig.add_axes([0.94, 0.12, 0.015, 0.76])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("time in episode (s)", color=t["ink"], fontsize=9)
        cb.ax.tick_params(colors=t["ink2"], labelsize=7)
        cb.outline.set_edgecolor(t["grid"])
        p = OUT / f"fig_return_map_{theme_name}.png"
        safe_savefig(fig, p, dpi=130, facecolor=t["surface"])
        plt.close(fig)
        print(f"wrote {p}")


def make_spectral_bar_fig(mv_df):
    for theme_name, t in THEME.items():
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=t["surface"], sharey=True)
        width = 0.19
        group_x = np.arange(len(TERRAINS))
        for ax, cmd in zip(axes, SPEEDS):
            ax.set_facecolor(t["surface"])
            sub = mv_df[mv_df.speed == cmd]
            for i, v in enumerate(VARIANTS):
                xs = group_x + (i - 1.5) * width
                heights, hatches = [], []
                for terr in TERRAINS:
                    row = sub[(sub.variant == v) & (sub.terrain == terr)]
                    if len(row) == 0 or pd.isna(row.spectral_radius.iloc[0]):
                        heights.append(0.0); hatches.append(True)
                    else:
                        heights.append(float(row.spectral_radius.iloc[0])); hatches.append(False)
                bars = ax.bar(xs, heights, width=width * 0.92, color=t["vcol"][v], label=v, zorder=3)
                for bar, h, missing in zip(bars, heights, hatches):
                    if missing:
                        ax.text(bar.get_x() + bar.get_width() / 2, 0.02, "x", ha="center",
                               va="bottom", fontsize=8, color=t["ink2"])
            ax.axhline(1.0, ls="--", lw=0.9, color=t["ink2"], zorder=2)
            ax.set_xticks(group_x); ax.set_xticklabels(TERRAINS, color=t["ink"])
            ax.set_title(f"vx={cmd:.2f} m/s", color=t["ink"], fontsize=10)
            ax.tick_params(colors=t["ink2"])
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ax.spines.values():
                s.set_color(t["grid"])
            ax.grid(axis="y", lw=0.5, color=t["grid"], zorder=0)
        axes[0].set_ylabel("spectral radius (dominant Floquet mult.)", color=t["ink"], fontsize=9)
        axes[0].legend(fontsize=7.5, frameon=False, loc="upper left", ncol=2)
        fig.suptitle("Spine (angle, rate) dominant Floquet multiplier by variant x terrain x "
                     "speed ('x' = not reported: rigid has no spine, underdetermined, or "
                     "disagreeing seeds)", color=t["ink"], fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        p = OUT / f"fig_spectral_radius_{theme_name}.png"
        safe_savefig(fig, p, dpi=130, facecolor=t["surface"])
        plt.close(fig)
        print(f"wrote {p}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
