"""Eval-time spine ablations: is the pitch-on-steps advantage mediated by
spine MECHANICS or by the leg controller that happened to be learned on a
spined body? No retraining -- every intervention acts on a frozen checkpoint
at evaluation time.

    python scripts/spine_ablation.py --variant pitch --terrain flat --condition c0 --seed 0
    python scripts/spine_ablation.py --variant pitch --terrain steps --condition c1 --seed 0
    python scripts/spine_ablation.py --variant pitch --terrain steps --condition c3 --alpha-pct 50 --seed 0

Conditions
----------
c0  baseline, unmodified.
c1  HARD LOCK: robot.write_joint_state_to_sim() forces the spine's qpos=q0,
    qvel=0 immediately before every policy step. This is a direct kinematic
    override, not a stiffened PD servo -- confirmed against mjlab's real
    Entity API (see the discovery notes in this file's docstring below).
c2  PASSIVE: the spine PD servo (kp=1.0, kd=0.045) stays active and can still
    deflect under load; only the *commanded target* is frozen at q0, by
    overwriting the spine action term's `_applied_targets` AND
    `_processed_actions` every step (see apply_c2_freeze for why both, not
    just the former, are required for this to actually change the physics).
c3  ACTIVE DRIVE AUTHORITY SWEEP (deliberately NOT a ROM clamp -- the spine
    has no physical joint limit at all; see "Discovery: no physical spine
    limit" below). Scales the policy's own commanded deviation from q0 by
    alpha = alpha_pct/100:

        tgt_effective = q0 + alpha * (tgt_policy - q0)

    PD gains and compliance are left completely untouched -- this isolates
    active policy driving specifically. alpha_pct=100 must reproduce C0
    exactly (tgt_effective = tgt_policy); alpha_pct=0 must reproduce C2
    exactly (tgt_effective = q0). Both are checked empirically, not just
    asserted by construction, before trusting the sweep.

Discovery: no physical spine limit
-----------------------------------
`env.unwrapped.sim.model.jnt_range` is `[0, 0]` and `jnt_limited=0` for the
spine joint in every compiled variant -- confirmed directly, not assumed. The
apparent "hard limit" (`ACTIVE_SPINE_HARD_LIMIT_RAD`) is enforced entirely in
software, inside the spine action term's own `_filter_target()`, never as a
real MuJoCo constraint. Tried forcing `jnt_limited=1` plus a tight
`jnt_range` directly on the live warp model, followed by
`env.unwrapped.sim.create_graph()` (mjlab's documented recovery for
CUDA-graph-stale array edits) -- the joint sailed straight through the
attempted clamp regardless, so this is not a graph-staleness bug: adding a
constraint that never existed at compile time isn't something a live
in-place array edit can retrofit; it would need the model rebuilt from a
modified `mujoco.MjSpec` before compilation. That approach (labeled (A) in
project discussion) was set aside in favor of the authority sweep above
(labeled (B)) as the primary design, per instruction -- (A) would install a
hard stop that never existed in training, which is a bigger distribution
shift than C1, not a gentler one, and would be uninterpretable if it produced
a performance cliff.

Discovery notes (see the eval-wrapper spec this script implements)
--------------------------------------------------------------------------
* spine_joint_pos / spine_joint_vel ARE separate observation terms (actor AND
  critic) for every spined variant, reading robot.data.joint_pos/joint_vel
  live each call -- so every condition above is automatically observed
  truthfully by the policy with no extra wiring.
* rigid has no spine joint or action term at all -- c1/c2/c3 are not
  applicable and this script refuses to run them for --variant rigid.
* Task id, seeding order, and cfg/env construction path
  (make_env_and_teacher) are copied verbatim from scripts/rollout_log.py's
  run_one() so a C0 run here is byte-identical to the existing baseline data
  for the same (variant, terrain, vx, seed) -- this is what makes paired
  comparison valid.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MICROTAUR_USE_JOYSTICK_COMMANDS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from microtaur_variants import VARIANTS, use_variant  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

TASK_ID = "Mjlab-Velocity-Yaw-Flat-Microtaur"
SPINE_VARIANTS = ("pitch", "yaw", "roll")
BL_M = 0.168
G = 9.81
OUT = Path("rollouts/spine_ablation")


# --------------------------------------------------------------------------- #
# Spine resolution + interventions
# --------------------------------------------------------------------------- #

def resolve_spine(env_u):
    """(robot, spine_action_term, joint_ids tensor) -- None, None, None for rigid."""
    if "spine_pos" not in env_u.action_manager._terms:  # rigid has no such term
        return None, None, None
    robot = env_u.scene["robot"]
    spine_term = env_u.action_manager.get_term("spine_pos")
    joint_ids = spine_term._target_ids
    return robot, spine_term, joint_ids


def get_spine_q(robot, joint_ids):
    return float(robot.data.joint_pos[0, joint_ids[0]].item())


def apply_c1_lock(robot, joint_ids, q0, device):
    """Force qpos=q0, qvel=0 on the spine DOF. Call every policy step, right
    before env.step() -- see module docstring for why this (not a stiffened
    PD servo) is the primary mechanism."""
    n = robot.data.joint_pos.shape[0]
    pos = torch.full((n, len(joint_ids)), float(q0), device=device)
    vel = torch.zeros((n, len(joint_ids)), device=device)
    robot.write_joint_state_to_sim(pos, vel, joint_ids=joint_ids, env_ids=None)


def apply_c2_freeze(spine_term, q0):
    """Freeze the commanded spine target at a constant.

    MicrotaurActiveSpinePositionAction.process_actions() sets TWO attributes:
    `_applied_targets` (rebound each call, exists only for external
    introspection/logging -- e.g. rollout_log.py's jm.applied_targets()) and
    `_processed_actions` (updated IN PLACE, `self._processed_actions[:] =
    applied`) -- the latter is what the inherited JointPositionAction.
    apply_actions() actually reads to call set_joint_position_target() and
    drive the real PD servo. Freezing only `_applied_targets` verifies as
    correct in logging but has ZERO effect on physics -- caught this via the
    C2 verification check (q_spine trace was bit-identical to an unfrozen
    run despite `spine_target_applied` correctly showing a frozen value; see
    module notes). Both must be frozen for the intervention to be real AND
    correctly self-reported.
    """
    spine_term._applied_targets[:] = float(q0)
    spine_term._processed_actions[:] = float(q0)


def apply_c3_authority(spine_term, q0, alpha):
    """ACTIVE DRIVE AUTHORITY SWEEP (not a ROM clamp -- the spine has no
    physical joint limit to clamp; see module notes). Scales the policy's own
    already-processed, already-delayed commanded deviation from q0 by alpha:

        tgt_effective = q0 + alpha * (tgt_policy - q0)

    PD gains and compliance are completely untouched -- this only reshapes
    what the policy is ALLOWED to actively command, same mechanism as C2
    (same two attributes, `_applied_targets` for introspection and
    `_processed_actions` for the real physics path -- see apply_c2_freeze),
    generalized with a continuous authority knob:

        alpha = 1  ->  tgt_effective = tgt_policy   ->  identical to C0
        alpha = 0  ->  tgt_effective = q0            ->  identical to C2

    Must be called AFTER the term's own (unpatched) process_actions has run
    for this step, so `_applied_targets` already holds tgt_policy.
    """
    tgt_policy = spine_term._applied_targets
    tgt_eff = q0 + alpha * (tgt_policy - q0)
    spine_term._applied_targets[:] = tgt_eff
    spine_term._processed_actions[:] = tgt_eff


def measure_natural_range(env, robot, joint_ids, policy, obs, n_steps, device):
    """Free (no intervention) rollout to measure C0's own spine ROM: q0 (mean)
    and half-range (max|q-q0|), used to center/scale c1/c2/c3.

    Takes the WRAPPED env (RslRlVecEnvWrapper), not env.unwrapped -- an
    earlier version passed the unwrapped env here, which skips the wrapper's
    own action-history bookkeeping (prev_action/prev_prev_action) that the
    policy's observations and the action-rate reward term depend on. That bug
    was caught by the c3-alpha=100-must-equal-c0 verification itself: the
    aggregate distance looked deceptively close (4.6808 vs 4.6808) but the
    per-step trace diverged by up to 0.095 rad in q_spine -- exactly the
    signature of accumulated small state differences compounding through a
    nonlinear feedback loop, not measurement noise.
    """
    from microtaur_velocity.distill_reliable.mjlab_utils import step_env
    qs = []
    for _ in range(n_steps):
        with torch.no_grad():
            raw = torch.nan_to_num(policy(obs), nan=0.0, posinf=6.0, neginf=-6.0)
        obs, _, _done, _ = step_env(env, raw)
        qs.append(get_spine_q(robot, joint_ids))
    q0 = float(np.mean(qs))
    half_range = float(np.max(np.abs(np.array(qs) - q0)))
    return q0, half_range, obs


# --------------------------------------------------------------------------- #
# Single rollout
# --------------------------------------------------------------------------- #

def run_one(variant, terrain, vx, seed, condition, alpha_pct, sim_seconds, device,
            q0_override=None, half_range_override=None, verbose=False):
    from microtaur_velocity.distill_reliable.mjlab_utils import (
        get_initial_obs, make_env_and_teacher, step_env,
    )
    from microtaur_velocity.env_cfgs import set_joystick_twist_command

    if condition != "c0" and variant not in SPINE_VARIANTS:
        raise ValueError(f"{variant} has no spine joint; only c0 is valid for it")

    torch.manual_seed(seed)
    np.random.seed(seed)

    hook = None
    if terrain != "flat":
        from microtaur_velocity.microtaur_terrains import apply_terrain
        def hook(env_cfg, _t=terrain):  # noqa: E306
            apply_terrain(env_cfg, _t)

    ckpt = VARIANTS[variant].ckpt
    env, _agent_cfg, policy = make_env_and_teacher(
        TASK_ID, str(ckpt), num_envs=1, device=device,
        robust=False, no_terminations=False, cfg_hook=hook,
    )
    env_u = env.unwrapped
    try:
        env_u.seed(seed)
    except Exception:  # noqa: BLE001
        pass

    robot = env_u.scene["robot"]
    spine_robot, spine_term, joint_ids = resolve_spine(env_u)
    policy_dt = float(getattr(env_u, "step_dt", 0.02))
    n_steps = int(round(sim_seconds / policy_dt))

    obs = get_initial_obs(env)
    set_joystick_twist_command(env, vx, 0.0)

    # q0 / natural ROM: measured from THIS run's own first ~1s under C0
    # dynamics (no intervention yet), matching "per-(variant,terrain,speed)
    # mean angle from the C0 baseline" -- done inline so every condition uses
    # a consistently-measured reference without depending on a prior run
    # having been executed first.
    q0, half_range = q0_override, half_range_override
    n_settle = int(round(1.0 / policy_dt))
    if condition != "c0" and (q0 is None or half_range is None):
        q0, half_range, obs = measure_natural_range(
            env, robot, joint_ids, policy, obs, n_settle, device)
        set_joystick_twist_command(env, vx, 0.0)
        if verbose:
            print(f"[measure] q0={q0:+.4f} rad, half_range={half_range:.4f} rad")

    if condition == "c2":
        orig_process = spine_term.process_actions
        def patched(actions, _orig=orig_process, _q0=q0):
            _orig(actions)
            apply_c2_freeze(spine_term, _q0)
        spine_term.process_actions = patched

    if condition == "c3":
        alpha = alpha_pct / 100.0
        orig_process = spine_term.process_actions
        def patched(actions, _orig=orig_process, _q0=q0, _alpha=alpha):
            _orig(actions)
            apply_c3_authority(spine_term, _q0, _alpha)
        spine_term.process_actions = patched

    if condition == "c1":
        # Re-assert the lock every 5ms PHYSICS SUBSTEP (env_u.sim.step, the
        # same hook rollout_log.py's --substep-log wraps), not just every 20ms
        # policy step -- a once-per-policy-step correction left ~0.017 rad
        # residual RMS (about half the natural 0.032 rad oscillation), nowhere
        # near "~0"; the joint was drifting freely for the full 20ms between
        # corrections. Correcting before each of the 4 substeps bounds the
        # free-drift window to 5ms instead.
        _sim_step = env_u.sim.step
        def _locked_step(*a, **kw):
            apply_c1_lock(robot, joint_ids, q0, device)
            return _sim_step(*a, **kw)
        env_u.sim.step = _locked_step

    leg_term = env_u.action_manager.get_term("joint_pos")
    leg_joint_ids = leg_term._target_ids

    rows = []
    for i in range(n_steps):
        with torch.no_grad():
            raw = torch.nan_to_num(policy(obs), nan=0.0, posinf=6.0, neginf=-6.0)
        obs, _, done, _ = step_env(env, raw)
        set_joystick_twist_command(env, vx, 0.0)

        p = robot.data.root_link_pos_w[0].cpu().numpy()
        vbx = float(robot.data.root_link_lin_vel_b[0, 0])
        row = dict(
            t=(i + 1) * policy_dt, done=int(bool(done[0].item())),
            base_x=float(p[0]), base_y=float(p[1]), base_z=float(p[2]),
            v_body_x=vbx,
        )
        # Raw action vector (pre-processing, exactly what's diffed for the
        # action_rate reward term) -- logged so action_rate (confusion-vs-
        # mechanics diagnostic) can be computed post-hoc without a rerun.
        raw_np = raw[0].detach().cpu().numpy()
        for k in range(raw_np.shape[0]):
            row[f"action_raw_{k}"] = float(raw_np[k])
        # Leg joint positions -- logged so the leg-observation distribution
        # under C1-steps vs C0-steps can be compared post-hoc.
        leg_q = robot.data.joint_pos[0, leg_joint_ids].detach().cpu().numpy()
        for k in range(leg_q.shape[0]):
            row[f"leg_q_{k}"] = float(leg_q[k])
        if joint_ids is not None:
            row["q_spine"] = get_spine_q(robot, joint_ids)
            row["qd_spine"] = float(robot.data.joint_vel[0, joint_ids[0]].item())
            row["spine_target_applied"] = float(spine_term.applied_targets[0, 0].item())
        rows.append(row)
        if bool(done[0].item()):
            break

    env.close()
    return dict(rows=rows, q0=q0, half_range=half_range,
                spine_present=joint_ids is not None)


# --------------------------------------------------------------------------- #
# CLI: one rollout, or a batch cell -> tidy CSV row(s)
# --------------------------------------------------------------------------- #

def summarize(result, variant, terrain, vx, seed, condition, alpha_pct, sim_seconds=16.0):
    """Windowing note: C0 logs t=[~0, sim_seconds] (starts immediately). Every
    other condition runs a ~1s measure_natural_range() settle phase BEFORE its
    own logged window, so it covers t=[~1, 1+sim_seconds] instead -- a
    DIFFERENT absolute segment of an actively-walking trajectory, not just a
    time-shifted copy of the same one. Comparing raw distance/velocity across
    conditions without correcting for this would bias every non-C0 condition
    (their window skips the messier post-reset settling C0's window still
    includes). Every other analysis in this project already discards the
    first 1.0s as transient (SETTLE_S=1.0); this restricts every condition's
    metrics to the overlapping t in [1.0, sim_seconds] window so the
    comparison is apples-to-apples without needing to rerun anything -- the
    raw per-step CSV still has the full trace if a different window is ever
    needed."""
    import pandas as pd
    full_rows = result["rows"]
    full_n = len(full_rows)
    df_full = pd.DataFrame(full_rows)
    df = df_full[(df_full.t >= 1.0) & (df_full.t <= sim_seconds)].reset_index(drop=True)
    n = len(df)
    # A fall is a property of the whole run, not just the analysis window --
    # check df_full, not the windowed df, so a fall just after sim_seconds
    # (outside the window) is still reported.
    fell = bool(df_full.done.iloc[-1]) if full_n else False
    fall_t = float(df_full.t.iloc[-1]) if fell else np.nan
    dist_bl = float(df.base_x.iloc[-1] - df.base_x.iloc[0]) / BL_M if n else np.nan
    v_ach = float(df.v_body_x.mean()) if n else np.nan
    # For c0, q0 was never measured (no intervention needs it) -- report the
    # trace's own mean post-hoc, purely as a reference value for comparison
    # against c1/c2/c3's pre-measured q0, not used to drive anything.
    q0 = result["q0"]
    if q0 is None and "q_spine" in df.columns and n:
        q0 = float(df.q_spine.mean())
    out = dict(variant=variant, terrain=terrain, vx=vx, seed=seed, condition=condition,
               alpha_pct=alpha_pct if condition == "c3" else np.nan,
               n_steps=n, fell=fell, fall_time_s=fall_t,
               distance_BL=dist_bl, v_achieved=v_ach,
               v_shortfall=vx - v_ach if not np.isnan(v_ach) else np.nan,
               q0_used=q0, half_range_used=result["half_range"],
               spine_present=result["spine_present"])
    if "q_spine" in df.columns:
        out.update(
            spine_rms=float(np.sqrt(np.mean(df.q_spine ** 2))),
            spine_rom=float(df.q_spine.max() - df.q_spine.min()),
            spine_peak_rate=float(df.qd_spine.abs().max()),
            spine_resid_from_q0_rms=float(np.sqrt(np.mean((df.q_spine - q0) ** 2))) if q0 is not None else np.nan,
        )
    action_cols = [c for c in df.columns if c.startswith("action_raw_")]
    if action_cols and n > 1:
        A = df[action_cols].to_numpy()
        out["action_rate_rms"] = float(np.sqrt(np.mean(np.sum(np.diff(A, axis=0) ** 2, axis=1))))
    else:
        out["action_rate_rms"] = np.nan
    out["n_steps_total"] = full_n  # unfiltered -- true run length, e.g. to a fall
    return out, df_full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["rigid", "pitch", "yaw", "roll"])
    ap.add_argument("--terrain", required=True, choices=["flat", "steps"])
    ap.add_argument("--condition", required=True, choices=["c0", "c1", "c2", "c3"])
    ap.add_argument("--alpha-pct", type=float, default=100.0)
    ap.add_argument("--vx", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    with use_variant(args.variant):
        result = run_one(args.variant, args.terrain, args.vx, args.seed, args.condition,
                         args.alpha_pct, args.seconds, args.device, verbose=args.verbose)

    summary, df = summarize(result, args.variant, args.terrain, args.vx, args.seed,
                            args.condition, args.alpha_pct, sim_seconds=args.seconds)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.variant}_{args.terrain}_{args.condition}"
    if args.condition == "c3":
        tag += f"_{int(args.alpha_pct)}pct"
    tag += f"_seed{args.seed}"
    df.to_csv(OUT / f"{tag}.csv", index=False)
    with open(OUT / f"{tag}.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
