# Microtaur spine study — 2026-09-25 update: what changed, what is still open, and the teacher-first plan

Branch `spine-study-isaac`. Supersedes the plan sections of
`MEMO_2026-09-23_spine_study.md` where they conflict. Every number here was
produced by running code on 2026-09-23/25; the script that produced it is named
next to it.

---

## 1. Summary

1. **The model to use is upstream's 2026-09-20 rigid config, not this repo's.**
   It lives in the private `aryan-chandra-cmu/Microtaur_RL` (HEAD `5880025`),
   total mass **0.540 kg** (verified). It is now reorganised into a standalone
   package, `microtaur_study/`, that is **bit-identical to upstream** on every
   deterministic part (§3).
2. **IsaacLab/PhysX can hold the closed chain**, but only with
   `solver_position_iteration_count = 16` and `dt = 0.0025 s` (§2). The
   verdict is not final: it was measured at `kd = 0.045`, and the 09-20 config
   uses `kd = 0`.
3. **A minimal reward (D0) was written and piloted three times.** It learns to
   walk and then gives it up, because with copper loss included walking at
   0.14 m/s costs more reward than it earns (§4). The fix is an energy-weight
   curriculum, not more reward terms.
4. **Plan changed to teacher-first** (§6): privileged actor = critic, a fixed
   Gaussian block map with random spawns for terrain B, a step course with flat
   margins for terrain C, amplitude ramped from 0 to 29.5 mm peak-to-peak.

---

## 2. IsaacLab closed-chain spike

Scripts: `tools/spike_mujoco_reference.py` (pure MuJoCo reference),
`tools/spike_isaac_closed_chain.py` (USD built from the compiled MjModel, PhysX
loop-closure joints). Isaac Sim 5.0.0 rc.45 pip, IsaacLab 2.3.0 clone at
`7b26eb4`, RTX 5080.

- **MuJoCo reference.** The training physics itself carries **0.24 mm** of
  closure error under load (soft `<connect>` constraints; 0 without gravity,
  independent of solver iterations). That is the baseline PhysX is judged
  against.
- **Representation.** Each leg's two `<connect>` sites share one axis, so one
  PhysX *revolute* loop joint per leg is exact. Two spherical joints (literal
  MJCF translation) jitter 7x more.
- **Result.** Closure gap, 512 envs, independent of env count:

| motion | pos_iters | dt | p99 | max |
|---|---|---|---|---|
| MuJoCo trot (training settings) | — | 0.005 | 0.49 mm | 0.50 mm |
| PhysX trot | 4 (default) | 0.005 | 5.0 mm | 6.5 mm |
| PhysX trot | 16 | 0.005 | 1.13 mm | 6.51 mm |
| **PhysX trot** | **16** | **0.0025** | **0.38 mm** | **1.16 mm** |

  Criterion: p99 below MuJoCo's, max below 20% of a 5 mm step. `16 / 0.0025`
  passes the first and misses the second by 16% (0.003% of samples > 1 mm).
  Cost: twice the physics steps. 150k env-steps/s at 2048 envs, dt 0.005 (idle
  GPU); ~75k at dt 0.0025 is extrapolated, not measured.
- **Replay.** The recorded rigid RL policy (`microtaur_runs/cpg_rl/results/data/rigid/rl_64s.npz`)
  replayed open-loop walks in PhysX with 0/32 falls over 60 s, but **13% faster
  and 4.5 mm taller** than in MuJoCo: a contact-model sim2sim gap that still
  needs calibrating.

---

## 3. The 09-20 config, reorganised: `microtaur_study/`

Upstream `env_cfgs.py` is 2529 lines in one file, with ten task aliases for one
config and ~40 environment-variable knobs. It is now split by concern:

| module | contents |
|---|---|
| `robot.py` | names, XL330 motor + electrical model, the one MjSpec patch, mass check (fails if total ≠ 0.540 kg) |
| `sim2real.py` | stage 0/1/2 table (was spread over six functions) |
| `actions.py` | 8 motor actions, predictive five-bar safety filter, delay |
| `observations.py` | actor 33 / critic 60 and the deployment-contract check |
| `sensors.py`, `events.py`, `commands.py`, `terminations.py`, `terrain.py` | one concern each |
| `rewards.py` | D0 (§4), one function per term |
| `env_cfg.py` | assembly only |
| `kinematics.py` | unchanged copy |

Tasks `Microtaur-Rigid-{Flat,Rough}-S{0,1,2}` are registered explicitly (no
env-var stage switch). Model files are byte-identical copies of upstream's.

**Equivalence, `tests/check_equivalence.py`.** mujoco-warp is *not*
bit-deterministic on GPU — upstream against itself diverges after 2 steps —
so trajectories cannot be compared. Instead:

- bit-exact (max diff 0.000): state after reset; requested / safe / applied /
  processed motor targets over 300 steps with upstream's and this package's
  action terms on the same env; the safety filter on 3 × 4096 random states;
- statistical: root height, tilt, feet in contact and reset counts over
  512 envs × 500 steps differ from upstream by the same order as upstream
  differs from itself.

Found while doing this, all in upstream:

- **`reset()` crashes on mjlab ≥ 1.6** (`.unsqueeze()` in the IK reset). Fixed
  here; upstream still has it.
- **The 30 Hz control-rate override never runs** (`hasattr(cfg.sim, "dt")` is
  always False), so 09-20 trains at **50 Hz** like everything before it. Kept at
  50 Hz here and set explicitly; changing it to the ~30 Hz hardware rate is a
  decision, not a bug fix.
- **The GAPS doc's spec-patch table was wrong:** armature / frictionloss /
  damping are already in the XML; only the motor range patch changes anything.
  (Corrected in `6f719a6`.)

---

## 4. Reward D0 and three pilots

`rewards.py`, checked by `tests/check_rewards.py` (every term matches a hand
computation on the same state):

| term | weight | form |
|---|---|---|
| track_lin_vel_xy | +1.0 | exp(−\|v_cmd − v\|² / 0.10²) |
| track_ang_vel_z | +0.5 | exp(−(ω_cmd − ω)² / 0.15²), always on |
| motor_energy | −0.025 | Σ_all motors (\|τ q̇\| + 159 τ²) / (m g · 0.15) |
| action_rate | −0.05/8 | Σ_all motors (Δa)² |
| termination | −2.0 per event | non-timeout terminations |

Every motor cost sums over all motors with one formula (`qfrc_actuator` is zero
on passive joints, so a spine motor is included automatically). There is no
posture term; posture is bounded by terminations only.

**Copper loss.** From the XL330-M077 datasheet at 5 V: R = 3.40 Ω,
k_t = 0.146 N·m/A, so R/k_t² = **159 W/(N·m)²**. On the 09 rigid RL gait,
copper loss is **5.07 W against 1.32 W mechanical** — ignoring it would
understate energy about 5x. ⚠ k_t from the stall point disagrees by 17% with
the back-EMF constant from the no-load speed; calibrate against the current
traces in the upstream hardware characterisation before trusting the absolute
value.

Pilots (flat, stage 0, 2048 envs, logs in `runs_local/`, not committed):

| pilot | change | outcome |
|---|---|---|
| 1 | pure D0 (no termination penalty, σ 0.04) | **suicide**: every episode ended in 0.1 s by dropping below the height limit. Under initial exploration noise, living costs ~15 W of copper per step; dying costs 0 |
| 2 | + termination −2.0 | **stands still** for 900 iterations: with σ 0.04 a standing robot scores 6e-6 on forward tracking, no gradient toward walking |
| 3 | + σ 0.10 | **walks at iteration 100** (forward 0.54, copper 13 W), **then returns to standing** by 200 and stays there to 1000 |

Pilot 3 is the informative one: walking adds +0.40 of tracking and costs −0.56
of energy, so at weight 0.025 standing is optimal for the inefficient gait PPO
finds first. The structure is fine; the energy weight must start at 0 and ramp
in once walking is established.

---

## 5. Open concerns

1. **Spike at kd = 0** not yet run (09-20 uses kd = 0; less damping may raise closure spikes and jitter).
2. ~~`frictionloss` has no PhysX mapping.~~ **Corrected:** on Isaac Sim 5.0 actuator `friction` is a constant torque; measured threshold 0.009–0.011 N·m for a 0.010 setting and 0.019–0.021 for 0.020 (`tools/spike_isaac_joint_friction.py`, CPU PhysX). The XML values carry over directly.
3. **Contact-model sim2sim gap** (+13% speed, +4.5 mm height in PhysX).
4. **Copper-loss constant** is datasheet-derived with a 17% internal inconsistency.
5. **Reward terms read body quantities one physics substep stale** — mjlab calls `forward()` only before observations. Documented mjlab behaviour, same for upstream; relevant when reading logs.
6. **Control rate** 50 Hz in sim vs ~30 Hz on hardware.
7. **Block-grid regression is not identifiable** (twist and curvature correlate +0.95 across grid maps; `TERRAIN_DESIGN.md` §3b). Per-axis answers must come from the typed test field.
8. **Checkpoints.** The only rigid policy checkpoint (`rigid_flat_fixed_m077/model_4499.pt`) is on naomio's machine, trained at 0.465 kg.
9. **Copied model.** `microtaur_study/assets/` holds a copy of upstream's 09-20 XML (differs from this repo's XML in one `<inertial>` line, whose values were already public in the GAPS doc).

---

## 6. Revised plan — teacher first

The student is deferred. The immediate goal is a privileged teacher that walks
on A, B and C across the full amplitude range.

### 6.1 Teacher observations

Actor and critic receive the same privileged set: the 60-dim critic vector
(base linear velocity, foot heights, air time, contacts, contact forces, plus the
deployable 33) **plus the height map**. The deployment contract check stays, but
applies to the future student, not the teacher.

Map encoding has to be easy for an MLP: a yaw-aligned grid of heights relative
to the root. With 70 mm cells the scan spacing must be ≤ 35 mm to resolve cell
edges (`TERRAIN_DESIGN.md` §6): 0.40 × 0.30 m at 35 mm → 12 × 9 = 108 values.

### 6.2 Terrains

| | geometry | randomness | commands |
|---|---|---|---|
| **A** | flat | — | full curriculum |
| **B** | flat-topped 70 mm cells on a 10 mm heightfield; cell heights **Gaussian**, clipped at the ceiling | **one fixed map** (fixed seed); variety comes from **random spawn position and yaw** on it | full curriculum |
| **C** | one step running along the path, left high / right low, with **flat margins** before and after the step section and ≥ 1 m each side of the seam | fixed geometry, mirrored in both directions | **straight only, no yaw** |

Amplitude ceiling (computed with `kinematics.py` at mid-stance, retract
26.22 mm, extend 32.86 mm, total 59.1 mm):

- **50% of each direction:** +13.1 mm up, −16.4 mm down, i.e. **29.5 mm
  peak-to-peak**, equivalently ±14.8 mm (±25% of total travel) around stand.
- Difficulty ramps **0% → 100%** of that ceiling through `terrain_levels`
  (rows = amplitude scale of the same fixed pattern).
- ⚠ Extension collapses with swing (11.1 mm available at 50% when
  |common| = 0.3, 4.7 mm at 0.5), so the "down" side is the one to watch.

### 6.3 Curricula

- **Commands:** three stages as in `TERRAIN_DESIGN.md` §5 (yaw from step 0,
  finished by iteration ~625). C envs get zero yaw always.
- **Terrain:** `terrain_levels` promote/demote, starting at level 0.
- **Energy weight:** 0 until forward tracking is established, then ramped to the
  target. The target is swept (upper bound from pilot 3: walking stays optimal
  only below ~0.018 for the first gait PPO finds: 0.40 tracking gain / 22.5 raw energy difference).

### 6.4 Order of work

1. Energy curriculum on flat; sweep 2–3 target weights (≈10 min each).
2. Teacher observation group (privileged actor, 35 mm scan).
3. Terrain B and C as mjlab heightfield sub-terrains wrapping `tools/terrain_geometry.py`; terrain-aware spawn (C: along the seam, no yaw) and command masking on C.
4. Teacher pilot ~6000 iterations to find where capability flattens; fix the budget from it.
5. Then the three spine variants on the same pipeline; student distillation after.

### 6.5 Decisions needed

- Gaussian σ for B relative to the ceiling (e.g. clip at 2σ so σ = 7.4 mm).
- Whether C's step section repeats along the course or is a single long step (the doc's negative-control argument favours a single sustained step).
- Energy curriculum trigger: fixed schedule vs tracking threshold.
- Sim2real stage for the teacher.

---

## Reproduce

```bash
# mjlab 1.6 venv (uv): mjlab==1.6.0, mujoco 3.11, rsl-rl-lib 5.4.2
uv pip install -e microtaur_study
python microtaur_study/tests/check_rewards.py
python microtaur_study/tests/check_equivalence.py --upstream-src <upstream>/src
train Microtaur-Rigid-Flat-S0 --env.scene.num-envs 2048 --agent.max-iterations 1000 --agent.logger tensorboard
```
