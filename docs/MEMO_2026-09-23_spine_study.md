# Microtaur spine study — state of the code, and the experiment we are going to run

2026-09-23 · branch `spine-study-isaac`

This memo has two halves. **Part 1** is what a person needs to know before touching the training
code — what is broken, what is not comparable, and what has been measured. **Part 2** is the
experiment design and the reasoning behind it.

Every number in Part 1 was re-verified against the code or against logged data on 2026-09-22/23,
not taken from prose. Where a previously-circulated claim turned out to be wrong, it is marked.

---

# Part 1 — state of the code

## 1.1 The four morphologies are not currently comparable

The project has four variants of one body: `rigid`, `roll` (a.k.a. `active_twist`), `pitch`, `yaw`,
differing in the axis of a single actuated trunk joint. Any claim of the form "morphology A beats
morphology B" is, as of today, unsupportable, because they differ in six ways beyond the spine axis.

**Reward weights — `yaw` alone is the outlier.** A previously circulated version of this said the
mismatch was between `roll` and `yaw`; it is not, `rigid`/`roll`/`pitch` agree exactly:

| | rigid / roll / pitch | **yaw** |
|---|---|---|
| `feet_air_time` | 0.35 | **0.15** |
| `moving_contact_pattern` | 0.25 | **0.20** |
| `normalized_acceleration` | −0.025 | **−0.01** |
| `action_rate` | −0.07 | **−0.04** |
| `AIR_TIME_TARGET_S` / `AIR_TIME_SIGMA_S` | 0.15 / 0.045 | **0.10 / 0.06** |

Sources: `rigid_env.txt:912-948`, `active_roll_env.txt:1069-1122`, `pitch_env.txt:1176-1222`,
`active_yaw_env_cfgs.txt:1051-1097`. Cross-checked against each run's own saved
`params/env.yaml`; `active_twist_077_motors/git/microtaur.diff:406-418` shows those four weights
were unchanged before that run (only comments were removed).

**Consequence:** `microtaur_runs/training_results/results.md` states *"Reward:
`microtaur_scaled_walk`, identical across variants."* That is **false for yaw**, so the table of
flat final returns (107.8 / 108.0 / 109.3 / 109.4) has yaw computed under a *different reward
function* and its value cannot be compared to the others.

**Joint range — `pitch` alone is the outlier.** `MICROTAUR_JOINT_HALF_RANGE_RAD` is 0.95 for pitch
and 0.75 for rigid, roll and yaw. (The dataclass default of 0.98 is never used.)

**The spine action term is not even the same kind of controller.**

| | roll | pitch | **yaw** |
|---|---|---|---|
| control law | learned position, `action_scale_rad` raised by curriculum | same | **`steering_gain_s = 0.8` feedforward from the yaw command + `residual_scale_rad = 0.1` residual** |
| policy authority | full | full | **±0.1 rad ≈ ±5.7° only** |
| `target_limit_rad` | 0.32 (18.3°) | 0.524 (30°) | 0.30 (17.2°) |
| standing-neutral term | `standing_spine_neutral` | `standing_spine_neutral` | `zero_yaw_spine_neutral` |
| `spine_mean_bias` | present (−0.05) | absent | absent |

For yaw, the spine target is mostly *prescribed* by the steering command. The premise "the spine
receives no positive reward, so whatever it does is emergent" therefore **does not hold for yaw**.
Roll and pitch are the clean cases.

**The spine unlock is a curriculum, and it differs per variant.** The command curriculum term also
raises the spine's `action_scale_rad`:

| | step 0 | 25k | 50k | 70k | 90k |
|---|---|---|---|---|---|
| roll | **0.000** (fully locked) | 0.087 (5°) | 0.157 (9°) | 0.227 (13°) | 0.297 (17°) |
| pitch | 0.140 (8°) | 0.209 (12°) | 0.314 (18°) | 0.419 (24°) | 0.524 (30°) |
| yaw | — no such entry — | | | | |

Two unrelated concerns (command difficulty, spine authority) share one `CurriculumTermCfg` and one
schedule. They need to be split before either can be varied independently.

**The spine motor model is wrong on roll and yaw.** From each run's own `params/env.yaml`:

| run | leg eff/sat/vel | spine eff/sat/vel |
|---|---|---|
| roll | 0.129 / 0.215 / 32.086 = M077 ✅ | 0.129 / 0.215 / 32.086 = **the leg M077 table** ❌ |
| yaw | same ✅ | same ❌ |
| pitch | same ✅ | 0.312 / 0.52 / 8.629 = M288 ✅ |

The physical spine is an XL330-M288 — 2.42× the torque and 1/3.72 the speed of the M077 used in
the legs. Only the pitch run split the two tables.

> **Note on method, because it cost us a wrong conclusion once.** An earlier analysis inferred from
> commit dates that the *legs* had been trained with the wrong motor model. That was wrong. Each
> run directory carries `git/microtaur.diff` recording the **uncommitted** working-tree edits used
> for that run, so commit time is not the time the code changed. **The ground truth for what a run
> used is that run's own `params/env.yaml`, never a commit date and never a directory name.**

**Config vintage.** `ENV_CFG_REVISION`: rigid `2026-09-20-rigid-hw-calibrated-sim2real-v3-observation-fix`,
roll `2026-09-05-active-twist-emergent-v4-mean-bias-fix`, pitch `2026-09-01-active-pitch-final-xml-v2`,
yaw `2026-08-30-active-yaw-spine-residual-v1`. The three spine variants are 15–23 days behind rigid
and have not received the 09-20 observation work.

## 1.2 Bugs confirmed by execution

**`decimation` has never been overridden. Every run is 50 Hz.**

```python
# rigid_env.txt:2189-2192 and the same lines in the 09-20 env_cfgs.py
# Run the policy at approximately 60 Hz while leaving the base physics
# timestep unchanged.
if hasattr(cfg.sim, "dt") and hasattr(cfg, "decimation"):
  cfg.decimation = max(1, int(round((1.0 / 60.0) / float(cfg.sim.dt))))
```

`SimulationCfg` has no `dt` field — the timestep is `cfg.sim.mujoco.timestep`. The guard is always
False. Building the config and printing gives `decimation = 4`, `timestep = 0.005`,
**control rate 50.00 Hz**. The 60 Hz in the old comment and the 30 Hz / 28.57 Hz in the 09-20
comment were **never implemented**. One-line fix: `cfg.sim.dt` → `cfg.sim.mujoco.timestep`.

This matters because the hardware control loop tops out at **~22–25 Hz** (Bridge RPC, ~20 ms per
call, two calls per tick), confirmed three independent ways: our own measurement (~22 Hz), the
`origin/hardware` branch's `motor_trace.csv` (dt 37–41 ms → 24–27 Hz), and the calibration
archive's Test 09/09B (25–33 Hz zero overrun, 35 Hz 0.28%, 37 Hz 2.70%, 40 Hz 96.74% overrun).
Network compute is not the constraint: 185,865 parameters, 0.153 ms total per tick.

**The environment cannot `reset()` on mjlab ≥ 1.6.**

```
RuntimeError: shape mismatch: value tensor of shape [128, 1]
cannot be broadcast to indexing result of shape [128, 1, 1]
  at mjlab/entity/entity.py:1072
  from microtaur_ik_consistent_reset -> robot.set_joint_position_target(
        spine_q, joint_ids=spine_ids.unsqueeze(0), env_ids=env_ids.unsqueeze(1))
```

Two call sites per env file (one for the spine, one for the eight leg motors). **The fix is already
in this repository**, in the per-variant run snapshots, with its reason:

> *"MJLab >= 1.6 outer-indexes 1-D env and joint IDs internally, so pass them unexpanded;
> pre-expanding them to [N, 1] and [1, 8] breaks broadcasting."*

Drop the two `.unsqueeze()` calls. Until this is done, nothing can be built, played, or validated
locally.

**`upright` is the dominant spine cost, and it is not labelled as one.**

`upright` (weight 0.75) is evaluated on **both** reconstructed trunk halves — the midpoint frame
rotated by ±q/2 about the spine axis — so a spine angle q tilts each half by q/2 and reduces the
reward directly. Assuming an upright midpoint frame and the 15° kernel:

| spine angle q | tilt per half | `upright` | loss at weight 0.75 |
|---|---|---|---|
| 0° | 0° | 1.000 | — |
| 10° | 5° | 0.900 | 0.075 |
| 18.3° (roll's travel limit) | 9.2° | 0.728 | **0.204** |
| 30° (pitch's travel limit) | 15° | 0.500 | 0.375 |

At roll's travel limit, `spine_limit_proximity` contributes 0.049. **`upright` is ~4× the largest
of the four named spine cost terms.** The claim "the spine appears in the reward only as a cost"
remains true, but the four named terms account for about a fifth of that cost. This is arguably
correct physics — tilting half the trunk 9° off vertical does impair locomotion — but it must be
named and quantified rather than left implicit, and it constrains how the emergence claim can be
worded.

Note also that the loophole the two-half form was written to close is already bounded by the
mechanical travel limit: maximum half-splay is 9.2° (roll), 8.6° (yaw), 15° (pitch).

**`spine_mean_bias` (−0.05) penalises exactly the postural use of the spine.** It low-passes the
spine angle (α = 0.02, ≈1 s) and charges a sustained one-sided offset **only while travelling
straight**; fast zero-mean oscillation averages to zero and is free. That is a deliberate and
reasonable design, but it means that on any terrain with a sustained lateral or longitudinal bias,
the "hold a constant offset to redistribute leg workspace" strategy is charged while the
"oscillate" strategy is not. Decide this explicitly per experiment rather than inheriting it.

## 1.3 The rough-terrain runs exist, and are not usable

Four rounds of rough training were run 2026-09-12 → 09-14 (logs under `microtaur_runs/`). The
final round, `*_rough_from_flat` (09-14 16:35–21:08), completed 4499 iterations for all four
variants, warm-started from the flat `model_4499.pt` with the 63 new height-scan input weights
initialised to zero.

| | flat final return | flat episode | flat T95 / T90 | rough final return | rough episode | final terrain level |
|---|---|---|---|---|---|---|
| rigid | 107.8 ± 0.5 | 1000 | 877 / 826 | 78.2 ± 2.1 | 873 | 1.40 |
| **roll** | 108.0 ± 0.6 | 1000 | **2729 / 1697** | **64.8 ± 4.2** | **727** | **1.06** |
| pitch | 109.3 ± 0.5 | 1000 | 1186 / 845 | 78.1 ± 2.7 | 856 | 1.33 |
| yaw | 109.4 ± 0.6 | 1000 | 863 / 820 | 82.0 ± 2.3 | 887 | 1.21 |

Two reasons they cannot be used:

1. **Their actors receive `height_scan`** — the 63-ray (9×7) grid that the 09-20 revision itself
   identified as privileged-information leakage. No hardware has a terrain-height sensor.
2. **No checkpoint was saved.** `.gitignore` excludes `logs/`; there are zero `.pt`/`.onnx` files
   from these runs anywhere, and no tensorboard events. Only the summary table above survives.

Note the terrain ladder has `num_rows = 4` and everything plateaued at level 1.06–1.40, i.e. at
the floor. The menu is millimetre-scale: 40% flat, 20% steps 5–10 mm, 10% inverted steps 0–15 mm,
20% random ±8 mm, 10% waves 0–9 mm. **On this ladder there is no dynamic range to separate the
variants** — a comparison there measures noise.

## 1.4 Measured facts worth having

**Bus read failures: 3.9–6.2% per joint, and almost never consecutive.** The firmware has been
logging this all along — `meas[i] = -1` means the read failed — but nobody had looked.

| capture | rows | per-joint failure | consecutive-failure run lengths | ticks with ≥1 joint failed | all joints failed |
|---|---|---|---|---|---|
| 2026-09-14 03:41 | 373 | 5.87% | 1×158, 2×15, 3×3 | 36.7% | 0 |
| 2026-09-14 04:01 | 334 | 3.93% | 1×104, 2×7 | 32.3% | 0 |
| 2026-09-17 00:59 | 652 | 6.17% | 1×249, 2×18, 3×3, 4×7 | 39.1% | 3 (0.46%) |

Three consequences. (a) The previously quoted "9-motor syncRead fails 26.5%" is **not a separate
phenomenon** — `1 − (1−0.06)^8 = 39%` reproduces the observed per-tick rate, so it is just the
per-joint rate compounded. (b) **Losses are essentially independent, not bursty**, so a domain
randomisation for this needs no burst model. (c) A closed-loop deployment that raises `SystemExit`
on a length mismatch — which the current `policy_core.py:160-164` does — dies within a few ticks
at a 32–39% per-tick rate. The correct mitigation is per-joint zero-order hold, and *that* is what
a domain randomisation should imitate: a stale observation, not a zero. Feeding 0 into
`joint_pos_rel` asserts "this joint is exactly at its stand position", which is a specific false
claim, not a missing-value token.

Also unaddressed and cheap: `return_delay_time` has never been set. The factory value 250 means
500 µs of turnaround per device. `set_rdt` sketches exist and have never been run.

**There is only one working IMU.** A previous note said the robots carry two BNO055. In every
capture that has the second-IMU columns, `gx2/gy2/gz2` are **identically zero** — 0 nonzero samples
out of 652 rows and out of 52 rows, while the front `gx` in the same file ranges −3.86 … +3.41.
The firmware's `IMU_SINGLE = false` only means it *probes* both 0x28 and 0x29. **In simulation there
is also only one IMU**: every variant's MJCF has a single `imu_site` on the freejoint body, and the
`front_trunk_site` / `rear_trunk_site` sites carry no sensors. The two "trunk halves" in the
`upright` reward are reconstructed analytically from that one gravity vector and the spine encoder
angle — not measured.

**On flat ground, an open-loop spine sweep is strictly harmful.** 132 configurations
(11 amplitudes × 12 phases), sinusoidal spine motion on top of a fixed leg gait, pitch axis:

| spine amplitude | mean CoT | CoT sd | mean speed | max tilt | max heading drift |
|---|---|---|---|---|---|
| 0.0° | 0.6466 | 0.0114 | 0.1401 | 4.35° | 62°/20 s |
| 5.0° | 0.6573 | 0.0236 | 0.1396 | 5.42° | 135° |
| 10.0° | 0.7141 | 0.0436 | 0.1376 | 6.81° | 254° |
| 15.0° | 0.8204 | 0.0918 | 0.1318 | 7.38° | 346° |
| 25.0° | **1.3003** | 0.3809 | **0.1082** | 10.62° | 486° |

Mean CoT rises **monotonically**, +101% at 25°. Speed falls 23%. Zero falls at any setting.
(Taking the *minimum* over the 12 phases appears to show a 3.2% CoT improvement at 5°; that is
selection bias — the standard deviation grows from 0.0114 to 0.3809, so the minimum of a widening
distribution falls while the mean rises.)

Caveats that bound this: open-loop sine on a *fixed* leg gait is the weakest possible form of spine
use, it is flat ground, it is one axis, and it is a single harmonic. It is not evidence that a
learned policy cannot do better. But it is the only direct evidence available, and it says the
flat-ground regime has no energy benefit to find and no stability dynamic range either.

**Other measured physical quantities** (from the hardware characterisation archive): total mass
**0.540 kg** (simulation had 0.465, 13.8% light), fore/aft split 52.99 / 47.01%, foot spacing
17.0 × 10.5 cm, unloaded step response onset 12 ms / t90 28 ms / overshoot 63.1% / settle 144 ms,
ground μ 1.1, contact time constant 0.04 s, IMU axes identity, stationary gyro σ 0.0011–0.0015
rad/s, fused gravity lagging the gyro by −10 ms. Nominal root height 70.17 mm; `base_too_low`
terminates at 52.17 mm.

## 1.5 Cost of transport is currently unmeasurable on hardware

**No current is logged anywhere.** The firmware reads only `PRESENT_POSITION` (4 bytes). Since
cost of transport is the primary metric for an energy claim, this blocks the hardware half of the
study.

XL330 control table: `PRESENT_CURRENT` 126 (2 B), `PRESENT_VELOCITY` 128 (4 B),
`PRESENT_POSITION` 132 (4 B) — **126…135 is one contiguous 10-byte block.** A single 10-byte read
returns all three and is strictly better than the present 4-byte read: electrical power becomes
bus voltage × Σ current, and joint velocity no longer has to be differentiated from position (which
also avoids amplifying `present_velocity`'s ~30 ms inherent filter lag through a difference). The
number of reads is unchanged, so the added cost is ~60 µs per read at 1 Mbps.

## 1.6 If the environment is ported to IsaacLab

An inventory was taken. The accounting, over the 3898 lines that constitute the rigid environment
(`env_cfgs.py` 2529 + `microtaur_kinematics.py` 601 + `microtaur_constants.py` 298 +
`microtaur_config.py` 303 + `rl_cfg.py` 53 + `__init__.py` 114):

| category | lines | share |
|---|---|---|
| portable unchanged | 1614 | 41% |
| portable with mechanical edits | 509 | 13% |
| needs rewrite (API remap) | 997 | 26% |
| **no counterpart — needs reimplementation** | ~283 | 7% |
| delete (dead code, version monkeypatch) | 420 | 11% |

The good news is specific: the closed-form five-bar kinematics (`microtaur_kinematics.py`,
601 lines) and the three-stage predictive limit filter (217 lines) import only
`math`/`numpy`/`torch` and reference no simulator at all. Together with the reward math (188) and
the reset sampling (128) that is **1134 lines of algorithm that survive any port untouched** — the
entire intellectual content of the environment. Terrain is nearly a find-and-replace: `Box*` →
`Mesh*` for two primitive classes, and `HfRandomUniformTerrainCfg` / `HfWaveTerrainCfg` are
*identical* class names in both frameworks.

**The blocker is not code volume. It is the closed kinematic loop.** Each leg is modelled as two
open 2-link chains stitched by MuJoCo equality constraints — eight `<connect>` elements at
`microtaur_xmls/*/robot_modified.xml:360-368`, two per leg. PhysX reduced-coordinate articulations
cannot express this. Both routes are costly:

- **Loop-closure D6 joints.** PhysX supports them, but they are solved as constraints rather than
  in reduced coordinates, so they are softer and can drift or jitter. Eight of them, on a 10 cm
  robot, over millimetre-scale terrain features, at 2048 environments. Whether this is usable
  cannot be known without trying it.
- **Serialise each leg** into a 2-DoF chain driven through the existing closed-form FK. Fast and
  stable, but the effective inertia and torque transmission ratio of a five-bar vary with
  configuration, so both become approximations — and the passive-link armature (1e-6) and
  frictionloss (1e-3) values, and the identified 0.4662 kg / COM patch, stop meaning what they
  meant. The sim-to-real identification would have to be redone.

Three further items have no counterpart and are easy to lose silently: `ContactMatch` /
`fields=("found","force")` / `reduce="netforce"` contact sensing (drives three reward terms and the
air-time state machine; must become force-magnitude thresholding on `net_forces_w`);
**`encoder_bias`**, which mjlab threads through *both* the observation (`joint_pos_rel(biased=True)`)
and the action (`apply_actions` subtracts it before writing the target) and which will vanish
without a trace if ported term-by-term; and `TerrainHeightSensorCfg` + `RingPatternCfg` +
MuJoCo-site raycasting for `foot_height_scan` (critic-only, so the cheapest answer is to drop it).

Rough estimate for the whole port, assuming Isaac Sim is already installed: **6–11 weeks of
engineering, producing no new results, with a genuine risk that the loop-closure step does not
yield acceptable physics.** By comparison, the experiment described in Part 2 is about 24–30
GPU-hours — one to two days of wall clock on a single modern GPU.

A relevant structural note: **mjlab is IsaacLab's manager-based API running on MuJoCo Warp.** Its
`utils/lab_api/` is a verbatim fork of IsaacLab utilities, its terrain modules are adapted from
`isaaclab.terrains`, and both frameworks drive the same `rsl_rl`. Most of what a migration would
buy in tooling terms is already present, and MuJoCo's native handling of closed loops is a
capability that would be given up rather than gained. Both frameworks require CUDA and Linux
equally.

---

# Part 2 — the experiment

## 2.1 The claim, and why the reward function is the instrument

The reward is identical across morphologies. It rewards forward-velocity tracking (weight 3.00,
σ 0.04 m/s), yaw-rate tracking (1.20), body height (1.00), uprightness of both trunk halves (0.75),
a standing pose (0.50), an air-time target (0.35), a locomotion-consistent contact pattern (0.25)
and forward progress (0.20); it penalises non-foot ground contact (−2.00, also a termination), yaw
rate when none is commanded (−0.30), lateral and vertical velocity, roll/pitch rate, safety-filter
correction, action rate, joint-limit proximity, torque and joint acceleration.

**The spine appears only as a cost.** There is no term rewarding spine motion, no target spine
trajectory and no imitation objective. It carries four penalties of its own — mechanical power
(−0.02), travel-limit proximity (−0.10), a neutral-posture term while standing (−0.05), and a
sustained one-sided mean deflection during straight travel (−0.05) — plus, as §1.2 established, a
larger implicit cost inside `upright`.

The methodological consequence is the point of the whole setup. **Because the only route by which
spine motion can increase return is through improved locomotion, any spine motion the policy adopts
has already paid for itself.** The optimiser moves the spine only when the locomotion benefit
exceeds the cost of doing so. The reward is therefore a *measurement instrument*, not a
specification: we are not telling the robot its spine is useful and then confirming that it agrees.
We are asking what a locomotion objective does with a degree of freedom it is charged for.

Three things follow:

**A null result is a result.** If the policy leaves the spine near neutral, that is the finding
that at this scale, with this actuator, one trunk degree of freedom does not earn its cost under
this objective. A design that builds in a positive spine reward, or that imitates a recorded
biological spine trajectory, cannot produce this outcome because it has assumed it away.

**Spine usage is graded.** Amplitude, phase and terrain-dependence are all observable, so we can
report *how much* the optimiser is willing to pay and *under what conditions* it becomes willing.

**Mechanical hypotheses become falsifiable.** If a roll-axis spine helps by keeping more feet
loaded on laterally uneven ground, the mean number of feet in contact should be measurably higher
there and not on a longitudinal feature. Each such prediction can come out the other way.

## 2.2 Four robots, all spines active

The comparison is `rigid` / `roll` / `pitch` / `yaw`, every spine active. **There is no
locked-spine or passive-spine condition.**

An earlier draft of this design used a within-morphology `locked` / `passive` / `free` ladder, on
the reasoning that comparing a spined robot against a separate rigid robot confounds the degree of
freedom with mass, inertia, actuator count and power budget. That reasoning is sound in general but
the `locked` condition is not a neutral control **here**:

> `locked` is implemented as `spine_action_scale_rad = 0.0`, which means the target angle is held
> at neutral **and the actuator keeps it there with its PD controller**. On irregular terrain there
> is a continuous disturbance torque, so a locked spine draws current continuously — and that
> current goes straight into the numerator of cost of transport. `rigid` pays no such cost, and a
> free spine can avoid it by yielding. So `locked` is worse than both of the things it sits between,
> and an apparent `free` > `locked` advantage would be partly an artefact of the control scheme
> rather than a property of having a spine.

The correct counterpart to "has a spine" is "**has no spine**", which is `rigid`. One of the
benefits of having a spine is not having to spend energy fighting it, and that benefit should be
counted.

What this costs: we can no longer make the *causal* statement "the spine caused this improvement",
only "the robot with a spine performed thus, and its spine did thus". The revealed-preference claim
in §2.1 is unaffected, because *how much the optimiser pays for the spine as terrain difficulty
rises* is a measurement within one robot across terrain, not a comparison across conditions.

What must still be reported as a caveat: rigid differs from the spined variants in total mass,
inertia distribution and actuator count. Report the mass and inertia tensor of every variant. If
the mass difference is large, adding dead mass to rigid at the spine motor's location is a cheap
partial equalisation (it matches total mass, not inertia distribution).

## 2.3 Terrain — three families, and why there are no per-axis courses

All terrain is **zero-mean**: no net grade. A net grade both contaminates cost of transport (with
potential-energy change, so CoT stops reflecting dissipation) and shifts the nominal posture that
`body_height`, `standing_pose` and `upright` are written against.

| family | geometry | difficulty scalar |
|---|---|---|
| **flat** | plane | — (reference for the CoT–speed curve) |
| **random** | grid of cells with randomised heights; amplitude tied to usable leg travel | cell height range |
| **step field** | **alternating** lateral steps, wavelength ≈1–2 body lengths; direction of travel reverses which side is high | step height, steps per metre |

**An earlier version of this design had one course per spine axis. That was wrong, and the reason
it was wrong is worth recording**, because it is the kind of mistake that survives review by
looking reasonable.

A *sustained* left-high / right-low step — left legs up on the step, right legs on the ground for
the whole traverse — is a **static** roll demand, and a rigid trunk satisfies it either by rolling
as a whole or by extending the legs asymmetrically. A roll spine only pays when the **front and
rear pairs need different roll angles at the same instant.** Therefore:

- The roll content of a lateral step lives entirely in its **entry and exit transients** — the
  ~1 body length where the front feet are on the step and the rear feet are still on the flat.
- The sustained middle contributes nothing.
- So **more transitions per metre beats a longer step**, which is why the step family is a field of
  alternating steps rather than one long one.

The same argument applies to a constant grade for a pitch spine: a rigid trunk simply pitches as a
whole, and what demands differential pitch is a *change* of slope. Designing a course to favour an
axis turned out to be easy to get wrong — so we stop designing courses per axis and instead
**measure** each map's geometry and regress (§2.6).

Two generator requirements that fail silently if skipped:

1. **Identifiability.** For the regression to separate the per-axis coefficients, the
   differential-roll, differential-pitch and path-curvature content of the maps must be
   **decorrelated across maps**. Randomise ridge orientation relative to travel (0° gives pure
   pitch content, 90° pure roll, 45° mixed) and spatial wavelength (0.5×–4× body length) while
   holding overall RMS amplitude comparable. The generation constraints are chosen for statistical
   identifiability, not for difficulty.
2. **Spawn so the feature is actually encountered.** Give the step family its own proportion and
   spawn on the flat approach, so every episode crosses at least one transition.

Re-scale the difficulty ladder so that the level at which the current policies plateau sits near
**level 2 of 4**, leaving headroom in both directions. Today everything saturates at 1.06–1.40.

Train on the generator; evaluate on a **fixed held-out set of 50 maps**, with the terrain seed
fixed across conditions and varied across trials, so every robot walks literally the same courses.

## 2.4 Teacher–student

Give the teacher the map; have the student copy it.

**Teacher** — privileged, simulation-only: the deployable observation plus the 63-ray `height_scan`
(9×7 grid, 0.40 × 0.30 m, 50 mm resolution, root body, yaw-aligned), plus base linear velocity,
foot contacts, foot heights and contact forces. Trained with PPO.

**Student** — the deployable observation only. Trained against the teacher.

This is the right mechanism for the privileged-information problem the project has been carrying
(§1.3), and it also fixes a separate defect. **The current actor is memoryless.** The 09-20 revision
removed `base_lin_vel` from the actor — correctly, since no hardware sensor provides it — but
replaced it with nothing: there is no observation history (`delay_min_lag = delay_max_lag = 0`,
`history_length` at its default of 0), no recurrence and no state estimator. The actor therefore has
**no way to estimate its own forward speed**, which is plausibly why a 128-point lookup table
reproduces it so faithfully (pitch CPG vs RL leg angles, R² = 1.00). A **recurrent** student fixes
this.

Use the machinery already installed in `rsl_rl 5.4.2` rather than writing a pipeline:

- `rsl_rl/algorithms/distillation.py` — `Distillation`. The student drives and the teacher labels,
  i.e. on-policy DAgger at β = 0. `loss_type ∈ {"mse","huber"}`, `gradient_length = 15` TBPTT.
- `rsl_rl/runners/distillation_runner.py` — `DistillationRunner(OnPolicyRunner)`.
- `rsl_rl/models/rnn_model.py` — `RNNModel`, so the student can be a GRU or LSTM.
- A PPO checkpoint loads directly as the teacher.

What has to be written is small: `mjlab/rl/config.py` has no distillation config dataclasses, but
every seam exists — `RslRlBaseRunnerCfg.obs_groups` goes from
`{"actor": ("actor",), "critic": ("critic",)}` to `{"student": ("actor",), "teacher": ("actor","critic")}`,
`RslRlModelCfg` already carries `rnn_type` / `rnn_hidden_dim` / `rnn_num_layers` / `class_name`, and
`mjlab/tasks/registry.py` already allows a per-task runner class. Two dataclasses, one `rl_cfg`
variant, one task registration.

**Do not use `src/microtaur_velocity/distill_reliable/`.** It is 1821 lines across 20 files,
complete but **never once executed** — zero datasets, zero student checkpoints, zero logs anywhere
in any branch. Its student is a stateless ~7k-parameter MLP with no recurrence and its DAgger loop
is human-in-the-loop by design. Keep only `mjlab_utils.py` (274 lines), which eight `analysis/`
scripts already import for teacher loading. The four generation-one scripts under `scripts/`
(`train_student_bc.py`, `play_student_policy.py`, `play_student_policy_template.py`,
`export_student_npz.py`) import `microtaur_velocity.distill.student_policy`, a module that exists in
no branch, and crash on import; delete them.

## 2.5 Training to capability

Make each robot capable first, then compare.

**Budget: an identical fixed iteration count for every robot.** The alternatives — a terrain-level
threshold, or a held-out success rate — make the amount of training a function of the condition,
which is a confound in precisely the comparison the study rests on. Capability curves are *reported
as results*, not used as stopping rules, so nothing is lost by fixing the budget.

**Set the number from a pilot, not a guess.** The existing evidence says 4500 was too *few* on
rough terrain, not too many: flat T95 ranged 863–2729, but on rough, warm-started, T90 arrived at
100–130 while T95 crept out to 1516 (roll) and 2780 (yaw) — the curves were **still rising at 4500**.
So: one pilot on the random terrain, one morphology, run to ~6000 iterations, find where the
capability curve actually flattens, then fix that number for all four.

**Compress the command curriculum at the same time.** Its five stages currently land at iterations
0 / 781 / 1562 / 2187 / 2812, so `ang_vel_z` is pinned at exactly zero for the **first 781
iterations** while `track_yaw_velocity` (weight 1.20, the second-largest positive term) is trained
against a zero target. The policy is rewarded for never turning and then has to unlearn it; the
return curve steps up at 781 and dips after each later change, which is consistent with this.
`MICROTAUR_CURRICULUM_START_STEP` is a global offset, so setting it to 90000 saturates the
curriculum **with no code change** and answers "is the ramp needed at all" in one run. Run that on
rigid first — on roll and pitch the same switch also unlocks the spine to full authority at step 0,
which is what the 0 → 17° ramp existed to prevent, so split the two schedules first.

Warm-starting is available once the observation layout is unified across variants: seed each spined
robot from a converged rigid policy with the extension columns zero-initialised, the same technique
used for the rough runs in §1.3. Roll was the slowest learner of the four, so this is where it pays.

## 2.6 Evaluation

Evaluate on the 50 held-out maps across a difficulty grid. Evaluation does not train and is cheap.

**Two curves, answering two different questions.** This matters because on hard enough terrain every
robot fails, and that outcome should be reported rather than designed around:

1. **Completion rate vs terrain difficulty.** Each robot's curve falls off at some difficulty, and
   that point is its capability boundary. If all four fall to zero at the same difficulty, that is
   the platform's boundary and is reported as such.
2. **Cost of transport vs speed, computed only over the difficulty range every robot completes.**
   This gives the efficiency comparison, and restricting it to the commonly-completed subset avoids
   comparing conditional means over differently-selected populations.

**Never episode return as a performance metric.** It is not a physical quantity, and it is not
comparable across configurations whose reward weights or action spaces differ — which, per §1.1, is
exactly our situation until they are equalised.

**Expect and report trade-offs.** A spined robot has nine actuators against rigid's eight, so it may
well show a higher cost of transport at a given speed while reaching a higher top speed, or the
reverse. That dissociation is a result to be measured, not an outcome to be avoided.

**Metrics.** Cost of transport `E_electrical / (m·g·d)`; speed; completion rate; heading drift; and
for mechanism: mean number of feet in contact, spine mechanical power as a fraction of total, spine
amplitude and phase relative to stride.

**Three figures.**

1. **CoT vs speed**, one panel per terrain family, one curve per robot.
2. **Robot × terrain-content matrix**, each cell the (ΔCoT, Δspeed) of a spined robot against rigid.
3. **Spine amplitude vs terrain difficulty** — the revealed-preference figure, and the study's
   signature. Because the spine earns no positive reward, the vertical axis reads directly as *how
   much the optimiser is willing to pay*.

**The regression that replaces per-axis courses.** For each held-out map, compute its geometric
content from the heightfield — differential-roll spectrum along the direction of travel,
differential-pitch spectrum, path-curvature demand — then fit

```
Δperformance(spined − rigid)  ~  β_roll·twist + β_pitch·curvature + β_path·pathcurv + ε
```

Axis specificity then becomes a coefficient **fitted on random terrain** rather than a course built
to produce it. Keep two or three extreme diagnostic maps (pure twist, pure curvature) as a positive
control — used to check that the regression predicts the extremes, not as a headline result.

**Statistical note.** To resolve a difference Δ at power 0.8 requires roughly `n ≈ 16(σ/Δ)²` trials.
With 2048 parallel environments the trial count is effectively free in simulation, so this binds
only on hardware.

## 2.7 Hardware

One morphology, one terrain, via the existing table pipeline: policy → rollout → 128-point gait
table → `gait_table.h` → firmware playback at 100 Hz with phase interpolation. Note that this
pipeline sits *below* the RL layer, so changes to the observation contract or reward regenerate the
tables but do not change the toolchain.

Depends on §1.5 (current logging, or there is no cost of transport) and on a forward-speed and
heading-drift measurement that has been open since 2026-09-14 and has never been taken on any of the
three calibrated robots.

Courses are all tabletop-scale, which is a genuine advantage of a 10 cm robot: a ≥2 m flat board for
the speed reference (≥10 body lengths), an alternating lateral-step field on 3D-printed wedges under
foam board with step height swept 3–15 mm, and a randomised-height grid. An overhead phone at 60 fps
with a ruler in frame and a marker on the body gives both speed and heading drift.

One thing to check before any hardware claim, costing about thirty seconds: **leg numbering has never
been verified by eye on any robot.** Upstream's measured rigid map differs from the one in use by
exactly a front/rear half swap, which maps each diagonal pair to itself — so walking cannot detect
it. The firmware's `h` key resolves it: see whether DXL 1 moves the front-left or the rear-right
corner.

## 2.8 Prerequisites before any of this produces a number

None of these produce a publishable result and all of them are mandatory.

1. Fix the mjlab ≥1.6 reset breakage (§1.2). Nothing can be built or validated until this is done.
2. Bring the 09-20 observation work into the working copy and apply it to the three spine templates.
   Their actors still receive `base_lin_vel`, which no hardware can measure.
3. Equalise the four variants: one reward-weight set, one `joint_half_range_rad`, one kind of spine
   action term, the M288 table for the roll and yaw spines, the measured 0.540 kg and fore/aft split,
   and the `decimation` fix with the control rate chosen deliberately.
4. Split the spine-unlock schedule from the command curriculum.
5. Move the anti-splay constraint out of `upright` and into a per-half tilt termination (threshold in
   the 45–50° range, above the 15° maximum half-splay and below the unrecoverable angle), widening
   or re-basing the `upright` kernel so it serves only as an exploration gradient. Run the original
   kernel as a one-condition ablation so the result can be shown not to be an artefact of it.
6. Unify the observation layout across variants so warm-starting and a single deploy contract become
   possible.
7. Re-scale the terrain difficulty ladder (§2.3) and verify that the regressors are decorrelated
   across the generated maps.
8. Add the 10-byte block read for current (§1.5).

## 2.9 Open risks

1. **Dynamic range may not be achievable.** Flat ground shows a zero-or-negative spine effect with
   no falls; the current rough ladder saturates at level ~1.2 of 4. If re-scaling the ladder and
   fixing the nominal model still leave no regime where the robots separate, the comparison cannot be
   made and the terrain design has to be revisited. This is the central assumption, and item 7 above
   is where it gets tested — deliberately, and early.
2. **The available flat-ground evidence points against the hypothesis** (§1.4). The framing in §2.1
   is what makes that a publishable finding rather than a failed experiment.
3. **Cross-morphology inertia cannot be equalised** (§2.2). Report it.
4. **`spine_mean_bias` designs out one of the two spine mechanisms** on terrain with a sustained bias
   (§1.2). Decide explicitly whether to gate it, and say so.
5. **Compute.** Training requires CUDA and Linux. A representative measurement: 1145 environment
   steps/s on an M2 Max CPU at 2048 environments, against 47 000–58 000 on an RTX 4070 Ti SUPER —
   41–51× slower, making a 4500-iteration run ~72 hours rather than ~1 h 50 m. Budget roughly
   24–30 GPU-hours for four robots × three seeds plus the pilot, and 150–200 including Phase-0
   validation runs, ablations and reruns.
