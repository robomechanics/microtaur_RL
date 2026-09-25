# Terrain and curriculum design

2026-09-25 · branch `spine-study-isaac`

Three training terrains, one command curriculum, and the numbers behind them.
Every limit below is derived from the leg kinematics or measured from a generated
heightfield, not chosen by eye; the derivations are in §1 and §2 so they can be
rechecked when the leg geometry or the joint limits change.

---

## 0. The three terrains and what each is for

| | terrain | purpose | difficulty axis |
|---|---|---|---|
| **A** | flat, always | reference for the cost-of-transport vs speed curve; sanity check that the four morphologies coincide when the terrain demands nothing | none |
| **B** | block grid — flat-topped square cells of uniform edge, random heights | general capability. Axis-neutral by construction (§3), so it favours no spine axis a priori | cell height range |
| **C** | lateral offset — left side of the path high, right side low, sustained | lateral balance training, **and the study's negative control** (§4) | left-right height difference |

Terrain C deserves the emphasis. Every spine in this project rotates the front
half of the trunk relative to the rear half, so none of the three axes can help
with a *sustained* left-right height difference: a rigid trunk satisfies that by
rolling as a whole, or by standing taller on one side. Measured, it is not a
matter of degree — see §4. So C is the obstacle on which **no morphology should
beat rigid**, and reporting that it does not is what defends the rest of the
study against the charge that the terrain set was chosen to produce the result.
If a spined variant *does* win on C, something is confounded, or the spine is
doing something the axis argument does not cover. Either is worth knowing.

---

## 1. The mechanical budget every number is derived from

Computed with `src/microtaur_velocity/microtaur_kinematics.py` (pure numpy, runs
without a simulator) at the rigid stand pose, under the joint limits the rigid
configuration actually enforces: single joint within stand ± 0.75 rad, and the
coupled limits `|common| ≤ 0.52`, `|diff| ≤ 0.52`, `|common| + |diff| ≤ 0.70`,
where — from the filter's own definition —

```
da = q_a − stand_a           de = q_e − stand_e
common = 0.5 · sign · (da + de)      # fore-aft swing
diff   = 0.5 · sign · (de − da)      # lift / extend
```

Stand foot position is **(−5.05, −73.47) mm** relative to the hip, i.e. a
73.47 mm leg. Foot is a sphere of radius **6.2 mm**.

Vertical foot travel, as a function of how far through the swing the leg is:

| `\|common\|` | `diff` limit | can **retract** | can **extend** | total |
|---|---|---|---|---|
| 0.00 | 0.52 | **26.22 mm** | 32.86 mm | 59.08 mm |
| 0.10 | 0.52 | 26.76 | 32.39 | 59.15 |
| 0.20 | 0.50 | 26.80 | 30.08 | 56.88 |
| 0.30 | 0.40 | 24.06 | 22.30 | 46.36 |
| 0.40 | 0.30 | 21.10 | 13.72 | 34.82 |
| 0.50 | 0.20 | 18.13 | **4.69** | 22.83 |

Two facts drive the whole design:

- **Retract is robust** — 26.2 mm at mid-stance, still 24.1 mm at
  `|common| = 0.3`, 18.1 mm at 0.5.
- **Extend collapses** — 32.9 mm at mid-stance down to 4.7 mm at `|common| = 0.5`.
  A foot reaching *down* into a hole while also near the end of its swing has
  almost no authority left.

**The binding limit for terrain height is retract**, because a foot standing on a
cell that is `h` higher must shorten that leg by `h`. So the ceiling is
**≈ 26 mm**, and a foot has to clear the rise as well as stand on it, so the
practical ceiling for reliably stepping *onto* a one-cell rise is lower — around
20 mm once a few millimetres of swing clearance are reserved.

⚠ Note also that operating near `|diff| = 0.52` is *at* the safety filter's
limit, which generates `filter_correction` penalty (weight −0.10), and beyond
80% of the single-joint half-range the `joint_limit_proximity` penalty
(−0.05) engages. Terrain that *forces* the robot to the limit is being charged
for it by the reward. The ceiling of the curriculum is meant to be a place the
robot struggles, not a place it operates.

---

## 2. Implementation: an oversampled heightfield, not instanced boxes

The terrain wanted is Minecraft-like: flat-topped cells of uniform edge, height
varying cell to cell, **no slopes**. The reason to avoid slopes is not
aesthetics:

- **A slope applies a tangential force for the whole stance.** The robot spends
  ~0.15 s per stance fighting a component along the surface. That is a
  *systematic* force, and it shows up in cost of transport and in heading drift —
  so a sloped terrain measures hill-climbing mixed in with rough-ground walking.
- **A cell edge is a one-or-two-timestep perturbation.** The foot rolls off it or
  lands beside it. That is noise, not bias, and a quadruped is statically stable
  on three legs, so a missed foothold is recoverable by stepping again.

The naive implementation of flat tops is a grid of box prims, which is expensive:
a 4 m × 4 m tile at 70 mm is 57 × 57 = **3249 boxes**, and with
`num_rows × num_cols = 4 × 8` that is roughly 10^5 collision primitives.

**There is a cheaper way that gives the same geometry.** A heightfield's
resolution and the cell size of the height *pattern* are independent parameters.
Set the heightfield fine and the pattern coarse:

| heightfield resolution | height pattern | resulting surface |
|---|---|---|
| 70 mm (= cell size) | 70 mm cells | **no flat tops at all.** Heights live at vertices and interpolate, so every cell boundary is a ramp: `atan(25/70)` = **19.7°**, and the robot is on a ramp for essentially its whole stance |
| **10 mm** | **70 mm cells** | 7 × 7 = 49 samples per cell at the same height → **genuinely flat tops**; transitions occupy one 10 mm sample interval → `atan(25/10)` = **68.2°**, effectively a wall. 6 of every 7 sample intervals per axis are flat, i.e. ~86% per axis |

Cost: 400 × 400 = 160k height samples per tile, and heightfield collision is a
**native primitive** in both MuJoCo (`hfield`) and PhysX. That is far cheaper
than 10^5 box prims and needs no new terrain class.

In the existing configuration this is two numbers:
`MICROTAUR_TERRAIN_SCAN_RESOLUTION`'s sibling on the terrain side —
`horizontal_scale`, currently **0.15 m** — goes to **0.01 m**, and the noise is
generated per 70 mm block rather than per sample.

The only thing lost is a truly vertical wall or an overhang, and this design
needs neither.

---

## 3. Terrain B — block grid

### Geometry

| | value | why |
|---|---|---|
| cell | **70 × 70 mm** | 11.3 × the 6.2 mm foot radius, so feet mostly land on tops rather than edges; 2.4 cells per body length, 1.5 per body width, so the left and right sides usually sit on different cells |
| heightfield resolution | **10 mm** | 7 samples per cell → flat tops, 68° transitions (§2) |
| cell heights | i.i.d. uniform, zero mean, peak-to-peak `A(level)` | zero mean because a net grade puts potential-energy change into the numerator of cost of transport and shifts the posture the reward terms are written against |

Cell size may be anywhere in **40–80 mm**. The lower bound is the foot radius:
below ~40 mm feet land on edges often enough that the edge contact stops being a
transient. **Cell size is not a sensitive parameter** — see §3b, it changes the
magnitude of the demand by a few percent and does not change which axis is
loaded. The height range is the parameter that matters.

### Difficulty ladder

`A` is the peak-to-peak of the cell height distribution. Adjacent-cell steps are
then triangular on ±A, with p95 ≈ 0.76 A.

| level | `A` | p95 adjacent step | fraction of the 26.2 mm retract budget | expected behaviour |
|---|---|---|---|---|
| 0 | **0 mm** | 0 | 0% | flat — the warm-up start Ben specified |
| 1 | **6 mm** | 4.6 | 23% | ≈ one foot radius; the foot can no longer ignore it |
| 2 | **12 mm** | 9.1 | 46% | one-cell steps are comfortable |
| 3 | **18 mm** | 13.7 | 69% | tight once swing clearance is reserved |
| 4 | **24 mm** | 18.2 | 92% | at the retract limit — **the ceiling, expect failures here** |

Use the existing **`terrain_levels` curriculum** rather than a step-counter
schedule. It already implements exactly the promote/degrade behaviour wanted:
promote when the robot walked further than `size[0] / 2` (2.0 m), demote when it
walked less than half the commanded distance, with `max_init_terrain_level = 1`
so everything starts near the bottom. Nothing new has to be written; the ladder
above replaces the amplitude numbers in the existing generator.

⚠ The current ladder is millimetre-scale and **all four morphologies plateaued at
terrain level 1.06–1.40 out of 4**, i.e. at the floor, where a comparison
measures noise. Re-scale so the plateau lands near level 2 of 4, leaving
headroom in both directions. The ladder above is sized for that.

### 3b. Measured: the block grid is axis-neutral, and cell size does not change that

Content measured with `tools/terrain_content.py`, at `A` = 25 mm:

| cell edge | cells per body length | twist | curvature | twist / curvature |
|---|---|---|---|---|
| 20 mm | 8.5 | 0.1396 | 0.1502 | 0.93 |
| 40 mm | 4.2 | 0.1395 | 0.1425 | 0.98 |
| 60 mm | 2.8 | 0.1350 | 0.1465 | 0.92 |
| **70 mm** | 2.4 | ~0.129 | ~0.142 | ~0.91 |
| 85 mm | 2.0 | 0.1225 | 0.1387 | 0.88 |
| 120 mm | 1.4 | 0.1221 | 0.1123 | 1.09 |
| 170 mm | 1.0 | 0.0987 | 0.0874 | 1.13 |
| 250 mm | 0.7 | 0.0641 | 0.0868 | 0.74 |

**The ratio is 0.74–1.13 over a 12× range of cell size** — essentially 1
throughout. Cell size changes the magnitude (0.14 down to 0.064) and not the
balance, because i.i.d. cell heights have no directional structure, so the field
is isotropic whatever shape the cells are.

**Consequence, and it is a real constraint on the analysis.** Because twist and
curvature move together across any set of i.i.d. block-grid maps, a regression of
performance on per-axis geometric content is **not identifiable** on this
terrain. Measured across 24 maps:

| map set | corr(twist, curvature) | |
|---|---|---|
| square cells, varying edge and difficulty | **+0.954** | unusable |
| rectangular bricks, varying edge, aspect ratio and difficulty | **+0.867** | unusable |

Rectangular cells do not fix it: aspect ratio moves the ratio over only a 2.2×
range (1.51 at 160 × 40 mm to 0.69 at 40 × 160 mm), while amplitude moves both
regressors together and dominates the variance.

This is **not a problem for this study**, because the per-axis question is
answered by a typed test field with a per-obstacle breakdown (§6), not by a
regression. It is recorded here so that nobody later tries to fit that regression
on grid data and believes the coefficients.

---

## 4. Terrain C — lateral offset

### Geometry

Two wide flat halves at different heights, seam running along the direction of
travel, robot spawned straddling the seam. Left side high and right side low, or
mirrored — running the course in both directions covers both.

| | value | why |
|---|---|---|
| half-plane width | **≥ 1 m each side** | the robot drifts laterally over a 20 s episode; it must not run off the seam |
| transition | step at the seam, spawn already straddling it | measured, the ramp-in length barely matters — see below |
| **no yaw command on this terrain** | | a yaw command steers the robot off the seam, which turns a sustained roll demand into an intermittent one. Terrain C is run straight-ahead only |

### Difficulty ladder

`Δ` is the left-minus-right height difference, so each leg carries `Δ/2`.

| level | `Δ` | per-leg offset | fraction of the 26.2 mm retract budget |
|---|---|---|---|
| 0 | **0 mm** | 0 | 0% |
| 1 | **4 mm** | ±2 | 8% |
| 2 | **8 mm** | ±4 | 15% |
| 3 | **12 mm** | ±6 | 23% |
| 4 | **14 mm** | ±7 | 27% |

The ceiling is about **half** the retract budget, which is deliberate and matches
the intent. Unlike terrain B, where a large step is transient and the leg returns
to mid-range on the next cell, terrain C's offset is **sustained for the whole
traverse**: the two legs on the high side hold their retraction continuously, so
that authority is unavailable for gait and for disturbance rejection the entire
time. Spending more than half the budget on a static posture leaves too little
for walking.

### Measured: C is a clean negative control

| how the offset is established | twist | curvature |
|---|---|---|
| step, robot spawned on it | **0.0000** | **0.0000** |
| ramped in over 1 body length | 0.0071 | 0.0061 |
| ramped in over 5 body lengths | 0.0050 | 0.0012 |

Against terrain B's 0.13 / 0.14, even the worst case is **5% of B's demand**. So
no spine axis has anything to work with here, the transient at the entry is
negligible, and the ramp length does not need care.

---

## 5. Command curriculum

Forward velocity plus yaw. **No reverse** — `lin_vel_x` stays positive and the
mirroring option (`MICROTAUR_MIRROR_FORWARD_VELOCITY`) stays off, which is its
default.

### Schedule

Three stages instead of five, and turning is available from the first step.
Steps are `common_step_counter`; at 32 steps per iteration:

| stage | step | iteration | `lin_vel_x` (m/s) | `ang_vel_z` (rad/s) | standing fraction |
|---|---|---|---|---|---|
| 0 | 0 | 0 | (0.10, 0.18) | **±0.10** | 0.05 |
| 1 | 8 000 | 250 | (0.08, 0.20) | ±0.18 | 0.10 |
| 2 | 20 000 | 625 | (0.08, 0.20) | **±0.25** | 0.10 |

`lin_vel_y` stays identically zero; `heading_command` stays off; resampling
interval stays 6–10 s.

### Why turning from step 0

The current schedule pins `ang_vel_z` at exactly zero for the **first 781
iterations**, while `track_yaw_velocity` — weight 1.20, the second-largest
positive term — is trained against a target of zero, and `zero_command_yaw_rate`
(−0.30) is also pushing yaw rate down. For 17% of training the policy is
rewarded for never turning, and then has to unlearn it. The published return
curve steps up at iteration 781 and dips after each later stage change, which is
what that looks like.

And the hard end of this curriculum is not hard. Turning radius `r = v / ω`:

| `v` | `ω` | radius | in body lengths (0.170 m) |
|---|---|---|---|
| 0.14 | 0.10 | 1.40 m | 8.2 |
| 0.14 | 0.18 | 0.78 m | 4.6 |
| 0.14 | 0.25 | 0.56 m | 3.3 |
| 0.08 | 0.25 | 0.32 m | 1.9 |

Even the tightest combination is a radius of about two body lengths. There is no
reason to withhold ±0.10 rad/s, which is an eight-body-length arc.

### Why front-loaded

With the command curriculum finished by iteration 625 and the terrain curriculum
adaptive thereafter, **only one thing changes after iteration 625**. If both
schedules were still moving, a regression in the return could not be attributed
to either. The two use different mechanisms on purpose: commands do not need to
adapt to the robot, so they are a fixed schedule; terrain does, so it uses the
promote/demote rule.

On a 3000-iteration budget the final stage gets **2375 iterations, 79%**, against
1688 of 4500 (37.5%) today.

### A free experiment worth running first

`MICROTAUR_CURRICULUM_START_STEP` is a global offset on the schedule. Setting it
to 20 000 (or 90 000 on the current five-stage table) starts training at the
final stage, which answers "is the ramp needed at all" **with no code change**.
Given that rigid, pitch and yaw all reached 90% of their final return within the
first stage on flat ground, the ramp may be doing nothing.

---

## 6. What the teacher sees, and how the axis question gets answered

### Height scan resolution has to match the cell size

The teacher is privileged: it receives the terrain height scan, the student does
not. The existing scan is **9 × 7 = 63 rays at 50 mm spacing** over a
0.40 × 0.30 m window, root-body frame, yaw-aligned.

With 70 mm cells that is **1.4 rays per cell** — near the Nyquist limit, so the
teacher sees a blurred field and cannot tell where the cell boundaries are. To
resolve the blocks, spacing must be at most half the cell:

| | spacing | window | rays | rays per cell |
|---|---|---|---|---|
| current | 50 mm | 0.40 × 0.30 | 63 | 1.4 |
| **recommended** | **35 mm** | 0.40 × 0.30 | 12 × 9 = **108** | 2.0 |
| narrower window, similar count | 35 mm | 0.35 × 0.245 | 11 × 8 = 88 | 2.0 |

108 rays is 45 more critic inputs than today. The teacher never runs on hardware,
so that cost is irrelevant.

⚠ `height_scan` must stay out of the actor. The rigid configuration's
`_validate_observation_contract()` lists it in `forbidden_actor_terms` and raises
at config-construction time if it leaks. **Port that validator early** — the four
existing rough policies are unusable precisely because the scan reached their
actors.

### The student will be worse than the teacher, and that is the measurement

A teacher that places its feet using a height scan is asking a blind student to
reproduce actions from information the student does not have. DAgger cannot
invent it; the student will average over the ambiguity. This is expected, not a
bug, and there are two things to do about it:

- The recurrent student's actual job here is **inferring terrain from
  proprioceptive history** — contact timing, joint deflection, IMU — not merely
  estimating its own forward speed. That is what makes recurrence non-optional.
- Measure and report the teacher-student gap. It is a result.

⭐ Worth noting that the block terrain is *friendly* to this inference in a way a
sloped terrain is not. Flat tops mean contact heights are drawn from a small
discrete set, so contact timing tells the student "I am one cell up" fairly
directly. On a continuously sloped surface there is no such signal.

### The per-axis question

Because terrain B is axis-neutral (§3b) and terrain C is a negative control (§4),
neither training terrain answers "which spine axis helps where". That is answered
by a **typed test field**: several courses, one score each, plus a total, with
**the scoring fixed in writing before the runs**. The test field must contain

- at least one obstacle per spine axis, and
- at least one obstacle on which no axis should win — terrain C serves this.

Report the per-obstacle breakdown, not only the total. The negative control is
what answers the reviewer who asks whether the obstacle set was chosen to produce
the result.

---

## 7. Open items

1. **Cell-size and ladder numbers assume the rigid joint limits.** The coupled
   limits `|common|, |diff| ≤ 0.52` and `|common| + |diff| ≤ 0.70` and the single
   joint ±0.75 are what produce the 26.2 mm retract budget. The pitch variant uses
   `joint_half_range_rad = 0.95` — which is the XML's own mechanical range, i.e.
   it does not narrow it — so its budget is larger and the ladder would have to be
   recomputed for a like-for-like comparison.
2. **Test field obstacle set and scoring are not yet specified.** They have to be
   written down before any test run.
3. **Copper loss was added to the reward and the weights changed** (2026-09-25).
   The reward tables in `docs/REWARD_TERMS_EN.md` and
   `docs/MEMO_2026-09-23_spine_study.md` are stale until the new set is recorded.
   The important constraint: the reward's energy term and the cost-of-transport
   metric must use the **same** energy model, or the thing being optimised is not
   the thing being scored. Note that the existing `normalized_torque` (−0.03,
   `(τ/τ_safe)²`) is already a copper-loss proxy since `I²R ∝ τ²`, so the two may
   be charging the same cost twice.
4. **Terrain C excludes yaw commands**, so the yaw curriculum applies to A and B
   only. If yaw on C is wanted, the seam has to become something drift-tolerant,
   which reintroduces transitions and costs C its negative-control property.
