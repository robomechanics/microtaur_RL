# What is worth taking from upstream `main`, as of 2026-10-28

A read of the three commits `aryan-chandra-cmu/Microtaur_RL` added to `main`
after `eef9f92` -- `5880025` (hardware characterization), `3c8892d` (35 ms
controller), `7961ade` (final rigid env). Nothing here was run or trained; this
is a diff review, so every claim is "the file says" and not "we measured".

The framework differs: upstream is mjlab (`src/microtaur_velocity/env_cfgs.py`)
while the active work here is IsaacLab (`microtaur_study/src/microtaur_isaac`).
Reward and terrain code therefore does not copy across, and several pieces are
already covered here in a different shape. What transfers is the measured
numbers and the designs, listed below with what this repo already has next to
them.

## 1. Adopt directly: the measured loop latency

```python
# env_cfgs.py:43-50
# The MCU frame is exactly 35 ms (28.57 Hz). Restrained armed testing measured
# 100% one-frame action receive/apply lag, ~54.0 ms mean motor-sample->action
# mailbox latency, and no sensing/Bridge overruns.
HARDWARE_CONTROL_PERIOD_S          = 0.035
HARDWARE_SENSOR_TO_ACTION_MEAN_S   = 0.0540
HARDWARE_SENSOR_TO_ACTION_MAX_S    = 0.0624
```

This is the one item with no equivalent here and no porting risk, because it is
a fact rather than code. Three things come with it:

- **35 ms is not a preference, it is a ceiling.** The accompanying
  characterization (`sim2real/README.md`) reports 25-33 Hz reliable, 35-37 Hz
  beginning to miss deadlines, and 40 Hz not deterministic. Upstream enforces
  the period with an integer decimation and raises if the product does not land
  on 35 ms exactly.
- **The action delay is one policy step, measured, not estimated.** "100%
  one-frame action receive/apply lag" is why the nominal is exactly one step
  rather than a fitted number, with a small stage-dependent extra-step
  probability only for scheduler tails.
- **54 ms is a different segment from the actuation delay.** It is motor sample
  to action mailbox, i.e. the sensing and compute path. The actuation segment --
  goal written to joint moving -- was measured separately here at 9.6-11.6 ms
  (`ACTUATOR_SYSID_2026-10-28.md` §2). The two do not overlap and should not be
  added into a single "delay" without saying which is which.

## 2. Reward shaping: the turning terms are new here, the gait terms are not

Upstream's reward work shows up as a block of tuned constants rather than new
terms. Comparing against `microtaur_rigid/rewards.py`:

| upstream constant | value | status here |
|---|---|---|
| `LINEAR_TRACK_SIGMA_M_S` | 0.050 | `track_lin_vel_xy(sigma)` exists; sigma is a tuning input |
| `YAW_TRACK_SIGMA_RAD_S` | 0.070 | `track_ang_vel_z(sigma)` exists |
| `YAW_TRACK_WEIGHT` | 1.20 | weight is tuned here separately |
| **`YAW_DIRECTION_WEIGHT`** | **0.25** | **no equivalent** |
| **`YAW_WRONG_SIGN_WEIGHT`** | **-0.35** | **no equivalent** |
| **`YAW_ACTIVE_THRESHOLD_RAD_S`** | **0.030** | **no equivalent** |
| `ACTION_RATE_WEIGHT` | -0.080 | standard term, tuned here |
| `TROT_WEIGHT_STRAIGHT` / `_TURN` | 2.000 / 1.600 | `trot_gait` exists; **the straight/turn split does not** |
| `TROT_PAIRS`, `TROT_MAX_ERR_S` | `((0,2),(1,3))`, 0.200 | `trot_gait(pairs, max_err)` already takes both |
| **`TROT_AIR_TARGET_S`** | **0.105** | `feet_air_time` exists; an explicit target does not |
| **`SWING_HEIGHT_TARGET_M` / `_SIGMA_M`** | **0.007 / 0.003** | **no equivalent** |
| **`SWING_PEAK_EMA_ALPHA`** | **0.25** | **no equivalent** |
| **`ROUGH_SWING_MIN_M` / `_SOFT_MAX_M`** | **0.010 / 0.020** | **no equivalent** |
| `ROUGH_TROT_WEIGHT_STRAIGHT` / `_TURN` | 1.25 / 1.00 | terrain-specific reweighting does not exist here |
| `TERMINATION_COST` | -2.0 | present here |

Three groups are worth lifting, and they are independent of each other.

**Wrong-direction turning is penalised separately from tracking error.** A
tracking reward alone is indifferent between turning too little and turning the
wrong way, because both are just error. `YAW_WRONG_SIGN_WEIGHT = -0.35` with
`YAW_ACTIVE_THRESHOLD_RAD_S = 0.030` makes the sign its own term, active only
once a turn is actually commanded. This repo's turning work
(`heading_tracking`, left/right symmetry augmentation, both off by default) does
not include it.

**Swing height is shaped against a target, with the peak tracked by an EMA.**
`SWING_HEIGHT_TARGET_M = 0.007` against `SWING_HEIGHT_SIGMA_M = 0.003`, and on
rough terrain a band rather than a point (`ROUGH_SWING_MIN_M = 0.010`,
`ROUGH_SWING_SOFT_MAX_M = 0.020`, with separate low and high sigmas). The EMA
(`alpha = 0.25`) is what makes a per-step reward see a per-stride peak. Here
there is `feet_air_time` and `feet_slide` but no clearance target, and foot
clearance is the quantity that decides whether a leg catches an edge.

**Gait and trot weights differ between straight and turning commands, and
again between flat and rough.** 2.000 / 1.600 straight-versus-turn and
1.25 / 1.00 on rough. This repo reweights rewards by policy step
(`reward_schedule`) but not by command type or terrain.

## 3. Terrain progression gated on competence, not distance

```python
TERRAIN_PROMOTE_FORWARD_QUALITY   = 0.75     TERRAIN_DEMOTE_FORWARD_QUALITY = 0.45
TERRAIN_PROMOTE_YAW_QUALITY       = 0.65     TERRAIN_DEMOTE_YAW_QUALITY     = 0.40
TERRAIN_PROMOTE_SURVIVAL_FRACTION = 0.95     TERRAIN_DEMOTE_SURVIVAL_FRACTION = 0.65
TERRAIN_PROMOTE_STREAK            = 2        TERRAIN_DEMOTE_STREAK          = 2
```

with the stated intent: *"Keep the deployable policy blind to terrain geometry
and adapt the already learned flat gait through progressively harder contact
disturbances. Terrain progression is based on completed-episode competence
rather than net distance."*

Three design choices are separable and all three are worth considering here:

- **Promotion requires forward quality, yaw quality AND survival fraction
  simultaneously**, so a robot cannot be promoted for driving straight through
  a course it cannot turn on.
- **Promote and demote thresholds are far apart** (0.75 against 0.45 forward,
  0.95 against 0.65 survival) and each needs a **streak of 2**. That is
  hysteresis in both directions, which is what stops a level from oscillating.
- **Progression is on completed-episode competence**, so an episode that ends in
  a fall contributes its failure rather than its distance.

This repo has a terrain curriculum (`steps_per_level`, commanded-distance
progression, a `min_level` floor, `terrain_level_tracking` logging) and already
evaluates blind (`isaac_eval_terrain --blind`). The quality-gated, hysteretic
form above is a different rule, not a missing feature, and the comparison worth
running is distance-gated against competence-gated on the same terrain set.

## 4. Three obstacle terrains that have no counterpart here

`src/microtaur_velocity/microtaur_terrains.py` adds, with robot-derived
dimensions (`TRACK_HALF_WIDTH_M = 0.043`, `WHEELBASE_M = 0.168`,
`NOMINAL_ROOT_HEIGHT_M = 0.0702`):

| class | what it is |
|---|---|
| `CurbTerrainCfg` | a road with a kerb -- a single edge to step up or along |
| `RandomStepsTerrainCfg` | irregular step field |
| `WeavePoleTerrainCfg` | weave poles, i.e. an obstacle that can only be passed by turning |

plus `TrainingPyramidStairsTerrainCfg` / `TrainingInvertedPyramidStairsTerrainCfg`
wrappers and `TRAINING_TERRAIN_NUM_LEVELS = 6`,
`TRAINING_MAX_SURFACE_DROP_M = 0.016`.

The terrains here (`TERRAIN_DESIGN.md`: flat, block grid, lateral offset) are
height-field disturbances. The three above are **obstacles**, and
`WeavePoleTerrainCfg` in particular is the only one in either repo that makes
turning necessary rather than merely commanded -- which is the thing the yaw
reward terms in §2 exist to shape.

## 5. What not to take

- **`microtaur_velocity_trot_reward`** -- `rewards.py:trot_gait(pairs, std, max_err)`
  here already has the same shape and arguments. Lift the constants, not the
  function.
- **Energy and copper loss** -- upstream has no equivalent of
  `motor_power_w` / `copper_power_w` / `motor_energy`, so this repo is ahead and
  nothing should be merged backwards over it.
- **Actuator constants** -- both repos currently carry `KD = 0.0`,
  `FRICTIONLOSS 0.010`, `EFFORT_LIMIT 0.60 * stall`. The measured corrections
  and the condition attached to `KD` are in
  `ACTUATOR_SYSID_2026-10-28.md` §4; upstream does not have them either.
- **The deployment app** (`sim2real/async_deployment_v1/`) -- worth reading for
  its architecture, which is how it gets past the Bridge round-trip: the MCU
  runs its own 100 Hz I/O loop and caches state, and the Linux side reads the
  cache at 35 ms instead of doing a synchronous round trip per tick. The
  companion `sync_hardware_deployment_bad/` is the version that did not work.
  Its firmware runs `POSITION_D_GAIN_LEG = 0`, which is the pairing discussed in
  `ACTUATOR_SYSID_2026-10-28.md`.
