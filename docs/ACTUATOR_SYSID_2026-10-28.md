# Actuator identification, and what it changes in the sim

2026-10-28 · branch `spine-study-isaac` · robot: `rigid`, XL330-M077 legs

> ⚠ **Read this first: every number here was measured with the servo's own
> `Position D Gain` set to 200**, copied from the App Lab firmware that was
> actually running on the robot (`uno_q_apps/microtaur_model13395_v1_8_4_motion_timeline`,
> `POSITION_D_GAIN_LEG = 200`).
>
> The upstream sim2real characterization used **D = 0**
> (`sim2real/microtaur_hw_characterization/arduino/.../microtaur_characterization.ino`,
> `POSITION_D_GAIN_LEG = 0`) and fitted `KD = 0.0` to it, reporting sim settling
> ~145 ms against hardware ~144 ms. Both sets of numbers are right; they
> describe two different controllers. **`KD` in sim and `D` in firmware are the
> same physical quantity and have to be set as a pair** -- see §4.

Measured on hardware, suspended, one joint driven at a time. Every number below
comes from a capture that is reproducible with the sketches named in §5, so it
can be rechecked when a motor is replaced or the supply changes. The three
simulation parameters this leads to are in §4; `damping` is the only one that
changes what a policy learns, and it is wrong by 5.1x.

## 1. The question this answers

The eight leg motors showed tracking errors that differed from each other by up
to 70 ticks (6 deg) while walking, which looked like unit-to-unit variation and
would have meant per-motor calibration.

It is not unit-to-unit variation. Suspended, the eight motors agree to within
10-17% of each other and their steady-state step error spans only -0.5 to +1.8
ticks (96 captures, 8 motors x 3 amplitudes x 2 directions x 2 repeats). The
error appears only under load, and it has two causes, both of them properties of
the actuator rather than of any individual unit:

1. **A P-only position loop has to sit at an error to hold weight up.** Its only
   output is `K_PP * err` with `K_PP = P/128`, so the holding torque and the
   error are the same quantity. Different legs carry different weight, so they
   sit at different errors. Measured holding Z79 loaded: 16-53 ticks at I=0,
   fitting `err ~= 9.9 + 1.41 * tau(mNm)` -- a friction intercept plus a
   load-proportional term. Raising the integral gain to 500 took the worst joint
   from 50 ticks to 4, and cut the total current across the eight from 932 mA to
   600 mA, because at I=0 every leg sags about 50 ticks and the five-bar sits off
   its design point, so holding the same weight costs more torque.
2. **A transport delay of about 10 ms, times whatever speed the command asks
   for.** Measured by sinusoidal tracking (§2): the phase lag expressed in TIME
   is flat across 1-6 Hz, and the amplitude ratio stays above 0.97. A flat lag in
   ms with no attenuation is a delay, not a bandwidth limit -- the joint does what
   it was told, late. Commanding a trajectory 20 ms early on the walking robot
   dropped the measured lag by 19.7 ms on six of eight joints, i.e. it cancels
   almost exactly, which no amount of noise would do.

## 2. Measured actuator spec

| quantity | measured | how |
|---|---|---|
| natural frequency `omega_n` | **11.0 Hz** | peak of the gain curve, 11-point sine sweep |
| damping ratio `zeta` | **0.292** | from the peak gain, `Q = 1/(2 zeta sqrt(1-zeta^2)) = 1.79` |
| resonant amplification | **1.79x at 10 Hz** | directly measured |
| transport delay | **9.6-11.6 ms** at command amplitudes >= 150 ticks | sine sweep 1-6 Hz, hold-corrected |
| transport delay, small motions | **17 ms** at amplitude 50 ticks | amplitude sweep at 2 Hz |
| free-running speed | **>= 16 900 tick/s (248 rpm)**, not saturated | full-travel 577-tick step, peak `PRESENT_VELOCITY` |
| stall torque | 0.215 N.m (datasheet, consistent with `kt * I_stall`) | - |
| static holding demand | **206 mA = 30.5 mNm = 14% of stall**, worst joint | loaded hold at Z79 |
| Coulomb-equivalent friction | **0.028 N.m = 13% of stall** | from the `err ~= 9.9 + ...` intercept |
| unit-to-unit friction spread | **about 2x** (id6 vs id1) | §3 |
| binding constraint on fast moves | **deceleration, not acceleration** | peak current 1404 mA occurred at 15 242 tick/s, not at standstill |

Two of these deserve a note.

**The velocity ceiling is a floor, not a value.** 16 900 tick/s is what a
full-travel step achieved, and the step had not saturated -- 577 ticks is simply
the largest move the leg has. Scaling the datasheet no-load speed by the
measured supply (383 rpm x 4.5/5.0) gives 23 500 tick/s, so the true ceiling is
somewhere between. Any claim that a commanded trajectory exceeds the motor
should cite 16 900 as the measured bound.

**Peak current during braking is the useful surprise.** A DC motor draws its
largest current at stall, but the largest current here appeared near top speed,
because a large step has to stop a fast-moving leg. Trajectories that decelerate
smoothly -- a sinusoid does, a step does not -- are much easier on the motor than
their peak speed suggests.

## 3. One joint is not all joints: dynamics match, friction does not

id6 (`leg3_a`) was swept under identical conditions to id1 (`leg1_a`), because a
static hold had shown it drawing 272 mA where its mirror id5 drew 32 mA.

| | id1 | id6 |
|---|---|---|
| peak frequency | 10 Hz | 10 Hz |
| peak gain | 1.793 | 1.651 |
| `zeta` | 0.292 | 0.320 |
| lag at 1 Hz | 11.6 ms | **28.2 ms** |
| lag at 8 Hz | 11.9 ms | **15.5 ms** |
| peak current at 2 Hz | 31 mA | **79 mA** |

The gain curves overlap below 8 Hz and both peak at 10 Hz, so `kp`, `damping`
and `armature` can stay single-valued across the eight joints. But id6's lag is
1.3-2.4x higher and its current about 2x higher at low frequency, converging by
8-10 Hz. That shape is friction: low frequency means low velocity, where
stiction dominates, and it stops mattering once inertial torque takes over. It is
the same mechanism as the amplitude dependence measured on id1 (17.1 ms at 50
ticks against 9.6 ms at 577).

So the friction spread is real device variation, not noise, and randomizing
`frictionloss` per joint is the honest response. It is also not worth more than
that: friction is under-modelled by 0.018 N.m, which is large next to the
holding torque but small next to the seed-to-seed spread this project has
already measured in trained policies.

## 3b. The spine is a different motor and a different kind of system

Measured on the **yaw** robot, spine only (motor id 9, model number read back as
1190 = XL330-M288, 288.4:1 against the legs' 77.3:1), driven at the robot's own
`SPINE_P_GAIN = 3000` -- twice the legs' 1500. Suspended, legs limp and never
commanded, goals bounded by the robot's own travel limit (1556..2496 from
`yaw/robot_config.h`) and a +/-180 tick window about the armed position, so the
test needs no pose calibration of any kind.

| f (Hz) | amp (tk) | \|out/in\| | lag (deg) | lag_srv (ms) | peak current (mA) |
|---|---|---|---|---|---|
| 1.5 | 240 | 1.087 | 12.0 | 19.3 | 422 |
| 2.0 | 177 | 1.200 | 18.8 | 23.1 | 473 |
| 2.5 | 113 | 1.405 | 28.4 | 28.5 | 522 |
| **3.0** | 79 | **1.559** | 55.3 | 48.2 | 670 |
| 3.5 | 58 | 1.380 | 90.4 | 68.8 | 672 |
| 4.0 | 44 | 0.955 | 107.7 | 71.8 | 418 |
| 5.0 | 28 | 0.410 | 112.2 | 59.3 | 143 |
| 6.0 | 20 | 0.307 | 110.4 | 48.0 | 70 |

Peak gain 1.559 at 3.0 Hz gives `zeta = 0.341` and `omega_n = 3.0/sqrt(1-2 zeta^2)
= 3.43 Hz`; the phase crosses -90 deg between 3.5 and 4.0 Hz, which puts
`omega_n` at about 3.5 Hz independently. **`omega_n = 3.4 Hz`, `zeta = 0.341`.**

| | legs (M077) | spine (M288) |
|---|---|---|
| `omega_n` | 11.0 Hz | **3.4 Hz** |
| `zeta` | 0.292 | 0.341 |
| `lag_srv` against frequency | **flat near 10 ms** -- a pure delay | **19 -> 48 -> 72 -> 48 ms, not flat** -- second-order dominated |
| lag at 4.431 Hz | 21-30 deg | **about 109 deg** |
| peak velocity reached | >= 16 900 tick/s (248 rpm), not saturated | 2 720 tick/s (39.8 rpm), current-clamped near 1300 mA |

The two rows that matter are the last two. The legs' phase lag is a transport
delay, so it is cancellable by commanding early and it is small in the gait
band. The spine's is its own second-order response: `lag_srv` is not constant,
so there is no single delay to cancel, and at 4.431 Hz the spine is about 109
deg behind its command -- near antiphase -- where a leg is 21-30 deg behind.

### `omega_n` is right in sim, by cancellation, and the torque is not

From the compiled model (`active_yaw_microtaur/scene.xml`, `mj_fullM` at the
default pose) the spine joint's inertia is `2.546e-3 kg.m^2` -- the `2e-4`
armature plus `2.35e-3` of trunk. With `ACTIVE_SPINE_POSITION_KP` defaulting to
the legs' `1.0`,

```
omega_n_sim = sqrt(1.0 / 2.546e-3) = 3.15 Hz     against 3.4 Hz measured, 7% off
```

So the simulated spine already has the right natural frequency. But `omega_n`
only fixes the ratio `k/J`, and the peak current separates them. At 2 Hz with a
120-tick half amplitude (0.184 rad) the hardware drew **1233 mA**:

| | stiffness `k` | inertia `J` | predicted current at that motion |
|---|---|---|---|
| sim values | 1.0 N.m/rad | 2.546e-3 | **180 mA** |
| from the measured current | 7.44 N.m/rad | 0.0163 | **1338 mA** |

`7.44` is the leg stiffness scaled by the doubled P gain and the gear ratio
(`2 x 3.72`), and `0.0163` is what `omega_n = 3.4 Hz` then requires. Only the
second row reproduces the measurement.

**So the simulated spine's stiffness and inertia are each about 6-7x too small,
and the error cancels in `omega_n` while leaving the torque wrong by the same
factor.** The same motion costs 180 mA in sim and 1233 mA on hardware. The
0.474 N.m it actually takes also exceeds `ACTIVE_SPINE_SAFE_EFFORT_NM`
(`0.60 * 0.52 = 0.312`) by 1.5x, so in simulation that motion would have been
clipped rather than merely cheap.

A caveat on how to split the inertia. The rotor's reflected contribution follows
the gear ratio squared: the legs' `2e-4` at 77.3:1 implies a rotor inertia of
`3.35e-8`, which at 288.4:1 gives `2.79e-3` for the spine. The measured total is
`0.0163`, so about `0.0135` would have to come from the trunk, against the
`2.35e-3` the XML currently carries. That points at the asset's mass properties
rather than at the actuator constants, and this measurement cannot distinguish
the two -- part of the 1233 mA is friction and stiffness rather than inertia. The
defensible statement is the total: `k/J` is right, both are about 6-7x low, and
the torque follows the absolute values rather than the ratio.

### Narrow usable amplitude

At 2 Hz, varying amplitude:

| amp (tk) | 30 | 90 | 180 | 320 |
|---|---|---|---|---|
| `lag_srv` (ms) | **59.0** | 27.5 | **20.0** | **86.0** |
| \|out/in\| | 0.899 | 1.134 | 1.192 | 1.346 |
| rms error (tk) | 7.9 | 14.4 | 23.9 | **147.4** |
| peak current (mA) | 83 | 206 | 363 | **1314** |

Stiction at the bottom -- 59 ms of lag at amplitude 30, three times the
large-amplitude value, where the legs showed a factor of 1.8 -- and torque
saturation at the top, with 320 ticks drawing 1314 mA and leaving 147 ticks of
rms error. The linear window is roughly 90 to 180 ticks, much narrower than the
legs'.

## 4. What to change in the sim

All of this lands in one file: `microtaur_study/src/microtaur_common/robot_constants.py`,
which both `microtaur_rigid` (mjlab) and `microtaur_isaac` import, so a single
edit covers both frameworks.

Effective joint inertia from the compiled MuJoCo model (`mj_fullM` at the
default pose) is `2.284e-4 kg.m^2` on every leg joint -- the `2e-4` armature
plus about 12% from the leg itself. With `KP = 1.0` that gives
`omega_n = sqrt(KP/J) = 10.53 Hz` against the measured 11.0 Hz, so **`KP` is
right to within 4% and should not move.** The damping is the problem.

| constant | current | set to | reason |
|---|---|---|---|
| `KD` | **0.0** | **0.0092 only if the firmware runs servo `D` = 200** | `KD` and the firmware `D` are one quantity; `0.0` is correct for the `D` = 0 robot upstream characterized. Do not change it alone -- see below. |
| `EFFORT_LIMIT_NM` | `0.60 * stall` | **`0.40 * stall`** | 0.60 of stall is 871 mA, 59% of stall current, high for continuous duty. 0.40 is 581 mA (40%), still 2.8x the worst measured static demand of 206 mA. |
| `FRICTIONLOSS_NM` | 0.010 | **0.028** | measured Coulomb equivalent. Optionally randomise per joint over [0.018, 0.050] to cover the 2x spread in §3. On Isaac also set `dynamic_friction` to the same value, as `microtaur_isaac/robot.py` already does. |
| `KP` | 1.0 | keep | measured `omega_n` within 4% |
| `ARMATURE` | 2e-4 | keep | equals BAM's identified 2.0005e-4 |
| `VELOCITY_LIMIT_RAD_S` | `0.80 * no_load` | keep | physical, not a safety cap -- see below |
| `XL330_STALL_TORQUE_NM` | 0.215 | keep | datasheet, consistent with `kt * I_stall` |
| `ACTIVE_SPINE_POSITION_KP` (old `env_cfgs.py`; absent from `robot_constants.py`) | defaults to the legs' 1.0 | **about 7.4**, with the spine inertia raised to match | §3b: `omega_n` is right only because `k` and `J` are both about 6-7x low. Fixing one alone moves `omega_n` off the measured 3.4 Hz. |

### `KD`: set it with the firmware's `D`, not on its own

That comment -- "never validated in isolation" -- is answered, but the answer is
conditional, and the condition is the firmware gain.

| firmware servo `D` | measured overshoot | measured settling | implied `zeta` | matching sim `KD` |
|---|---|---|---|---|
| **0** (upstream characterization) | 63% | 144 ms | **0.146** | **0.0** |
| **200** (the App Lab firmware actually deployed) | 43.7% | 93 ms | **0.292** | **0.0092** |

`zeta = KD/(2 sqrt(KP*J))` with `J = 2.284e-4` and `KP = 1.0` gives
`KD = 2 * 0.292 * 0.015113 = 0.0092` for the D=200 robot. For the D=0 robot the
existing `KD = 0.0` is right, and upstream's fit confirms it independently:
simulator Test 18 swept `KD = 0, 0.01, 0.02, 0.03, 0.045` and settling matched
hardware to 1 ms at `KD = 0`.

**So do not change `KD` alone.** Changing it without changing the firmware `D`
breaks the pairing in exactly the way sim2real work exists to prevent, and the
same is true in reverse: the robot has been running `D = 200` under the App Lab
firmware while the simulator was fitted at `D = 0`, which is already a mismatch
in the deployed direction.

Two things are worth deciding rather than inheriting:

**The D=200 robot is measurably better behaved** -- 43.7% overshoot against 63%,
93 ms settling against 144 ms, and twice the damping ratio. Upstream's KD sweep
never tried raising the firmware `D` and refitting; it took `D = 0` as given.
If the pair `(D = 200, KD = 0.0092)` is adopted, the resonant peak drops from
the `KD = 0` case and the 8-13 Hz band costs something without the plant being
over-damped.

**Whichever pair is chosen, the resonance is real and should be checked against
the measurement.** At D=200 the joint amplifies **1.79x at 10 Hz** and draws
1328 mA there, against 86 mA for the same amplitude at 4.4 Hz -- fifteen times
the current. `KD = 0.0` with the D=0 robot amplifies more than that, not less,
so neither configuration makes high-frequency command content free. What makes
it free is an over-damped `KD`, which is what the older
`Microtaur_RL-main/src/microtaur_velocity/env_cfgs.py` still carries at
`XL330_POSITION_KD = 0.045`: that puts `zeta` at 1.489, above `1/sqrt(2)`, and
removes the peak entirely. That value matches neither robot.

⚠ **Check the closed chain if `KD` moves.** `microtaur_isaac/robot.py` records
that 16 position iterations at a 2.5 ms step hold loop closure within MuJoCo's
error on trot at KD 0.045, and that at KD 0 the p99 still passes but the maximum
rises to about 5 mm. Re-run that spike rather than assuming it carries over.

⚠ **`0.0092` solves the linear term alone.** Raising `FRICTIONLOSS_NM` to 0.028
adds damping on top of it, so verify instead of trusting the arithmetic: replay
a single-joint sine sweep in sim and compare against

```
f (Hz)    1.0    2.0    3.0   4.431   6.0    8.0   10.0
|out/in| 1.009  1.025  1.050 1.103  1.210  1.531 1.793
lag(deg)  5.2    9.2   13.9   21.1   29.3   42.5    -
```

measured at D=200. `tools/isaac_servo_curves.py` already does the
command-versus-response half of this in sim and is the natural place to add the
sweep.

### `velocity_limit` is not the knob for current stress

In `DcMotorActuatorCfg` it is the zero-torque speed, i.e. the slope of the
torque-speed line, so lowering it weakens the motor at **every** speed rather
than capping anything. `EFFORT_LIMIT_NM` is the one protection setting, and it
maps directly to a current through `I = tau/kt`. The current value of
`0.80 * 383 rpm = 306 rpm` sits between the measured floor of 248 rpm and the
voltage-scaled datasheet figure of about 345 rpm, so leave it.

### `kt`: the 17% disagreement, resolved

`robot_constants.py` flags that the datasheet-derived `kt = tau_stall/I_stall =
0.146 N.m/A` disagrees by 17% with the back-EMF constant `V/omega_no_load =
0.125 V.s/rad`, and says to treat it as an estimate until the hardware current
traces are available.

They are available now, and the answer comes from an independent source rather
than from them: BAM (Rhoban) identified this exact motor on a torque-current
rig and reports **`kt = 0.1481 N.m/A`**, which agrees with the datasheet stall
point to 1.2% and disagrees with the back-EMF figure by the same 17%.

The disagreement is real and expected, not an error to be averaged away: for an
ideal DC motor `kt` equals `ke`, but the datasheet's stall torque is measured at
the output shaft **after** the gearbox, so gearbox friction has already been
subtracted from it, while the no-load speed is reached **against** that same
friction and is therefore lower than ideal. Both datasheet points are pushed in
the direction that widens the gap. **Use the identified `kt = 0.1481` for the
copper-loss reward; do not use `V/omega_no_load`.**

One honesty note: the hardware traces measured here cannot independently confirm
`kt`, because every torque figure quoted in this document was obtained by
multiplying a measured current **by** `kt`. BAM's identification is the
independent measurement; this document only shows it is self-consistent.

### Delay, for completeness

The recorded training configs have `min_action_delay_steps: 1`,
`max_action_delay_steps: 2`, which at decimation 4 (60 Hz control) is 16.7-33.3
ms. That brackets the measured actuation chain and should stay as it is. The same
configs have `delay_min_lag: 0` on every observation term, i.e. no observation
delay, while the real closed-loop path carries about 20 ms of transport before
the policy sees a state. The amplitude dependence in §2 matters here too: a
closed-loop policy making small corrections meets 17 ms, not 10 ms.

## 5. Reproducing this

| test | sketch / key | what it gives |
|---|---|---|
| step response, per motor | `dxl_check/rigid/rigid_step`, `1`-`8` / `A` | rise, overshoot, settle, steady-state error, current |
| integral gain sweep under load | same, `H` | the 50 -> 4 tick result |
| sine tracking, gain and phase | `dxl_check/rigid/rigid_track`, `L` / `B` | delay, amplitude ratio |
| velocity and torque envelope | same, `V` | free-running speed, braking current |
| frequency sweep to 25 Hz | same, `S` | `omega_n`, `zeta`, resonant peak |
| amplitude sweep (linearity) | same, `A` | the 17 ms vs 9.6 ms result |

Two measurement notes that cost time here and will cost it again:

**Demodulate by least squares against `[1, cos, sin]`, not by a single-bin
lock-in.** The standard lock-in assumes a uniform sample grid, and this bus does
not give one: per-read loss reaches 15-17% once a syncWrite shares the loop, the
loss is correlated with motion, and the position signal's DC is about ten times
its AC amplitude. The first version of this measurement returned amplitude
ratios of 1.05-1.11 -- impossible for a position servo -- and a lag that went
negative. Against synthetic data under the same conditions the lock-in was off
by up to 69.8 deg of phase; solving for the DC term as a third unknown was exact.

**Timestamp the measurement, and subtract the command's own hold.** Reusing the
phase that built the command labels each sample earlier than it happened, which
*understates* the lag; and a staircase command contributes `update_interval/2` of
delay that belongs to the rig, not the joint. Both corrections together moved
the reported figure by about 4 ms.
