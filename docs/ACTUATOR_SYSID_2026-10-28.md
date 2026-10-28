# Actuator identification, and what it changes in the sim

2026-10-28 · branch `spine-study-isaac` · robot: `rigid`, XL330-M077 legs

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
| `KD` | **0.0** | **0.0092** | `zeta = KD/(2 sqrt(KP*J))`, so `KD = 0` is zero controller damping; measured `zeta` is 0.292. See below. |
| `EFFORT_LIMIT_NM` | `0.60 * stall` | **`0.40 * stall`** | 0.60 of stall is 871 mA, 59% of stall current, high for continuous duty. 0.40 is 581 mA (40%), still 2.8x the worst measured static demand of 206 mA. |
| `FRICTIONLOSS_NM` | 0.010 | **0.028** | measured Coulomb equivalent. Optionally randomise per joint over [0.018, 0.050] to cover the 2x spread in §3. On Isaac also set `dynamic_friction` to the same value, as `microtaur_isaac/robot.py` already does. |
| `KP` | 1.0 | keep | measured `omega_n` within 4% |
| `ARMATURE` | 2e-4 | keep | equals BAM's identified 2.0005e-4 |
| `VELOCITY_LIMIT_RAD_S` | `0.80 * no_load` | keep | physical, not a safety cap -- see below |
| `XL330_STALL_TORQUE_NM` | 0.215 | keep | datasheet, consistent with `kt * I_stall` |

### `KD = 0.0  # never validated in isolation`

That comment is what this measurement answers. `KD` is the only linear damping
in the model, so `KD = 0` makes the simulated joint ring with nothing to stop it
but Coulomb friction. The measured joint is underdamped but not undamped:
`zeta = 0.292`, a 1.79x peak at 10 Hz. Solving `zeta = KD/(2 sqrt(KP*J))` for the
measured value gives **`KD = 2 * 0.292 * sqrt(1.0 * 2.284e-4) = 0.0092`**.

Note the direction, because the older `Microtaur_RL-main/src/microtaur_velocity/env_cfgs.py`
carries `XL330_POSITION_KD = 0.045`, which is the opposite error: it puts
`zeta` at 1.489, above `1/sqrt(2)`, so that joint has **no resonant peak at
all** -- its gain never exceeds 1 at any frequency. Whichever file a run
imports from decides whether the plant can resonate, and neither value is
right. 0.0092 sits between them and is the measured one.

Why it matters beyond fidelity: the real joint amplifies 1.79x at 10 Hz and
draws 1328 mA there, against 86 mA for the same amplitude at 4.4 Hz -- fifteen
times the current. Under `KD = 0` the simulated joint amplifies more than that,
under `KD = 0.045` it does not amplify at all, and in neither case does the
8-13 Hz band cost what it costs on hardware. Setting `KD` from the measurement
is what makes a policy's spectral choices carry their real price.

⚠ **Check the closed chain after changing it.** `microtaur_isaac/robot.py`
records that 16 position iterations at a 2.5 ms step hold loop closure within
MuJoCo's error on trot **at KD 0.045**, and that at **KD 0** the p99 still
passes but the maximum rises to about 5 mm. 0.0092 is nearer 0 than 0.045, so
re-run that spike rather than assuming the solver settings carry over.

⚠ **`0.0092` solves the linear term alone.** Raising `FRICTIONLOSS_NM` to 0.028
adds damping on top of it, so verify instead of trusting the arithmetic: replay
a single-joint sine sweep in sim and compare against

```
f (Hz)    1.0    2.0    3.0   4.431   6.0    8.0   10.0
|out/in| 1.009  1.025  1.050 1.103  1.210  1.531 1.793
lag(deg)  5.2    9.2   13.9   21.1   29.3   42.5    -
```

If the simulated peak overshoots 1.79, `KD` is too low; if it falls short, too
high. `tools/isaac_servo_curves.py` already does the command-versus-response
half of this in sim and is the natural place to add the sweep.

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
