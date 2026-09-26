# Microtaur spine study — 2026-09-25: state, findings, and what is left before the teacher

Branch `spine-study-isaac`. Supersedes the plan sections of
`MEMO_2026-09-23_spine_study.md` where they conflict. Every number was produced
by running code on 2026-09-23/25; the script is named next to it. Figures are in
`docs/figures/2026-09-25/`.

---

## 1. Summary

1. **Model.** Upstream's 2026-09-20 rigid config (private
   `aryan-chandra-cmu/Microtaur_RL`, HEAD `5880025`), total mass 0.540 kg,
   reorganised into `microtaur_study/` and bit-identical to upstream on every
   deterministic part (§3).
2. **Reward.** A minimal reward (tracking + one motor-cost model for every
   motor, copper loss included) **learns to walk** once the energy weight is
   introduced by a curriculum (§4).
3. **But the gait is not good yet** (§5): an asymmetric, short-stride trot that
   tops out at ~0.17 m/s, with the servo model saturated a third of the time.
4. **IsaacLab/PhysX is viable** (§2): closed chain holds (pos_iters 16,
   dt 0.0025); `frictionloss` **does** map on Isaac Sim 5.0 (corrected);
   the contact-model difference is judged unimportant for this robot. One check
   left: the closed chain at `kd = 0`.
5. **Terrains A/B/C are designed and rendered** (§6); the teacher sees
   privileged state plus a height map (§7). §8 is the checklist of what is left.

---

## 2. IsaacLab / PhysX

Scripts: `tools/spike_mujoco_reference.py`, `tools/spike_isaac_closed_chain.py`,
`tools/spike_isaac_joint_friction.py`. Isaac Sim 5.0.0 rc.45 (pip), IsaacLab
2.3.0 at `7b26eb4`, RTX 5080.

**Closed chain.** One PhysX revolute loop joint per leg (the two MJCF `<connect>`
sites share an axis). MuJoCo's own training physics carries 0.24 mm of closure
error under load; that is the baseline.

| motion | pos_iters | dt | p99 | max |
|---|---|---|---|---|
| MuJoCo trot | — | 0.005 | 0.49 mm | 0.50 mm |
| PhysX trot | 4 (default) | 0.005 | 5.0 mm | 6.5 mm |
| PhysX trot | 16 | 0.005 | 1.13 mm | 6.51 mm |
| **PhysX trot** | **16** | **0.0025** | **0.38 mm** | **1.16 mm** |

Cost: 2x physics steps; 150k env-steps/s at 2048 envs and dt 0.005 (idle GPU),
~75k at dt 0.0025 extrapolated, not measured. ⚠ Measured at `kd = 0.045`; the
09-20 servo model is `kd = 0` (§2.1).

**Joint friction — corrected.** An earlier version of this memo said
`frictionloss` had no PhysX mapping. On Isaac Sim 5.0 IsaacLab's actuator
`friction` is an effort (it was a coefficient on 4.5), and measured it is a
constant Coulomb torque: set 0.010, a joint holds under 0.009 N·m and slips at
0.011; set 0.020, holds up to 0.015 and slips fully at 0.021. XML values carry
over. ⚠ Further correction (Isaac Sim 5.1, CPU = GPU): that threshold is only
the static part; IsaacLab's `friction=F` leaves `dynamic_friction` at 0, so a
slipping joint has no friction. `friction=F, dynamic_friction=F` reproduces
MuJoCo's frictionloss.

**Contact model.** Open-loop replay of the same motor targets ran 13% faster and
4.5 mm taller in PhysX. Judged unimportant here (four small spherical feet on
simple ground, and policies are retrained in the target simulator); it matters
only insofar as the hardware contact calibration (Test 20) was done in MuJoCo.

### 2.1 The servo model: KP = 1, KD = 0

- Firmware writes Position P = 1500, I = 0, **D = 0** at start-up (the XL330
  factory default D is 400). So `KD = 0` matches the servo.
- From the datasheet (KPP = P/128 on PWM output, 4096 pulses/rev, PWM full scale
  885, stall 0.215 N·m at 5 V) the static stiffness is ≈ 1.86 N·m/rad,
  saturating at 6.6° of error. Test 18 identified **KP = 1.0** from the
  hardware step response (settling 145 ms sim vs 144 ms hardware); the
  identified value is the one to trust.
- A voltage-driven DC motor has back-EMF damping even with D = 0:
  k_t k_e / R ≈ 0.146 · 0.125 / 3.40 ≈ **0.005 N·m·s/rad**. The sim's DC-motor
  model represents back-EMF only at the torque-speed envelope, and Test 18's KD
  grid (0, 0.01, …) skipped 0.005. KD = 0 plus the later 0.010 N·m frictionloss
  fits the step response well, so it stays; adding 0.005 to that sweep would
  settle it.

---

## 3. `microtaur_study/`: the 09-20 config, one module per concern

`robot.py` (names, XL330 motor + electrical model, the motor-range patch, mass
check), `sim2real.py` (stage 0/1/2 table), `actions.py` (8 motor actions,
predictive five-bar filter, delay), `observations.py` (actor 33 / critic 60 +
contract check), `sensors.py`, `events.py`, `commands.py`, `terminations.py`,
`rewards.py`, `terrain.py`, `terrain_maps.py` (A/B/C heightfields, pure numpy),
`env_cfg.py`, `kinematics.py` (unchanged). Tasks `Microtaur-Rigid-{Flat,Rough}-S{0,1,2}`.

`tests/check_equivalence.py`: bit-exact against upstream on the post-reset
state, the action term over 300 steps on a shared env, and the safety filter;
rollout statistics agree within run-to-run noise. (mujoco-warp is not
bit-deterministic on GPU, so whole trajectories cannot be compared.)

Found in upstream along the way: `reset()` crashes on mjlab ≥ 1.6 (fixed here);
the GAPS doc's spec-patch table was wrong (only the motor range patch changes
the model); and the **30 Hz control-rate override never runs**, so training has
always been 50 Hz although the hardware characterisation chose a 35 ms period
(§8, item 1).

---

## 4. Reward

`rewards.py`, checked term-by-term by `tests/check_rewards.py`.

| term | weight | form |
|---|---|---|
| track_lin_vel_xy | +1.0 | exp(−\|v_cmd − v\|² / 0.10²) |
| track_ang_vel_z | +0.5 | exp(−(ω_cmd − ω)² / 0.15²), always on |
| motor_energy | 0 → target (curriculum) | Σ_all motors (\|τ q̇\| + 159 τ²) / (m g · 0.15) |
| action_rate | −0.05/8 | Σ_all motors (Δa)² |
| termination | −2.0 per event | non-timeout terminations |

Every motor cost uses one formula summed over all motors (`qfrc_actuator` is
zero on passive joints, so a spine motor is included automatically). No
posture term. Copper loss R/k_t² = 159 W/(N·m)² from the datasheet
(k_t = 0.146 N·m/A is also the datasheet value); on the 09 RL gait copper loss
is 4x the mechanical power.

How it got here (flat, 2048 envs):

| run | energy weight | outcome |
|---|---|---|
| pilot 1 | −0.025 from step 0, no termination penalty | suicide: episodes ended in 0.1 s |
| pilot 2 | + termination −2, σ 0.04 | stood still (no gradient from a standing start) |
| pilot 3 | + σ 0.10 | walked at it. 100, gave it up by 200 (walking +0.40 tracking, −22.5 × w energy) |
| **sweep** | **0 for 200 it., linear ramp to target by 500** | **walks and keeps walking** |

Energy-curriculum sweep, 700 iterations (runs under a shared, busy GPU):

| target weight | forward tracking | speed | copper | mechanical |
|---|---|---|---|---|
| 0.005 | 0.95 | 0.139 m/s | 10.3 W | 2.2 W |
| 0.010 | 0.88 | 0.133 m/s | 8.5 W | 1.7 W |
| 0.018 | 0.88 (it. 446, ramp not finished) | 0.128 m/s | 10.7 W | 2.6 W |

---

## 5. Gait evaluation: not good enough yet

`microtaur_study/scripts/eval_gait.py` runs a checkpoint at fixed straight
commands (play config) and reports tracking, gait, body and effort metrics.
Policy: target weight 0.005, iteration 699.

| cmd | achieved | gait | stride | duty RR/RL/FL/FR | body height | tracking err | torque saturated |
|---|---|---|---|---|---|---|---|
| 0.10 | 0.102 | irregular ≈ trot | 4.5 Hz | .73 / .43 / .72 / .15 | 85 mm | 10.7° | 26% |
| 0.15 | 0.149 | irregular ≈ trot | 5.0 Hz | .70 / .36 / .70 / .20 | 88 mm | 10.6° | 27% |
| 0.20 | 0.171 | irregular ≈ pace | 5.0 Hz | .70 / .33 / .69 / .26 | 88 mm | 10.5° | 36% |
| 0.25 | 0.173 | irregular ≈ pace | 5.0 Hz | .68 / .46 / .66 / .33 | 87 mm | 10.7° | 37% |
| 0.30 | 0.137 | irregular ≈ pace | 5.9 Hz | .59 / .62 / .66 / .30 | 84 mm | 11.8° | 37% |

(`docs/figures/2026-09-25/gait_0.15.png`, `frames_0.15.png`.)

- **Limping trot**: one diagonal (RR + FL) carries the load; the other mostly
  hovers, the front-right foot lifting 20+ mm while the rest clear 5–8 mm.
- **Short strides** (~30 mm at 5 Hz), and **no speed above ~0.17 m/s**, so it is
  not only that the commands were too low.
- **Actuator-limited**: 10–12° rms target-to-actual error and 26–37% of samples
  at the 0.129 N·m cap; the policy commands exaggerated targets to get torque.
- **Standing tall**: 84–88 mm against the 70 mm nominal, legs near straight
  (possibly to reduce holding torque; not verified).
- **Undertrained**: 700 iterations against 4500 for the old policies.

---

## 6. Terrains

`terrain_maps.py`, rendered by `scripts/viz_terrains.py`
(`docs/figures/2026-09-25/terrains_overview.png`, `render_B.png`,
`render_C_entry.png`, `render_C_mid.png`).

- **A** flat.
- **B** one fixed map of 70 mm flat-topped cells, heights Gaussian
  (σ = 6.55 mm) clipped to **+13.1 / −16.4 mm**; variety from random spawn
  position and heading. Heightfield spacing **2.5 mm**: a heightfield
  interpolates between samples, and at 10 mm the median cell edge was a 32°
  slope (71% of edges under 45°); at 2.5 mm the median edge is 68° and the ramp
  is narrower than the 6.2 mm foot radius. Cost: 2.56 M samples per 4 × 4 m.
- **C** low flat 0.5 m → **left half raised for 4 m** (right half stays on the
  base level) → low flat 0.5 m. 100% = 29.5 mm, so a lowered body puts the left
  legs at ~13.1 mm retract and the right at ~16.4 mm extend. Straight commands
  only. Rendered mid-course: left feet +29.4 mm, right −0.2 mm.
- **Ceiling** = 50% of retract / extend at the stand pose (+13.1 / −16.4 mm),
  ramped 0 → 100% by `terrain_levels` (same map, heights scaled).
  ⚠ The recorded rigid gait keeps the feet **behind** the hip (foot x −36 to
  −16 mm, common +0.26 at touchdown, +0.38 at liftoff), where the travel is
  smaller: 50% there is +10.8 / −7.6 mm. Pure extension is nearly vertical
  (3.9° from vertical); the backward-downward motion comes from the gait.

---

## 7. Teacher

Actor and critic both get the privileged set: the 60-dim critic vector plus a
yaw-aligned height map, 0.40 × 0.30 m at 35 mm (12 × 9 = 108 values, two samples
per 70 mm cell). Terrain A + B + C, `terrain_levels` from level 0; command
curriculum as `TERRAIN_DESIGN.md` §5 (C envs never get yaw); energy curriculum
as §4. The student is deferred.

---

## 8. What is left before training the teacher

**Decisions**

1. **Control rate.** Hardware characterisation chose a 35 ms policy period
   (28.6 Hz; Test 09: 25–33 Hz reliable, 40 Hz not deterministic), and the
   action delay and 0.10 s filter horizon were tuned for it, but every policy
   so far trained at 50 Hz. Fix it (decimation 7 at 5 ms) before the teacher,
   or state why not.
2. **Energy target weight** — pick from the sweep after a longer run.
3. **Actuator cap.** Training caps torque at 0.6 × stall (0.129 N·m) and the gait
   saturates there 26–37% of the time. Confirm the cap matches what the
   hardware can sustain; it is the likeliest limit on speed and gait quality.
4. **Command range.** Training goes to 0.20 m/s; the policy saturates at ~0.17.
   Keep, or raise if the hardware can go faster.
5. **Terrain ceiling**: stand-pose (+13.1 / −16.4 mm, current) or stance-pose
   (≈ +11 / −8 mm).
6. **B map size**: a 4 m map is exactly one episode's maximum travel; tile the
   same pattern (preferred) or enlarge.
7. **Framework for the teacher**: mjlab now, IsaacLab after the reward/terrain
   are frozen (recommended), or IsaacLab directly.

**Checks**

8. Longer flat run (~3000 iterations) to see whether the asymmetric gait
   resolves with training.
9. Closed chain in PhysX at `kd = 0` (stand / trot / drop, ~1 h).
10. Throughput at dt 0.0025, pos_iters 16, on an idle GPU.

**Engineering (framework-independent unless noted)**

11. Teacher observation group (privileged actor, 35 mm height scan).
12. B and C as heightfield sub-terrains wrapping `terrain_maps.py`, with
    `terrain_levels` rows = difficulty.
13. Terrain-aware reset (C: at the start of the course, heading +x; B: random
    position and heading) and zero yaw commands on C.
14. `eval_gait.py` on B and C.
15. IsaacLab only: USD asset (friction values from the XML), loop joints,
    DC motor with KP 1 / KD 0, contact-sensor rewrite (`net_forces_w`
    threshold), encoder bias on observation and action, port of the action
    term, reset, rewards and terrain (pure torch / numpy already).

**Not blocking the teacher**

16. Copper-loss constant from the hardware current traces (the datasheet k_t
    and back-EMF constant disagree by 17%).
17. Checkpoints of the old policies are on naomio's machine only (0.465 kg).
18. `microtaur_study/assets/` is a copy of upstream's 09-20 XML (one
    `<inertial>` line differs from this repo's; its values were already public
    in the GAPS doc).

---

## Reproduce

```bash
# mjlab 1.6 venv (uv): mjlab==1.6.0, mujoco 3.11, rsl-rl-lib 5.4.2
uv pip install -e microtaur_study matplotlib
python microtaur_study/tests/check_rewards.py
python microtaur_study/tests/check_equivalence.py --upstream-src <upstream>/src
python microtaur_study/scripts/train_energy_sweep.py --energy-weight -0.01 --iters 700
DISPLAY=:1 MUJOCO_GL=glfw python microtaur_study/scripts/eval_gait.py --checkpoint <model.pt> --out <dir>
DISPLAY=:1 MUJOCO_GL=glfw python microtaur_study/scripts/viz_terrains.py --out <dir>
# isaaclab conda env (Isaac Sim 5.0)
python tools/spike_isaac_joint_friction.py --usd <microtaur_rigid_revolute.usda> --device cpu
```
