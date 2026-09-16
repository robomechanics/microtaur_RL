# Open-loop CPG gaits for all four Microtaur robots (2026-09-16)

Each robot gets an **open-loop gait made only of sine waves**: every motor follows an offset plus a few sine terms on one shared clock,
with no sensors and no neural network. Each gait is fitted to that robot's flat-ground RL policy and checked against it in
simulation, within what the hardware can do: XL330 servos taking position commands from the Uno Q at 50 Hz.

- **Plots and the combined video:** [`results/`](results/) (index in [`results/INDEX.md`](results/INDEX.md)). The video
  `results/00_all_robots_rl_vs_final_vs_old_cpg.mp4` has one row per robot and columns for RL, final CPG and old CPG.
- **Pitch in detail** (the first robot done, with the full story): [`pitch_match/README.md`](pitch_match/README.md).
- **Per-robot files:** `pitch_match/`, `yaw_match/`, `roll_match/` (active twist), `rigid_match/`.

Everything is simulation (MuJoCo / mjlab) on flat ground, walking straight at 0.15 m/s.

## Results

Scores are R² of the *average stride* (1 = identical), on equal 20 s windows against two independent RL runs. Ranges cover 3 windows
× 2 RL references. In brackets is the ceiling: RL windows scored the same way, the best any gait can reach.

| Robot | Final gait file | Sines | Legs | Spine | IMU angles | Gyro | Torque | Body speed CPG / RL | Falls |
|---|---|---|---|---|---|---|---|---|---|
| **Pitch** | [`gait_pitch_final.json`](pitch_match/gait_pitch_final.json) | 6 | 1.00 [1.00] | 0.97–1.00 [0.98–1.00] | 0.92–0.95 [0.89–0.99] | 0.92–0.96 [0.97–0.99] | 0.92–0.93 [1.00] | 0.146 / 0.144–0.145 | 0 |
| **Yaw** | [`gait_yaw_final.json`](yaw_match/gait_yaw_final.json) | 6 | 0.98–0.99 [1.00] | 0.99 [1.00] | 0.80–0.88 [0.78–0.94] | 0.79–0.80 [0.98–0.99] | 0.92–0.93 [1.00] | 0.164 / 0.164–0.165 | 0 |
| **Roll** | [`gait_roll_final.json`](roll_match/gait_roll_final.json) | 5 | 0.99 [1.00] | 0.98–0.99 [1.00] | 0.95–0.98 [0.96–1.00] | pitch/roll rate 0.97 / 0.94 * | 0.84–0.86 [1.00] | 0.144 / 0.150 | 0 |
| **Rigid** | [`gait_rigid_final.json`](rigid_match/gait_rigid_final.json) | 5 | 0.99 [1.00] | – | 0.92–0.96 [0.95–0.99] | pitch/roll rate 0.97 / 0.99 * | 0.83–0.86 [1.00] | 0.149 / 0.147 | 0 |
| *old CPGs* | `*_match/gait_old_cpg.json` | 1 per leg | 0.80–0.89 | −1.2 to 0.4 | −9.3 to −0.6 | strongly negative | 0.11–0.58 | varies | 0 |

\* **Roll and rigid gyro:** the yaw-rate part is left out. Its stride curve is tiny (under 5°/s peak to peak), and the RL's version
is irregular, so the RL's average stride has almost none of it while the CPG's has 2× more; the total yaw-rate wobble is the same size
(roll 3.8 vs 4.2°/s, rigid 2.5 vs 2.6°/s).

**What this does and doesn't show.**

- **Legs and spine** match at the ceiling on every robot.
- **Body motion (IMU)** is at or near the ceiling for pitch, roll and rigid, and inside the ceiling's range for yaw.
- **Torque is the remaining gap.** The RL corrects every stride from its sensors, so only 62–75 % of its torque signal (and 54–81 % of
  its IMU signal) repeats from one stride to the next. A gait of fixed sine waves repeats 90–100 %. The scores compare *average
  strides*, and no open-loop gait can copy that stride-to-stride variation.
- **Roll walks 4 % slower than its RL.** The others match speed within 1 %.

## What "3 sines" or "6 sines" means

It is how many sine waves are added together to make each motor's motion.

- **1 sine.** A smooth, symmetric swing: the same speed going one way as coming back.
- **Why that's not enough.** The RL's leg stroke is lopsided: a slow sweep, then a fast snap back. One sine can't make that shape.
- **More sines.** Adding waves at 2×, 3×, … the stride rate, each with its own size and timing, bends the wave into the RL's shape:

  ```
  1 sine:              3 sines added:           RL:
    .--.      .--.       .-.        .-.          |\      |\
   /    \    /    \     /   `.     /   `.        | \     | \
  /      \__/      \   /      `._./      `._     |  `.___|  `.___
   same speed          slow sweep, fast return   slow sweep, fast return
   both ways
  ```

- **Example (pitch, leg 1's `a` motor).** Centered at 44°, it adds waves of about 16.8° (4 Hz), 9.2° (8 Hz), 5.4° (12 Hz), 2.0°, 0.7°
  and 0.1°. The first three do most of the work: 3 sines already fix the body motion, and 5–6 sines mostly improve the torque shape
  ([`results/09_pitch_1_vs_3_vs_6_sines_angles_imu.png`](results/09_pitch_1_vs_3_vs_6_sines_angles_imu.png)).
- **On the board.** Every 20 ms each motor computes `angle = center + wave1 + … + waveK`, clips the leg angle to stand ±30°, and
  sends it to the servo. That is about 100 cos/sin evaluations per tick.

## How the gaits were made

1. **Record the RL** (32 simulated robots, 24 s) with the motor command delay fixed at 20 ms. The sim normally randomizes it per
   robot between 20 and 40 ms, and the RL's stride rate follows it, so a fixed delay gives one clean stride rate. Only
   robot-measurable signals are logged (IMU, 9 motors' angle, speed and torque), plus the motor commands.
2. **Fit the gait** (`fit_harmonic_gait.py`): per robot, offset + K sines per motor on the RL's commands, time-aligned and averaged.
   - **Sine count:** K is capped so every sine stays below 24.5 Hz, for the 50 Hz board loop. That gives 6 sines for pitch and yaw
     and 5 for roll and rigid, which step faster.
   - **Leg scale ×1.04:** an empirical speed calibration. It was tuned against the RL's body speed, and a fresh, independent RL run
     confirmed it for pitch, yaw and rigid.
3. **Yaw only: one round of learning correction** (`refine_gait_ilc.py`). Run the gait, compare each motor's *actual* stride with
   the RL's, and add 0.7× the difference to that motor's sine waves; then legs ×1.03 to restore speed. This cut the back-knee errors
   by 35–50 % and improved legs, spine, IMU and torque against independent RL runs. It is still pure sine waves, but the correction is
   tuned to the simulated servos. More rounds fitted one run's noise and cost speed, so only one was kept.
4. **Play the gait open-loop in sim** through the RL's own motor pipeline. That includes its **±30° leg command range**, which the
   board player applies too, so sim and board send the same commands.

## Verification

| Check | Result |
|---|---|
| Plots of every robot (stride curves, torque) | final CPG overlays the RL; see `results/01`–`08` |
| Videos, frames every 2 s over 20 s | RL and final CPG in matching poses on all four robots; no falls. The old CPGs turn away. |
| 60 s runs, 32 robots | no falls; speed steady. Heading drift per 60 s, CPG / RL: pitch +27 ± 18 / +30 ± 29°, yaw +95 ± 111 / +127 ± 79°, roll +17 ± 11 / +37 ± 34°, rigid −17 ± 19 / −21 ± 30°. Time at the torque cap and peak joint speeds are within the RL's. |
| Board start (hold stand 1 s, ramp in over 2 s), simulated, falls counted from t = 0 | 0 of 32 falls on every robot; speed after 4 s matches |
| Fresh RL run as independent reference | speed within 0.002 m/s for pitch, yaw, rigid; roll 4 % slow |
| 50 Hz board loop | top sine at 24.0 / 23.3 / 20.8 / 22.2 Hz (pitch / yaw / roll / rigid) |
| Pitch with 25 Hz updates or 10 Hz servo smoothing | still walks with no falls; 6 sines degrade least |
| Board player dry runs and tests | pitch, yaw, roll inside every servo's tick range; leg clip shares match the sim (36 %, 28 %, 24 %); all tests pass |
| Two independent reviews (method validity, code correctness) | no hidden feedback and no invalid step; their bugs and overstatements are fixed below |

**Fixed after the reviews.**

- **Leg clip.** The sim clipped leg commands at stand ±30° and the board didn't. The board now does.
- **Old-CPG baselines** had been replayed with that clip. They were re-run through the original CPG code.
- **Scoring.** The "RL vs RL" ceiling compared unequal run lengths; everything is now scored on equal windows.
- **Cost of transport and spine sweep.** Both divided by distance along the start direction, which counted turning as "slower". They
  now use body speed, and the sweep also requires walking straight.
- **Spine timing** was mislabelled.
- **Roll's fit** dropped 5 of 32 robots.
- **×1.04** had been explained as averaging shrinkage. It is a speed calibration.

## Hardware constraints

| Constraint | How the gaits respect it |
|---|---|
| **Open loop: no sensors, no network** | A gait file is 1 frequency plus, per motor, an offset and K cos/sin pairs, and the replay reads no sensor. The only rule applied on top is the fixed ±30° leg clip. |
| **50 Hz position commands** | Every sine term stays below 24.5 Hz. |
| **±30° leg command range** | Clipped in sim and in the board player alike. The gaits touch it on 24–36 % of leg samples, by at most 1.2–2.9°, where the RL's own commands sit at the same limit. |
| **XL330 speed and torque** | The sim uses the XL330 motor model. Fastest leg joint speed CPG / RL over 60 s: pitch 14.7 / 16.0, yaw 8.5 / 8.8, roll 9.3 / 12.8, rigid 15.5 / 16.1 rad/s. Time at the torque cap is within 1 point of the RL. Commands step faster than 32 rad/s on about 1 % of steps; that is where the motor model's torque reaches zero, so the joint just lags, as with the RL (which commands up to 49–52 rad/s). |
| **Servo tick ranges** | Pitch, yaw and roll stay inside every servo's allowed ticks, with no tick clamping. Rigid has 8 servos and the 9-servo player in [`board/`](board/) doesn't cover it. |

**Caveats for later hardware work** (not needed for the sim results):

- The sim servo is a soft PD capped at 60 % of stall torque, while a real XL330 is stiffer. Keep Profile Velocity at 0.
- The yaw correction is tuned to the sim servo.
- Pitch legs 2 and 3 are mirrored front-to-back in the sim model; check the wiring with a one-joint jog before running a pitch gait.
- Servo directions and spine IDs and zero positions still need measuring on each robot.

## Per robot

- **Pitch** (4.00 Hz, 6 sines): closest overall match. Details, spine options and hardware notes are in
  [`pitch_match/README.md`](pitch_match/README.md).
- **Yaw** (3.89 Hz, 6 sines, one correction round): **the yaw RL policy itself turns left** about 2°/s at a zero turn command, and
  only 55 % of its body motion repeats from one stride to the next. The final gait follows it (+95° per 60 s, RL +112 to +127°). The
  back knees (leg1_e, leg2_e) still over-swing by about 3°, and body roll runs slightly ahead of the RL.
  [`results/15_yaw_before_vs_after_correction_angles_imu.png`](results/15_yaw_before_vs_after_correction_angles_imu.png) shows the
  correction.
- **Roll** (4.16 Hz, 5 sines): near-exact stride match, but 4 % slow. A larger leg scale closes the speed gap and lowers the IMU match
  (×1.06: speed 0.147, IMU 0.92).
- **Rigid** (4.43 Hz, 5 sines): near-exact; the old rigid CPG spun about 175° per minute.

## Reproduce

From this folder, with `PY="env -u PYTHONPATH /home/naomio/anaconda3/envs/microtaur/bin/python"` and `R` = `pitch`, `yaw`, `roll` or
`rigid`:

```bash
$PY openloop_spine_cpg.py $R policy --delay-steps 1 --num-envs 32 --seconds 24 --series --out ${R}_match/rl_d1.npz        # record the RL
$PY fit_harmonic_gait.py ${R}_match/rl_d1.npz ${R}_match/gait_${R}_final.json --harmonics 6 --leg-scale 1.04               # fit
# yaw only: one correction round from a 40 s rollout of the fitted gait (kept as gait_yaw_final_before_ilc.json), then
# legs x1.03 -> yaw_match/ilc/gait_it1_legs103.json, copied to gait_yaw_final.json
$PY refine_gait_ilc.py yaw_match/rl_d1.npz yaw_match/final_40s.npz yaw_match/gait_yaw_final_before_ilc.json yaw_match/ilc/gait_it1.json --gain 0.7
$PY openloop_spine_cpg.py $R harmonic --gait ${R}_match/gait_${R}_final.json --delay-steps 1 --repeats 32 --seconds 64 --series --out ${R}_match/long_final.npz
$PY openloop_spine_cpg.py $R policy --delay-steps 1 --num-envs 32 --seconds 64 --series --out ${R}_match/rl_fresh.npz     # independent RL run
$PY score_gaits.py --ref "RL=${R}_match/rl_d1.npz" --ref "RL fresh=${R}_match/rl_fresh.npz@0:20" \
    "RL fresh 20-40 s (ceiling)=${R}_match/rl_fresh.npz@20:40" "Final 0-20 s=${R}_match/long_final.npz@0:20"                # equal-window scores
$PY compare_rl_cpg.py ${R}_match/compare_final "RL=${R}_match/rl_d1.npz" "Old CPG=${R}_match/old_cpg.npz" "Final CPG=${R}_match/long_final.npz@0:20"
$PY openloop_spine_cpg.py $R cpg --delay-steps 1 --repeats 32 --seconds 24 --series --out ${R}_match/old_cpg.npz          # old CPG baseline
$PY openloop_spine_cpg.py $R harmonic --gait ${R}_match/gait_${R}_final.json --delay-steps 1 --repeats 32 --seconds 24 --settle 0 --board-start --series --out ${R}_match/board_start.npz
$PY openloop_spine_cpg.py $R harmonic --gait ${R}_match/gait_${R}_final.json --delay-steps 1 --seconds 20 --video ${R}_match/video_cpg_final.mp4
$PY grid_video.py results/00_all_robots_rl_vs_final_vs_old_cpg.mp4 --columns "RL" "Final CPG" "Old CPG" --row "Pitch=..." ...
```

| Script | What it does |
|---|---|
| `openloop_spine_cpg.py` | Sim runner. `policy` runs an RL checkpoint; `harmonic` plays a gait file open-loop; `cpg` plays the old CPG. Also: delay, sweeps, trim, board start, 25 Hz hold, smoothing. |
| `fit_harmonic_gait.py` | Fits a gait file from an RL rollout (capped below 24.5 Hz). |
| `refine_gait_ilc.py` | One learning-correction round against the RL's actual joint motion. |
| `compare_rl_cpg.py`, `score_gaits.py` | Stride plots, tables, equal-window scores, stride repeatability. |
| `spine_sweep_report.py`, `apply_lr_trim.py` | Spine sweeps and left/right steering trim (pitch spine options). |
| `side_by_side.py`, `grid_video.py` | Labeled side-by-side and grid videos. |
| `board/` | Uno Q open-loop player for pitch, yaw and roll gait files, with tests. |
