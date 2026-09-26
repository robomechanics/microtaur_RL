# Microtaur IsaacLab flat tuning log

Each round: train (2048 envs), evaluate (`criteria.txt`: speed / trot / stride /
clearance / yaw; five-bar FK leg space), then check consecutive close-up frames,
the top-view path and the videos before judging. Folders are
`figures/isaac_eval/<run>_it<N>/`; videos `figures/isaac_play/<run>_*.mp4`.

## r0 flat_pilot2 (1000 it): D1 reward, yaw sigma 0.15, sensor contact timers
- Numbers: speed stuck ~0.17 m/s at every command, no turning, "7 Hz" stride.
- Frames: feet on the ground almost all the time, legs rock/drag about the contact point; splayed posture.
- Cause found: PhysX gives a resting foot zero impulse in ~43% of substeps -> contact timers restarted every ~10 ms -> GaitReward scored chatter as a trot. Fixed (contact per policy step, `mdp/contact.py`).
- FK: swing retraction ~0 (-1.5..+3 mm): no retract/extend stepping.

## r1 r1_contact_yaw010 (1000 it): contact fix + yaw sigma 0.10
- Numbers: retraction +3..+12 mm, clearance 11-13 mm, speed 0.21-0.28 (cmd 0.10-0.35), turning ~0.4x with -0.12 rad/s drift, stride 7.1 Hz.
- Frames: walks on ONE diagonal (RL+FR stance, RR+FL mostly airborne, FL flails to 38 mm); body held at pitch -12 deg, roll -7 deg; heading drifts 40-55 deg in 6 s on straight commands.
- Reading: GaitReward has no period; "one pair always up, one always down" makes both timers grow equally and scores as a trot. The retraction number is partly the flailing leg.
- Learning curve flat after ~600 it -> variants screened at 600 it.

## r2 variants (600 it each, one change vs r1)
### r2a_trackup: trot weight 2.0 -> 1.0, yaw weight 0.5 -> 1.0
- Numbers: duty 0.8, retraction <= 0, clearance 3 mm, speed ~0.15 at every command, no turning.
- Frames: frozen splayed stance (RR -30 deg, FR +17 deg), all feet on the ground in 20 consecutive steps, micro-shuffle. Top view straight at +0.25 rad/s.
- Better: less tilt (roll -4.5, pitch -3), less drift (-11 deg / 6 s).
- Verdict: worse. Lower trot weight -> back to dragging; more yaw weight does not produce turning.

### r2b_smooth: action_rate weight -0.05/8 -> -0.05 (8x)  ** best so far **
- Numbers: "trot" at every straight command, duty 0.50 on all four legs, swing retraction 8-10 mm on every leg, clearance 12 mm, body 72 mm (upright), tilt 3-4 deg; speed 0.13-0.25 m/s (cmd 0.10-0.35); stride 7.1 Hz; yaw error rms 0.86 rad/s.
- Frames / contact diagram: a clean diagonal trot (RR+FL alternate with RL+FR, 50/50); every leg lifts 10-23 mm and is visibly shorter in swing than in stance (FK r ~61 vs ~79 mm). This is the retract/extend stepping we want.
- Remaining: stride far too fast (each diagonal 2 policy steps = 70 ms); straight commands curve right (~30 deg in 6 s, top view); speed saturates ~0.25 m/s; turning not tracked.
- r2c_tilt20 cancelled at start and r2d not run: both were r1-based; the next round builds on r2b.

## r3 variants (600 it, one change vs r2b)
- r3a_smoother: action_rate -0.05 -> -0.15 (slower stride)
- r3b_yaw: track_ang_vel_z weight 0.5 -> 1.0 (drift / turning)
- r3c_gaitwide: GaitReward std 0.025 -> 0.1, max_err 0.1 -> 0.2
- r3d_linsig: track_lin_vel_xy sigma 0.10 -> 0.07 (speed modulation)

### r3a_smoother: action_rate -0.05 -> -0.15
- Numbers: stride 5.7 Hz (from 7.1), still labelled trot, clearance ~10 mm; but duty asymmetric (RR/FL 0.41, RL/FR 0.61), speed ~0.16 m/s at every command, pitch -13 deg, heading drifts +30..45 deg / 6 s.
- Frames: alternation still visible, but FL flails to 35-39 mm while RL stays crouched (r 45-56 mm) carrying the body: sliding back toward r1's lean-on-one-diagonal gait.
- Verdict: worse than r2b. -0.15 slows the stride but brings back the diagonal lean; the useful range is between -0.05 and -0.15 (try ~-0.08 later on the best r3 base).

### r3b_yaw: r2b + track_ang_vel_z weight 0.5 -> 1.0
- Numbers: trot, clearance 16 mm, retraction 5-12 mm, stride 5.7 Hz, body upright (pitch -4); duty asymmetric 0.60/0.40 (the other diagonal this time); speed ~0.18 at every command; no turning at any yaw command.
- Frames: regular 5-step cycle, legs retract in swing, upright body. Top view at +0.25 rad/s curves left, but so does the straight command (bias, not command following).
- Why no variant turns: the trot yaws the body +-0.5 rad/s every step (yaw error rms 0.52 rad/s at zero command, mean 0.01). That noise is larger than the commands (<= 0.25), so tracking the mean yaw rate changes the yaw reward by only ~10%; more weight or a different sigma does not change that. Also: the command curriculum reaches yaw +-0.25 only at iteration 625, so 600-iteration screens never trained on the commands the eval tests.
- Next: yaw range +-0.25 from the start (numbers in the command stages); keep looking for less body yaw oscillation.

### r3c_gaitwide: r2b + GaitReward std 0.025 -> 0.1, max_err 0.1 -> 0.2
- Numbers: best speed tracking so far (0.107 / 0.174 / 0.297 m/s at cmd 0.10 / 0.20 / 0.35, ratio 0.83-1.07), trot, duty ~0.51 on all legs, clearance 8-12 mm, body 67 mm, torque saturation 0.12-0.29 (lower). But turns left +0.3..0.37 rad/s at EVERY command (100-120 deg in 6 s).
- Frames / contact diagram: clean alternating diagonals; RL lifts ~8 mm but its leg angle stays at ~0 deg (FK arc 1.1 mm, extension 1.8 mm) -- it steps in place and does not push, so the robot yaws left.
- Pattern across runs: which leg misbehaves is random (r1: RL over-swings; r3c: RL does not swing; r2b drifts right, r3b left). Left/right symmetry is broken by chance each run; numbers alone do not fix that. Candidate (needs approval: training-algorithm change, not a reward term): rsl_rl symmetry augmentation with a left/right mirror (also mirrors yaw commands).

### r3d_linsig: r2b + track_lin_vel_xy sigma 0.10 -> 0.07
- Numbers: trot, stride 5.7 Hz, clearance 16 mm, swing retraction 12-18 mm on every leg, body 75 mm upright, least heading drift so far (+16 deg / 6 s); first hint of turning (heading difference between +-0.25 commands ~35 deg / 6 s, ~0.2x). But duty 0.40/0.60 (diagonals unequal), speed 0.18-0.24 (weak modulation), torque saturation 42%, FL lifts 35-39 mm (more than needed).
- Frames / contact diagram: very regular 5-step (175 ms) trot, every leg cycles through clear retract (RR r 52 mm) -> extend (r 76 mm); FR 70 <-> 95 mm.

## r3 summary
- r2b stays the cleanest symmetric trot; r3c has the best speed tracking but a dead RL leg and constant left turn; r3d the best-looking stride and least drift but high effort. action_rate -0.15 (r3a) is too strong; yaw weight 1.0 alone (r3b) does not produce turning.

## r4 variants (600 it; all yaw commands +-0.25 from step 0; base r2b action_rate -0.05)
- r4a_combo: gaitwide (r3c) + lin sigma 0.07 (r3d)
- r4b_linsig_yaw: r3d + yaw range from start (isolates the yaw-range effect)
- r4c_combo_smooth: r4a + action_rate -0.08

### r4a_combo: r2b + gaitwide + lin sigma 0.07 + yaw +-0.25 from start
- Numbers: speed tracking as good as r3c (0.119 / 0.179 / 0.299 at cmd 0.10 / 0.20 / 0.35), trot at 0.15-0.35, duty ~0.51 symmetric, clearance 8-14 mm, retraction 4-10 mm, stride 7.1 Hz, torque saturation 0.18-0.30. Turning still weak: mean yaw rate +0.18 / +0.12 at +-0.25 commands (difference 0.06 rad/s) and +0.15 on straight commands (left bias); roll +5.6 deg.
- Frames / contact diagram: clean alternating diagonals, all legs lift; RR is the weak leg this time (lifts ~7 mm, angle stays ~0 deg, arc 1.5-2.7 mm) -> left bias. Same random left/right asymmetry as r1 / r3c.
- Verdict: best all-round so far (speed + symmetric trot), but no turning. Yaw range from step 0 did not fix turning by itself.

### r4b_linsig_yaw: r3d + yaw +-0.25 from start (no gaitwide)
- Numbers: trot at every straight command, duty 0.50 on all four legs, retraction 8-10 mm on every leg, clearance 11-14 mm, upright (roll -1, pitch -4), speed 0.111 / 0.180 / 0.284 at cmd 0.10 / 0.20 / 0.35. Yaw: -0.18..-0.21 rad/s at EVERY command (straight, +0.25, -0.25): constant right turn, no command following.
- Frames / contact diagram / top view: textbook 50/50 diagonal trot, every leg lifts 10-22 mm; the path is a right-hand arc (~70 deg in 4 s) on a straight command.
- Verdict: cleanest gait + good speed, but the random yaw bias again (right this time). Across r1-r4 the turn direction is random per run and never follows the command -> symmetry is the missing piece, not a weight.

## r5 variants (queued after r4c; yaw commands +-0.5 rad/s from step 0 so the turning signal exceeds the +-0.5 rad/s per-step body yaw)
- r5a_r4b_yaw05: r4b + yaw +-0.5
- r5b_r4a_yaw05: r4a + yaw +-0.5
- r5c_r4b_yaw05_w1: r4b + yaw +-0.5 + yaw weight 1.0

### r4c_combo_smooth: r4a + action_rate -0.05 -> -0.08
- Numbers: best speed tracking so far (0.119 / 0.187 / 0.320 at cmd 0.10 / 0.20 / 0.35, ratio 0.92-1.19), trot everywhere, duty ~0.51, lowest torque saturation (0.14 at 0.20 m/s), clearance 7-12 mm. Yaw -0.21 rad/s at every command (right turn, no command following); stride still 7.1 Hz.
- Frames: FR cycles strongly (r 71 <-> 84 mm), RR only 53-62 mm with its angle stuck at +0..+7 deg: RR is the weak leg -> right turn.

## r4 summary
- Speed tracking is solved numerically (gaitwide + lin sigma 0.07: r4a / r4c ratio ~0.9-1.2); the trot is clean and symmetric in contact timing (duty 0.50).
- Every run turns at a constant, random-direction yaw rate (one leg pushes less) and ignores yaw commands: r4a left, r4b right, r4c right. Starting yaw commands at +-0.25 did not change that.

### r5a_r4b_yaw05: r4b + yaw commands +-0.5 from start
- Numbers: trot, duty 0.50, clearance 16 mm, retraction 9-12 mm, upright; speed 0.15 / 0.19 / 0.23 (weaker than r4b). Yaw +0.47..+0.50 rad/s at EVERY command -- a constant left spin (~190 deg in 6 s), bigger than r4b's bias.
- Top view: at cmd -0.25 rad/s (right) it walks a tight LEFT circle (~180 deg in 4 s). RL arc ~3 mm (the non-pushing leg).
- Verdict: worse. A wider yaw range makes the random turn bias larger instead of teaching command following.

### r5b_r4a_yaw05: r4a + yaw commands +-0.5 from start
- Numbers: speed tracking good (0.124 / 0.194 / 0.290), trot, duty ~0.50, clearance 11-14 mm, retraction 6-11 mm; yaw +0.37..+0.38 rad/s at every yaw command (left spin, ~130 deg in 6 s), except ~0 at 0.35 m/s.
- Top view: S-shaped, mostly left path at cmd +0.25; not following the command. RL arc ~2 mm (the weak leg again).
- Note: the non-pushing leg has always been a REAR leg (r3c RL, r4a RR, r4c RR, r5a RL, r5b RL); which side is random.

### r5c_r4b_yaw05_w1: r4b + yaw +-0.5 + yaw weight 1.0
- Numbers: trot, duty 0.50, clearance 12-16 mm; speed 0.15 / 0.20 / 0.23; yaw +0.44..+0.47 at every command (left spin); RL arc ~3 mm.
- Top view: left arc at cmd -0.25 (right). Same as r5a.

## r5 summary / loop stopped (2026-09-26)
- Widening yaw commands to +-0.5 made the random constant turn larger (+0.37..+0.48 rad/s), never command following. Numbers-only tuning has not produced turning in 13 variants (r1-r5).

## Where things stand (numbers-only)
- Solved: foot-contact signal (bug fix); a clean, symmetric-in-timing diagonal trot with retract/extend stepping (duty 0.50, swing retraction 6-12 mm, clearance 10-16 mm); speed tracking 0.10-0.35 m/s (best: r4c_combo_smooth, ratio 0.92-1.19, torque saturation 0.14).
- Not solved: (1) every run turns at a constant, random-direction yaw rate because one REAR leg barely swings, and none follows yaw commands; (2) stride 5.7-7.1 Hz (fast).
- Best candidate to build on: r4c_combo_smooth (action_rate -0.08, GaitReward std 0.1 / max_err 0.2, lin sigma 0.07, yaw sigma 0.10, yaw +-0.25 from start).
- Proposed next step (needs approval, not a reward term): rsl_rl symmetry augmentation with a left/right mirror of observations and actions (mirrors yaw commands too), which targets exactly the random left/right asymmetry.
