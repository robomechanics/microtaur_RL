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

## r6 (user approved 2026-09-26: rescale the curriculum to the screen length, add straight commands; user also flagged: stride too fast, tapping/rocking instead of extension)
- Found: 600-it screens never reached the last curriculum stage (step 20000 = it 625), so vx 0.30-0.35 (and yaw 0.25 in r1-r3) were never trained but were evaluated.
- Found: the r4c policy's requested motor offsets sit at the action bound (|common|+|diff| = max motor offset, p50 0.49-0.52 rad = 30 deg x 1) while the safety filter never changes them by > 1 deg. The +-30 deg action scale, not the filter, caps extension (joint range +-43 deg, filter diamond 0.70 rad).
- New config values: commands.twist.rel_straight_envs (fraction of yaw = 0 commands), actions.joint_pos.action_scale_rad (hardware must use the same value).
- r6a_straight: r4c + curriculum stages at step 0 / 4000 / 10000 + 30% straight
- r6b_scale40: r6a + action scale 30 -> 40 deg (0.70 rad)
- r6c_scale40_smooth: r6b + action_rate -0.08 -> -0.11

## r7 (queued after r6): second gait-shaping term, approved by the user
- feet_air_time = IsaacLab Spot air_time_reward (user: use the standard open-source trot method, not upstream's): each foot earns the duration of its current air/contact phase, capped at mode_time, and 0 once the phase is longer -> phases pushed toward mode_time (period ~2 x mode_time). GaitReward sets which legs pair; air_time sets how long each step is.
- On the r4c tapping gait it earns 0.21 per step of a possible 0.6 (mode 0.15) / 1.2 (mode 0.3): lengthening phases pays 3-6x.
- r7a_air02: r6a + air_time weight 1.0, mode_time 0.2 s (Spot ratio air:gait = 1:2)
- r7b_air03: r6a + air_time weight 1.0, mode_time 0.3 s (Spot's value)
- r7c_air02_w2: r6a + air_time weight 2.0, mode_time 0.2 s

### r6a_straight: r4c + curriculum rescaled (final stage from it 312) + 30% straight commands
- Numbers: best speed tracking so far (0.116 / 0.182 / 0.327 at cmd 0.10 / 0.20 / 0.35, ratio 0.91-1.16 at every speed); trot, duty ~0.51; drift much smaller at 0.20 m/s (mean yaw -0.02..-0.04 rad/s at every yaw command; env 0 heading -22 deg / 6 s) but +0.27 rad/s at 0.35; still no command following (+-0.25 -> -0.03); stride 7.1 Hz; torque saturation 0.21 (0.38 at 0.35).
- Top view: nearly straight (~20 deg right in 4 s) -- first run that walks roughly straight. Close-ups: RR stays crouched (r 50-58 mm, angle ~0 deg) and taps; FR moderate 73-87 mm. Still tapping.
- Verdict: straight commands + reaching the final curriculum stage fixed most of the drift and the 0.35 m/s tracking. Turning and tapping remain (-> r7 air_time).

## r6 stopped by the user after r6a (r6b_scale40 killed at ~it 180, r6c not run); r7 started directly.

### r7a_air02: r6a + Spot air_time weight 1.0, mode_time 0.2 s
- Numbers: stride 5.7 Hz (from 7.1), clearance 10-17 mm, swing retraction up (FR 17-20 mm, RR 10-12 mm; RL 4-7, FL 6-8); speed 0.114 / 0.185 / 0.304 (ratio 0.87-1.14); some yaw response for the first time with a left bias (+0.165 at +0.25, +0.077 at -0.25, +0.12 straight; spread 0.09 rad/s); duty rear 0.60 / front 0.44-0.49 at <= 0.25 m/s -> "irregular (closest trot)"; pitch -8; torque saturation 0.26 (0.39 at 0.35), CoT 14.
- Frames (checked 2026-09-26 12:55): FR swings forward and retracts/extends clearly (r 70 <-> 98 mm, angle +2..+28 deg); RR uses both motors; FL kicks to 30-36 mm every step (too high); rear legs stay down longer than front (duty 0.60 vs 0.44-0.49), hence "irregular".

### r7b_air03: r6a + Spot air_time weight 1.0, mode_time 0.3 s (Spot's value)
- Numbers: stride 5.7 Hz (same as r7a; mode 0.2-0.3 s would mean ~1.7-2.5 Hz), trot at 0.15-0.35, speed ratio ~1.0 (0.136 / 0.201 / 0.300), retraction RR 11, FR 13-15, FL 9, RL 4-6 mm; strong left bias again (+0.30..+0.34 at every yaw command; heading +113 deg / 6 s straight); pitch -11, roll -6.5; torque saturation 0.27, CoT 13.
- Frames: not viewed (viewer was down; r7a / r8a / r8b checked instead, see "Frame check of r7 / r8")
- Reading so far: air_time lowers the stride from 7.1 to 5.7 Hz and lengthens retraction, but at weight 1.0 phases stay far below mode_time; the random turn bias is back (r6a had it mostly removed).

### r7c_air02_w2: r6a + air_time weight 2.0, mode 0.2
- Numbers: same as r7b: stride 5.7 Hz, trot, speed 0.121 / 0.189 / 0.277, left bias +0.29, RL weak (arc 2-4 mm), pitch -11.
- Frames: not viewed (viewer was down; r7a / r8a / r8b checked instead, see "Frame check of r7 / r8")

## r7 summary
- The air_time term earns the SAME per-step value at weight 1 and 2 (0.254 of 0.8 at mode 0.2 = 32%; 21% of 1.2 at mode 0.3) while trot_gait is ~saturated (1.79-1.89 / 2.0): the policy cannot lengthen the phases, not that it does not want to.
- r7a's requested motor offsets sit at the +-30 deg action bound (|common|+|diff| p50 0.52 rad on three legs, max 0.52 on all four; filter changes nothing > 1 deg). At a given speed a slower stride needs a longer stroke, which the action scale does not allow -> the scale caps stride length and forces the high stride rate.
- The random turn bias came back with air_time (r7b/c +0.3 rad/s); r7a had a small yaw response.

## r8 (launched without the r7 frame check -- the viewer is down; motivated by the measured action-bound saturation, not by the r7 frames)
- r8a_air02_scale40: r7a + action scale 40 deg (0.70 rad = filter diamond)
- r8b_air02w2_scale40: r7c + action scale 40 deg
- r8c_air03_scale40: r7b + action scale 40 deg

### r8a_air02_scale40: r7a + action scale 40 deg
- Numbers: straightest run so far (heading +2 deg in 6 s at 0.20 m/s; mean yaw -0.02..-0.04 at every yaw command) but no command following; stride still 5.7 Hz and air_time still 0.256 (identical to r7) -> the +-30 deg action bound was NOT what capped the phase length (hypothesis rejected). Motor travel p95 now 0.9-1.0 of the joint half-range on several motors; FR leg angle -27 deg (foot behind the hip) vs FL +18 deg; duty 0.62 / 0.44 on the two diagonals; speed 0.156 / 0.205 / 0.288; pitch -10; torque saturation 0.30.
- Frames (checked 2026-09-26 12:55): clean diagonal alternation, both RR motors move (a 0..45 deg, e -65..-35 deg), feet lift 8-23 mm; but FR stays long (73-96 mm) and angled BACKWARD (-21..-32 deg) the whole time; diagonals unequal (0.62 / 0.45).
- Open question: what keeps every phase at ~2.5 policy steps (stride 5 steps) across r3-r8 regardless of air_time weight, mode_time or action scale. Candidates to test: tight tracking sigmas (0.07 m/s) rewarding smooth body velocity, action_rate, the 1-step action delay + 0.10 s filter horizon, the compliant KP=1 servo.

### r8b_air02w2_scale40: r7c (air weight 2.0, mode 0.2) + action scale 40 deg
- Numbers: first stride below 5 Hz (4.76 Hz = 6-step period) and first straight commands that PASS all criteria (0.20 and 0.25 m/s: trot, stride, clearance, speed); air_time per unit weight 0.285 (from 0.254). Straight: mean yaw within +-0.02 rad/s (heading -21 deg / 6 s env 0). But speed no longer follows the command (0.21-0.23 m/s at every command: 0.10 -> 2.1x, 0.35 -> 0.66x); legs split by side (RR / FR mean angle -21 / -24 deg, RL / FL +9 / +15 deg); pitch -13 deg; no turning.
- So air weight 2.0 lowers the stride only together with the 40 deg action scale (r7c at 30 deg stayed at 5.7 Hz).
- Frames (checked 2026-09-26 12:55): regular 6-step cycle, BUT RR's a motor sits pinned at its joint limit (~66 deg, target flat) and only the e motor steps -> a one-motor leg, not five-bar retract/extend; FR nearly straight (82-101 mm) and angled back -14..-34 deg; RR crouched 46-63 mm; FL kicks to 30-35 mm; body nose-up 13 deg. Passing two straight commands in the table does NOT mean a usable gait.

### r8c_air03_scale40: r7b (air weight 1.0, mode 0.3) + action scale 40 deg
- Numbers: stride 5.7 Hz, trot, speed tracking good (0.117 / 0.199 / 0.298), right bias -0.14 rad/s (heading -39 deg / 6 s), yaw spread between +-0.25 commands only 0.04 rad/s; retraction 7-13 mm; RL mean angle -26 deg vs RR +7; air_time 0.243.
- Frames: not viewed (viewer was down; r7a / r8a / r8b checked instead, see "Frame check of r7 / r8")

## r8 summary
- Only r8b (air weight 2.0 + 40 deg scale) lowered the stride (4.76 Hz) and passed two straight commands, but it lost speed modulation (0.21-0.23 m/s at every command). r8a is the straightest walker (+2 deg / 6 s) but still 5.7 Hz. None follows yaw commands. Loop paused after r8: image viewer down (no frame checks since r7) and the phase-length limit needs a diagnosis rather than more numeric variants.

## Frame check of r7 / r8 (done 2026-09-26 12:55, viewer back)
- The 40 deg action scale lets the policy park motors at joint limits (r8b: RR a-motor pinned) and hold FR straight and swept back (r8a, r8b). r8b's lower stride comes with a distorted posture -> not a candidate.
- Most usable-looking so far: r6a (30 deg, straight commands; nearly straight path) and r7a (30 deg + air_time 0.2; FR clearly retracts/extends, but FL over-kicks and rear/front duty unequal).

## Body lean-back (user: "is the robot leaning back too much? check")
- Nose-up pitch per run (0.20 m/s): r2b 4.3, r4b 3.7, r6a 7.1, r7a 8.1, r7b/c 10.7-10.9, r1 12.4, r3a 12.8, r8b 12.6 deg; nominal stand 0.
- Fully explained by leg lengths (five-bar FK): front legs 76-90 mm, rear 50-65 mm vs 73.5 nominal; atan((front-rear)/168 mm hip spacing) matches the measured pitch within ~1 deg.
- The worst gaits lean most; the cleanest (r2b, r4b) lean < 5 deg. Rear legs crouched at 50-55 mm have no room to retract in swing, front legs at 86-90 mm are nearly straight -> both ends at the workspace edge -> short fast steps. Likely the real cause of the capped stride.
- The reward has no posture term (D0 removed upright / height; only the 70 deg tilt termination).

## Standard reward sets (looked up 2026-09-26)
- legged_gym LeggedRobotCfg defaults: tracking_lin_vel 1.0, tracking_ang_vel 0.5, lin_vel_z -2.0, ang_vel_xy -0.05, orientation 0, torques -1e-5, dof_acc -2.5e-7, base_height 0, feet_air_time 1.0, collision -1, action_rate -0.01; ANYmal-C flat: orientation -5.0, feet_air_time 2.0.
- IsaacLab velocity RewardsCfg: track_lin_vel_xy_exp 1.0, track_ang_vel_z_exp 0.5, lin_vel_z_l2 -2.0, ang_vel_xy_l2 -0.05, dof_torques_l2 -1e-5, dof_acc_l2 -2.5e-7, action_rate_l2 -0.01, feet_air_time 0.125, undesired_contacts -1, flat_orientation_l2 0 (flat configs: -2.5 A1/Go1/Go2, -5.0 ANYmal).
- Posture terms taken: flat_orientation_l2 (= legged_gym orientation), lin_vel_z_l2, ang_vel_xy_l2 (user approved: "reward needs body posture").

## r9 (base r7a) body posture
- r9a_orient2p5: flat_orientation_l2 -2.5
- r9b_orient5: flat_orientation_l2 -5.0
- r9c_posture_full: orientation -5.0 + lin_vel_z -2.0 + ang_vel_xy -0.05
- r9d_orient10: flat_orientation_l2 -10.0 (8 deg lean costs only ~0.1/s at -5 vs tracking 1.0)
- Next (r10, on the best r9): turning as two separate experiments: (a) a dedicated turn term, (b) rsl_rl left/right symmetry augmentation.

### r9a_orient2p5: r7a + flat_orientation_l2 -2.5
- Numbers: lean-back 8.1 -> 3.0 deg (front legs 72 mm, rear 64 mm vs 79 / 58 in r7a; nominal 73.5); best speed tracking of all runs (ratio 0.94-1.06 at EVERY speed 0.10-0.35); trot, duty 0.50 on all legs, clearance 11-13 mm, retraction 6-12 mm, torque saturation 0.17. But stride back to 7.1 Hz (r7a 5.7), air_time 0.21 (lower), right bias -0.25 rad/s (no turning).
- Frames: body visibly level; clean 50/50 diagonals; all legs lift 8-25 mm; both motors of RR move; FR 70-85 mm swinging +14..+27 deg, RR 55-69 mm.
- Reading: the orientation term fixes the lean and helps speed tracking, but a level body did NOT lengthen the stride -> the lean was not the stride limiter (hypothesis rejected).

### r9b_orient5: r7a + flat_orientation_l2 -5.0
- Numbers: level body (nose-up 2.8, roll -2; front 71 / rear 63 mm), speed ratio 0.88-1.17 (0.117 / 0.199 / 0.309), trot, duty ~0.50-0.56, retraction 4-11 mm, clearance 10-13 mm, torque saturation 0.19; stride 7.1 Hz; small left bias (+0.05..+0.07 rad/s; env 0 heading +37 deg / 6 s), yaw response ~0 (+0.069 at +0.25 vs +0.049 at -0.25).
- Frames (checked 14:32): body level; RL retracts/extends 58 <-> 74 mm but sits behind the hip (-18..-25 deg), FL 67 <-> 79 mm swinging +11..+26 deg; regular 4-step cycle; top view gently curving left (~25 deg in 4 s).
- Same picture as r9a: posture fixed, stride and turning not.

### r9c_posture_full: r7a + orientation -5 + lin_vel_z -2 + ang_vel_xy -0.05 (legged_gym posture set)
- Numbers: level (nose-up 2.0, roll -0.2) and nearly straight (env 0 +11 deg / 6 s), but WORSE gait: speed no longer follows the command (0.195 / 0.203 / 0.216 at 0.10 / 0.20 / 0.35), "irregular" at every command (FL duty 0.62), front legs barely retract (FL +1.5, FR -5 mm: longer in swing), clearance 8-9 mm, torque saturation 0.35, trot_gait 1.67 (lower); no turning.
- Frames (checked 14:32): feet lift only 5-14 mm (mostly < 10), FL stance blocks lengthen irregularly, small fast tip-toe steps.
- Reading: lin_vel_z_l2 -2 costs -0.046/s here (the tiny robot bounces relative to its size); with ang_vel_xy it suppresses the body motion a step needs. Standard weights are tuned for 30-50 kg robots; not a fit at these weights.

## r9 end (r9d stopped at ~it 300) -- user: foot lift / retraction is fine for now, leave it to rough terrain (no reason for the move to emerge on flat); start the terrain curriculum.

# Terrain curriculum
## c1_teacher (Microtaur-Isaac-Teacher-v0, 2048 envs, 1500 it)
- Terrain: A flat / B Gaussian blocks / C flat-step-flat at 20/50/30 %, 5 levels, promotion = walked > half a tile from the spawn, demotion = < half the commanded distance; teacher actor and critic see the critic set + 35 mm height map.
- Reward = flat best: r6a base (action_rate -0.08, GaitReward std 0.1 / max_err 0.2, lin sigma 0.07, yaw sigma 0.10) + air_time 1.0 (mode 0.2) + flat_orientation_l2 -2.5 (r9a; IsaacLab switches it off on rough, -2.5 is a compromise) + heading_tracking 1.0; commands: rescaled curriculum, yaw +-0.25 from start, 30 % straight.
- Turning: heading_tracking AND left/right symmetry augmentation together (user: do both at once).

(2026-09-26 15:00, reorganised: flat run folders moved to figures/isaac_eval/flat/<run>_it<N>/, flat videos to figures/isaac_play/flat/, superseded flat checkpoints to runs_local/isaac/archive/flat_checkpoints/. Paths above written as figures/isaac_eval/<run> now live under figures/isaac_eval/flat/. Layout: /home/rml3/Documents/ben/spine/README.md)
