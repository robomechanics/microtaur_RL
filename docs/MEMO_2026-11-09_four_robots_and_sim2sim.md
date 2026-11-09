# Microtaur spine study, 2026-11-09: four robots under one recipe, and the MuJoCo sim2sim check

Branch `spine-study-isaac`, HEAD `ab38987`. Status as of 11-09 16:00 (EST). Supersedes the status parts of the 09-25
memo. All numbers come from the eval scripts named next to them.

This memo has two parts:
- §1–4: where the experiment stands.
- §5: the brief for a separate session that will run the trained policies in plain MuJoCo (a sim2sim check).

---

## 1. Summary

1. **Question.** Does an active-spine Microtaur (spine axis yaw, roll or pitch) use its spine on its own when it is trained
   exactly like the rigid one?
   - Fairness rule (Ben): one recipe for all robots and no spine-specific reward.
   - Any change applies to every robot.
2. **Four robots, same recipe.** Each run is a teacher (PPO, 6000 iterations, 2048 envs) followed by a student
   (distillation, 300 iterations).
   - Robots: `rigid_uno` (ru1), `yaw_uno` (yu1), `roll_uno` (roll1) and `pitch_uno` (pitch1).
   - Battery kept, MKR board removed, UNO Q and IMUs at the yaw / rigid pose.
   - ru1, yu1 and roll1 are done.
   - pitch1 is training: iteration ~2450 at 16:00, teacher ETA ~19:45, then the student and the evals.
3. **Every robot walks.** All four use a 4.8 Hz diagonal trot with no flight phase. The rigid robot is still the best on
   terrain.
4. **Turning.**
   - **Spine use:**
     - Yaw uses its spine to steer: a static bend of up to 13°.
     - Roll uses a small roll of 1–4°. Locking it halves the turning, so the roll matters.
     - Pitch holds a large *constant* bend of −9° to −18°, the same sign for both turn directions. It is posture, not
       steering, and the pitch robot turns about as badly as the rigid one.
   - **Turning decays during training for every robot.** The decay follows the terrain-level curriculum (levels start at
     iteration 500), so final-checkpoint comparisons understate turning.
5. **Open problems:**
   - The roll student drifts in heading (41° in 20 s, against 11° for its teacher).
   - The yaw student falls on C level 0 on the GPU but not on the CPU (a known yaw GPU-physics issue).
   - The yu1 teacher is slow at low commands.
   - Turning decay: a proposal to give B levels ≥ 3 a zero yaw command for all robots awaits Ben's approval.

---

## 2. Robots (registry `microtaur_study/src/microtaur_common/robot_models.py`)

Asset directories are under `microtaur_study/assets/`. The MJCFs are MuJoCo-native. Each USD was built from its MJCF and
checked against MuJoCo by the build scripts (`*.usda.check.txt`).

| variant | MJCF | mass kg | root body (= Isaac "base") | spine axis, sign | 2nd IMU body × site quat (wxyz) | foot bodies (leg 1..4) |
|---|---|---|---|---|---|---|
| rigid_uno | rigid_microtaur/robot_uno_battery.xml | 0.552 | battery | none | none (slot zeros) | leglink2_v2_with_leg_down{,_2,_3,_4} |
| yaw_uno | active_yaw_microtaur/robot_uno_battery.xml | 0.739 | s_microtaur_bottom_half_motor_mount | yaw, −1 | s_microtaur_top_half_motor_mount × (0,0,1,0) | as rigid |
| roll_uno | active_twist_microtaur/robot_uno_battery.xml | 0.494 | twisting_microtaur_bottom_half_motor_mount (**rear** half) | roll (axis −x), −1 | twisting_microtaur_top_half × (0.707,0,0.707,0) | **…_3, …_4, …, …_2** (another order) |
| pitch_uno | active_pitch_microtaur/robot_uno_battery.xml | 0.509 | pitch_microtaur_top_half_1 (front half) | pitch (axis +y), +1 | pitch_microtaur_rear_half × (0,0,0.707,0.707) | leg{1..4}_d_link |

The sign column is `spine_mirror_sign`, the factor the spine angle gets under left/right mirroring.

Spine sign conventions:
- **yaw:** + = bent toward a left turn (yu1 holds +12° in a left turn).
- **roll:** + = front half rolled left side down (checked in MuJoCo).
- **pitch:** pitch1 holds −9° to −18°. Its rear legs are about 15 mm shorter than the front ones, which suggests the rear
  half is tail-down. This is inferred from leg lengths, not yet checked in frames.

Model notes:
- **Leg order.** Leg 1 is rear right, 2 rear left, 3 front left, 4 front right. Joint names are canonical in every model
  (`leg{i}_{a,e}_joint_act` active, `leg{i}_{b,d}_joint` passive). Only roll names its *foot bodies* in another order.
- **Closed chains.** Each leg is a five-bar, closed in the MJCF by 8 `<connect>` equality constraints. In Isaac it is a
  PhysX articulation loop: 16 position iterations, 2.5 ms step, closure gap p99 ~0.15 mm.
- **roll_uno:** the front half stands 11.5 mm higher (4° nose down).
- **pitch_uno:** taken from upstream `c6d3dcd` (11-08) with our fixes:
  - the leg 1/2 b-link inertials were not transformed; we replaced every b link by the mirror mean of legs 3/4;
  - names that start with a digit were renamed for the USD.
  - Build script: `/home/rml3/Documents/ben/spine/models_uno_imu_2026-11-09/build_uno_imu.py`.
  - The two halves are still 0.85 mm apart in y (CAD).
- **yaw_uno** is heavy (0.739 kg). Two-leg stance needs about 94 % of the 0.129 N m effort cap.

---

## 3. Training recipe (identical for all four)

**Tasks.** `Microtaur-Isaac-{RigidUno,YawUno,RollUno,PitchUno}-Teacher-Cur-v0` (teacher) and `…-Teacher-Cur-Student-v0`
(student), all with `leg_polar_obs=True`. Run them with
`microtaur_study/pipeline.py --robot <variant>`, which trains, runs the early gait checks and then all evals.

**Teacher.**
- PPO, MLP 512-256-128 with ELU.
- Observations are privileged: the critic terms, a 108-value height scan, and foot (x, z) plus virtual-leg (r, θ) for
  each leg. Input size is 184 for rigid and 187 for spine robots.
- 6000 iterations × 32 steps × 2048 envs.

**Terrain** (`microtaur_isaac/terrains.py`, scale 2.0):
- 20 % A flat, 50 % B blocks, 30 % C (a railed curb-straddle lane, walked straight only, yaw command 0).
- 7 levels plus a run-out row.
- Gate "complete" from iteration 500, per-env "speed" moves, `c_reverse_frac` 0.5.

**Commands** (`task_params.COMMAND_STAGES`):
- vx 0.10–0.35 m/s, wz ±0.25 rad/s (the final stage starts at step 20 000), vy always 0.
- No heading mode; resampled every 6–10 s.

**Rewards.** Byte-identical across runs: tracking, trot_gait, heading_tracking, motor_energy (final weight −0.002),
action_rate, feet terms.

**Student.**
- Rsl_rl DistillationRunner, 300 iterations.
- MLP 256-128 with ELU, empirical observation normalisation.
- 15-frame observation history (layout in §5.2).

**Runs.** Checkpoints are under `/home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/`.
Each run directory has `params/env.yaml` and `params/agent.yaml`.

| run | teacher dir (model_5999.pt) | student dir (model_299.pt) | eval summary |
|---|---|---|---|
| ru1 | 2026-11-06_11-26-27_ru1 | 2026-11-07_00-12-55_ru1_student | runs_local/pipeline/ru1/summary.md |
| yu1 | 2026-11-06_11-22-28_yu1 | 2026-11-07_00-38-07_yu1_student | runs_local/pipeline/yu1/summary.md |
| roll1 | 2026-11-05_00-14-37_roll1 | 2026-11-05_10-51-51_roll1_student | runs_local/pipeline/roll1/summary.md |
| pitch1 | 2026-11-09_13-34-22_pitch1 (training) | created by the pipeline after the teacher | runs_local/pipeline/pitch1/summary.md (when done) |

---

## 4. Results (Isaac, flat unless stated)

### 4.1 Final checkpoints

| | ru1 rigid | yu1 yaw | roll1 roll | pitch1 pitch |
|---|---|---|---|---|
| teacher speed at cmd 0.10 / 0.20 / 0.35 m/s | 0.063 / 0.208 / 0.332 | 0.023 / 0.133 / 0.312 | 0.075 / 0.208 / 0.336 | pending (it 1000: 0.091 / 0.174 / 0.325) |
| teacher mastery: first level < 100 % (C fwd / C rev / B) | none / none / none | none / 6 / 5 | 6 / 6 / none | pending |
| teacher heading drift, A flat, 20 s straight | 5.5° | 9.2° | 11.0° | pending |
| turn ratio (final) | −0.01 | 0.17 | 0.28 | pending |
| student speed | 0.081 / 0.230 / 0.341 | 0.030 / 0.135 / 0.292 | 0.093 / 0.197 / 0.314 | pending |
| student mastery (C fwd / C rev / B) | 6 / 6 / none | **0 / 0** / none | 4 / 6 / none | pending |
| student heading drift | 3.6° | 10.2° | **40.8°** | pending |

### 4.2 Turning over training

Turn ratio is the measured yaw rate divided by the commanded one. The value in parentheses is the mean spine angle in the
left-turn segment.

Measurement: `scripts/isaac_turn_legs.py`, 20 A-flat envs, tour of straight 3 s, left 0.25 rad/s 6 s, straight 2 s,
right 6 s, straight 3 s; statistics over the last 4 s of each segment; ratio is the mean of left and right.

| it | 500 | 1000 | 1500 | 2000 | 3000 | 4000 | 5000 | 5999 |
|---|---|---|---|---|---|---|---|---|
| ru1 | 0.18 | 0.02 | 0.03 | −0.01 | 0.00 | 0.00 | 0.00 | −0.01 |
| yu1 | 1.13 (+13°) | 1.21 (+12°) | 0.89 (+11°) | 0.99 (+11°) | 0.59 (+8°) | 0.49 (+5°) | 0.25 (+4°) | 0.17 (+4°) |
| roll1 | 0.52 (+4.4°) | 0.30 (+2.7°) | 0.42 (+1.9°) | 0.35 (+1.8°) | 0.46 (+1.5°) | 0.32 (+1.2°) | 0.32 (+0.9°) | 0.28 (+0.7°) |
| pitch1 | −0.03 (−11°) | 0.12 (−14°) | 0.12 (−16°) | −0.01 (−18°) | queued | queued | queued | queued |

- **Rigid** turns at iteration 500 by stride difference alone (right legs sweeping ~77 mm/s faster). It forgets this by
  iteration 1000, when the terrain levels start rising.
- **Yaw** turns with a static spine bend; locking the spine removes most of its turning.
- **Roll**, in S tours (`scripts/isaac_spine_turn.py`): spine free 0.17 vs locked 0.09, fast turns 0.21 vs 0.10.
- **Pitch** turns no better than rigid.

Videos (2×2 grid, same iteration for all four, oblique camera, path inset):
- `/home/rml3/Documents/ben/spine/figures/isaac_play/four_robots_2026-11-09/four_robots_turn_tour_it2000.mp4`
- The `…_it5999.mp4` version is generated automatically after pitch1 ends.

---

## 5. Sim2sim brief: run the Isaac-trained policies in MuJoCo

### 5.1 Goal and deliverables

**Goal.** Check that the IsaacLab (PhysX) policies behave the same in an independent simulator, MuJoCo, on the *same MJCF*.
This tests whether gait, speed, turning and spine use are robot behaviour or PhysX artefacts. It is an evaluation only:
no retraining and no change to the training code.

- **Scope:** students first, because their observations are deployable and need no height scan or contact sensors.
  Teachers are optional: on flat ground the height scan is a constant, but their privileged contact terms would need
  MuJoCo equivalents.
- **Simulator:** plain MuJoCo on CPU (the `mujoco` Python bindings, already in conda env `spine`). Torch is also in
  `spine`, so checkpoints load without rsl_rl. mjlab / MuJoCo-Warp in env `spine-mjlab` is possible but not needed.
- **Deliverables:**
  - A small package, suggested at `microtaur_study/src/microtaur_mujoco/`, containing: model loading and patching per
    variant, the actuator model, the observation builder per variant, a checkpoint loader, and a stepping loop.
  - Eval scripts that mirror the Isaac ones in §5.4.
  - One table: Isaac vs MuJoCo for all four robots (students, and teachers if done).
  - Videos into `/home/rml3/Documents/ben/spine/figures/` (no GUI windows on this machine).

### 5.2 The interface contract (must match Isaac exactly)

Source of truth for the code: `microtaur_study/src/microtaur_isaac/` (`env_cfg.py`, `robot.py`, `mdp/actions.py`,
`mdp/observations.py`) and `microtaur_study/src/microtaur_common/`.

**Reference implementation.** `microtaur_study/deploy/unoq/microtaur_policy.py` is a numpy implementation of the *rigid*
47-D student: observation history, normaliser and action pipeline. It was checked bit-for-bit against Isaac recordings by
`deploy/unoq/test_golden.py`. Extend it rather than starting from scratch. The `*_uno` students add `leg_polar` and, on
spine robots, the spine and IMU 2 terms.

**Timing.**
- Policy period 35 ms. Isaac uses physics 2.5 ms × 14; the old mjlab env used 5 ms × 7.
- The MJCFs have no `<option>`, so set the timestep explicitly.

**Joint order.** The canonical order `leg1_a, leg1_e, …, leg4_a, leg4_e`, then `spine_joint_act` on spine robots, sets
the action and joint-observation order. Joint zero and sign are the MJCF's own (the USD keeps them).

**Default pose.**
- `STAND_A = (0.45, −0.45, −0.45, 0.45)`, `STAND_E = −STAND_A`, spine 0.
- Reset is IK-consistent (`microtaur_common/reset.py`), with the root at `nominal_root_height_m`.

**Leg actions** (8):
- Use `microtaur_common.walk_action.WalkActionCore` directly. It is framework-free torch and the same code mjlab and Isaac
  ran.
- `requested = stand + 0.5236 rad (30°) × clip(a, −1, 1)`, then a safety filter, then a 1-policy-step delay.
- Eval settings: gain 1, bias 0.
- Call `step(actions[:, :8], q, qd, reset)` once per policy step. It returns `(raw, applied)`; `applied` is the leg
  target for the next 14 physics substeps.

**Spine action** (9th, `mdp/actions.py`):
- `requested = clamp(clip(a, −1, 1) × 0.35, ±0.33 rad)`.
- The applied target is the *previous* step's request (one-step hold). At reset the hold is seeded with the current spine
  angle.

**Actuators.** The MJCF has no `<actuator>` block and its joint parameters are **not** the trained ones (§5.3). Apply
torques in Python each substep, as IsaacLab's `DCMotor`:
- `τ* = KP (q_target − q) − KD q̇`;
- `τ_max = clip(S (1 − q̇/V), 0, E)`, `τ_min = clip(S (−1 − q̇/V), −E, 0)`, `τ = clip(τ*, τ_min, τ_max)`.

| motors | KP | KD | E (effort) | S (saturation) | V (rad/s) | armature | frictionloss (static = dynamic) |
|---|---|---|---|---|---|---|---|
| legs (XL330) | 1.0 | 0.0092 | 0.129 | 0.215 | 32.08 | 2e-4 | 0.028 |
| spine (XL330-M288) | 7.44 | 0.238 | 0.50 | 0.52 | 8.63 | 0.01395 | 0.010 |
| passive joints | 0 | damping 1e-4 | – | – | – | 1e-6 | 0.001 |

All values come from `microtaur_common/robot_constants.py` and `microtaur_isaac/robot.py`. Isaac additionally scales soft
joint limits by 0.95.

**Student observation**, one frame in this order (`mdp/observations.py`, `STUDENT_TERMS` + `leg_polar`):

| term | dim | definition |
|---|---|---|
| projected_gravity | 3 | gravity unit vector in the **root body frame** (not the imu_site frame; they coincide only on rigid) |
| base_ang_vel | 3 | root body angular velocity in the root frame |
| imu2_projected_gravity | 3 | spine robots: in the frame `other_half_body × imu2_quat`; rigid: zeros |
| imu2_base_ang_vel | 3 | spine robots: other half's angular velocity in that frame; rigid: zeros |
| joint_pos | 8 / 9 | motor angle − default (eval: no encoder bias) |
| joint_vel | 8 / 9 | motor velocities |
| foot_fk | 8 | `five_bar_for(variant).forward_torch(q_a, q_e)` per leg → (x, z) |
| actions | 8 / 9 | last policy output, **unclipped** (`env.action_manager.action`) |
| command | 3 | (vx, 0, wz) |
| leg_polar | 8 | per leg r = \|(x, z)\| and θ = atan2(x, −z) from the same FK |

- Frame size: 55 for rigid, 58 for spine robots. Every term is clipped to ±100.

**History and network.**
- 15 frames per term, flattened **term-major**: `[term0 × 15 frames oldest first, term1 × 15, …]`. Input size is 825 for
  rigid and 870 for spine robots.
- At reset each term's buffer is filled with the first frame (`ObsHistory.push` in the reference file).
- Normalisation: `(x − mean) / (std + eps)`. The keys are `student_obs_normalizer._mean` / `._std` in
  `model_state_dict`; the reference file reads `eps` from the bundle.
- Network: `student.0/2/4` (Linear, ELU, Linear, ELU, Linear), deterministic output = action.
- Teachers: `actor.0/2/4/6`, no normaliser.

**Commands in eval.** Fixed per segment; no noise, randomisation or pushes (the Play configs).

### 5.3 Known mismatches to handle (or measure on purpose)

1. **Foot friction.**
   - Isaac's ground is static = dynamic 1.0 with combine mode "max", so the effective friction is 1.0.
   - In MuJoCo the foot geoms have `priority=1` and `friction="0.7 0.02 0.001"` with `condim=4`, so MuJoCo would use
     **0.7**, plus torsional friction PhysX does not have.
   - For the matched comparison set the foot friction to 1.0. Then run 0.7 as a sensitivity point.
2. **XML joint parameters differ from training.**
   - Legs: the XML has `frictionloss 0.01` and range −0.5..1.4; training used 0.028 (sysid 10-28) and stand ± 0.75 rad
     (the mjlab env patched the range through MjSpec, see `microtaur_rigid/robot.py`).
   - Spine: the XML has armature 2e-4 and damping 0.002; training used armature 0.01395, the explicit KD, and no joint
     damping.
   - Patch all of these from `robot_constants`.
3. **Root frame vs IMU site.** Isaac's "base" terms use the root *body* frame. Compute them from `xquat` / `cvel` of the root
   body (`mj_objectVelocity(..., mjOBJ_BODY, id, flg_local=1)`), not from the MJCF `imu_site` sensors. Check that the two
   agree before using either.
4. **Closed-chain solver.**
   - Isaac's loop closure has p99 0.15 mm.
   - In MuJoCo the closure comes from the XML's equality constraints, and the models differ:
     - `rigid_microtaur/robot_uno_battery.xml` sets `solref="0.01 1"` (plus `solimp`) on its 8 `<connect>`;
     - the yaw, roll and pitch XMLs set none, so they get MuJoCo's default `0.02 1`, which is softer.
   - Use the same stiffness for all four (the rigid one), or report both settings.
   - Report the closure gap in MuJoCo too.
5. **Contact.** Isaac uses `contact_offset` 2 mm, `rest_offset` 0, with no self-collision. Keep MuJoCo's defaults but report
   foot penetration.
6. **The yaw robot already differs between PhysX GPU and CPU.** The yaw student falls on C level 0 on the GPU, not on the
   CPU. A MuJoCo result is a third data point, not a verdict.

### 5.4 Suggested order of work

**1. Plumbing before physics.**
- Record an Isaac rollout of each student: per policy step the full state, observation frames, network input and output.
  `scripts/export_student_bundle.py` shows how to build `golden.npz`.
- Load those states into MuJoCo (`qpos` / `qvel` by joint name; root from the root body pose), build the observation, and
  run the network.
- Observation and action must match to float precision. This catches frame, order and sign errors (roll's foot order,
  spine signs, IMU 2 frames) before any physics difference muddies them.

**2. Open loop.** Play the same scripted joint targets in both simulators:
- stand-and-settle height;
- the scripted IK trot of `scripts/isaac_reference_gait.py` / `isaac_diff_stride_turn.py`;
- a free spine step response (measured 3.4 Hz, ζ 0.34).

Compare root height, forward speed, yaw rate and spine angle.

**3. Closed loop, students.** For all four robots, mirror the Isaac evals:

| measurement | Isaac script | setup |
|---|---|---|
| gait + speed | `isaac_eval_gait.py` | commands 0.10 / 0.20 / 0.35 m/s |
| turn tour | `isaac_turn_legs.py` | ratio, spine angle, per-side stride |
| heading drift | `isaac_eval_terrain.py` | 20 s straight at 0.20 m/s, A flat |

- For trajectory comparison, `isaac_render_policy.py --tour --trace` saves an npz per step (command, yaw rate, root xy /
  yaw, spine angle, contacts).

**4. Optional:** teachers on flat ground, and B / C terrains (would need the terrain meshes from `terrains.py`).

**Suggested pass criteria** (Ben to confirm):
- same gait type (trot, ~4.8 Hz);
- no falls on flat;
- speeds within ~15 %;
- the turning order preserved: yaw > roll > rigid ≈ pitch, with the spine bends of the same sign and similar size.

### 5.5 Working rules on this machine (also in the session memory)

- **Environment.**
  - Conda env `spine` (`/home/rml3/anaconda3/envs/spine/bin/python`); `spine-mjlab` only for mjlab.
  - Never use other people's envs.
- **Isaac reference evals on the GPU.**
  - Check `nvidia-smi` first.
  - At most two of our runs at once.
  - While user3 has a process on the GPU, set `MT_GPU_CAP_MIB` to 9216; otherwise 15000.
  - Never edit `~/.config/microtaur/gpu_cap_mib`.
  - MuJoCo runs on the CPU and has no GPU limit.
- **Don't touch running jobs.** Running now: the pitch1 pipeline (PID 644987, train 645033), the pitch1 turn sweep and the
  final 2×2 video job. Stop jobs only by PID.
- **Git.**
  - Commit only your own files: no `git add -A` / `-a`, no checkpoints.
  - Author Ben Gu <gu1669318914@gmail.com>; never mention Claude / Anthropic in commits or files.
  - Upstream `aryan-chandra-cmu/Microtaur_RL` is read-only.
- **Reporting.** Reply to Ben in Chinese with English technical terms and full absolute paths. Ask before structural
  changes.

---

## 6. Open items

- **pitch1:** teacher, student and evals tonight; then the turn sweep to 5999, the final 2×2 video, and pitch rows in §4.
- **Turning decay:** zero yaw command on B levels ≥ 3 for all robots, tested with one rigid_uno and one yaw_uno run. Awaits
  Ben.
- **roll student heading drift (40.8°):**
  - First check: IMU 2 observation frame on roll vs ground truth.
  - Alternative: distillation length.
- **yu1:** slow at low commands; yaw student GPU falls on C level 0.
- **Pitch spine posture** (−9° to −18°, the same sign for both turn directions): confirm the tail-down reading from frames.
- **Hardware side:** the deployment bundle exists only for the old rigid robot; the `*_uno` student layout (leg_polar,
  spine) is not in it yet.
