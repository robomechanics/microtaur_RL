# Data behind the results (2026-09-16)

These are copies of every simulation log, gait file and table used for the plots and scores in [`../`](../), plus CSV exports of
exactly what each plot draws. The originals stay in `../../<robot>_match/`. To regenerate this folder, run
`../../export_results_data.py`.

All data comes from simulation (MuJoCo / mjlab), flat ground, command 0.15 m/s straight, motor command delay 20 ms unless the name
says otherwise. Nothing here was used to train a neural network. The gaits are fitted to recorded RL runs, and the RL policies were
trained earlier (`old_distilling/<run>/model_4499.pt`).

## Layout

```
plot_curves/   CSVs of the plotted curves (open in Excel, MATLAB, pandas)
pitch/  yaw/  roll/  rigid/   raw logs (.npz), gait files (.json), tables (.md, .txt) per robot
```

## plot_curves/: what each plot draws

| CSV | Plot(s) | Content |
|---|---|---|
| `01_02_pitch_rl_vs_final_vs_old_cpg.csv` | 01, 02 | average stride of RL, old CPG and final CPG |
| `03_04_yaw_rl_vs_final_vs_old_cpg.csv` | 03, 04 | same, yaw |
| `05_06_roll_rl_vs_final_vs_old_cpg.csv` | 05, 06 | same, roll |
| `07_08_rigid_rl_vs_final_vs_old_cpg.csv` | 07, 08 | same, rigid (no spine columns) |
| `09_10_pitch_1_vs_3_vs_6_sines.csv` | 09, 10 | RL vs gaits with 1, 3 and 6 sines per motor |
| `11_pitch_rl_stride_rate_by_motor_delay.csv` | 11 | stride rate of each simulated RL robot at 20 and 40 ms delay |
| `12_pitch_spine_sweep_grid.csv` | 12 | every spine setting in the sweep: body speed, heading drift, real spine swing, tilt, cost of transport, falls |
| `13_14_pitch_spine_versions.csv` | 13, 14 | RL vs final CPG vs ±15° and ±17.5° spine gaits with trim |
| `15_16_yaw_before_vs_after_correction.csv` | 15, 16 | RL vs yaw gait before and after the learning correction |

**Stride-curve CSVs** have 48 rows, one per stride bin. The `stride_pct` column is the bin center, where 0 % is leg 1's `a` motor at
its peak. Every other column is `<run> | <signal>`:

- `<motor> angle (deg)` for the 8 leg motors and the spine;
- `IMU pitch (deg)`, `IMU roll (deg)`, `pitch rate (deg/s)`, `roll rate (deg/s)`, `yaw rate (deg/s)`;
- `<motor> torque (N m)`.

Each value is the average over all simulated robots that didn't fall. For the RL, 32 robots over 20 s; the other runs are listed below.

## Per-robot files

| Name pattern | What it is |
|---|---|
| `rl_24s.npz` | RL policy, 32 robots, 24 s. **The run each gait was fitted to**, and the reference in the plots. |
| `rl_64s.npz`, `rl_fresh_64s.npz` | two more independent RL runs, 32 robots, 64 s, used for the equal-window scores and the ceiling |
| `rl_24s_40ms_delay.npz` (pitch) | RL at 40 ms motor delay, which gives the slower stride rate in plot 11 |
| `old_cpg_24s.npz` | old CPG (one sine per leg, fitted earlier), run through the original CPG code; 32 robots, 24 s |
| `old_cpg_fit.json` | the old CPG's parameters (`fit_<robot>.json`) |
| `final_cpg_40s.npz`, `final_cpg_64s.npz` | final gait, 32 robots, 40 s and 64 s |
| `gait_<robot>_final.json` | **the final gait**: the file the board player plays |
| `board_start_24s.npz` | final gait started like the board player (hold stand 1 s, ramp in over 2 s), falls counted from t = 0 |
| `fair_scores.txt` | equal-window scores against two RL runs (the table in `../INDEX.md`) |
| `compare_*_table.md` | per-plot tables: speeds, falls, tilt, cost of transport, clip share, per-motor match |
| pitch: `cpg_1sine_24s.npz`, `cpg_3sine_24s.npz`, `gait_1sine.json`, `gait_3sine.json`, `gait_3sine_40ms_delay.json`, `gait_4sine_backup.json` | earlier and backup gaits (16 robots, 24 s runs) |
| pitch: `spine_sweep_part1.npz`, `spine_sweep_part2.npz`, `spine_sweep_table.md` | spine sweep: 12 amplitudes × 12 timings × 3 robots, metrics only |
| pitch: `spine15_trim_64s.npz`, `spine175_trim_64s.npz`, `gait_spine15_trim.json`, `gait_spine175_trim.json` | spine-using gaits with left/right steering trim |
| pitch: `robust_{3,4,6}sine_{25hz,smoothed}.npz` | same gaits with 25 Hz command updates or 10 Hz command smoothing (16 robots, 24 s) |
| yaw: `final_cpg_before_correction_64s.npz`, `gait_yaw_before_correction.json` | yaw gait before the learning correction |

Run lengths are simulated time. The first 4 s of each run are discarded before logging (not for `board_start`), so a "24 s" file
holds 20 s = 1000 steps at 50 Hz.

## .npz format

```python
import json, numpy as np
z = np.load("pitch/final_cpg_64s.npz")
print(z.files)
meta = json.loads(str(z["meta"]))               # variant, mode, checkpoint or gait file, mass_kg, ...
q = z["s_q"]                                    # joint angles [T steps, N robots, motors], rad
names = [str(n) for n in z["motor_names"]]      # leg1_a, leg1_e, ... leg4_e, spine (canonical order)
dt = float(z["dt"])                             # 0.02 s
```

**Time series** `s_*`, shape `[T, N, ...]`, 50 Hz, motors in `motor_names` order. Only the IMU and motor signals exist on the real
robot; `s_truth_*` are simulator-only.

| Key | Meaning |
|---|---|
| `s_q`, `s_qd`, `s_tau` | joint angle (rad), joint speed (rad/s), motor torque (N m) |
| `s_requested`, `s_safe`, `s_target` | motor command as requested, after the sim's safety filter, and as applied after the command delay (rad) |
| `s_filter_correction` | requested − safe (rad) |
| `s_imu_gravity`, `s_imu_gyro` | IMU: gravity direction in the body frame; angular rate (rad/s) in the body frame |
| `s_phase_leg1` | gait clock phase (rad), gait and old-CPG runs only |
| `s_action` | action sent to the sim's action terms |
| `s_truth_pos`, `s_truth_vel_b` | simulator-only: base position (m) and body-frame velocity (m/s), used to score walking |

**Per-robot metrics** `m_*`, shape `[N]`:

| Key | Meaning |
|---|---|
| `forward_m_s` / `lateral_m_s` | distance along / across the starting direction ÷ time |
| `body_vx_mean_m_s` | mean forward speed along the body axis; use this for "speed" |
| `heading_drift_deg` | total turn over the logged run |
| `falls` | number of falls |
| `imu_tilt_mean_deg`, `imu_tilt_max_deg` | body tilt |
| `power_w`, `cot` | positive mechanical power; cost of transport |
| `leg_torque_rms_nm`, `leg_torque_peak_nm` | leg motor torque |
| `leg_target_speed_peak_rad_s` | fastest change in the leg command |
| `leg_filter_correction_deg`, `leg_filter_correction_max_deg` | safety filter corrections |
| `leg_clip_share`, `leg_clip_max_deg` | share of leg commands clipped at stand ±30°, and the largest overshoot |
| `spine_mean_deg`, `spine_amp_deg`, `spine_torque_*` | spine robots only |

Sweeps also carry `params` / `param_names`, and every file has `delay_steps`.

**Older logs.** The fixes were made at 17:58. Files recorded before then — every `rl_24s` (including `rl_24s_40ms_delay`) and `rl_64s`,
pitch and rigid `final_cpg_64s`, the pitch 1-sine / 3-sine / spine / robust runs (sweep parts included), and yaw
`final_cpg_before_correction_64s` — differ in four ways:

- `m_cot` divides by distance along the start direction instead of body speed;
- `m_forward_m_s` reads about 0.1 % low;
- `s_phase_leg1` is one step early;
- they have no `m_leg_clip_*`.

The plots and tables recompute cost of transport from `s_tau`, `s_qd` and `s_truth_vel_b`, so they are not affected.

Videos are not duplicated here. The combined grid is [`../00_all_robots_rl_vs_final_vs_old_cpg.mp4`](../00_all_robots_rl_vs_final_vs_old_cpg.mp4),
and the single-robot clips are in `../../<robot>_match/video_*.mp4`.
