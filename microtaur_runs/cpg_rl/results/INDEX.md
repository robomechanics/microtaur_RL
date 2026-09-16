# Results: open-loop CPG vs RL, all four robots (2026-09-16)

Copies of the current plots and one combined video, all from simulation on flat ground walking straight at 0.15 m/s. The originals and
every log are in `../<robot>_match/`. Method, verification and caveats: [`../README.md`](../README.md) and
[`../pitch_match/README.md`](../pitch_match/README.md).

## Data

**[`data/`](data/README.md):** every simulation log, gait file and table used here, organised per robot, plus CSV exports of exactly
what each plot draws (`data/plot_curves/`). See `data/README.md` for file meanings and how to load them.

## Video

**`00_all_robots_rl_vs_final_vs_old_cpg.mp4`:** one grid, 20 s, all synced from t = 0, same camera.

| Row | Robot | Columns |
|---|---|---|
| 1 | pitch spine | RL (neural network), final CPG (sine waves only), old CPG |
| 2 | yaw spine | same |
| 3 | roll (twist) spine | same |
| 4 | rigid (no spine) | same |

## Plots

All stride plots show one average stride over the simulated robots. Blue is always the RL.

| File | What it shows |
|---|---|
| `01`–`08` | Per robot (pitch, yaw, roll, rigid): RL vs final CPG vs old CPG. Odd numbers: joint angles + IMU. Even numbers: motor torque. |
| `09`, `10` | Pitch: 1 vs 3 vs 6 sine waves per motor. This is why the final gaits use several. |
| `11` | Pitch: the RL's stride rate per robot at 20 vs 40 ms motor delay. The delay sets the rate. |
| `12` | Pitch: spine sweep. Body speed per spine amplitude and timing; a box marks a setting that walks straight with enough spine swing. |
| `13`, `14` | Pitch: RL-like spine vs ±15° and ±17.5° spine with steering trim (the spine-using options). |
| `15`, `16` | Yaw: final gait before vs after the learning correction. |

## Scores

R² of the average stride (1 = identical), on equal 20 s windows against two independent RL runs. Ranges cover 3 windows × 2 RL
references. In brackets is the ceiling: RL windows scored the same way, i.e. the best any gait can reach.

| Robot | Sines | Legs | Spine | IMU angles | Gyro | Torque | Body speed CPG / RL | Falls |
|---|---|---|---|---|---|---|---|---|
| Pitch | 6 | 1.00 [1.00] | 0.97–1.00 [0.98–1.00] | 0.92–0.95 [0.89–0.99] | 0.92–0.96 [0.97–0.99] | 0.92–0.93 [1.00] | 0.146 / 0.144–0.145 | 0 |
| Yaw | 6, corrected | 0.98–0.99 [1.00] | 0.99 [1.00] | 0.80–0.88 [0.78–0.94] | 0.79–0.80 [0.98–0.99] | 0.92–0.93 [1.00] | 0.164 / 0.164–0.165 | 0 |
| Roll | 5 | 0.99 [1.00] | 0.98–0.99 [1.00] | 0.95–0.98 [0.96–1.00] | pitch/roll rate 0.97 / 0.94 | 0.84–0.86 [1.00] | 0.144 / 0.150 | 0 |
| Rigid | 5 | 0.99 [1.00] | – | 0.92–0.96 [0.95–0.99] | pitch/roll rate 0.97 / 0.99 | 0.83–0.86 [1.00] | 0.149 / 0.147 | 0 |
| Old CPGs | 1 per leg | 0.80–0.89 | −1.2 to 0.4 | −9.3 to −0.6 | strongly negative | 0.11–0.58 | varies | 0 |

- **Roll and rigid gyro leave out yaw rate.** Its stride curve is tiny and the RL's is irregular; the total yaw-rate wobble of CPG and RL
  is the same size.
- **Torque is the remaining gap.** Only 62–75 % of the RL's torque signal repeats from one stride to the next, because the RL corrects
  each stride from its sensors. No open-loop gait can copy that.
- **The yaw RL itself turns left** about 2°/s at a zero turn command, and the final yaw gait follows it.
