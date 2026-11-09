## roll1_teacher
checkpoint `/home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-05_00-14-37_roll1/model_5999.pt`

**Gait (flat):** flight 0.00 four 0.00 diag +1.00 same_end -1.00 same_side -1.00 stride 4.8 Hz
**Speed at command 0.10 / 0.20 / 0.35 m/s:** 0.075 / 0.208 / 0.336

**Terrain (straight 0.20 m/s, 20 s):**
```
checkpoint /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-05_00-14-37_roll1/model_5999.pt
Teacher-Cur terrain; terrain scale 2.0; straight 0.2 m/s for 20.0 s; promotion distance (half a tile) 3.15 m

terrain   lvl    n   fall  trunc  progress m >= half tile  speed m/s  |dyaw| deg  |lateral| m  causes
A_flat      0    8   0.00   0.00        4.15         1.00      0.203         8.2         0.55  {}
A_flat      1    7   0.00   0.00        4.20         1.00      0.204        14.4         0.16  {}
A_flat      2   13   0.00   0.00        4.17         1.00      0.204        14.3         0.26  {}
A_flat      3   13   0.00   0.00        4.25         1.00      0.206         5.7         0.06  {}
A_flat      4   13   0.00   0.00        4.18         1.00      0.204        11.5         0.56  {}
A_flat      5   23   0.00   0.00        4.20         1.00      0.204        11.3         0.26  {}
A_flat      6    9   0.00   0.00        4.22         1.00      0.205         9.4         0.40  {}
A_flat      7   14   0.00   0.00        4.17         1.00      0.204        13.2         0.41  {}
B_blocks    0   28   0.00   0.00        4.22         1.00      0.205         9.6         0.34  {}
B_blocks    1   35   0.00   0.00        4.28         1.00      0.208         7.6         0.36  {}
B_blocks    2   34   0.00   0.00        4.30         1.00      0.209         8.5         0.28  {}
B_blocks    3   36   0.00   0.00        4.16         1.00      0.202         8.3         0.31  {}
B_blocks    4   27   0.00   0.00        3.99         1.00      0.195        11.0         0.37  {}
B_blocks    5   27   0.00   0.00        3.72         0.96      0.182        10.7         0.34  {}
B_blocks    6   31   0.00   0.00        3.47         0.87      0.171        13.4         0.47  {}
B_blocks    7   32   0.00   0.16        3.61         0.91      0.179        11.2         0.43  {'out_of_terrain_bounds': 5}
C_step      0   17   0.00   0.00        4.13         1.00      0.203         2.2         0.01  {}
C_step      1   19   0.00   0.00        4.19         1.00      0.205         1.4         0.00  {}
C_step      2   17   0.00   0.00        4.38         1.00      0.213         0.9         0.01  {}
C_step      3   24   0.00   0.00        4.46         1.00      0.215         2.0         0.00  {}
C_step      4   23   0.00   0.00        4.31         1.00      0.204         2.1         0.00  {}
C_step      5   21   0.00   0.00        3.99         1.00      0.188         0.9         0.00  {}
C_step      6   15   0.00   0.00        3.50         1.00      0.164         0.4         0.00  {}
C_step      7   14   0.00   0.00        3.54         1.00      0.166         0.8         0.00  {}
```
_(C step posture, turning + virtual legs and S tours rerun 11-05 with 470fb82: the eval scripts had used the rigid foot body names, which on roll_uno swap front / rear legs; old outputs in roll1_teacher/old_rigid_foot_names/)_

**C step posture:**
```
checkpoint /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-05_00-14-37_roll1/model_5999.pt
C lane, straight 0.2 m/s, 400 steps; on-step = root x inside [-1.85, 2.00] m
lvl | step mm | n samples | roll deg mean (+ left up) | stance r left - right mm | r_left mm | r_right mm | resets
  0 |     0.0 |      2035 |     -0.2 |     +0.0 |   +89.2 |   +89.1 | 0
  3 |    18.4 |      2049 |     -3.1 |    -20.9 |   +77.7 |   +98.6 | 0
  6 |    36.8 |      1990 |     +0.6 |    -29.5 |   +73.5 |  +103.0 | 0
target on a level body: r_left - r_right ~= -step height (legs on the raised left side shorter), roll ~ 0
```
**Turning + virtual legs:**
```
roll1_teacher: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-05_00-14-37_roll1/model_5999.pt
roll_uno robot, 20 A_flat envs, turn tour; envs that reset: 0
segment  | cmd vx  cmd wz | yaw rate rad/s (ratio) | vx m/s | spine deg mean +- std
straight | 0.20  +0.00  | +0.004               | 0.186  | -0.00 +- 0.65
left     | 0.15  +0.25  | +0.071 (0.28)        | 0.162  | +0.66 +- 0.70
straight | 0.20  +0.00  | -0.001               | 0.204  | +0.02 +- 0.66
right    | 0.15  -0.25  | -0.066 (0.27)        | 0.164  | -0.75 +- 0.70
straight | 0.20  +0.00  | +0.004               | 0.207  | -0.04 +- 0.64

virtual leg r (mm), straight trotting; stand pose r = 74.70 mm; root height 90.9 mm
leg | stance mean  min   max | swing min (retracted) | theta range deg
 1  |   79.1   73.5  85.1 |   58.7                | -22.3 ..  -4.4
 2  |   79.2   73.5  85.5 |   58.7                | -22.3 ..  -4.8
 3  |  102.7   98.2 105.8 |   77.5                |  -6.2 .. +13.2
 4  |  102.5   89.3 105.9 |   77.3                |  -6.6 .. +13.3
```
**Tight turns, S tours (spine free / locked; plot `/home/rml3/Documents/ben/spine/runs_local/pipeline/roll1/roll1_teacher/spine_turn_roll1_teacher.png`):**
```
roll1_teacher: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-05_00-14-37_roll1/model_5999.pt
roll_uno robot, 12 envs per group (rigid: 24), S tour = 2 s straight + 4 arcs of 6 s (+ - + -), stats over the last 4 s of every arc, sign-corrected (+ = toward the commanded turn)
set      cmd vx  wz   R_cmd | spine  | envs ok | yaw rate (ratio) | speed | radius m | spine deg mean +- std | osc deg @ Hz | no-slip: bend for R deg, R for bend m
in_train 0.10  0.25  0.40  | free   | 12/12   | +0.042 (0.17)    | 0.104 |   2.46   | +0.83 +- 0.71       | 0.94 @ 4.8   |  3.89     nan
in_train 0.10  0.25  0.40  | locked | 12/12   | +0.022 (0.09)    | 0.098 |   4.42   | +0.05 +- 0.74       | 1.03 @ 4.8   |  2.16     nan
1L_fast  0.10  0.50  0.20  | free   | 12/12   | +0.106 (0.21)    | 0.103 |   0.96   | +2.03 +- 0.82       | 1.01 @ 4.8   |  9.88     nan
1L_fast  0.10  0.50  0.20  | locked | 12/12   | +0.048 (0.10)    | 0.098 |   2.05   | +0.09 +- 0.71       | 0.99 @ 4.8   |  4.66     nan
1L_slow  0.05  0.25  0.20  | free   | 12/12   | +0.023 (0.09)    | 0.033 |   1.41   | +0.86 +- 0.77       | 1.04 @ 4.8   |  6.74     nan
1L_slow  0.05  0.25  0.20  | locked | 12/12   | +0.032 (0.13)    | 0.035 |   1.09   | +0.07 +- 0.75       | 1.06 @ 4.8   |  8.73     nan
spin     0.00  0.50  0.00  | free   | 12/12   | -0.030 (-0.06)    | 0.040 |    inf   | +1.47 +- 0.92       | 1.19 @ 4.8   |  0.00     nan
spin     0.00  0.50  0.00  | locked | 12/12   | +0.027 (0.05)    | 0.031 |   1.15   | +0.11 +- 0.72       | 1.01 @ 4.8   |  8.29     nan
```
**Terrain mastery (straight 0.20 m/s for 40 s, 30 envs per level; success = got through without a fall, any speed (C: up on the raised half and no fall on the course, B: 2 m without a fall); mastered = 100 %):**
```
roll1_teacher_C_fwd: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-05_00-14-37_roll1/model_5999.pt
C_step, straight 0.2 m/s for 40 s, 30 envs per level; success = climbed: body 0.5 m past the entry riser, then no fall until it leaves the course, any speed
level | step mm | success | course 4.4 m | falls | progress m (mean, min) | course speed m/s | old 20 s rule | not counted | fall causes
    0 |     0.0 |    100% |         100% |    0% | 4.70, 4.70 |            0.207 |          100% |           0 | -
    1 |     6.1 |    100% |         100% |    0% | 4.70, 4.70 |            0.211 |          100% |           0 | -
    2 |    12.3 |    100% |         100% |    0% | 4.70, 4.70 |            0.221 |          100% |           0 | -
    3 |    18.4 |    100% |         100% |    0% | 4.70, 4.70 |            0.226 |          100% |           0 | -
    4 |    24.5 |    100% |         100% |    0% | 4.70, 4.70 |            0.213 |          100% |           0 | -
    5 |    30.7 |    100% |         100% |    0% | 4.70, 4.70 |            0.180 |          100% |           0 | -
    6 |    36.8 |     87% |          87% |   13% | 4.62, 4.10 |            0.157 |           87% |           0 | base_too_low 4
first level below 100 %: 6
roll1_teacher_C_rev: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-05_00-14-37_roll1/model_5999.pt
C_step REVERSED (raised side on the right), straight 0.2 m/s for 40 s, 30 envs per level; success = climbed: body 0.5 m past the entry riser, then no fall until it leaves the course, any speed
level | step mm | success | course 4.4 m | falls | progress m (mean, min) | course speed m/s | old 20 s rule | not counted | fall causes
    0 |     0.0 |    100% |         100% |    0% | 4.70, 4.70 |            0.208 |          100% |           0 | -
    1 |     6.1 |    100% |         100% |    0% | 4.70, 4.70 |            0.210 |          100% |           0 | -
    2 |    12.3 |    100% |         100% |    0% | 4.70, 4.70 |            0.220 |          100% |           0 | -
    3 |    18.4 |    100% |         100% |    0% | 4.70, 4.70 |            0.226 |          100% |           0 | -
    4 |    24.5 |    100% |         100% |    0% | 4.70, 4.70 |            0.215 |          100% |           0 | -
    5 |    30.7 |    100% |         100% |    0% | 4.70, 4.70 |            0.189 |          100% |           0 | -
    6 |    36.8 |     80% |          80% |   20% | 4.58, 4.10 |            0.157 |           80% |           0 | base_too_low 6
first level below 100 %: 6
roll1_teacher_B: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-05_00-14-37_roll1/model_5999.pt
B_blocks, straight 0.2 m/s for 40 s, 30 envs per level; success = 2.0 m before any fall, any speed
level | step mm | success | falls | progress m (mean, min) | old 20 s rule | not counted | fall causes
    0 |       - |    100% |    0% | 2.00, 2.00 |          100% |           0 | -
    1 |       - |    100% |    0% | 2.00, 2.00 |          100% |           0 | -
    2 |       - |    100% |    0% | 2.00, 2.00 |          100% |           0 | -
    3 |       - |    100% |    0% | 2.00, 2.00 |          100% |           0 | -
    4 |       - |    100% |    0% | 2.00, 2.00 |          100% |           0 | -
    5 |       - |    100% |    0% | 2.00, 2.00 |          100% |           0 | -
    6 |       - |    100% |    0% | 2.00, 2.00 |           87% |           0 | -
first level below 100 %: none
```
**Videos:** `/home/rml3/Documents/ben/spine/runs_local/pipeline/roll1/roll1_teacher/videos/roll1_teacher_A_flat.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/roll1/roll1_teacher/videos/roll1_teacher_B_lvl6.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/roll1/roll1_teacher/videos/roll1_teacher_C_lvl6.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/roll1/roll1_teacher/videos/roll1_teacher_flat_side.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/roll1/roll1_teacher/videos/roll1_teacher_turn_tour.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/roll1/roll1_teacher/videos/roll1_teacher_turn_tour_top.mp4`

## roll1_student
checkpoint `/home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-05_10-51-51_roll1_student/model_299.pt`

**Gait (flat):** flight 0.00 four 0.00 diag +0.96 same_end -0.96 same_side -0.96 stride 4.8 Hz
**Speed at command 0.10 / 0.20 / 0.35 m/s:** 0.093 / 0.197 / 0.314

**Terrain (straight 0.20 m/s, 20 s):**
```
checkpoint /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-05_10-51-51_roll1_student/model_299.pt
Teacher-Cur terrain; terrain scale 2.0; straight 0.2 m/s for 20.0 s; promotion distance (half a tile) 3.15 m

terrain   lvl    n   fall  trunc  progress m >= half tile  speed m/s  |dyaw| deg  |lateral| m  causes
A_flat      0   14   0.00   0.00        4.01         1.00      0.208        40.0         1.15  {}
A_flat      1    9   0.00   0.00        3.93         1.00      0.208        41.0         1.36  {}
A_flat      2   19   0.00   0.00        3.98         1.00      0.208        40.3         1.29  {}
A_flat      3   11   0.00   0.00        3.98         1.00      0.208        42.8         1.29  {}
A_flat      4   14   0.00   0.00        3.97         1.00      0.208        43.2         1.33  {}
A_flat      5   10   0.00   0.00        3.97         1.00      0.208        41.9         1.28  {}
A_flat      6   13   0.00   0.00        3.97         1.00      0.208        41.9         1.31  {}
A_flat      7   10   0.00   0.00        4.06         1.00      0.208        34.9         1.11  {}
B_blocks    0   35   0.00   0.00        4.03         1.00      0.207        34.3         1.12  {}
B_blocks    1   25   0.00   0.00        4.11         1.00      0.205        24.5         0.80  {}
B_blocks    2   27   0.00   0.00        4.12         1.00      0.200        11.2         0.31  {}
B_blocks    3   44   0.00   0.00        3.92         1.00      0.191         8.7         0.31  {}
B_blocks    4   25   0.00   0.00        3.71         1.00      0.181        10.5         0.31  {}
B_blocks    5   32   0.00   0.00        3.43         0.91      0.168        13.3         0.43  {}
B_blocks    6   24   0.00   0.00        3.18         0.67      0.156        13.6         0.39  {}
B_blocks    7   38   0.05   0.16        3.15         0.71      0.165        18.2         0.42  {'out_of_terrain_bounds': 6, 'illegal_contact': 1, 'base_too_low': 1}
C_step      0   11   0.00   0.00        3.76         1.00      0.181         3.9         0.05  {}
C_step      1   28   0.00   0.00        3.68         1.00      0.177         2.2         0.06  {}
C_step      2   25   0.00   0.00        3.34         0.76      0.161         4.0         0.06  {}
C_step      3   24   0.00   0.00        3.38         0.92      0.161         6.9         0.04  {}
C_step      4   15   0.00   0.00        2.65         0.00      0.126         4.3         0.04  {}
C_step      5   19   0.32   0.00        1.60         0.00      0.096         2.7         0.04  {'base_too_low': 6}
C_step      6   11   0.18   0.00        1.21         0.00      0.058         2.0         0.04  {'base_too_low': 2}
C_step      7   17   0.41   0.00        0.76         0.00      0.038         2.3         0.04  {'base_too_low': 7}
```
**C step posture:**
```
checkpoint /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-05_10-51-51_roll1_student/model_299.pt
C lane, straight 0.2 m/s, 400 steps; on-step = root x inside [-1.85, 2.00] m
lvl | step mm | n samples | roll deg mean (+ left up) | stance r left - right mm | r_left mm | r_right mm | resets
  0 |     0.0 |      2031 |     +2.8 |     +6.9 |   +92.5 |   +85.6 | 0
  3 |    18.4 |      2015 |     -3.5 |    -18.8 |   +78.8 |   +97.6 | 0
  6 |    36.8 |       359 |     +0.8 |    -29.1 |   +73.5 |  +102.6 | 1
target on a level body: r_left - r_right ~= -step height (legs on the raised left side shorter), roll ~ 0
```
**Turning + virtual legs:**
```
roll1_student: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-05_10-51-51_roll1_student/model_299.pt
roll_uno robot, 20 A_flat envs, turn tour; envs that reset: 0
segment  | cmd vx  cmd wz | yaw rate rad/s (ratio) | vx m/s | spine deg mean +- std
straight | 0.20  +0.00  | -0.025               | 0.194  | +0.02 +- 0.69
left     | 0.15  +0.25  | +0.041 (0.16)        | 0.164  | +0.51 +- 0.69
straight | 0.20  +0.00  | +0.001               | 0.204  | -0.05 +- 0.70
right    | 0.15  -0.25  | -0.094 (0.37)        | 0.154  | -0.95 +- 0.70
straight | 0.20  +0.00  | -0.013               | 0.204  | -0.09 +- 0.67

virtual leg r (mm), straight trotting; stand pose r = 74.70 mm; root height 91.0 mm
leg | stance mean  min   max | swing min (retracted) | theta range deg
 1  |   79.4   74.1  85.3 |   59.3                | -22.5 ..  -2.8
 2  |   79.5   73.3  86.5 |   58.7                | -22.2 ..  -5.4
 3  |  102.0   95.1 105.0 |   76.9                |  -7.6 .. +13.3
 4  |  103.1   98.6 106.1 |   74.8                |  -6.3 .. +14.6
```
**Tight turns, S tours (spine free / locked; plot `/home/rml3/Documents/ben/spine/runs_local/pipeline/roll1/roll1_student/spine_turn_roll1_student.png`):**
```
roll1_student: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-05_10-51-51_roll1_student/model_299.pt
roll_uno robot, 12 envs per group (rigid: 24), S tour = 2 s straight + 4 arcs of 6 s (+ - + -), stats over the last 4 s of every arc, sign-corrected (+ = toward the commanded turn)
set      cmd vx  wz   R_cmd | spine  | envs ok | yaw rate (ratio) | speed | radius m | spine deg mean +- std | osc deg @ Hz | no-slip: bend for R deg, R for bend m
in_train 0.10  0.25  0.40  | free   | 12/12   | +0.058 (0.23)    | 0.095 |   1.66   | +0.86 +- 0.70       | 0.92 @ 4.8   |  5.76     nan
in_train 0.10  0.25  0.40  | locked | 12/12   | +0.014 (0.05)    | 0.090 |   6.61   | +0.04 +- 0.71       | 1.00 @ 4.8   |  1.44     nan
1L_fast  0.10  0.50  0.20  | free   | 12/12   | +0.083 (0.17)    | 0.086 |   1.04   | +1.54 +- 0.81       | 1.04 @ 4.8   |  9.19     nan
1L_fast  0.10  0.50  0.20  | locked | 12/12   | +0.032 (0.06)    | 0.077 |   2.41   | +0.08 +- 0.67       | 0.95 @ 4.8   |  3.95     nan
1L_slow  0.05  0.25  0.20  | free   | 12/12   | +0.043 (0.17)    | 0.043 |   0.99   | +1.06 +- 0.75       | 1.01 @ 4.8   |  9.62     nan
1L_slow  0.05  0.25  0.20  | locked | 12/12   | +0.013 (0.05)    | 0.043 |   3.22   | +0.06 +- 0.74       | 1.05 @ 4.8   |  2.97     nan
spin     0.00  0.50  0.00  | free   | 12/12   | +0.045 (0.09)    | 0.041 |   0.90   | +2.25 +- 0.84       | 1.12 @ 4.8   | 10.58     nan
spin     0.00  0.50  0.00  | locked | 12/12   | +0.038 (0.08)    | 0.036 |   0.95   | +0.09 +- 0.72       | 1.02 @ 4.8   | 10.02     nan
```
**Terrain mastery (straight 0.20 m/s for 40 s, 30 envs per level; success = got through without a fall, any speed (C: up on the raised half and no fall on the course, B: 2 m without a fall); mastered = 100 %):**
```
roll1_student_C_fwd: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-05_10-51-51_roll1_student/model_299.pt
C_step, straight 0.2 m/s for 40 s, 30 envs per level; success = climbed: body 0.5 m past the entry riser, then no fall until it leaves the course, any speed
level | step mm | success | course 4.4 m | falls | progress m (mean, min) | course speed m/s | old 20 s rule | not counted | fall causes
    0 |     0.0 |    100% |         100% |    0% | 4.70, 4.70 |            0.187 |           97% |           0 | -
    1 |     6.1 |    100% |         100% |    0% | 4.70, 4.70 |            0.177 |           83% |           0 | -
    2 |    12.3 |    100% |         100% |    0% | 4.70, 4.70 |            0.169 |           67% |           0 | -
    3 |    18.4 |    100% |         100% |    0% | 4.70, 4.70 |            0.165 |           73% |           0 | -
    4 |    24.5 |     93% |          93% |    7% | 4.41, 0.29 |            0.137 |            0% |           0 | base_too_low 2
    5 |    30.7 |     77% |          73% |   23% | 3.65, 0.28 |            0.122 |            0% |           0 | base_too_low 7
    6 |    36.8 |     57% |           0% |   43% | 1.92, 0.28 |                - |            0% |           0 | base_too_low 13
first level below 100 %: 4
roll1_student_C_rev: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-05_10-51-51_roll1_student/model_299.pt
C_step REVERSED (raised side on the right), straight 0.2 m/s for 40 s, 30 envs per level; success = climbed: body 0.5 m past the entry riser, then no fall until it leaves the course, any speed
level | step mm | success | course 4.4 m | falls | progress m (mean, min) | course speed m/s | old 20 s rule | not counted | fall causes
    0 |     0.0 |    100% |         100% |    0% | 4.70, 4.70 |            0.188 |          100% |           0 | -
    1 |     6.1 |    100% |         100% |    0% | 4.69, 4.16 |            0.183 |           83% |           1 | -
    2 |    12.3 |    100% |         100% |    0% | 4.70, 4.63 |            0.154 |           37% |           0 | -
    3 |    18.4 |    100% |          80% |    0% | 4.60, 3.92 |            0.137 |            7% |           0 | -
    4 |    24.5 |    100% |          86% |    0% | 4.52, 2.04 |            0.142 |           10% |           1 | -
    5 |    30.7 |    100% |          80% |    0% | 4.58, 4.00 |            0.127 |            0% |           0 | -
    6 |    36.8 |     87% |           3% |   13% | 3.51, 0.23 |            0.115 |            0% |           0 | base_too_low 3, illegal_contact 1
first level below 100 %: 6
roll1_student_B: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-05_10-51-51_roll1_student/model_299.pt
B_blocks, straight 0.2 m/s for 40 s, 30 envs per level; success = 2.0 m before any fall, any speed
level | step mm | success | falls | progress m (mean, min) | old 20 s rule | not counted | fall causes
    0 |       - |    100% |    0% | 2.00, 2.00 |          100% |           0 | -
    1 |       - |    100% |    0% | 2.00, 2.00 |          100% |           0 | -
    2 |       - |    100% |    0% | 2.00, 2.00 |          100% |           0 | -
    3 |       - |    100% |    0% | 2.00, 2.00 |          100% |           0 | -
    4 |       - |    100% |    0% | 2.00, 2.00 |          100% |           0 | -
    5 |       - |    100% |    0% | 2.00, 2.00 |           80% |           0 | -
    6 |       - |    100% |    0% | 2.00, 2.00 |           33% |           0 | -
first level below 100 %: none
```
**Videos:** `/home/rml3/Documents/ben/spine/runs_local/pipeline/roll1/roll1_student/videos/roll1_student_A_flat.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/roll1/roll1_student/videos/roll1_student_B_lvl6.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/roll1/roll1_student/videos/roll1_student_C_lvl6.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/roll1/roll1_student/videos/roll1_student_flat_side.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/roll1/roll1_student/videos/roll1_student_turn_tour.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/roll1/roll1_student/videos/roll1_student_turn_tour_top.mp4`

