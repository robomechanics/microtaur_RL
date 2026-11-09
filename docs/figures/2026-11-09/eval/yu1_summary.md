## yu1_teacher
checkpoint `/home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-06_11-22-28_yu1/model_5999.pt`

**Gait (flat):** flight 0.00 four 0.00 diag +1.00 same_end -1.00 same_side -1.00 stride 4.8 Hz
**Speed at command 0.10 / 0.20 / 0.35 m/s:** 0.023 / 0.133 / 0.312

**Terrain (straight 0.20 m/s, 20 s):**
```
checkpoint /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-06_11-22-28_yu1/model_5999.pt
Teacher-Cur terrain; terrain scale 2.0; straight 0.2 m/s for 20.0 s; promotion distance (half a tile) 3.15 m

terrain   lvl    n   fall  trunc  progress m >= half tile  speed m/s  |dyaw| deg  |lateral| m  causes
A_flat      0   11   0.00   0.00        3.04         0.00      0.146         7.9         0.23  {}
A_flat      1    8   0.00   0.00        2.97         0.00      0.142         7.3         0.17  {}
A_flat      2    9   0.00   0.00        3.01         0.00      0.144         8.3         0.21  {}
A_flat      3   10   0.00   0.00        2.99         0.00      0.143         6.7         0.06  {}
A_flat      4   19   0.00   0.00        3.01         0.00      0.145        15.2         0.33  {}
A_flat      5    9   0.00   0.00        3.01         0.00      0.144         6.6         0.32  {}
A_flat      6   15   0.00   0.00        3.01         0.00      0.146         9.2         0.30  {}
A_flat      7   19   0.00   0.00        2.95         0.00      0.142        12.2         0.30  {}
B_blocks    0   27   0.00   0.00        2.96         0.00      0.142         9.4         0.28  {}
B_blocks    1   44   0.00   0.00        2.92         0.00      0.141         7.2         0.22  {}
B_blocks    2   32   0.00   0.00        2.99         0.16      0.145         7.5         0.22  {}
B_blocks    3   27   0.00   0.00        3.12         0.41      0.151         7.0         0.22  {}
B_blocks    4   29   0.00   0.00        3.22         0.72      0.158         9.4         0.26  {}
B_blocks    5   26   0.00   0.00        3.11         0.35      0.152        10.0         0.22  {}
B_blocks    6   27   0.00   0.04        2.89         0.11      0.146        12.6         0.35  {'physics_unstable': 1}
B_blocks    7   38   0.00   0.00        3.00         0.18      0.147        10.8         0.32  {}
C_step      0   12   0.00   0.00        2.28         0.00      0.109         4.3         0.00  {}
C_step      1   22   0.00   0.00        2.43         0.00      0.117         1.7         0.00  {}
C_step      2   24   0.00   0.00        2.90         0.00      0.140         1.5         0.00  {}
C_step      3   18   0.00   0.00        3.30         1.00      0.159         0.7         0.00  {}
C_step      4   19   0.00   0.00        3.33         1.00      0.161         1.4         0.01  {}
C_step      5   20   0.00   0.00        3.11         0.00      0.149         2.5         0.00  {}
C_step      6   16   0.00   0.00        2.12         0.00      0.101         0.3         0.01  {}
C_step      7   19   0.00   0.00        2.10         0.00      0.099         1.8         0.01  {}
```
**C step posture:**
```
checkpoint /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-06_11-22-28_yu1/model_5999.pt
C lane, straight 0.2 m/s, 400 steps; on-step = root x inside [-1.85, 2.00] m
lvl | step mm | n samples | roll deg mean (+ left up) | stance r left - right mm | r_left mm | r_right mm | resets
  0 |     0.0 |      1844 |     +0.0 |     -0.0 |   +98.9 |   +98.9 | 0
  3 |    18.4 |      1866 |     -0.2 |    -18.5 |   +81.1 |   +99.7 | 0
  6 |    36.8 |       919 |     +1.4 |    -32.9 |   +68.1 |  +101.0 | 0
target on a level body: r_left - r_right ~= -step height (legs on the raised left side shorter), roll ~ 0
```
**Turning + virtual legs:**
```
yu1_teacher: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-06_11-22-28_yu1/model_5999.pt
yaw_uno robot, 20 A_flat envs, turn tour; envs that reset: 0
segment  | cmd vx  cmd wz | yaw rate rad/s (ratio) | vx m/s | spine deg mean +- std
straight | 0.20  +0.00  | +0.006               | 0.145  | +0.01 +- 0.81
left     | 0.15  +0.25  | +0.044 (0.17)        | 0.062  | +3.62 +- 0.85
straight | 0.20  +0.00  | -0.022               | 0.126  | +0.34 +- 1.21
right    | 0.15  -0.25  | -0.038 (0.15)        | 0.059  | -3.62 +- 0.85
straight | 0.20  +0.00  | -0.002               | 0.127  | -0.20 +- 1.09

stride per side (last 4 s of each segment): stance sweep speed mm/s (planted foot moving back) | skid-free yaw rate from it, (v_R - v_L) / 0.1 m
straight | cmd wz +0.00 | left  112.1  right  110.9  (R - L   -1.2) | skid-free -0.012 rad/s vs measured +0.006
left     | cmd wz +0.25 | left   36.6  right   40.4  (R - L   +3.8) | skid-free +0.038 rad/s vs measured +0.044
straight | cmd wz +0.00 | left  102.5  right  102.8  (R - L   +0.3) | skid-free +0.003 rad/s vs measured -0.022
right    | cmd wz -0.25 | left   37.2  right   33.9  (R - L   -3.4) | skid-free -0.034 rad/s vs measured -0.038
straight | cmd wz +0.00 | left  104.3  right  104.1  (R - L   -0.2) | skid-free -0.002 rad/s vs measured -0.002

virtual leg r (mm), straight trotting; stand pose r = 74.70 mm; root height 96.2 mm
leg | stance mean  min   max | swing min (retracted) | theta range deg
 1  |   99.0   90.0 103.1 |   62.3                | -11.6 ..  +6.2
 2  |   99.0   91.6 102.7 |   65.7                | -13.1 ..  +6.1
 3  |   95.1   87.9 101.2 |   60.5                | -14.6 ..  +4.8
 4  |   95.1   88.1 101.9 |   60.9                | -13.5 ..  +5.2
```
**Tight turns, S tours (spine free / locked; plot `/home/rml3/Documents/ben/spine/runs_local/pipeline/yu1/yu1_teacher/spine_turn_yu1_teacher.png`):**
```
yu1_teacher: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-06_11-22-28_yu1/model_5999.pt
yaw_uno robot, 12 envs per group (rigid: 24), S tour = 2 s straight + 4 arcs of 6 s (+ - + -), stats over the last 4 s of every arc, sign-corrected (+ = toward the commanded turn)
set      cmd vx  wz   R_cmd | spine  | envs ok | yaw rate (ratio) | speed | radius m | spine deg mean +- std | osc deg @ Hz | no-slip: bend for R deg, R for bend m
in_train 0.10  0.25  0.40  | free   | 12/12   | +0.037 (0.15)    | 0.042 |   1.13   | +3.80 +- 0.86       | 1.22 @ 4.8   |  8.43    2.51
in_train 0.10  0.25  0.40  | locked | 12/12   | +0.030 (0.12)    | 0.028 |   0.92   | -0.00 +- 0.02       | 0.01 @ 4.8   | 10.33     inf
1L_fast  0.10  0.50  0.20  | free   | 12/12   | +0.104 (0.21)    | 0.071 |   0.69   | +5.94 +- 1.44       | 2.03 @ 4.8   | 13.83    1.60
1L_fast  0.10  0.50  0.20  | locked | 12/12   | +0.075 (0.15)    | 0.031 |   0.41   | +0.00 +- 0.02       | 0.02 @ 4.8   | 22.82     inf
1L_slow  0.05  0.25  0.20  | free   | 12/12   | +0.036 (0.14)    | 0.031 |   0.85   | +3.93 +- 0.87       | 1.23 @ 4.8   | 11.16    2.43
1L_slow  0.05  0.25  0.20  | locked | 12/12   | +0.027 (0.11)    | 0.015 |   0.54   | -0.01 +- 0.02       | 0.02 @ 4.8   | 17.40     inf
spin     0.00  0.50  0.00  | free   | 12/12   | +0.055 (0.11)    | 0.035 |   0.63   | +6.31 +- 1.48       | 2.10 @ 4.8   | 15.03    1.51
spin     0.00  0.50  0.00  | locked | 12/12   | +0.014 (0.03)    | 0.017 |   1.23   | +0.05 +- 0.02       | 0.02 @ 4.8   |  7.78     inf
```
**Terrain mastery (straight 0.20 m/s for 40 s, 30 envs per level; success = got through without a fall, any speed (C: up on the raised half and no fall on the course, B: 2 m without a fall); mastered = 100 %):**
```
yu1_teacher_C_fwd: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-06_11-22-28_yu1/model_5999.pt
C_step, straight 0.2 m/s for 40 s, 30 envs per level; success = climbed: body 0.5 m past the entry riser, then no fall until it leaves the course, any speed
level | step mm | success | course 4.4 m | falls | progress m (mean, min) | course speed m/s | old 20 s rule | not counted | fall causes
    0 |     0.0 |    100% |          77% |    0% | 4.55, 4.30 |            0.115 |            0% |           0 | -
    1 |     6.1 |    100% |         100% |    0% | 4.70, 4.63 |            0.121 |            0% |           0 | -
    2 |    12.3 |    100% |         100% |    0% | 4.70, 4.70 |            0.148 |            3% |           0 | -
    3 |    18.4 |    100% |         100% |    0% | 4.70, 4.70 |            0.163 |           37% |           0 | -
    4 |    24.5 |    100% |         100% |    0% | 4.70, 4.70 |            0.168 |           80% |           0 | -
    5 |    30.7 |    100% |         100% |    0% | 4.70, 4.70 |            0.157 |            0% |           0 | -
    6 |    36.8 |    100% |          83% |    0% | 4.54, 2.65 |            0.123 |            0% |           0 | -
first level below 100 %: none
yu1_teacher_C_rev: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-06_11-22-28_yu1/model_5999.pt
C_step REVERSED (raised side on the right), straight 0.2 m/s for 40 s, 30 envs per level; success = climbed: body 0.5 m past the entry riser, then no fall until it leaves the course, any speed
level | step mm | success | course 4.4 m | falls | progress m (mean, min) | course speed m/s | old 20 s rule | not counted | fall causes
    0 |     0.0 |    100% |          97% |    0% | 4.59, 4.40 |            0.115 |            0% |           0 | -
    1 |     6.1 |    100% |         100% |    0% | 4.70, 4.70 |            0.124 |            0% |           0 | -
    2 |    12.3 |    100% |         100% |    0% | 4.70, 4.70 |            0.154 |            7% |           0 | -
    3 |    18.4 |    100% |         100% |    0% | 4.70, 4.70 |            0.170 |           90% |           0 | -
    4 |    24.5 |    100% |         100% |    0% | 4.70, 4.70 |            0.169 |           83% |           0 | -
    5 |    30.7 |    100% |         100% |    0% | 4.70, 4.70 |            0.158 |            0% |           0 | -
    6 |    36.8 |     97% |          53% |    3% | 4.06, 0.32 |            0.118 |            0% |           0 | base_too_low 1
first level below 100 %: 6
yu1_teacher_B: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-06_11-22-28_yu1/model_5999.pt
B_blocks, straight 0.2 m/s for 40 s, 30 envs per level; success = 2.0 m before any fall, any speed
level | step mm | success | falls | progress m (mean, min) | old 20 s rule | not counted | fall causes
    0 |       - |    100% |    0% | 2.00, 2.00 |            0% |           0 | -
    1 |       - |    100% |    0% | 2.00, 2.00 |            0% |           0 | -
    2 |       - |    100% |    0% | 2.00, 2.00 |            7% |           0 | -
    3 |       - |    100% |    0% | 2.00, 2.00 |           17% |           0 | -
    4 |       - |    100% |    0% | 2.00, 2.00 |           43% |           0 | -
    5 |       - |     97% |    3% | 1.94, 0.22 |           40% |           0 | illegal_contact 1
    6 |       - |    100% |    0% | 2.00, 2.00 |           10% |           0 | -
first level below 100 %: 5
```
**Videos:** `/home/rml3/Documents/ben/spine/runs_local/pipeline/yu1/yu1_teacher/videos/yu1_teacher_A_flat.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/yu1/yu1_teacher/videos/yu1_teacher_B_lvl6.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/yu1/yu1_teacher/videos/yu1_teacher_C_lvl6.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/yu1/yu1_teacher/videos/yu1_teacher_flat_side.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/yu1/yu1_teacher/videos/yu1_teacher_turn_tour.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/yu1/yu1_teacher/videos/yu1_teacher_turn_tour_top.mp4`

## yu1_student
checkpoint `/home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-07_00-38-07_yu1_student/model_299.pt`

**Gait (flat):** flight 0.00 four 0.00 diag +0.98 same_end -0.97 same_side -0.97 stride 4.7 Hz
**Speed at command 0.10 / 0.20 / 0.35 m/s:** 0.030 / 0.135 / 0.292

**Terrain (straight 0.20 m/s, 20 s):**
```
checkpoint /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-07_00-38-07_yu1_student/model_299.pt
Teacher-Cur terrain; terrain scale 2.0; straight 0.2 m/s for 20.0 s; promotion distance (half a tile) 3.15 m

terrain   lvl    n   fall  trunc  progress m >= half tile  speed m/s  |dyaw| deg  |lateral| m  causes
A_flat      0   11   0.00   0.00        2.89         0.00      0.139         9.1         0.23  {}
A_flat      1   14   0.00   0.00        2.91         0.00      0.140         6.7         0.14  {}
A_flat      2   17   0.00   0.00        2.88         0.00      0.139        13.6         0.32  {}
A_flat      3   10   0.00   0.00        2.93         0.00      0.141         7.0         0.19  {}
A_flat      4   13   0.00   0.00        2.88         0.00      0.139        11.1         0.28  {}
A_flat      5   13   0.00   0.00        2.89         0.00      0.139        11.1         0.26  {}
A_flat      6   12   0.00   0.00        2.89         0.00      0.139        12.5         0.28  {}
A_flat      7   10   0.00   0.00        2.88         0.00      0.139        10.2         0.25  {}
B_blocks    0   32   0.00   0.00        2.86         0.00      0.138        11.6         0.28  {}
B_blocks    1   17   0.00   0.00        2.81         0.00      0.135        10.7         0.23  {}
B_blocks    2   29   0.00   0.00        2.78         0.00      0.134         9.8         0.23  {}
B_blocks    3   41   0.00   0.00        2.77         0.00      0.134         8.2         0.21  {}
B_blocks    4   29   0.00   0.00        2.80         0.00      0.136         8.3         0.19  {}
B_blocks    5   36   0.00   0.00        2.71         0.00      0.132         8.3         0.20  {}
B_blocks    6   30   0.07   0.00        2.53         0.00      0.127        10.0         0.17  {'illegal_contact': 1, 'base_too_low': 1}
B_blocks    7   36   0.06   0.03        2.38         0.00      0.122        12.3         0.32  {'physics_unstable': 1, 'base_too_low': 2}
C_step      0   19   0.00   0.00        2.33         0.00      0.110         7.8         0.05  {}
C_step      1   16   0.00   0.00        2.28         0.00      0.108        10.1         0.04  {}
C_step      2   21   0.05   0.00        2.09         0.00      0.100        11.8         0.05  {'fell_over': 1}
C_step      3   13   0.08   0.00        1.66         0.00      0.077        17.6         0.04  {'base_too_low': 1}
C_step      4   22   0.05   0.00        1.65         0.00      0.078        19.0         0.04  {'illegal_contact': 1}
C_step      5   26   0.04   0.00        1.39         0.00      0.065        12.2         0.03  {'base_too_low': 1}
C_step      6   16   0.62   0.00        0.25         0.00      0.012        21.6         0.01  {'illegal_contact': 10}
C_step      7   17   0.59   0.00        0.23         0.00      0.014        21.6         0.01  {'illegal_contact': 10}
```
**C step posture:**
```
checkpoint /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-07_00-38-07_yu1_student/model_299.pt
C lane, straight 0.2 m/s, 400 steps; on-step = root x inside [-1.85, 2.00] m
lvl | step mm | n samples | roll deg mean (+ left up) | stance r left - right mm | r_left mm | r_right mm | resets
  0 |     0.0 |      1668 |     -1.0 |     -2.0 |   +91.8 |   +93.8 | 1
  3 |    18.4 |      1834 |     +2.3 |    -10.2 |   +86.2 |   +96.4 | 0
  6 |    36.8 |        21 |     +8.2 |    -21.7 |   +77.3 |   +99.0 | 4
target on a level body: r_left - r_right ~= -step height (legs on the raised left side shorter), roll ~ 0
```
**Turning + virtual legs:**
```
yu1_student: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-07_00-38-07_yu1_student/model_299.pt
yaw_uno robot, 20 A_flat envs, turn tour; envs that reset: 0
segment  | cmd vx  cmd wz | yaw rate rad/s (ratio) | vx m/s | spine deg mean +- std
straight | 0.20  +0.00  | -0.000               | 0.129  | -0.07 +- 0.77
left     | 0.15  +0.25  | +0.043 (0.17)        | 0.085  | +3.25 +- 0.82
straight | 0.20  +0.00  | -0.006               | 0.131  | +0.25 +- 1.16
right    | 0.15  -0.25  | -0.046 (0.19)        | 0.080  | -3.28 +- 0.82
straight | 0.20  +0.00  | +0.013               | 0.135  | -0.25 +- 1.01

stride per side (last 4 s of each segment): stance sweep speed mm/s (planted foot moving back) | skid-free yaw rate from it, (v_R - v_L) / 0.1 m
straight | cmd wz +0.00 | left   95.5  right   96.9  (R - L   +1.5) | skid-free +0.015 rad/s vs measured -0.000
left     | cmd wz +0.25 | left   58.6  right   62.7  (R - L   +4.1) | skid-free +0.041 rad/s vs measured +0.043
straight | cmd wz +0.00 | left  107.7  right  108.0  (R - L   +0.3) | skid-free +0.003 rad/s vs measured -0.006
right    | cmd wz -0.25 | left   58.7  right   54.5  (R - L   -4.2) | skid-free -0.042 rad/s vs measured -0.046
straight | cmd wz +0.00 | left  111.2  right  110.8  (R - L   -0.3) | skid-free -0.003 rad/s vs measured +0.013

virtual leg r (mm), straight trotting; stand pose r = 74.70 mm; root height 96.0 mm
leg | stance mean  min   max | swing min (retracted) | theta range deg
 1  |   99.0   89.4 102.5 |   64.3                | -11.5 ..  +7.6
 2  |   98.6   87.7 102.2 |   63.0                | -11.5 ..  +6.7
 3  |   94.9   84.9 101.1 |   57.9                | -14.1 ..  +6.7
 4  |   95.0   85.7 100.7 |   56.5                | -14.5 ..  +4.6
```
**Tight turns, S tours (spine free / locked; plot `/home/rml3/Documents/ben/spine/runs_local/pipeline/yu1/yu1_student/spine_turn_yu1_student.png`):**
```
yu1_student: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-07_00-38-07_yu1_student/model_299.pt
yaw_uno robot, 12 envs per group (rigid: 24), S tour = 2 s straight + 4 arcs of 6 s (+ - + -), stats over the last 4 s of every arc, sign-corrected (+ = toward the commanded turn)
set      cmd vx  wz   R_cmd | spine  | envs ok | yaw rate (ratio) | speed | radius m | spine deg mean +- std | osc deg @ Hz | no-slip: bend for R deg, R for bend m
in_train 0.10  0.25  0.40  | free   | 12/12   | +0.034 (0.13)    | 0.048 |   1.42   | +3.49 +- 0.85       | 1.20 @ 4.8   |  6.70    2.74
in_train 0.10  0.25  0.40  | locked | 12/12   | +0.021 (0.08)    | 0.036 |   1.73   | +0.02 +- 0.02       | 0.02 @ 4.8   |  5.51     inf
1L_fast  0.10  0.50  0.20  | free   | 12/12   | +0.078 (0.16)    | 0.075 |   0.97   | +5.91 +- 1.26       | 1.77 @ 4.8   |  9.85    1.61
1L_fast  0.10  0.50  0.20  | locked | 12/12   | +0.051 (0.10)    | 0.050 |   1.00   | +0.03 +- 0.03       | 0.02 @ 4.8   |  9.57     inf
1L_slow  0.05  0.25  0.20  | free   | 12/12   | +0.030 (0.12)    | 0.034 |   1.13   | +3.65 +- 0.85       | 1.20 @ 4.8   |  8.47    2.62
1L_slow  0.05  0.25  0.20  | locked | 12/12   | +0.017 (0.07)    | 0.019 |   1.10   | +0.02 +- 0.02       | 0.02 @ 4.8   |  8.69     inf
spin     0.00  0.50  0.00  | free   | 12/12   | +0.082 (0.16)    | 0.043 |   0.52   | +6.22 +- 1.24       | 1.76 @ 4.8   | 18.30    1.53
spin     0.00  0.50  0.00  | locked | 12/12   | +0.029 (0.06)    | 0.016 |   0.57   | +0.03 +- 0.02       | 0.02 @ 4.8   | 16.56     inf
```
**Terrain mastery (straight 0.20 m/s for 40 s, 30 envs per level; success = got through without a fall, any speed (C: up on the raised half and no fall on the course, B: 2 m without a fall); mastered = 100 %):**
```
yu1_student_C_fwd: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-07_00-38-07_yu1_student/model_299.pt
C_step, straight 0.2 m/s for 40 s, 30 envs per level; success = climbed: body 0.5 m past the entry riser, then no fall until it leaves the course, any speed
level | step mm | success | course 4.4 m | falls | progress m (mean, min) | course speed m/s | old 20 s rule | not counted | fall causes
    0 |     0.0 |     83% |          20% |   17% | 3.94, 1.62 |            0.122 |            0% |           0 | fell_over 3, base_too_low 2
    1 |     6.1 |     70% |          17% |   30% | 3.60, 1.08 |            0.130 |            0% |           0 | illegal_contact 1, base_too_low 4, fell_over 4
    2 |    12.3 |     64% |           7% |   36% | 3.37, 1.52 |            0.114 |            0% |           2 | fell_over 2, illegal_contact 4, base_too_low 4
    3 |    18.4 |     93% |           0% |    7% | 3.14, 1.66 |                - |            0% |           0 | base_too_low 2
    4 |    24.5 |     97% |           0% |    3% | 3.05, 2.09 |                - |            0% |           0 | base_too_low 1
    5 |    30.7 |     93% |           0% |    7% | 2.56, 0.35 |                - |            0% |           0 | base_too_low 2
    6 |    36.8 |     17% |           0% |   73% | 0.51, 0.13 |                - |            0% |           0 | illegal_contact 20, base_too_low 2
first level below 100 %: 0
yu1_student_C_rev: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-07_00-38-07_yu1_student/model_299.pt
C_step REVERSED (raised side on the right), straight 0.2 m/s for 40 s, 30 envs per level; success = climbed: body 0.5 m past the entry riser, then no fall until it leaves the course, any speed
level | step mm | success | course 4.4 m | falls | progress m (mean, min) | course speed m/s | old 20 s rule | not counted | fall causes
    0 |     0.0 |     93% |          26% |    7% | 3.87, 1.29 |            0.119 |            0% |           3 | base_too_low 1, fell_over 1
    1 |     6.1 |     87% |          60% |   13% | 4.22, 2.29 |            0.131 |            0% |           0 | base_too_low 2, illegal_contact 2
    2 |    12.3 |     83% |          31% |   17% | 3.97, 1.65 |            0.121 |            0% |           1 | base_too_low 3, fell_over 2
    3 |    18.4 |     85% |           4% |   15% | 3.07, 1.25 |            0.112 |            0% |           3 | illegal_contact 1, base_too_low 3
    4 |    24.5 |     97% |           0% |    3% | 2.83, 1.77 |                - |            0% |           1 | base_too_low 1
    5 |    30.7 |     67% |           0% |   33% | 1.85, 0.13 |                - |            0% |           0 | illegal_contact 3, base_too_low 7
    6 |    36.8 |      0% |           0% |   63% | 0.17, 0.14 |                - |            0% |           0 | illegal_contact 19
first level below 100 %: 0
yu1_student_B: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-07_00-38-07_yu1_student/model_299.pt
B_blocks, straight 0.2 m/s for 40 s, 30 envs per level; success = 2.0 m before any fall, any speed
level | step mm | success | falls | progress m (mean, min) | old 20 s rule | not counted | fall causes
    0 |       - |    100% |    0% | 2.00, 2.00 |            0% |           0 | -
    1 |       - |    100% |    0% | 2.00, 2.00 |            0% |           0 | -
    2 |       - |    100% |    0% | 2.00, 2.00 |            0% |           0 | -
    3 |       - |    100% |    0% | 2.00, 2.00 |            0% |           0 | -
    4 |       - |    100% |    0% | 2.00, 2.00 |            0% |           0 | -
    5 |       - |    100% |    0% | 2.00, 2.00 |            0% |           0 | -
    6 |       - |    100% |    0% | 2.00, 2.00 |            0% |           0 | -
first level below 100 %: none
```
**Videos:** `/home/rml3/Documents/ben/spine/runs_local/pipeline/yu1/yu1_student/videos/yu1_student_A_flat.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/yu1/yu1_student/videos/yu1_student_B_lvl6.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/yu1/yu1_student/videos/yu1_student_C_lvl6.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/yu1/yu1_student/videos/yu1_student_flat_side.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/yu1/yu1_student/videos/yu1_student_turn_tour.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/yu1/yu1_student/videos/yu1_student_turn_tour_top.mp4`

