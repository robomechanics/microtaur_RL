## ru1_teacher
checkpoint `/home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-06_11-26-27_ru1/model_5999.pt`

**Gait (flat):** flight 0.00 four 0.00 diag +1.00 same_end -1.00 same_side -1.00 stride 4.8 Hz
**Speed at command 0.10 / 0.20 / 0.35 m/s:** 0.063 / 0.208 / 0.332

**Terrain (straight 0.20 m/s, 20 s):**
```
checkpoint /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-06_11-26-27_ru1/model_5999.pt
Teacher-Cur terrain; terrain scale 2.0; straight 0.2 m/s for 20.0 s; promotion distance (half a tile) 3.15 m

terrain   lvl    n   fall  trunc  progress m >= half tile  speed m/s  |dyaw| deg  |lateral| m  causes
A_flat      0   14   0.00   0.00        4.86         1.00      0.231         6.3         0.40  {}
A_flat      1   11   0.00   0.00        4.86         1.00      0.231         5.5         0.23  {}
A_flat      2   12   0.00   0.00        4.86         1.00      0.231         4.5         0.19  {}
A_flat      3   14   0.00   0.00        4.86         1.00      0.231         2.3         0.16  {}
A_flat      4   10   0.00   0.00        4.87         1.00      0.231         2.8         0.06  {}
A_flat      5   18   0.00   0.00        4.87         1.00      0.231         3.8         0.22  {}
A_flat      6    8   0.00   0.00        4.84         1.00      0.231         7.9         0.40  {}
A_flat      7   13   0.00   0.00        4.86         1.00      0.232        10.8         0.44  {}
B_blocks    0   41   0.00   0.00        4.80         1.00      0.228         3.0         0.18  {}
B_blocks    1   29   0.00   0.00        4.68         1.00      0.222         2.6         0.12  {}
B_blocks    2   39   0.00   0.00        4.53         1.00      0.216         3.4         0.17  {}
B_blocks    3   23   0.00   0.00        4.29         1.00      0.205         6.2         0.22  {}
B_blocks    4   26   0.00   0.00        3.92         1.00      0.188         6.3         0.31  {}
B_blocks    5   35   0.00   0.00        3.69         1.00      0.178         7.1         0.28  {}
B_blocks    6   28   0.00   0.00        3.45         0.93      0.168        11.6         0.40  {}
B_blocks    7   29   0.00   0.24        3.70         0.97      0.188         7.4         0.31  {'out_of_terrain_bounds': 7}
C_step      0   19   0.00   0.00        3.20         0.37      0.151         0.7         0.01  {}
C_step      1   18   0.00   0.00        3.73         1.00      0.176         1.3         0.02  {}
C_step      2   18   0.00   0.00        4.31         1.00      0.204         1.6         0.03  {}
C_step      3    8   0.00   0.00        4.54         1.00      0.211         0.7         0.04  {}
C_step      4   21   0.00   0.00        4.34         1.00      0.202         1.0         0.04  {}
C_step      5   18   0.00   0.00        4.20         1.00      0.196         4.3         0.03  {}
C_step      6   26   0.00   0.00        4.10         1.00      0.192         0.3         0.04  {}
C_step      7   22   0.00   0.00        4.03         1.00      0.189         0.1         0.04  {}
```
**C step posture:**
```
checkpoint /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-06_11-26-27_ru1/model_5999.pt
C lane, straight 0.2 m/s, 400 steps; on-step = root x inside [-1.85, 2.00] m
lvl | step mm | n samples | roll deg mean (+ left up) | stance r left - right mm | r_left mm | r_right mm | resets
  0 |     0.0 |      1986 |     -0.6 |     -0.7 |   +98.8 |   +99.5 | 0
  3 |    18.4 |      1988 |     -0.2 |    -19.2 |   +77.9 |   +97.1 | 0
  6 |    36.8 |      1866 |     +4.5 |    -29.4 |   +68.5 |   +97.9 | 0
target on a level body: r_left - r_right ~= -step height (legs on the raised left side shorter), roll ~ 0
```
**Turning + virtual legs:**
```
ru1_teacher: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-06_11-26-27_ru1/model_5999.pt
rigid_uno robot, 20 A_flat envs, turn tour; envs that reset: 0
segment  | cmd vx  cmd wz | yaw rate rad/s (ratio) | vx m/s | spine deg mean +- std
straight | 0.20  +0.00  | -0.001               | 0.210  | n/a (rigid)
left     | 0.15  +0.25  | +0.001 (0.00)        | 0.173  | n/a (rigid)
straight | 0.20  +0.00  | -0.001               | 0.229  | n/a (rigid)
right    | 0.15  -0.25  | +0.002 (-0.01)        | 0.175  | n/a (rigid)
straight | 0.20  +0.00  | -0.001               | 0.231  | n/a (rigid)

stride per side (last 4 s of each segment): stance sweep speed mm/s (planted foot moving back) | skid-free yaw rate from it, (v_R - v_L) / 0.1 m
straight | cmd wz +0.00 | left  189.2  right  186.4  (R - L   -2.8) | skid-free -0.028 rad/s vs measured -0.001
left     | cmd wz +0.25 | left  151.8  right  152.1  (R - L   +0.3) | skid-free +0.003 rad/s vs measured +0.001
straight | cmd wz +0.00 | left  212.8  right  212.4  (R - L   -0.5) | skid-free -0.005 rad/s vs measured -0.001
right    | cmd wz -0.25 | left  154.7  right  153.8  (R - L   -1.0) | skid-free -0.010 rad/s vs measured +0.002
straight | cmd wz +0.00 | left  215.7  right  214.9  (R - L   -0.9) | skid-free -0.009 rad/s vs measured -0.001

virtual leg r (mm), straight trotting; stand pose r = 73.65 mm; root height 94.2 mm
leg | stance mean  min   max | swing min (retracted) | theta range deg
 1  |   98.9   96.4 102.2 |   84.7                | -11.1 ..  +9.7
 2  |   99.2   97.0 102.4 |   85.1                | -10.9 ..  +9.8
 3  |   97.1   94.4 100.4 |   80.8                | -11.2 ..  +7.5
 4  |   96.7   93.0 100.7 |   80.8                | -11.1 ..  +7.5
```
**Tight turns, S tours (spine free / locked; plot `/home/rml3/Documents/ben/spine/runs_local/pipeline/ru1/ru1_teacher/spine_turn_ru1_teacher.png`):**
```
ru1_teacher: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-06_11-26-27_ru1/model_5999.pt
rigid_uno robot, 12 envs per group (rigid: 24), S tour = 2 s straight + 4 arcs of 6 s (+ - + -), stats over the last 4 s of every arc, sign-corrected (+ = toward the commanded turn)
set      cmd vx  wz   R_cmd | spine  | envs ok | yaw rate (ratio) | speed | radius m | spine deg mean +- std | osc deg @ Hz | no-slip: bend for R deg, R for bend m
in_train 0.10  0.25  0.40  | rigid  | 24/24   | +0.001 (0.00)    | 0.085 |    inf   | n/a                   | n/a          |  0.00
1L_fast  0.10  0.50  0.20  | rigid  | 24/24   | +0.002 (0.00)    | 0.100 |  44.53   | n/a                   | n/a          |  0.21
1L_slow  0.05  0.25  0.20  | rigid  | 24/24   | +0.001 (0.00)    | 0.019 |    inf   | n/a                   | n/a          |  0.00
spin     0.00  0.50  0.00  | rigid  | 24/24   | +0.003 (0.01)    | 0.020 |   6.00   | n/a                   | n/a          |  1.59
```
**Terrain mastery (straight 0.20 m/s for 40 s, 30 envs per level; success = got through without a fall, any speed (C: up on the raised half and no fall on the course, B: 2 m without a fall); mastered = 100 %):**
```
ru1_teacher_C_fwd: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-06_11-26-27_ru1/model_5999.pt
C_step, straight 0.2 m/s for 40 s, 30 envs per level; success = climbed: body 0.5 m past the entry riser, then no fall until it leaves the course, any speed
level | step mm | success | course 4.4 m | falls | progress m (mean, min) | course speed m/s | old 20 s rule | not counted | fall causes
    0 |     0.0 |    100% |         100% |    0% | 4.70, 4.70 |            0.169 |           47% |           0 | -
    1 |     6.1 |    100% |         100% |    0% | 4.70, 4.70 |            0.181 |           87% |           0 | -
    2 |    12.3 |    100% |         100% |    0% | 4.70, 4.70 |            0.218 |          100% |           0 | -
    3 |    18.4 |    100% |         100% |    0% | 4.70, 4.70 |            0.227 |          100% |           0 | -
    4 |    24.5 |    100% |         100% |    0% | 4.70, 4.70 |            0.216 |          100% |           0 | -
    5 |    30.7 |    100% |         100% |    0% | 4.70, 4.70 |            0.209 |          100% |           0 | -
    6 |    36.8 |    100% |         100% |    0% | 4.70, 4.70 |            0.203 |          100% |           0 | -
first level below 100 %: none
ru1_teacher_C_rev: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-06_11-26-27_ru1/model_5999.pt
C_step REVERSED (raised side on the right), straight 0.2 m/s for 40 s, 30 envs per level; success = climbed: body 0.5 m past the entry riser, then no fall until it leaves the course, any speed
level | step mm | success | course 4.4 m | falls | progress m (mean, min) | course speed m/s | old 20 s rule | not counted | fall causes
    0 |     0.0 |    100% |         100% |    0% | 4.70, 4.70 |            0.169 |           43% |           0 | -
    1 |     6.1 |    100% |         100% |    0% | 4.70, 4.70 |            0.168 |           47% |           0 | -
    2 |    12.3 |    100% |         100% |    0% | 4.70, 4.70 |            0.211 |          100% |           0 | -
    3 |    18.4 |    100% |         100% |    0% | 4.70, 4.70 |            0.235 |          100% |           0 | -
    4 |    24.5 |    100% |         100% |    0% | 4.70, 4.70 |            0.219 |          100% |           0 | -
    5 |    30.7 |    100% |         100% |    0% | 4.70, 4.70 |            0.212 |          100% |           0 | -
    6 |    36.8 |    100% |         100% |    0% | 4.70, 4.70 |            0.203 |          100% |           0 | -
first level below 100 %: none
ru1_teacher_B: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-06_11-26-27_ru1/model_5999.pt
B_blocks, straight 0.2 m/s for 40 s, 30 envs per level; success = 2.0 m before any fall, any speed
level | step mm | success | falls | progress m (mean, min) | old 20 s rule | not counted | fall causes
    0 |       - |    100% |    0% | 2.00, 2.00 |          100% |           0 | -
    1 |       - |    100% |    0% | 2.00, 2.00 |          100% |           0 | -
    2 |       - |    100% |    0% | 2.00, 2.00 |          100% |           0 | -
    3 |       - |    100% |    0% | 2.00, 2.00 |          100% |           0 | -
    4 |       - |    100% |    0% | 2.00, 2.00 |          100% |           0 | -
    5 |       - |    100% |    0% | 2.00, 2.00 |          100% |           0 | -
    6 |       - |    100% |    0% | 2.00, 2.00 |           83% |           0 | -
first level below 100 %: none
```
**Videos:** `/home/rml3/Documents/ben/spine/runs_local/pipeline/ru1/ru1_teacher/videos/ru1_teacher_A_flat.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/ru1/ru1_teacher/videos/ru1_teacher_B_lvl6.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/ru1/ru1_teacher/videos/ru1_teacher_C_lvl6.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/ru1/ru1_teacher/videos/ru1_teacher_flat_side.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/ru1/ru1_teacher/videos/ru1_teacher_turn_tour.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/ru1/ru1_teacher/videos/ru1_teacher_turn_tour_top.mp4`

## ru1_student
checkpoint `/home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-07_00-12-55_ru1_student/model_299.pt`

**Gait (flat):** flight 0.00 four 0.00 diag +1.00 same_end -1.00 same_side -1.00 stride 4.8 Hz
**Speed at command 0.10 / 0.20 / 0.35 m/s:** 0.081 / 0.230 / 0.341

**Terrain (straight 0.20 m/s, 20 s):**
```
checkpoint /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-07_00-12-55_ru1_student/model_299.pt
Teacher-Cur terrain; terrain scale 2.0; straight 0.2 m/s for 20.0 s; promotion distance (half a tile) 3.15 m

terrain   lvl    n   fall  trunc  progress m >= half tile  speed m/s  |dyaw| deg  |lateral| m  causes
A_flat      0    8   0.00   0.00        4.84         1.00      0.230         4.0         0.25  {}
A_flat      1   19   0.00   0.00        4.84         1.00      0.230         3.2         0.18  {}
A_flat      2   12   0.00   0.00        4.83         1.00      0.229         3.9         0.21  {}
A_flat      3   14   0.00   0.00        4.84         1.00      0.230         3.7         0.20  {}
A_flat      4   14   0.00   0.00        4.83         1.00      0.229         2.7         0.13  {}
A_flat      5    9   0.00   0.00        4.84         1.00      0.229         3.1         0.11  {}
A_flat      6    9   0.00   0.00        4.82         1.00      0.229         5.5         0.28  {}
A_flat      7   15   0.00   0.00        4.83         1.00      0.229         3.0         0.15  {}
B_blocks    0   36   0.00   0.00        4.79         1.00      0.227         2.9         0.16  {}
B_blocks    1   27   0.00   0.00        4.68         1.00      0.223         3.5         0.17  {}
B_blocks    2   25   0.00   0.00        4.53         1.00      0.216         4.4         0.17  {}
B_blocks    3   37   0.00   0.00        4.28         1.00      0.205         7.1         0.26  {}
B_blocks    4   26   0.00   0.00        3.94         1.00      0.189         6.3         0.23  {}
B_blocks    5   31   0.00   0.00        3.67         1.00      0.178        12.5         0.36  {}
B_blocks    6   34   0.00   0.00        3.42         0.88      0.167        12.5         0.42  {}
B_blocks    7   34   0.00   0.15        3.55         0.94      0.177        13.9         0.57  {'out_of_terrain_bounds': 5}
C_step      0   16   0.00   0.00        4.77         1.00      0.226         1.0         0.03  {}
C_step      1   16   0.00   0.00        4.52         1.00      0.216         0.7         0.04  {}
C_step      2   23   0.00   0.00        4.66         1.00      0.222         1.5         0.03  {}
C_step      3   19   0.00   0.00        4.76         1.00      0.223         1.4         0.03  {}
C_step      4   15   0.00   0.00        4.42         1.00      0.206         1.2         0.04  {}
C_step      5   14   0.00   0.00        4.30         1.00      0.200         5.1         0.03  {}
C_step      6   24   0.04   0.00        3.86         0.96      0.185         0.5         0.04  {'illegal_contact': 1}
C_step      7   23   0.00   0.00        4.03         1.00      0.189         0.3         0.04  {}
```
**C step posture:**
```
checkpoint /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-07_00-12-55_ru1_student/model_299.pt
C lane, straight 0.2 m/s, 400 steps; on-step = root x inside [-1.85, 2.00] m
lvl | step mm | n samples | roll deg mean (+ left up) | stance r left - right mm | r_left mm | r_right mm | resets
  0 |     0.0 |      2098 |     -0.0 |     -0.2 |   +97.8 |   +98.0 | 0
  3 |    18.4 |      2042 |     +2.2 |    -14.0 |   +83.5 |   +97.5 | 0
  6 |    36.8 |      1936 |     +4.7 |    -28.7 |   +68.8 |   +97.5 | 0
target on a level body: r_left - r_right ~= -step height (legs on the raised left side shorter), roll ~ 0
```
**Turning + virtual legs:**
```
ru1_student: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-07_00-12-55_ru1_student/model_299.pt
rigid_uno robot, 20 A_flat envs, turn tour; envs that reset: 0
segment  | cmd vx  cmd wz | yaw rate rad/s (ratio) | vx m/s | spine deg mean +- std
straight | 0.20  +0.00  | -0.000               | 0.210  | n/a (rigid)
left     | 0.15  +0.25  | -0.001 (-0.00)        | 0.168  | n/a (rigid)
straight | 0.20  +0.00  | +0.000               | 0.225  | n/a (rigid)
right    | 0.15  -0.25  | +0.000 (-0.00)        | 0.168  | n/a (rigid)
straight | 0.20  +0.00  | +0.002               | 0.227  | n/a (rigid)

stride per side (last 4 s of each segment): stance sweep speed mm/s (planted foot moving back) | skid-free yaw rate from it, (v_R - v_L) / 0.1 m
straight | cmd wz +0.00 | left  188.4  right  188.5  (R - L   +0.1) | skid-free +0.001 rad/s vs measured -0.000
left     | cmd wz +0.25 | left  142.9  right  142.8  (R - L   -0.1) | skid-free -0.001 rad/s vs measured -0.001
straight | cmd wz +0.00 | left  207.3  right  205.9  (R - L   -1.4) | skid-free -0.014 rad/s vs measured +0.000
right    | cmd wz -0.25 | left  146.4  right  144.8  (R - L   -1.6) | skid-free -0.016 rad/s vs measured +0.000
straight | cmd wz +0.00 | left  210.3  right  208.9  (R - L   -1.4) | skid-free -0.014 rad/s vs measured +0.002

virtual leg r (mm), straight trotting; stand pose r = 73.65 mm; root height 94.4 mm
leg | stance mean  min   max | swing min (retracted) | theta range deg
 1  |   99.0   96.8 102.1 |   84.4                | -11.3 ..  +9.9
 2  |   99.2   97.0 101.8 |   85.2                | -11.3 ..  +9.9
 3  |   97.3   92.9 101.1 |   80.2                | -10.6 ..  +7.3
 4  |   96.7   92.9 100.0 |   80.7                | -11.2 ..  +7.3
```
**Tight turns, S tours (spine free / locked; plot `/home/rml3/Documents/ben/spine/runs_local/pipeline/ru1/ru1_student/spine_turn_ru1_student.png`):**
```
ru1_student: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-07_00-12-55_ru1_student/model_299.pt
rigid_uno robot, 12 envs per group (rigid: 24), S tour = 2 s straight + 4 arcs of 6 s (+ - + -), stats over the last 4 s of every arc, sign-corrected (+ = toward the commanded turn)
set      cmd vx  wz   R_cmd | spine  | envs ok | yaw rate (ratio) | speed | radius m | spine deg mean +- std | osc deg @ Hz | no-slip: bend for R deg, R for bend m
in_train 0.10  0.25  0.40  | rigid  | 24/24   | +0.000 (0.00)    | 0.090 |    inf   | n/a                   | n/a          |  0.00
1L_fast  0.10  0.50  0.20  | rigid  | 24/24   | +0.002 (0.00)    | 0.105 |  69.60   | n/a                   | n/a          |  0.14
1L_slow  0.05  0.25  0.20  | rigid  | 24/24   | -0.001 (-0.00)    | 0.020 |    inf   | n/a                   | n/a          |  0.00
spin     0.00  0.50  0.00  | rigid  | 24/24   | +0.006 (0.01)    | 0.016 |   2.84   | n/a                   | n/a          |  3.36
```
**Terrain mastery (straight 0.20 m/s for 40 s, 30 envs per level; success = got through without a fall, any speed (C: up on the raised half and no fall on the course, B: 2 m without a fall); mastered = 100 %):**
```
ru1_student_C_fwd: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-07_00-12-55_ru1_student/model_299.pt
C_step, straight 0.2 m/s for 40 s, 30 envs per level; success = climbed: body 0.5 m past the entry riser, then no fall until it leaves the course, any speed
level | step mm | success | course 4.4 m | falls | progress m (mean, min) | course speed m/s | old 20 s rule | not counted | fall causes
    0 |     0.0 |    100% |         100% |    0% | 4.70, 4.70 |            0.236 |          100% |           0 | -
    1 |     6.1 |    100% |         100% |    0% | 4.70, 4.70 |            0.228 |          100% |           0 | -
    2 |    12.3 |    100% |         100% |    0% | 4.70, 4.70 |            0.234 |          100% |           0 | -
    3 |    18.4 |    100% |         100% |    0% | 4.70, 4.70 |            0.239 |          100% |           0 | -
    4 |    24.5 |    100% |         100% |    0% | 4.70, 4.70 |            0.223 |          100% |           0 | -
    5 |    30.7 |    100% |         100% |    0% | 4.70, 4.70 |            0.211 |          100% |           0 | -
    6 |    36.8 |     97% |          97% |    3% | 4.57, 0.68 |            0.196 |           97% |           0 | illegal_contact 1
first level below 100 %: 6
ru1_student_C_rev: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-07_00-12-55_ru1_student/model_299.pt
C_step REVERSED (raised side on the right), straight 0.2 m/s for 40 s, 30 envs per level; success = climbed: body 0.5 m past the entry riser, then no fall until it leaves the course, any speed
level | step mm | success | course 4.4 m | falls | progress m (mean, min) | course speed m/s | old 20 s rule | not counted | fall causes
    0 |     0.0 |    100% |         100% |    0% | 4.70, 4.70 |            0.237 |          100% |           0 | -
    1 |     6.1 |    100% |         100% |    0% | 4.70, 4.70 |            0.223 |          100% |           0 | -
    2 |    12.3 |    100% |         100% |    0% | 4.70, 4.70 |            0.228 |          100% |           0 | -
    3 |    18.4 |    100% |         100% |    0% | 4.70, 4.70 |            0.230 |          100% |           0 | -
    4 |    24.5 |    100% |         100% |    0% | 4.70, 4.70 |            0.224 |          100% |           0 | -
    5 |    30.7 |    100% |         100% |    0% | 4.70, 4.70 |            0.213 |          100% |           0 | -
    6 |    36.8 |     97% |          97% |    3% | 4.58, 0.88 |            0.196 |           97% |           0 | illegal_contact 1
first level below 100 %: 6
ru1_student_B: /home/rml3/Documents/ben/spine/runs_local/pipeline/logs/rsl_rl/microtaur_isaac_teacher/2026-11-07_00-12-55_ru1_student/model_299.pt
B_blocks, straight 0.2 m/s for 40 s, 30 envs per level; success = 2.0 m before any fall, any speed
level | step mm | success | falls | progress m (mean, min) | old 20 s rule | not counted | fall causes
    0 |       - |    100% |    0% | 2.00, 2.00 |          100% |           0 | -
    1 |       - |    100% |    0% | 2.00, 2.00 |          100% |           0 | -
    2 |       - |    100% |    0% | 2.00, 2.00 |          100% |           0 | -
    3 |       - |    100% |    0% | 2.00, 2.00 |          100% |           0 | -
    4 |       - |    100% |    0% | 2.00, 2.00 |          100% |           0 | -
    5 |       - |    100% |    0% | 2.00, 2.00 |          100% |           0 | -
    6 |       - |    100% |    0% | 2.00, 2.00 |           93% |           0 | -
first level below 100 %: none
```
**Videos:** `/home/rml3/Documents/ben/spine/runs_local/pipeline/ru1/ru1_student/videos/ru1_student_A_flat.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/ru1/ru1_student/videos/ru1_student_B_lvl6.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/ru1/ru1_student/videos/ru1_student_C_lvl6.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/ru1/ru1_student/videos/ru1_student_flat_side.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/ru1/ru1_student/videos/ru1_student_turn_tour.mp4`, `/home/rml3/Documents/ben/spine/runs_local/pipeline/ru1/ru1_student/videos/ru1_student_turn_tour_top.mp4`

