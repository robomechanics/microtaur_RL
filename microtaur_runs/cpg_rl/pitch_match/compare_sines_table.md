| | RL | 1 sine per motor | 3 sines per motor | 6 sines per motor (final) |
|---|---|---|---|---|
| robots | 32/32 | 16/16 | 16/16 | 32/32 |
| stride rate (Hz) | 4.00 (3.95-4.05) | 4.00 | 4.00 | 4.00 |
| body speed (m/s) | 0.145 | 0.133 | 0.139 | 0.145 |
| speed along start direction (m/s) | 0.146 | 0.129 | 0.142 | 0.144 |
| falls | 0 | 0 | 0 | 0 |
| IMU tilt max (deg) | 4.1 | 6.9 | 2.9 | 3.3 |
| cost of transport | 1.11 | 0.66 | 0.64 | 0.96 |
| leg commands clipped at stand ±30° (share / max) | - | - | - | 36% / 1.8° |
| spine swing (deg) | ±2.7 | ±2.3 | ±2.2 | ±2.3 |
| peak leg target speed (rad/s) | 49 | 13 | 25 | 34 |
| safety filter correction mean / max (deg) | 0.01 / 0.2 | 0.00 / 0.2 | 0.00 / 0.0 | 0.00 / 0.2 |
| match to RL: leg angles (R²) | 1.00 | 0.84 | 0.99 | 1.00 |
| match to RL: spine angle (R²) | 1.00 | 0.90 | 0.99 | 0.97 |
| match to RL: IMU pitch + roll (R²) | 1.00 | -3.12 | 0.94 | 0.91 |
| match to RL: gyro pitch/roll/yaw rate (R²) | 1.00 | -7.70 | 0.62 | 0.92 |
| match to RL: leg torque (R²) | 1.00 | 0.34 | 0.75 | 0.93 |
| leg1_a angle (R²) | 1.00 | 0.89 | 1.00 | 1.00 |
| leg1_e angle (R²) | 1.00 | 0.79 | 0.98 | 0.99 |
| leg2_a angle (R²) | 1.00 | 0.83 | 1.00 | 1.00 |
| leg2_e angle (R²) | 1.00 | 0.99 | 1.00 | 1.00 |
| leg3_a angle (R²) | 1.00 | 0.71 | 0.99 | 1.00 |
| leg3_e angle (R²) | 1.00 | 0.96 | 1.00 | 1.00 |
| leg4_a angle (R²) | 1.00 | 0.74 | 1.00 | 1.00 |
| leg4_e angle (R²) | 1.00 | 0.80 | 0.99 | 0.99 |
| spine angle (R²) | 1.00 | 0.90 | 0.99 | 0.97 |
