# Microtaur walking policy: deployment bundle for the Arduino UNO Q

A learned walking controller for the **rigid Microtaur** (four five-bar legs, eight XL330 servos). It was trained in simulation (IsaacLab) and has **not been run on hardware yet**. This bundle contains everything needed to run it on the UNO Q's Linux side, plus a test that proves a port computes exactly what the simulator computed.

The policy is blind: it has no terrain map. It reads only the front IMU, the eight servo encoders and the velocity command, and outputs eight servo position targets at 28.6 Hz.

## Files

| File | What it is |
|---|---|
| `policy.onnx` | The network: input `obs` [1, 705] float32 → output `actions` [1, 8]. Input normalisation is included. The output is **not** clipped. |
| `policy_weights.npz` | The same network as plain numpy arrays (no ONNX runtime needed). |
| `microtaur_policy.py` | Reference implementation, numpy only: builds the observation, keeps the history, runs the network, applies the safety filter, and converts between ticks and radians. |
| `test_golden.py` | Replays two simulator recordings (`golden.npz`) through `microtaur_policy.py` and compares every stage. |
| `golden.npz` | 2 × 300 policy steps recorded in the simulator: flat ground (straight, then turning left) and the 18 mm side step. |
| `SOURCE.txt` | Which checkpoint this was exported from. |

## Quick start

```bash
pip install numpy            # optional: pip install onnxruntime
python test_golden.py        # must print ALL PASS (add --onnx to also check policy.onnx)
```

`ALL PASS` means the numpy pipeline reproduces the simulator to about 1e-7 at every stage. The errors are measured per stage: each observation term, the 705-value input, the network output, and the servo target. **Re-run it after porting or changing anything** (C++, another runtime, a different history buffer). It is the only check that the observation layout is right; a wrong layout does not crash, the robot just walks badly.

Control loop, once every 35 ms:

```python
import microtaur_policy as mp
ctrl = mp.Controller("policy_weights.npz")       # or Controller(..., onnx_path="policy.onnx")
ctrl.reset()                                     # at start, and after any stop / pick-up
while running:
    q  = mp.rad_from_ticks(present_position_ticks)   # 8 values, canonical joint order (below)
    qd = present_velocity_rad_s                      # 8 values, rad/s
    target = ctrl.step(gravity_b, gyro_b, q, qd, command=(vx, 0.0, wz))
    write_goal_position(mp.ticks_from_rad(target))   # send immediately (see "Latency")
```

## Interface

### Timing
- **Policy period:** 35 ms (28.6 Hz). The simulator ran 14 physics steps of 2.5 ms per policy step.
- **Compute cost:** about 0.2 M multiply-adds per step (network 705-256-128-8). On the QRB2210's Cortex-A53 that should be well under 1 ms (an estimate, not measured).
- **The loop period matters more than compute speed.** Keep the 35 ms period steady; the servo bus read/write will likely take longer than the network.

### Joint order and hardware mapping
The canonical order is used everywhere (observations, actions, targets):

| index | joint | leg | DXL ID | stand (rad) | stand tick | tick sign |
|---|---|---|---|---|---|---|
| 0 | leg1_a | rear right | 5 | +0.45 | 3590 | −1 |
| 1 | leg1_e | rear right | 6 | −0.45 | 625 | +1 |
| 2 | leg2_a | rear left | 3 | −0.45 | 3538 | +1 |
| 3 | leg2_e | rear left | 4 | +0.45 | 532 | −1 |
| 4 | leg3_a | front left | 1 | −0.45 | 3473 | +1 |
| 5 | leg3_e | front left | 2 | +0.45 | 491 | −1 |
| 6 | leg4_a | front right | 7 | +0.45 | 3685 | −1 |
| 7 | leg4_e | front right | 8 | −0.45 | 606 | +1 |

`rad = stand_rad + sign × (tick − stand_tick) × 2π/4096`.

These constants are copied from upstream `sim2real/microtaur_hw_characterization/config/hardware_config.py`, **which marks them `CALIBRATION_VERIFIED = False`**. The IDs, signs and stand ticks must be confirmed on the robot before the policy runs (upstream Test 01, `01_mapping_and_sign.py`). If any value is wrong, the only file to change is `microtaur_policy.py` (`JOINT_TO_DXL_ID`, `HW_SIGN`, `STAND_TICK`).

**Sanity check after calibration:** hold the stand ticks. Every leg should look identical, with the foot straight under the hip, and `mp.foot_fk(q)` should read about (x, z) = (−5.1, −73.5) mm for all four legs.

### Observation: 47 values per frame, 15 frames → 705 inputs
| term | size | source on the robot |
|---|---|---|
| `projected_gravity` | 3 | Front IMU: unit gravity direction in the body frame. About (0, 0, −1) when level. From the IMU orientation estimate, or `−acc/|acc|` when nearly static. |
| `base_ang_vel` | 3 | Front IMU gyro, body frame, rad/s |
| `imu2_projected_gravity` | 3 | **Zeros** (rear-IMU slot, reserved for the spine robot) |
| `imu2_base_ang_vel` | 3 | **Zeros** |
| `joint_pos` | 8 | motor angle − stand angle (rad) |
| `joint_vel` | 8 | motor velocity (rad/s) |
| `foot_fk` | 8 | `mp.foot_fk(q)`: five-bar forward kinematics, foot (x, z) per leg in metres |
| `actions` | 8 | the **previous raw network output, not clipped** (zeros after reset) |
| `command` | 3 | (vx m/s, 0, yaw rate rad/s) |

- **Body frame:** x forward, y left, z up. The IMU axes must be rotated into this frame; check the signs with upstream `09_imu_axes.py`. For example, nose-down pitch must give a positive gravity x, and a left turn must give a positive gyro z.
- **Clipping:** each term is clipped to ±100 before it enters the history. This never triggers in normal operation.
- **History layout is term-major, oldest first:** `[gravity × 15 frames, gyro × 15, …, command × 15]`, **not** frame by frame. After `reset()` the first frame fills all 15 slots. `ObsHistory` does exactly this.

### Action → servo target
The network outputs 8 numbers. `ActionPipeline.step` turns them into targets:
1. clip each output to [−1, 1];
2. `requested = stand + 30° × action`;
3. safety filter, identical to training:
   - per-joint limits stand ± 0.75 rad, with a 0.1 s time-to-limit blend;
   - per-leg limits on the "common" and "diff" coordinates of the two motors, 0.52 rad each;
   - `|common| + |diff| ≤ 0.70 rad`.

The filter uses the measured q and qd from the same step.

**Latency:** the simulator applies each target one policy step late (35 ms), to stand in for the real command-to-motion delay. On the robot, write the target as soon as it is computed; do **not** add another delay. If the measured end-to-end latency (upstream `04b_end_to_end_step_latency.py`) is far from 35 ms, tell us.

### Commands
Training covered vx 0.10–0.35 m/s and yaw rate −0.25…+0.25 rad/s, with vy always 0. Stay inside these ranges; there is no standing or walking-backwards command.

## What to expect (simulation only)
- **Gait:** diagonal trot at about 7 Hz, no flight phase, body bob about 3 mm.
- **Speed:** the robot walks faster than slow commands: 0.17 m/s at 0.10, 0.24 at 0.20, 0.33 at 0.35.
- **Flat ground:** no falls. Heading drifts over time; on flat ground the drift is about 10–30° per 20 s.
- **Random blocks up to ±2 cm:** 3% falls at the hardest level; slows to 0.13 m/s.
- **Side step (one side raised 18–37 mm):** no falls, 0.13–0.20 m/s. The legs on the raised side shorten by about 29 mm on a 37 mm step, keeping the body within about 4° of level.

## Not validated / known gaps
- **The servo model is not identified.** The simulator models the XL330 as a compliant position servo (P gain 1, torque cap 0.129 N·m, velocity limit 32 rad/s). These are datasheet-based values; the real step response is the biggest sim-to-real unknown.
- **The policy actively uses the servo's step response**, and about 20% of actions sit at the ±30° bound.
- **Calibration constants are unverified** (see above).
- **The IMU axes must match the body frame** described above.

## Suggested first tests
1. Run `test_golden.py` on the UNO Q.
2. Suspend the robot; command the stand pose; check `foot_fk` and the IMU signs.
3. Still suspended, run the loop with vx = 0.10. The legs should trot (diagonal pairs moving together) without hitting the limits.
4. On flat ground on a tether, at 0.10–0.20 m/s. Stop immediately if the body tilts more than about 30°, or if a servo overheats or hits its current limit.
