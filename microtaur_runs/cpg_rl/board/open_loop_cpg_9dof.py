"""
open_loop_cpg_9dof.py - open-loop gait player for the 9-servo Microtaur (8 legs + spine): pitch, yaw
or roll (= twist). Every 1/control_hz s: 9 joint angles from a clock -> legs clipped to stand +-30 deg
(the sim's leg command range) -> ticks, clamped to the safe ranges -> one Bridge call to
closed_loop_9dof.ino (unchanged). No IMU, no model. Hold stand 1 s, ramp in over 2 s (harmonics x
0->1, offsets stand->gait), play; Ctrl-C ramps back to stand over 1 s. t comes from time.monotonic().

    python3 open_loop_cpg_9dof.py gait.json          # robot (App Lab: open_loop_cpg_9dof.main(["gait.json"]))
    python3 open_loop_cpg_9dof.py gait.json --dry-run --seconds 10 --out ticks.csv   # PC, no Bridge

Gait JSON (example_gait.json, fit_harmonic_gait.py): absolute joint angles in rad, simulator canonical
order and frame (the action terms' targets), "robot" = pitch | yaw | roll | twist, f = frequency_hz:
    angle_j(t) = offset_rad[j] + sum_k cos_rad[j][k-1]*cos(2*pi*k*f*t) + sin_rad[j][k-1]*sin(2*pi*k*f*t)
"""

import argparse
import json
import math
import time

# Copied from old_distilling/tinyRL-main/UNO_Q_FILES/closed_loop_controller/closed_loop_teleop_9dof.py
# (line numbers are in that file). Not imported: it loads Bridge and the TFLite runtime at import.
# test_open_loop_cpg_9dof.py checks every copy against that file.
ROBOT = "pitch"          # line 48: the robot this board drives, "pitch" | "yaw" | "roll" (= "twist")
SPINE_DXL_ID = None      # TODO e.g. 9 = MOTOR_IDS[8] in the sketch     lines 56-58
SPINE_ZERO_TICK = None   # TODO tick reading with the spine flat (0 rad)
SPINE_TICK_SIGN = +1.0   # TODO -1.0 if rising ticks give negative spine angles
STAND_A_OFFSETS = (0.45, -0.45, -0.45, 0.45)                     # line 90
STAND_E_OFFSETS = (-0.45, 0.45, 0.45, -0.45)                     # line 91
LEG_JOINTS = tuple(f"leg{i}_{m}_joint_act" for i in (1, 2, 3, 4) for m in ("a", "e"))  # lines 93-98
TICKS_PER_REV = 4096                                             # line 100
RAD_TO_TICK = TICKS_PER_REV / (2.0 * math.pi)                    # line 101
# TICK CONVENTION: ZERO_TICK is the tick at the STANDING pose (+-0.45 rad), not at 0 rad.
# STAND_TICKS (line 176) is the same numbers by DXL id and is held as the stand (lines 508-513);
# tick_to_rad (lines 196-199) reads it as 0, i.e. stand-relative; open_loop.py labels the numbers
# "#stand" (lines 79-89). raw_action_to_ticks still adds the absolute stand angle (lines 219-220,
# 230), landing 293 ticks (25.8 deg) off its own stand on every leg joint. This player subtracts
# the stand angle first, so standing angles give exactly STAND_TICKS. Spine stand is 0 (= line 234).
ZERO_TICK = {"leg1_a_joint_act": 3473, "leg1_e_joint_act": 491,   # lines 103-112
             "leg2_a_joint_act": 606, "leg2_e_joint_act": 3685,
             "leg3_a_joint_act": 625, "leg3_e_joint_act": 3590,
             "leg4_a_joint_act": 3538, "leg4_e_joint_act": 532}
HW_SIGN = {name: 1.0 for name in LEG_JOINTS}                     # line 114
JOINT_TO_DXL_ID = {"leg1_a_joint_act": 1, "leg1_e_joint_act": 2,  # lines 116-125
                   "leg2_a_joint_act": 8, "leg2_e_joint_act": 7,
                   "leg3_a_joint_act": 6, "leg3_e_joint_act": 5,
                   "leg4_a_joint_act": 3, "leg4_e_joint_act": 4}
SAFE_TICK_RANGE = {n: (ZERO_TICK[n] - 700, ZERO_TICK[n] + 700) for n in LEG_JOINTS}  # lines 127-130
SPINE_HARD_LIMIT_RAD = 0.35                                      # line 151
SPINE_TARGET_LIMIT_RAD = {"yaw": 0.30, "twist": 0.32, "pitch": 0.5235987755982988}  # SPINE_CFG, lines 153-161
CONTROL_HZ = 50.0                                                # line 172 (JSON control_hz wins)

# The sim replays gaits through the RL's leg action term, requested = stand + scale * clamp(action, -1, 1),
# so every leg target is clipped to stand +- this (default MICROTAUR_OLYMPUS_ACTION_SCALE_DEG = 30.0 in
# env_cfgs.py: active_pitch lines 1116-1118, active_yaw 991-993, active_twist 1009-1011). The spine is not.
OLYMPUS_WALK_ACTION_SCALE_RAD = math.radians(30.0)
CLIP_WARN_RAD = math.radians(5.0)   # startup warning when a gait overshoots that range by more

ROBOT_KEY = {"pitch": "pitch", "yaw": "yaw", "roll": "twist", "twist": "twist"}  # name -> SPINE_CFG key
SPINE_JOINT = "spine_joint_act"
JOINT_ORDER = LEG_JOINTS + (SPINE_JOINT,)
STAND_RAD = tuple(x for pair in zip(STAND_A_OFFSETS, STAND_E_OFFSETS) for x in pair) + (0.0,)
HOLD_S, RAMP_S, STOP_S, PRINT_HZ = 1.0, 2.0, 1.0, 2.0
DRY_RUN_SPINE = (9, 2048)   # stand-in spine id / zero tick, used by --dry-run only


def load_gait(path):
    with open(path) as f:
        gait = json.load(f)
    gait.setdefault("control_hz", CONTROL_HZ)
    cos_c, sin_c = gait.get("cos_rad", []), gait.get("sin_rad", [])
    ok = (gait.get("robot") in ROBOT_KEY and gait.get("joint_order") == list(JOINT_ORDER)
          and len(gait.get("offset_rad", [])) == len(cos_c) == len(sin_c) == 9
          and len({len(row) for row in cos_c + sin_c}) == 1
          and gait.get("frequency_hz", 0) > 0 and gait["control_hz"] > 0
          and all(math.isfinite(x) for x in [gait["frequency_hz"], gait["control_hz"]]
                  + gait["offset_rad"] + sum(cos_c + sin_c, [])))
    if not ok:
        raise SystemExit(f"Bad gait file {path}. Need robot in {sorted(ROBOT_KEY)}, joint_order {list(JOINT_ORDER)}, "
                         "offset_rad[9], cos_rad[9][K], sin_rad[9][K] (finite), frequency_hz > 0.")
    return gait


def gait_angles(gait, t, scale=1.0):
    """The 9 joint angles (rad, JOINT_ORDER) at time t. scale 0 = stand, 1 = the gait."""
    w = 2.0 * math.pi * gait["frequency_hz"] * t
    angles = []
    for j, stand in enumerate(STAND_RAD):
        angle = gait["offset_rad"][j]
        for k, (c, s) in enumerate(zip(gait["cos_rad"][j], gait["sin_rad"][j]), start=1):
            angle += c * math.cos(k * w) + s * math.sin(k * w)
        angles.append(stand + scale * (angle - stand))
    return angles


def start_scale(t):
    """Soft start: 0 while holding stand (t <= HOLD_S), then linear to 1 over RAMP_S."""
    return min(1.0, max(0.0, (t - HOLD_S) / RAMP_S))


def spine_limit_rad(robot):
    """The spine angle limit closed_loop_teleop_9dof.py applies for this robot (lines 189, 192)."""
    return min(SPINE_TARGET_LIMIT_RAD[ROBOT_KEY[robot]], SPINE_HARD_LIMIT_RAD - 1e-4)


def servo_table(robot):
    """(joint, dxl_id, zero_tick, sign, stand_rad, lo, hi) sorted by DXL id. Legs: the closed-loop safe
    ranges. Spine: the ticks the closed-loop script sends at this robot's angle limit (lines 189-193 and
    234-237). All cut to 0..4095, the valid goals in position mode (leg1_e's range would reach -209)."""
    half = round(RAD_TO_TICK * spine_limit_rad(robot))
    rows = [(name, JOINT_TO_DXL_ID[name], ZERO_TICK[name], HW_SIGN[name], s) + SAFE_TICK_RANGE[name]
            for name, s in zip(LEG_JOINTS, STAND_RAD)]
    rows.append((SPINE_JOINT, SPINE_DXL_ID, SPINE_ZERO_TICK, SPINE_TICK_SIGN, 0.0,
                 SPINE_ZERO_TICK - half, SPINE_ZERO_TICK + half))
    return sorted((r[:5] + (max(0, r[5]), min(TICKS_PER_REV - 1, r[6])) for r in rows), key=lambda r: r[1])


def clip_legs(angles):
    """Legs to stand +- OLYMPUS_WALK_ACTION_SCALE_RAD, as the sim's leg action term does. Spine untouched."""
    limit = OLYMPUS_WALK_ACTION_SCALE_RAD
    return [min(s + limit, max(s - limit, a)) for a, s in zip(angles[:8], STAND_RAD)] + list(angles[8:])


def angles_to_ticks(angles, robot, clamp=True):
    """9 angles (rad, JOINT_ORDER) -> 9 ticks sorted by DXL id, the order the sketch expects (line 239):
    legs clipped (clip_legs), then ticks rounded and clamped to servo_table(robot) (lines 231, 237).
    clamp=False skips both and gives the raw floats."""
    if clamp:
        angles = clip_legs(angles)
    angle_of = dict(zip(JOINT_ORDER, angles))
    ticks = []
    for name, _, zero, sign, stand, lo, hi in servo_table(robot):
        raw = zero + sign * RAD_TO_TICK * (angle_of[name] - stand)
        ticks.append(min(hi, max(lo, int(round(raw)))) if clamp else raw)
    return ticks


def ticks_to_angles(ticks):
    """Inverse of angles_to_ticks: ticks sorted by DXL id -> angles (rad, JOINT_ORDER)."""
    table = servo_table(ROBOT)                                   # only zero/sign/stand are used
    angle_of = {row[0]: row[4] + (tick - row[2]) / (row[3] * RAD_TO_TICK) for row, tick in zip(table, ticks)}
    return [angle_of[name] for name in JOINT_ORDER]


def leg_clip_report(gait, samples=500):
    """Per leg joint over one full-scale gait period: share of samples beyond stand +- 30 deg, worst overshoot (rad)."""
    share, over = [0.0] * 8, [0.0] * 8
    for i in range(samples):
        angles = gait_angles(gait, i / (samples * gait["frequency_hz"]))
        for j in range(8):
            excess = abs(angles[j] - STAND_RAD[j]) - OLYMPUS_WALK_ACTION_SCALE_RAD
            if excess > 0:
                share[j] += 1.0 / samples
                over[j] = max(over[j], excess)
    return share, over


def clamp_report(gait, samples=500):
    """{joint: worst ticks past its tick range, after the leg clip} over one full-scale period; {} if none."""
    over = {}
    for i in range(samples):
        angles = clip_legs(gait_angles(gait, i / (samples * gait["frequency_hz"])))
        raw = angles_to_ticks(angles, gait["robot"], clamp=False)
        for (name, _, _, _, _, lo, hi), tick in zip(servo_table(gait["robot"]), raw):
            if max(lo - tick, tick - hi) > 0:
                over[name] = max(over.get(name, 0.0), lo - tick, tick - hi)
    return over


class FakeClock:
    """Virtual time for --dry-run and tests: calling it reads the time, sleep() advances it."""
    def __init__(self):
        self.now = 0.0
    def __call__(self):
        return self.now
    def sleep(self, seconds):
        self.now += seconds


def play(gait, send, seconds=None, clock=time.monotonic, sleep=time.sleep, quiet=False):
    """send(ticks) at control_hz until Ctrl-C (or `seconds`), then ramp to stand over STOP_S."""
    dt = 1.0 / gait["control_hz"]
    t0 = next_tick = clock()
    late, scale, next_print = 0, 0.0, 0.0

    def wait():
        nonlocal next_tick, late
        next_tick += dt                  # fixed deadlines, so the send rate does not drift
        pause = next_tick - clock()
        if pause > 0:
            sleep(pause)
        else:                            # fell behind: skip ahead rather than burst
            late += 1
            next_tick = clock()

    try:
        while seconds is None or clock() - t0 < seconds - 1e-9:
            t = clock() - t0
            scale = start_scale(t)
            send(angles_to_ticks(gait_angles(gait, t, scale), gait["robot"]))
            if not quiet and t >= next_print:
                print(f"\rt={t:8.2f} s  scale={scale:.2f}  late={late}   ", end="", flush=True)
                next_print = t + 1.0 / PRINT_HZ
            wait()
    except KeyboardInterrupt:
        pass
    if not quiet:
        print("\nRamping back to stand...")
    t_stop = clock() - t0
    try:
        while True:                      # the gait keeps its clock; only its scale goes to 0
            t = clock() - t0
            left = 0.0 if t - t_stop >= STOP_S - 1e-9 else 1.0 - (t - t_stop) / STOP_S
            send(angles_to_ticks(gait_angles(gait, t, scale * left), gait["robot"]))
            if left == 0.0:
                break
            wait()
    except KeyboardInterrupt:
        pass                             # second Ctrl-C: stop now, servos hold the last target
    return late


def main(argv=None):
    global SPINE_DXL_ID, SPINE_ZERO_TICK
    parser = argparse.ArgumentParser(description="Open-loop 9-servo gait player.")
    parser.add_argument("gait", help="gait JSON")
    parser.add_argument("--dry-run", action="store_true", help="no Bridge, virtual clock, just ticks")
    parser.add_argument("--seconds", type=float, default=10.0, help="dry run: gait time before the stop ramp")
    parser.add_argument("--out", help="dry run: write the ticks CSV here instead of printing it")
    args = parser.parse_args(argv)
    gait = load_gait(args.gait)
    robot = gait["robot"]
    if not args.dry_run and ROBOT_KEY.get(ROBOT) != ROBOT_KEY[robot]:
        raise SystemExit(f"{args.gait} is a {robot!r} gait, but this board is set up for ROBOT = {ROBOT!r}.\n"
                         "Load that robot's gait, or set ROBOT and the spine calibration at the top of this file.")

    if SPINE_DXL_ID is None or SPINE_ZERO_TICK is None:
        if not args.dry_run:
            raise SystemExit(
                "SPINE_DXL_ID and SPINE_ZERO_TICK are not set.\n"
                "Find the spine servo's Dynamixel ID, put the spine at its neutral (flat)\n"
                "pose, read its present position, and put both at the top of this file.\n"
                "Running without them would drive the spine to an arbitrary tick.")
        SPINE_DXL_ID, SPINE_ZERO_TICK = DRY_RUN_SPINE
        print(f"[DRY RUN] spine not calibrated: stand-in id {SPINE_DXL_ID}, zero tick {SPINE_ZERO_TICK}")
    if SPINE_DXL_ID in JOINT_TO_DXL_ID.values():
        raise SystemExit(f"SPINE_DXL_ID {SPINE_DXL_ID} is already a leg servo.")
    print(f"{args.gait}: {robot} gait, {gait['frequency_hz']:.3f} Hz, {len(gait['cos_rad'][0])} harmonic(s), "
          f"spine limit +-{spine_limit_rad(robot):.4f} rad, sent at {gait['control_hz']} Hz to DXL ids "
          f"{[row[1] for row in servo_table(robot)]}")
    share, over = leg_clip_report(gait)
    print(f"Legs clipped to stand +-{math.degrees(OLYMPUS_WALK_ACTION_SCALE_RAD):.0f} deg in {100 * sum(share) / 8:.1f}% "
          f"of samples over one period, max overshoot {math.degrees(max(over)):.2f} deg; per joint: "
          + ", ".join(f"{name[:6]} {100 * s:.0f}%/{math.degrees(o):.1f}deg" for name, s, o in zip(LEG_JOINTS, share, over)))
    if max(over) > CLIP_WARN_RAD:
        print(f"[WARN] the gait overshoots the sim's leg command range by up to {math.degrees(max(over)):.1f} deg "
              f"(more than {math.degrees(CLIP_WARN_RAD):.0f}); the board plays it clipped, as the sim does.")
    for name, excess in clamp_report(gait).items():
        print(f"[WARN] {name} leaves its safe range by up to {excess:.0f} ticks; it will be clamped.")

    if not args.dry_run:
        from arduino.app_utils import Bridge
        print("Holding stand 1 s, then ramping in over 2 s. Ctrl-C to stop.")
        late = play(gait, lambda ticks: Bridge.call("set_motor_positions", *ticks))
        print(f"Stopped at stand. Late ticks: {late}")
        return

    clock, frames = FakeClock(), []
    play(gait, lambda ticks: frames.append((clock.now, ticks)), args.seconds, clock, clock.sleep, quiet=True)
    lines = ["t_s," + ",".join(f"id{row[1]}" for row in servo_table(robot))]
    lines += [f"{t:.3f}," + ",".join(map(str, ticks)) for t, ticks in frames]
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(lines) + "\n")
    else:
        print("\n".join(lines))
    for col, (name, dxl, _, _, _, lo, hi) in enumerate(servo_table(robot)):
        column = [ticks[col] for _, ticks in frames]
        print(f"  id{dxl} {name:17s} ticks {min(column)}..{max(column)}  allowed {lo:.0f}..{hi:.0f}")
    print(f"[DRY RUN] {len(frames)} frames, last at t={frames[-1][0]:.2f} s" + (f", in {args.out}" if args.out else ""))


if __name__ == "__main__":
    main()
