"""
test_open_loop_cpg_9dof.py - PC checks for open_loop_cpg_9dof.py on pitch, yaw and roll. No hardware, no Bridge.

    env -u PYTHONPATH /home/naomio/anaconda3/envs/microtaur/bin/python test_open_loop_cpg_9dof.py

PYTHONPATH is unset because this machine's ROS setup leaks Python 3.10 paths into it.
Plain asserts: prints PASS lines (and the dry-run tick ranges) and stops at the first failure.
"""

import contextlib
import csv
import importlib.util
import io
import json
import math
import os
import random
import re
import sys
import tempfile
import types

sys.dont_write_bytecode = True   # no __pycache__ here or next to the closed-loop script

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import open_loop_cpg_9dof as player  # noqa: E402

CPG_RL = os.path.dirname(HERE)
CLOSED_LOOP = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "old_distilling", "tinyRL-main", "UNO_Q_FILES",
    "closed_loop_controller", "closed_loop_teleop_9dof.py"))
EXAMPLE = os.path.join(HERE, "example_gait.json")
GAITS = {   # robot -> gait files to dry-run; the final fits must exist, the earlier d1_k3 fits are used if present
    robot: [os.path.join(CPG_RL, n) for n in names if "final" in n or os.path.exists(os.path.join(CPG_RL, n))]
    for robot, names in (("pitch", ["pitch_match/gait_pitch_final.json"]),
                         ("yaw", ["yaw_match/gait_yaw_final.json", "yaw_match/gait_d1_k3.json"]),
                         ("roll", ["roll_match/gait_roll_final.json", "roll_match/gait_d1_k3.json"]))
}
GAITS["pitch"].insert(0, EXAMPLE)
# Leg clip measured in the sim replays (coordinator): % of leg samples past stand +-30 deg, max overshoot
# in deg, and the allowed difference for each. Roll is being refit, so it gets more room.
EXPECTED_CLIP = {
    "gait_pitch_final.json": (36.0, 1.8, 3.0, 0.3),
    "gait_yaw_final.json": (28.0, 2.1, 3.0, 0.3),  # corrected yaw gait (sim: 28% of leg samples, max 2.1 deg)
    "gait_roll_final.json": (24.0, 1.2, 8.0, 1.0),
}
CL_NAME = {"pitch": "pitch", "yaw": "yaw", "roll": "twist"}     # the closed-loop script's ROBOT names
FAKE_SPINE_ID, FAKE_SPINE_ZERO = 9, 2100
R = player.RAD_TO_TICK
IDS = sorted(player.JOINT_TO_DXL_ID.values()) + [FAKE_SPINE_ID]


def load_closed_loop():
    """Import the closed-loop board script with Bridge and the TFLite runtime stubbed out."""
    class NoBridge:
        @staticmethod
        def call(*args):
            raise AssertionError("the test must not talk to hardware")

    names = ("arduino", "arduino.app_utils", "ai_edge_litert", "ai_edge_litert.interpreter")
    stubs = {name: types.ModuleType(name) for name in names}
    stubs["arduino.app_utils"].Bridge = NoBridge
    stubs["arduino"].app_utils = stubs["arduino.app_utils"]
    stubs["ai_edge_litert"].interpreter = stubs["ai_edge_litert.interpreter"]
    sys.modules.update(stubs)
    spec = importlib.util.spec_from_file_location("closed_loop_teleop_9dof", CLOSED_LOOP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SPINE_DXL_ID, module.SPINE_ZERO_TICK = FAKE_SPINE_ID, FAKE_SPINE_ZERO
    return module


def spine_half_ticks(cl, robot):
    """Ticks either side of the spine zero the closed-loop script can send for this robot (its lines 189-193, 234)."""
    cfg = cl.SPINE_CFG[CL_NAME[robot]]
    return round(cl.RAD_TO_TICK * min(cfg["target_limit_rad"], cl.SPINE_HARD_LIMIT_RAD - 1e-4))


def check_copied_constants(cl):
    for name in ("ROBOT", "SPINE_TICK_SIGN", "STAND_A_OFFSETS", "STAND_E_OFFSETS", "RAD_TO_TICK",
                 "ZERO_TICK", "HW_SIGN", "JOINT_TO_DXL_ID", "SAFE_TICK_RANGE",
                 "SPINE_HARD_LIMIT_RAD", "CONTROL_HZ"):
        assert getattr(player, name) == getattr(cl, name), f"{name} differs from closed_loop_teleop_9dof.py"
    assert player.LEG_JOINTS == cl.TRAINED_JOINT_ORDER
    assert player.SPINE_TARGET_LIMIT_RAD == {k: cfg["target_limit_rad"] for k, cfg in cl.SPINE_CFG.items()}
    for variant in ("active_pitch", "active_yaw", "active_twist"):                # the leg command range
        with open(os.path.join(CPG_RL, "..", variant, "src", "microtaur_velocity", "env_cfgs.py")) as f:
            deg = re.search(r'OLYMPUS_WALK_ACTION_SCALE_RAD = math\.radians\(\s*float\(os\.environ\.get\('
                            r'"MICROTAUR_OLYMPUS_ACTION_SCALE_DEG", "([0-9.]+)"\)\)', f.read()).group(1)
        assert player.OLYMPUS_WALK_ACTION_SCALE_RAD == math.radians(float(deg)), (variant, deg)
    assert set(player.ROBOT_KEY.values()) == set(cl.SPINE_CFG)
    print("PASS copied constants equal closed_loop_teleop_9dof.py (incl. per-robot SPINE_CFG limits)")


def check_a_formula():
    gait = {"frequency_hz": 2.0, "offset_rad": list(player.STAND_RAD[:8]) + [0.10],
            "cos_rad": [[0.0, 0.0] for _ in range(9)], "sin_rad": [[0.0, 0.0] for _ in range(9)]}
    gait["cos_rad"][0], gait["sin_rad"][0] = [0.10, 0.02], [0.05, -0.03]
    gait["cos_rad"][8], gait["sin_rad"][8] = [0.00, 0.04], [0.20, 0.00]
    # t = 0.0625 s at 2 Hz: 2*pi*f*t = pi/4, so k=1: cos = sin = 0.7071068; k=2: cos = 0, sin = 1.
    #   leg1_a: 0.45 + 0.10*0.7071068 + 0.05*0.7071068 + 0.02*0 - 0.03*1 = 0.5260660
    #   spine:  0.10 + 0.00*0.7071068 + 0.20*0.7071068 + 0.04*0 + 0.00*1 = 0.2414214
    angles = player.gait_angles(gait, 0.0625)
    assert abs(angles[0] - 0.5260660) < 1e-6, angles[0]
    assert abs(angles[8] - 0.2414214) < 1e-6, angles[8]
    assert all(abs(angles[j] - player.STAND_RAD[j]) < 1e-12 for j in range(1, 8))
    assert abs(player.gait_angles(gait, 0.0625 + 0.5)[0] - 0.5260660) < 1e-6   # one period later
    assert abs(player.gait_angles(gait, 0.0625, 0.5)[0] - (0.45 + 0.5 * 0.0760660)) < 1e-6
    print("PASS (a) angle formula matches the hand-computed value")


def check_b_stand(cl, robot):
    cl.ROBOT = CL_NAME[robot]                                    # selects its spine transform
    stand = player.angles_to_ticks(player.STAND_RAD, robot)
    closed_loop_stand = cl.STAND_TICKS + [cl.SPINE_ZERO_TICK]    # sent at its lines 509-513
    assert stand == closed_loop_stand, (stand, closed_loop_stand)

    # Why the player subtracts the stand angle: at zero action (the same standing angles) the
    # closed-loop policy path lands 293 ticks from its own stand on every leg joint.
    offset = [z - s for z, s in zip(cl.raw_action_to_ticks([0.0] * 9, 0.0), closed_loop_stand)]
    assert [abs(d) for d in offset[:8]] == [293] * 8 and offset[8] == 0, offset
    # Otherwise identical: legs = closed-loop ticks - RAD_TO_TICK * stand; spine exactly equal, with this
    # robot's spine transform and clamp (full spine action and yaw commands reach the clamp on pitch and yaw).
    stand_of_id = {player.JOINT_TO_DXL_ID[n]: s for n, s in zip(player.LEG_JOINTS, player.STAND_RAD)}
    stand_of_id[FAKE_SPINE_ID] = 0.0
    rng = random.Random(0)
    for _ in range(1000):
        act = [rng.uniform(-0.5, 0.5) for _ in range(8)] + [rng.uniform(-1.0, 1.0)]
        cmd_yaw = rng.uniform(-cl.YAW_MAX, cl.YAW_MAX)
        angles = []
        for leg in range(4):                                     # its lines 212-220
            swing = cl.SWING_OFFSET + cl.SWING_SCALE * act[2 * leg]
            lift = cl.LIFT_OFFSET + cl.LIFT_SCALE * act[2 * leg + 1]
            sign = cl.LEG_SIGNS[leg]
            angles += [cl.STAND_A_OFFSETS[leg] + sign * (swing - lift),
                       cl.STAND_E_OFFSETS[leg] + sign * (swing + lift)]
        angles.append(cl.spine_action_to_rad(act[8], cmd_yaw))
        cl_ticks = cl.raw_action_to_ticks(act, cmd_yaw)
        got = player.angles_to_ticks(angles, robot)
        assert all(abs(g - (c - R * stand_of_id[i])) <= 1.0 for g, c, i in zip(got[:8], cl_ticks, IDS)), (got, cl_ticks)
        assert got[8] == cl_ticks[8], (robot, angles[8], got[8], cl_ticks[8])
    print(f"PASS (b) {robot}: standing angles -> {stand} = closed-loop stand ticks; spine ticks equal the "
          f"closed-loop's for 1000 random commands; legs differ only by the stand offset it double-counts {offset[:2]}...")


def check_c_round_trip(cl, robot):
    rng = random.Random(1)
    limit = player.spine_limit_rad(robot) - 0.005
    worst = 0.0
    for _ in range(2000):   # stay inside the clips: legs at stand +- 30 deg (0.524 rad), spine at its limit
        angles = [s + rng.uniform(-0.5, 0.5) for s in player.STAND_RAD[:8]] + [rng.uniform(-limit, limit)]
        ticks = player.angles_to_ticks(angles, robot)
        back = player.ticks_to_angles(ticks)
        worst = max(worst, max(abs(b - a) * R for a, b in zip(angles, back)))
        assert player.angles_to_ticks(back, robot) == ticks
        for j, name in enumerate(player.LEG_JOINTS):             # closed-loop read path, lines 196-199
            tick = ticks[IDS.index(player.JOINT_TO_DXL_ID[name])]
            assert abs(cl.tick_to_rad(name, tick, j) - (back[j] - player.STAND_RAD[j])) < 1e-9
        assert abs(cl.spine_tick_to_rad(ticks[8]) - back[8]) < 1e-9
    assert worst <= 1.0, worst
    print(f"PASS (c) {robot}: angle -> tick -> angle within {worst:.3f} tick (limit 1), spine to +-{limit:.3f} rad")


def check_clamp(cl, robot):
    cl.ROBOT = CL_NAME[robot]
    half = spine_half_ticks(cl, robot)
    hi = {player.JOINT_TO_DXL_ID[n]: min(4095, player.ZERO_TICK[n] + 700) for n in player.LEG_JOINTS}
    lo = {player.JOINT_TO_DXL_ID[n]: max(0, player.ZERO_TICK[n] - 700) for n in player.LEG_JOINTS}
    hi[FAKE_SPINE_ID], lo[FAKE_SPINE_ID] = FAKE_SPINE_ZERO + half, FAKE_SPINE_ZERO - half
    saved = player.OLYMPUS_WALK_ACTION_SCALE_RAD
    player.OLYMPUS_WALK_ACTION_SCALE_RAD = 10.0                  # leg clip off: the tick clamps on their own
    try:
        assert player.angles_to_ticks([s + 3.0 for s in player.STAND_RAD], robot) == [hi[i] for i in IDS]
        assert player.angles_to_ticks([s - 3.0 for s in player.STAND_RAD], robot) == [lo[i] for i in IDS]
    finally:
        player.OLYMPUS_WALK_ACTION_SCALE_RAD = saved
    # The closed-loop script's own spine tick at full spine action and full left yaw command:
    top = cl.raw_action_to_ticks([0.0] * 8 + [1.0], cl.YAW_MAX)[8] - FAKE_SPINE_ZERO
    assert top == half if robot != "roll" else top < half, (robot, top, half)
    note = "its limit" if top == half else f"only {top} (policy scale {cl.SPINE_CFG['twist']['action_scale_rad']:.4f} rad)"
    print(f"PASS extra {robot}: tick clamps (leg clip off) = closed-loop safe ranges cut to 0..4095; spine to +-{half} "
          f"ticks (+-{player.spine_limit_rad(robot):.4f} rad); closed-loop full spine command reaches {note}")


def check_leg_clip(robot):
    limit, stand = player.OLYMPUS_WALK_ACTION_SCALE_RAD, player.STAND_RAD
    assert limit == math.radians(30.0)
    n = round(R * limit)                                         # 341 ticks
    for j in range(8):
        for sign in (+1.0, -1.0):
            edge = stand[j] + sign * limit
            for value, expected in ((stand[j] + sign * (limit + 0.2), edge), (stand[j] + sign * (limit + 1e-9), edge),
                                    (edge, edge), (stand[j] + sign * (limit - 1e-9), stand[j] + sign * (limit - 1e-9))):
                angles = list(stand)
                angles[j] = value
                clipped = player.clip_legs(angles)
                assert clipped[j] == expected and clipped[:j] + clipped[j + 1:] == list(stand[:j] + stand[j + 1:]), (j, value)
    zeros = [player.ZERO_TICK[name] for name in sorted(player.LEG_JOINTS, key=player.JOINT_TO_DXL_ID.get)]
    for sign in (+1.0, -1.0):
        far = [s + sign * (limit + 0.2) for s in stand[:8]] + [sign * 0.9]
        assert player.clip_legs(far)[8] == sign * 0.9                        # the spine is not clipped at 30 deg
        ticks = player.angles_to_ticks(far, robot)
        assert [t - z for t, z in zip(ticks[:8], zeros)] == [sign * n] * 8, ticks
        assert ticks[8] - FAKE_SPINE_ZERO == sign * round(R * player.spine_limit_rad(robot)), ticks[8]
    print(f"PASS clip {robot}: legs clip at exactly stand +-30 deg (+-{n} ticks), untouched just inside; "
          f"spine ignores it and stops at its own +-{round(R * player.spine_limit_rad(robot))} ticks")


def check_d_dry_run(cl, robot, path):
    gait = player.load_gait(path)
    assert player.ROBOT_KEY[gait["robot"]] == player.ROBOT_KEY[robot], (path, gait["robot"])
    clamped = player.clamp_report(gait)
    assert clamped == {}, f"{path} would be clamped (ticks past the range): {clamped}"
    share, over = player.leg_clip_report(gait)
    share_pct, over_deg = 100 * sum(share) / 8, math.degrees(max(over))
    expected = EXPECTED_CLIP.get(os.path.basename(path))
    if expected:
        pct, deg, pct_tol, deg_tol = expected
        assert abs(share_pct - pct) <= pct_tol and abs(over_deg - deg) <= deg_tol, (path, share_pct, over_deg, expected)
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "ticks.csv")
        player.main([path, "--dry-run", "--seconds", "10", "--out", out])
        with open(out) as f:
            rows = list(csv.reader(f))
    ranges = {player.JOINT_TO_DXL_ID[n]: player.SAFE_TICK_RANGE[n] for n in player.LEG_JOINTS}
    half = spine_half_ticks(cl, robot)
    ranges[FAKE_SPINE_ID] = (FAKE_SPINE_ZERO - half, FAKE_SPINE_ZERO + half)
    assert rows[0] == ["t_s"] + [f"id{i}" for i in IDS], rows[0]
    data = [[float(x) for x in row] for row in rows[1:]]
    assert data[-1][0] >= 10.0 and len(data) >= 11 * 50, (data[-1][0], len(data))
    zeros = [player.ZERO_TICK[name] for name in sorted(player.LEG_JOINTS, key=player.JOINT_TO_DXL_ID.get)]
    for row in data:
        for i, tick in zip(IDS, row[1:]):
            assert ranges[i][0] <= tick <= ranges[i][1] and 0 <= tick <= 4095, (row[0], i, tick, ranges[i])
    widest = max(abs(tick - z) for row in data for tick, z in zip(row[1:9], zeros))
    assert widest <= round(R * player.OLYMPUS_WALK_ACTION_SCALE_RAD), widest       # legs never past 30 deg
    assert (widest == round(R * player.OLYMPUS_WALK_ACTION_SCALE_RAD)) == (share_pct > 0), (widest, share_pct)
    if path == EXAMPLE:
        for col, i in enumerate(IDS, start=1):                   # 0.05 rad = 32.6 ticks each way
            p2p = max(r[col] for r in data) - min(r[col] for r in data)
            assert 60 <= p2p <= 70, (i, p2p)
    assert [int(x) for x in data[-1][1:]] == player.angles_to_ticks(player.STAND_RAD, robot)
    target = f" (sim: {expected[0]:.0f}%, {expected[1]:.1f} deg)" if expected else ""
    print(f"PASS (d) {robot}: {len(data)} dry-run frames of {os.path.relpath(path, CPG_RL)} inside the safe ranges, "
          f"no tick clamp needed, legs clipped in {share_pct:.1f}% of samples, max overshoot {over_deg:.2f} deg{target}, "
          "stop ramp ends at stand")


def check_e_soft_start():
    assert player.start_scale(1.0) == 0.0
    assert player.start_scale(2.0) == 0.5
    assert player.start_scale(3.0) == 1.0
    assert player.start_scale(0.0) == 0.0 and player.start_scale(60.0) == 1.0
    print("PASS (e) soft-start scale is 0 at 1 s, 0.5 at 2 s, 1 at 3 s")


def check_robot_names():
    def outcome(argv):
        """A real (non-dry) run: its refusal message, or None if it got as far as the stubbed Bridge."""
        try:
            player.main(argv)
        except SystemExit as err:
            return str(err)
        except AssertionError as err:
            assert "must not talk to hardware" in str(err), err
            return None
        raise AssertionError("main() returned without refusing or sending")

    roll = player.load_gait(GAITS["roll"][0])
    with tempfile.TemporaryDirectory() as tmp:
        twist_path = os.path.join(tmp, "gait_twist.json")
        with open(twist_path, "w") as f:
            json.dump(dict(roll, robot="twist"), f)
        twist = player.load_gait(twist_path)
        for t in (0.0, 0.37, 5.5):
            assert (player.angles_to_ticks(player.gait_angles(twist, t), "twist")
                    == player.angles_to_ticks(player.gait_angles(roll, t), "roll"))
        player.ROBOT = "pitch"
        assert "ROBOT = 'pitch'" in outcome([GAITS["yaw"][0]])
        assert "ROBOT = 'pitch'" in outcome([twist_path])
        player.ROBOT = "roll"
        assert outcome([twist_path]) is None                     # "twist" gait accepted on a roll board
        player.ROBOT = "pitch"
    print("PASS extra: 'twist' gait = 'roll' gait; a real run refuses another robot's gait before any Bridge call")


def check_clip_warning():
    base = player.load_gait(EXAMPLE)
    for amplitude, warned in ((0.60, False), (0.62, True)):     # leg1_a overshoots 30 deg by 4.4 / 5.5 deg
        gait = json.loads(json.dumps(base))
        gait["sin_rad"][0] = [amplitude]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gait.json")
            with open(path, "w") as f:
                json.dump(gait, f)
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                player.main([path, "--dry-run", "--seconds", "1", "--out", os.path.join(tmp, "ticks.csv")])
        text = printed.getvalue()
        assert "Legs clipped to stand +-30 deg" in text and ("[WARN] the gait overshoots" in text) == warned, text
    print("PASS extra: the startup clip line is always printed; it warns at 5.5 deg of overshoot, not at 4.4 deg")


def check_no_drift():
    gait, clock, frames = player.load_gait(EXAMPLE), player.FakeClock(), []

    def slow_send(ticks):
        frames.append((clock.now, ticks))
        clock.sleep(0.005)                                       # pretend each Bridge call takes 5 ms

    late = player.play(gait, slow_send, seconds=10.0, clock=clock, sleep=clock.sleep, quiet=True)
    gait_frames = [(t, ticks) for t, ticks in frames if t < 10.0 - 1e-6]
    assert len(gait_frames) == 500 and late == 0, (len(gait_frames), late)
    for t, ticks in gait_frames:
        assert ticks == player.angles_to_ticks(player.gait_angles(gait, t, player.start_scale(t)), gait["robot"])
    print(f"PASS extra: {len(gait_frames)} frames in 10 s with 5 ms send latency "
          "(sleeping 20 ms per frame would give 400); phase follows the clock")


def check_ctrl_c():
    gait, clock, frames, fired = player.load_gait(EXAMPLE), player.FakeClock(), [], []

    def send(ticks):
        if clock.now >= 5.0 and not fired:
            fired.append(clock.now)
            raise KeyboardInterrupt                               # Ctrl-C at t = 5 s, mid-gait
        frames.append((clock.now, ticks))

    player.play(gait, send, clock=clock, sleep=clock.sleep, quiet=True)   # no time limit
    t_stop = fired[0]
    ramp = [(t, ticks) for t, ticks in frames if t >= t_stop]
    assert abs(ramp[-1][0] - t_stop - player.STOP_S) < 1e-6, ramp[-1][0]
    assert ramp[-1][1] == player.angles_to_ticks(player.STAND_RAD, gait["robot"])
    for t, ticks in ramp:                                         # gait keeps its phase, scale 1 -> 0
        left = max(0.0, 1.0 - (t - t_stop) / player.STOP_S)
        expected = player.angles_to_ticks(player.gait_angles(gait, t, left), gait["robot"])
        assert max(abs(a - b) for a, b in zip(ticks, expected)) <= 1
    print(f"PASS extra: Ctrl-C at t=5 s ramps to stand in {ramp[-1][0] - t_stop:.2f} s ({len(ramp)} frames) and stops")


if __name__ == "__main__":
    assert os.path.exists(CLOSED_LOOP), f"closed-loop script not found: {CLOSED_LOOP}"
    for files in GAITS.values():
        assert all(os.path.exists(p) for p in files), files
    player.SPINE_DXL_ID, player.SPINE_ZERO_TICK = FAKE_SPINE_ID, FAKE_SPINE_ZERO
    closed_loop = load_closed_loop()
    check_copied_constants(closed_loop)
    check_a_formula()
    check_e_soft_start()
    for name in ("pitch", "yaw", "roll"):
        print(f"--- {name}")
        check_b_stand(closed_loop, name)
        check_c_round_trip(closed_loop, name)
        check_clamp(closed_loop, name)
        check_leg_clip(name)
        for gait_path in GAITS[name]:
            check_d_dry_run(closed_loop, name, gait_path)
    print("--- robot names, clip warning, timing, Ctrl-C")
    check_robot_names()
    check_clip_warning()
    check_no_drift()
    check_ctrl_c()
    print("ALL PASS")
