from __future__ import annotations

import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from mjlab.entity.entity import Entity
from microtaur_velocity.microtaur_constants import (
    ACTION_JOINT_NAMES,
    MICROTAUR_STAND_A_OFFSETS,
    MICROTAUR_STAND_E_OFFSETS,
    MINITAUR_PAIR_SWING_SCALE,
    MINITAUR_PAIR_EXTENSION_SCALE,
    get_microtaur_robot_cfg,
)


# ---------------------------------------------------------------------
# True bound gait parameters
# ---------------------------------------------------------------------

FREQ_HZ = 1.4

# Raw policy-style amplitudes.
# These get multiplied by MINITAUR_PAIR_SWING_SCALE / EXTENSION_SCALE.
SWING_RAW_AMP = 0.50
EXTENSION_RAW_AMP = 0.50
EXTENSION_RAW_BIAS = 0.0

# Flip if robot moves backward.
SWING_SIGN = -1.0

# Correct leg-wise signs from the working Microtaur gait code:
#   A_SIGN = {1:+1, 2:-1, 3:-1, 4:+1}
#   E_SIGN = {1:-1, 2:+1, 3:+1, 4:-1}
A_SIGN = np.array([+1.0, -1.0, -1.0, +1.0], dtype=np.float64)
E_SIGN = np.array([-1.0, +1.0, +1.0, -1.0], dtype=np.float64)

# TRUE BOUND:
#   front pair = legs 1 and 2
#   rear pair  = legs 3 and 4
#   front and rear are 180 degrees out of phase
BOUND_PHASE = np.array(
    [
        0.0,        # leg1 front
        0.0,        # leg2 front
        math.pi,    # leg3 rear
        math.pi,    # leg4 rear
    ],
    dtype=np.float64,
)

RAMP_TIME_S = 2.0
REALTIME = True


# ---------------------------------------------------------------------
# Convention
# ---------------------------------------------------------------------
#
# Raw action order:
#   [
#     leg1_swing, leg1_extension,
#     leg2_swing, leg2_extension,
#     leg3_swing, leg3_extension,
#     leg4_swing, leg4_extension,
#   ]
#
# Correct signed mapping:
#   a_target = stand_a + A_SIGN[leg] * swing
#   e_target = stand_e + E_SIGN[leg] * extension
#
# This matches the working rigid Microtaur gait structure:
#   a_delta = A_SIGN[leg] * swing_amp * fore_aft
#   e_delta = E_SIGN[leg] * lift_amp * lift
# ---------------------------------------------------------------------


def make_bound_raw_action(t: float) -> np.ndarray:
    """Return raw 8D action for a true front/rear bound.

    Bound pattern:
      front pair: legs 1,2
      rear pair:  legs 3,4
      front and rear are 180 degrees out of phase.
    """

    base_phase = 2.0 * math.pi * FREQ_HZ * t
    raw = np.zeros(8, dtype=np.float64)

    for leg_idx in range(4):
        phi = base_phase + BOUND_PHASE[leg_idx]

        # Fore-aft swing.
        fore_aft = math.sin(phi)

        # Half-wave lift/compression.
        # Same structure as rigid_microtaur_cpg_ctrl:
        #   lift = max(0, sin(phi + pi/2)) = max(0, cos(phi))
        lift = max(0.0, math.cos(phi))

        swing_raw = SWING_SIGN * SWING_RAW_AMP * fore_aft
        extension_raw = EXTENSION_RAW_BIAS + EXTENSION_RAW_AMP * lift

        raw[2 * leg_idx] = float(np.clip(swing_raw, -1.0, 1.0))
        raw[2 * leg_idx + 1] = float(np.clip(extension_raw, -1.0, 1.0))

    return raw


def raw_action_to_joint_targets(raw_action: np.ndarray) -> dict[str, float]:
    """Convert raw 8D swing/extension action to direct MuJoCo joint targets."""

    stand_a = np.array(MICROTAUR_STAND_A_OFFSETS, dtype=np.float64)
    stand_e = np.array(MICROTAUR_STAND_E_OFFSETS, dtype=np.float64)

    targets: dict[str, float] = {}

    for leg_idx in range(4):
        swing_raw = raw_action[2 * leg_idx]
        extension_raw = raw_action[2 * leg_idx + 1]

        swing = MINITAUR_PAIR_SWING_SCALE * swing_raw
        extension = MINITAUR_PAIR_EXTENSION_SCALE * extension_raw

        a_target = stand_a[leg_idx] + A_SIGN[leg_idx] * swing
        e_target = stand_e[leg_idx] + E_SIGN[leg_idx] * extension

        a_name = ACTION_JOINT_NAMES[2 * leg_idx]
        e_name = ACTION_JOINT_NAMES[2 * leg_idx + 1]

        targets[a_name] = float(a_target)
        targets[e_name] = float(e_target)

    return targets


def set_joint_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    value: float,
) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Could not find joint: {joint_name}")

    qpos_addr = model.jnt_qposadr[joint_id]
    data.qpos[qpos_addr] = value


def set_actuator_ctrl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_name: str,
    value: float,
) -> None:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)

    if actuator_id < 0:
        raise ValueError(
            f"Could not find actuator named {actuator_name}. "
            f"Check whether mjlab names the position actuator the same as the joint."
        )

    data.ctrl[actuator_id] = value


def initialize_stand(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put robot into its configured stand pose."""

    stand_targets: dict[str, float] = {}

    for leg_idx in range(4):
        a_name = ACTION_JOINT_NAMES[2 * leg_idx]
        e_name = ACTION_JOINT_NAMES[2 * leg_idx + 1]

        stand_targets[a_name] = MICROTAUR_STAND_A_OFFSETS[leg_idx]
        stand_targets[e_name] = MICROTAUR_STAND_E_OFFSETS[leg_idx]

    for joint_name, value in stand_targets.items():
        set_joint_qpos(model, data, joint_name, value)
        set_actuator_ctrl(model, data, joint_name, value)

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def print_debug(model: mujoco.MjModel) -> None:
    print("\nActuators:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  {i}: {name}")

    print("\nAction joints:")
    for joint_name in ACTION_JOINT_NAMES:
        print(f"  {joint_name}")


def main() -> None:
    print("Building Microtaur MuJoCo model through mjlab Entity...")

    robot = Entity(get_microtaur_robot_cfg())
    model = robot.spec.compile()
    data = mujoco.MjData(model)

    initialize_stand(model, data)

    print("Loaded model.")
    print(f"Action joints: {ACTION_JOINT_NAMES}")
    print(f"Stand A offsets: {MICROTAUR_STAND_A_OFFSETS}")
    print(f"Stand E offsets: {MICROTAUR_STAND_E_OFFSETS}")
    print(f"Swing scale: {MINITAUR_PAIR_SWING_SCALE}")
    print(f"Extension scale: {MINITAUR_PAIR_EXTENSION_SCALE}")
    print(f"A_SIGN: {A_SIGN}")
    print(f"E_SIGN: {E_SIGN}")
    print(f"BOUND_PHASE: {BOUND_PHASE}")
    print(f"FREQ_HZ: {FREQ_HZ}")
    print(f"SWING_RAW_AMP: {SWING_RAW_AMP}")
    print(f"EXTENSION_RAW_AMP: {EXTENSION_RAW_AMP}")

    print_debug(model)

    print("\nTuning notes:")
    print("  If robot moves backward: set SWING_SIGN = -1.0")
    print("  If it rocks without moving: increase SWING_RAW_AMP slightly")
    print("  If it hops/vibrates: reduce EXTENSION_RAW_AMP first")
    print("  If bound phase looks wrong: verify leg numbering/front-rear mapping")
    print("  Trot phase would be [0, pi, 0, pi]; bound phase is [0, 0, pi, pi]")

    sim_start_model = data.time

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            loop_start = time.time()

            t = data.time - sim_start_model

            ramp = min(max(t / RAMP_TIME_S, 0.0), 1.0)

            raw_action = make_bound_raw_action(t)
            raw_action *= ramp

            joint_targets = raw_action_to_joint_targets(raw_action)

            for actuator_name, target in joint_targets.items():
                set_actuator_ctrl(model, data, actuator_name, target)

            mujoco.mj_step(model, data)
            viewer.sync()

            if REALTIME:
                dt = model.opt.timestep
                elapsed = time.time() - loop_start
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)


if __name__ == "__main__":
    main()