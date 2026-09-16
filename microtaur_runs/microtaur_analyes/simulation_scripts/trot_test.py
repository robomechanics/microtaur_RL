from __future__ import annotations

import copy
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

from mjlab.entity.entity import Entity
from mjlab.terrains.terrain_generator import TerrainGenerator

from microtaur_velocity.microtaur_constants import (
    LEG_JOINT_NAMES,
    MICROTAUR_STAND_A_OFFSETS,
    MICROTAUR_STAND_E_OFFSETS,
    get_microtaur_robot_cfg,
)

from microtaur_velocity.microtaur_config import (
    MICRO_ROUGH_TERRAINS_CFG,
)


# ---------------------------------------------------------------------
# Script settings
# ---------------------------------------------------------------------

GAIT_TYPE = "trot"

# Trot timing.
GAIT_FREQ_HZ = float(os.environ.get("MICROTAUR_TEST_FREQ_HZ", "1.45"))
RAMP_TIME_S = float(os.environ.get("MICROTAUR_TEST_RAMP_TIME_S", "2.0"))
REALTIME = os.environ.get("MICROTAUR_TEST_REALTIME", "1").lower() in {
    "1",
    "true",
    "yes",
}

# Action amplitudes in raw policy-action coordinates [-1, 1].
#
# Action vector:
#   [
#     leg1_swing, leg1_lift,
#     leg2_swing, leg2_lift,
#     leg3_swing, leg3_lift,
#     leg4_swing, leg4_lift,
#   ]
SWING_RAW_AMP = float(os.environ.get("MICROTAUR_TEST_SWING_RAW_AMP", "0.75"))
LIFT_RAW_AMP = float(os.environ.get("MICROTAUR_TEST_LIFT_RAW_AMP", "0.85"))

# Actual motor target scale.
#
# These should match your training action config:
#   MicrotaurSwingLiftActionCfg(swing_scale=..., lift_scale=...)
SWING_SCALE = float(os.environ.get("MICROTAUR_TEST_SWING_SCALE", "0.26"))
LIFT_SCALE = float(os.environ.get("MICROTAUR_TEST_LIFT_SCALE", "0.21"))

# Positive lift_raw should mean "lift the leg" in this script.
#
# For the current Microtaur a/e geometry, the old working scripted trot used
# negative differential during swing, so this default converts:
#
#   positive lift action -> negative internal lift/diff effect
#
# If the lift motion is inverted in the viewer, run with:
#   $env:MICROTAUR_TEST_LIFT_DIRECTION="1"
LIFT_DIRECTION = float(os.environ.get("MICROTAUR_TEST_LIFT_DIRECTION", "-1.0"))

# Mirror signs for physical legs:
#   leg1 +, leg2 -, leg3 -, leg4 +
LEG_SIGNS = np.array([+1.0, -1.0, -1.0, +1.0], dtype=np.float64)

# Trot phase:
#   leg1 + leg3 together
#   leg2 + leg4 together
#   diagonal pairs opposite
TROT_PHASE_OFFSETS = np.array(
    [
        0.0,
        np.pi,
        0.0,
        np.pi,
    ],
    dtype=np.float64,
)

# Spawn / terrain.
USE_ROUGH_TERRAIN = os.environ.get("MICROTAUR_TEST_ROUGH", "1").lower() in {
    "1",
    "true",
    "yes",
}

SPAWN_ROW = int(os.environ.get("MICROTAUR_SPAWN_ROW", "0"))
SPAWN_COL = int(os.environ.get("MICROTAUR_SPAWN_COL", "0"))

ROOT_Z_CLEARANCE = float(os.environ.get("MICROTAUR_INIT_Z", "0.075"))

USE_TERRAIN_CURRICULUM_LAYOUT = os.environ.get(
    "MICROTAUR_SCRIPT_TERRAIN_CURRICULUM",
    "1",
).lower() in {"1", "true", "yes"}

# Initial yaw. Default is 90 deg left, matching your previous request.
ROOT_YAW_DEG = float(os.environ.get("MICROTAUR_TEST_YAW_DEG", "90.0"))


# ---------------------------------------------------------------------
# Gait helpers
# ---------------------------------------------------------------------

def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def make_trot_swing_lift_action(t: float) -> np.ndarray:
    """Return raw 8D swing/lift action.

    Output order:

      [
        leg1_swing, leg1_lift,
        leg2_swing, leg2_lift,
        leg3_swing, leg3_lift,
        leg4_swing, leg4_lift,
      ]

    swing:
      smooth forward/back leg sweep

    lift:
      positive during swing phase, near zero during stance phase
    """
    base_phase = 2.0 * np.pi * GAIT_FREQ_HZ * t

    action = np.zeros(8, dtype=np.float64)

    for leg_idx in range(4):
        phi = (base_phase + TROT_PHASE_OFFSETS[leg_idx]) % (2.0 * np.pi)

        if phi < np.pi:
            # Swing phase.
            u = phi / np.pi
            s = smoothstep(u)

            # Sweep from back -> front through swing.
            swing_shape = -1.0 + 2.0 * s

            # Lift is positive during swing, strongest mid-swing.
            lift_shape = np.sin(np.pi * u)
        else:
            # Stance phase.
            u = (phi - np.pi) / np.pi
            s = smoothstep(u)

            # Sweep from front -> back during stance.
            swing_shape = 1.0 - 2.0 * s

            # No intentional lift during stance.
            lift_shape = 0.0

        swing_raw = SWING_RAW_AMP * swing_shape
        lift_raw = LIFT_RAW_AMP * lift_shape

        action[2 * leg_idx] = float(np.clip(swing_raw, -1.0, 1.0))
        action[2 * leg_idx + 1] = float(np.clip(lift_raw, -1.0, 1.0))

    return action


# ---------------------------------------------------------------------
# Action conversion
# ---------------------------------------------------------------------

def swing_lift_action_to_joint_targets(action: np.ndarray) -> dict[str, float]:
    """Convert raw swing/lift action to MuJoCo actuator targets.

    Raw action order:

      [
        leg1_swing, leg1_lift,
        leg2_swing, leg2_lift,
        leg3_swing, leg3_lift,
        leg4_swing, leg4_lift,
      ]

    Internal motor mapping:

      swing = swing_scale * swing_raw
      lift  = lift_direction * lift_scale * lift_raw

      a_target = stand_a + sign * swing - sign * lift
      e_target = stand_e + sign * swing + sign * lift
    """
    if action.shape[0] != 8:
        raise ValueError(f"Expected action shape (8,), got {action.shape}")

    action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
    action = np.clip(action, -1.0, 1.0)

    stand_a = np.array(MICROTAUR_STAND_A_OFFSETS, dtype=np.float64)
    stand_e = np.array(MICROTAUR_STAND_E_OFFSETS, dtype=np.float64)

    targets: dict[str, float] = {}

    for leg_idx in range(4):
        swing_raw = action[2 * leg_idx]
        lift_raw = action[2 * leg_idx + 1]

        swing = SWING_SCALE * swing_raw
        lift = LIFT_DIRECTION * LIFT_SCALE * lift_raw

        sign = LEG_SIGNS[leg_idx]

        a_target = stand_a[leg_idx] + sign * swing - sign * lift
        e_target = stand_e[leg_idx] + sign * swing + sign * lift

        a_name = LEG_JOINT_NAMES[2 * leg_idx]
        e_name = LEG_JOINT_NAMES[2 * leg_idx + 1]

        targets[a_name] = float(a_target)
        targets[e_name] = float(e_target)

    return targets


# ---------------------------------------------------------------------
# MuJoCo helpers
# ---------------------------------------------------------------------

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
    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_name,
    )

    if actuator_id < 0:
        raise ValueError(
            f"Could not find actuator named {actuator_name}. "
            f"Check whether the actuator is named the same as the joint."
        )

    data.ctrl[actuator_id] = float(value)


def set_root_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pos: np.ndarray,
    yaw_deg: float = ROOT_YAW_DEG,
) -> None:
    """Set first freejoint root pose to pos with yaw rotation."""
    free_joint_id = None

    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            free_joint_id = joint_id
            break

    if free_joint_id is None:
        print("WARNING: no free joint found; could not set root pose.")
        return

    qpos_addr = model.jnt_qposadr[free_joint_id]

    data.qpos[qpos_addr + 0] = float(pos[0])
    data.qpos[qpos_addr + 1] = float(pos[1])
    data.qpos[qpos_addr + 2] = float(pos[2])

    yaw = np.deg2rad(yaw_deg)

    # MuJoCo freejoint quaternion order: w, x, y, z
    data.qpos[qpos_addr + 3] = np.cos(yaw / 2.0)
    data.qpos[qpos_addr + 4] = 0.0
    data.qpos[qpos_addr + 5] = 0.0
    data.qpos[qpos_addr + 6] = np.sin(yaw / 2.0)


def initialize_stand(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spawn_pos: np.ndarray,
) -> None:
    """Put robot into stand pose at the selected spawn position."""
    set_root_pose(model, data, spawn_pos)

    stand_targets: dict[str, float] = {}

    for leg_idx in range(4):
        a_name = LEG_JOINT_NAMES[2 * leg_idx]
        e_name = LEG_JOINT_NAMES[2 * leg_idx + 1]

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

    print("\nLeg action joints:")
    for joint_name in LEG_JOINT_NAMES:
        print(f"  {joint_name}")

    print("\nAction space:")
    print("  0: leg1_swing")
    print("  1: leg1_lift")
    print("  2: leg2_swing")
    print("  3: leg2_lift")
    print("  4: leg3_swing")
    print("  5: leg3_lift")
    print("  6: leg4_swing")
    print("  7: leg4_lift")


# ---------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------

def build_model() -> tuple[mujoco.MjModel, mujoco.MjData, np.ndarray]:
    """Build robot, optionally with rough terrain."""
    print("Building Microtaur MuJoCo model through mjlab Entity...")
    robot = Entity(get_microtaur_robot_cfg())
    spec = robot.spec

    spawn_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    if USE_ROUGH_TERRAIN:
        print("Adding MICRO_ROUGH_TERRAINS_CFG into the MuJoCo spec...")
        terrain_cfg = copy.deepcopy(MICRO_ROUGH_TERRAINS_CFG)

        if USE_TERRAIN_CURRICULUM_LAYOUT:
            terrain_cfg.curriculum = True

        terrain_generator = TerrainGenerator(terrain_cfg, device="cpu")
        terrain_generator.compile(spec)

        terrain_origins = terrain_generator.terrain_origins
        if terrain_origins is not None:
            num_rows, num_cols = terrain_origins.shape[:2]
            row = int(np.clip(SPAWN_ROW, 0, num_rows - 1))
            col = int(np.clip(SPAWN_COL, 0, num_cols - 1))
            spawn_origin = np.array(terrain_origins[row, col], dtype=np.float64)

            print(f"Selected terrain spawn row={row}, col={col}")
            print(f"Terrain origin: {spawn_origin}")
    else:
        print("Using flat/default model terrain only.")

    spawn_pos = spawn_origin.copy()
    spawn_pos[2] += ROOT_Z_CLEARANCE

    print("Compiling MuJoCo model...")
    model = spec.compile()
    data = mujoco.MjData(model)

    return model, data, spawn_pos


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    model, data, spawn_pos = build_model()

    initialize_stand(model, data, spawn_pos)

    print("\nLoaded model.")
    print(f"GAIT_TYPE: {GAIT_TYPE}")
    print(f"GAIT_FREQ_HZ: {GAIT_FREQ_HZ}")
    print(f"SWING_RAW_AMP: {SWING_RAW_AMP}")
    print(f"LIFT_RAW_AMP: {LIFT_RAW_AMP}")
    print(f"SWING_SCALE: {SWING_SCALE}")
    print(f"LIFT_SCALE: {LIFT_SCALE}")
    print(f"LIFT_DIRECTION: {LIFT_DIRECTION}")
    print(f"RAMP_TIME_S: {RAMP_TIME_S}")
    print(f"REALTIME: {REALTIME}")
    print(f"USE_ROUGH_TERRAIN: {USE_ROUGH_TERRAIN}")
    print(f"ROOT_YAW_DEG: {ROOT_YAW_DEG}")
    print(f"Leg joints: {LEG_JOINT_NAMES}")
    print(f"Stand A offsets: {MICROTAUR_STAND_A_OFFSETS}")
    print(f"Stand E offsets: {MICROTAUR_STAND_E_OFFSETS}")
    print(f"LEG_SIGNS: {LEG_SIGNS}")

    print_debug(model)

    sim_start_time = data.time

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            loop_start = time.time()

            t = data.time - sim_start_time
            ramp = min(max(t / RAMP_TIME_S, 0.0), 1.0)

            action = make_trot_swing_lift_action(t)
            action *= ramp

            joint_targets = swing_lift_action_to_joint_targets(action)

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