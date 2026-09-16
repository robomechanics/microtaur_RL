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

from microtaur_velocity.microtaur_config import MICRO_ROUGH_TERRAINS_CFG


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

USE_ROUGH_TERRAIN = os.environ.get("MICROTAUR_VIEW_ROUGH", "0").lower() in {
    "1",
    "true",
    "yes",
}

SPAWN_ROW = int(os.environ.get("MICROTAUR_SPAWN_ROW", "0"))
SPAWN_COL = int(os.environ.get("MICROTAUR_SPAWN_COL", "0"))

ROOT_Z_CLEARANCE = float(os.environ.get("MICROTAUR_INIT_Z", "0.075"))
ROOT_YAW_DEG = float(os.environ.get("MICROTAUR_TEST_YAW_DEG", "90.0"))
REALTIME = os.environ.get("MICROTAUR_VIEW_REALTIME", "1").lower() in {
    "1",
    "true",
    "yes",
}


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
    data.qpos[qpos_addr] = float(value)


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
    """Set stand qpos and initial actuator ctrl once.

    After this, the script does NOT overwrite data.ctrl again, so the MuJoCo
    GUI control sliders can move the actuators.
    """
    set_root_pose(model, data, spawn_pos)

    for leg_idx in range(4):
        a_name = LEG_JOINT_NAMES[2 * leg_idx]
        e_name = LEG_JOINT_NAMES[2 * leg_idx + 1]

        a_val = MICROTAUR_STAND_A_OFFSETS[leg_idx]
        e_val = MICROTAUR_STAND_E_OFFSETS[leg_idx]

        set_joint_qpos(model, data, a_name, a_val)
        set_joint_qpos(model, data, e_name, e_val)

        # Initial actuator targets only. These become the initial slider values.
        set_actuator_ctrl(model, data, a_name, a_val)
        set_actuator_ctrl(model, data, e_name, e_val)

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def print_debug(model: mujoco.MjModel) -> None:
    print("\nJoints:")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"  {i}: {name}")

    print("\nActuators:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  {i}: {name}")

    print(f"\nmodel.nu = {model.nu}")
    print("\nUse the MuJoCo GUI Control sliders. Do not move qpos joint sliders.")


# ---------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------

def build_model() -> tuple[mujoco.MjModel, mujoco.MjData, np.ndarray]:
    print("Building Microtaur MuJoCo model through MJLab Entity...")
    robot = Entity(get_microtaur_robot_cfg())
    spec = robot.spec

    spawn_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    if USE_ROUGH_TERRAIN:
        print("Adding MICRO_ROUGH_TERRAINS_CFG terrain...")
        terrain_cfg = copy.deepcopy(MICRO_ROUGH_TERRAINS_CFG)
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
        print("Using default flat XML/scene terrain only.")

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
    print_debug(model)

    if model.nu == 0:
        raise RuntimeError(
            "No actuators found. The MJLab Entity did not add actuators to the model."
        )

    print("\nLaunching MuJoCo viewer.")
    print("Open the Control panel/sliders to move each actuator.")
    print("This script will NOT overwrite data.ctrl after launch.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            loop_start = time.time()

            # Important:
            # Do not set data.ctrl here.
            # The GUI sliders control data.ctrl.

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