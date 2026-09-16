from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.entity.entity import Entity
from microtaur_velocity.microtaur_constants import (
    ACTION_JOINT_NAMES,
    FOOT_SITE_NAMES,
    MICROTAUR_STAND_A_OFFSETS,
    MICROTAUR_STAND_E_OFFSETS,
    get_microtaur_robot_cfg,
)


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

ROOT_BODY_NAME = "battery"

PERTURB = 0.12          # rad actuator target perturbation
SETTLE_STEPS = 500      # steps to let actuator/constraints settle
GRAVITY_OFF = True      # isolate kinematics from falling
PRINT_ALL_ACTUATORS = True


@dataclass
class LegResponse:
    leg_idx: int
    mode_name: str
    da: float
    de: float
    dx: float
    dy: float
    dz: float


# ---------------------------------------------------------------------
# MuJoCo helpers
# ---------------------------------------------------------------------

def joint_qpos_addr(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name,
    )
    if joint_id < 0:
        raise ValueError(f"Missing joint: {joint_name}")

    return int(model.jnt_qposadr[joint_id])


def actuator_id(model: mujoco.MjModel, actuator_name: str) -> int:
    act_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_name,
    )
    if act_id < 0:
        raise ValueError(
            f"Missing actuator: {actuator_name}. "
            "Check whether actuator names match ACTION_JOINT_NAMES."
        )

    return int(act_id)


def site_id(model: mujoco.MjModel, site_name: str) -> int:
    sid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )
    if sid < 0:
        raise ValueError(f"Missing site: {site_name}")

    return int(sid)


def body_id(model: mujoco.MjModel, body_name: str) -> int:
    bid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )
    if bid < 0:
        raise ValueError(f"Missing body: {body_name}")

    return int(bid)


def set_actuator_ctrl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_name: str,
    value: float,
) -> None:
    data.ctrl[actuator_id(model, actuator_name)] = float(value)


def foot_pos_body_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_site_name: str,
) -> np.ndarray:
    sid = site_id(model, foot_site_name)
    bid = body_id(model, ROOT_BODY_NAME)

    foot_w = np.array(data.site_xpos[sid], dtype=np.float64)
    body_w = np.array(data.xpos[bid], dtype=np.float64)
    body_R = np.array(data.xmat[bid], dtype=np.float64).reshape(3, 3)

    return body_R.T @ (foot_w - body_w)


def step_settle(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    steps: int = SETTLE_STEPS,
) -> None:
    for _ in range(steps):
        mujoco.mj_step(model, data)


# ---------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------

def initialize_stand_qpos_and_ctrl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    """Set qpos and actuator controls to the configured stand pose."""

    for leg_idx in range(4):
        a_name = ACTION_JOINT_NAMES[2 * leg_idx]
        e_name = ACTION_JOINT_NAMES[2 * leg_idx + 1]

        a_stand = MICROTAUR_STAND_A_OFFSETS[leg_idx]
        e_stand = MICROTAUR_STAND_E_OFFSETS[leg_idx]

        data.qpos[joint_qpos_addr(model, a_name)] = a_stand
        data.qpos[joint_qpos_addr(model, e_name)] = e_stand

        set_actuator_ctrl(model, data, a_name, a_stand)
        set_actuator_ctrl(model, data, e_name, e_stand)

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def reset_to_settled_stand(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    """Reset simulation to stand and let constraints settle."""

    mujoco.mj_resetData(model, data)

    if GRAVITY_OFF:
        model.opt.gravity[:] = 0.0

    initialize_stand_qpos_and_ctrl(model, data)
    step_settle(model, data, steps=SETTLE_STEPS)


# ---------------------------------------------------------------------
# Dynamic actuator perturbation
# ---------------------------------------------------------------------

def perturb_leg_with_actuators(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    leg_idx: int,
    da: float,
    de: float,
) -> np.ndarray:
    """Perturb one leg by actuator targets and return its settled foot pos.

    This is different from directly changing qpos. It lets the position
    actuators and closed-chain equality constraints solve dynamically.
    """

    reset_to_settled_stand(model, data)

    a_name = ACTION_JOINT_NAMES[2 * leg_idx]
    e_name = ACTION_JOINT_NAMES[2 * leg_idx + 1]

    a_target = MICROTAUR_STAND_A_OFFSETS[leg_idx] + da
    e_target = MICROTAUR_STAND_E_OFFSETS[leg_idx] + de

    set_actuator_ctrl(model, data, a_name, a_target)
    set_actuator_ctrl(model, data, e_name, e_target)

    step_settle(model, data, steps=SETTLE_STEPS)

    return foot_pos_body_frame(model, data, FOOT_SITE_NAMES[leg_idx])


def print_debug(model: mujoco.MjModel) -> None:
    print("\nActuators:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            i,
        )
        print(f"  {i}: {name}")

    print("\nAction joints:")
    for joint_name in ACTION_JOINT_NAMES:
        print(f"  {joint_name}")

    print("\nFoot sites:")
    for foot_site_name in FOOT_SITE_NAMES:
        print(f"  {foot_site_name}")


# ---------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------

def main() -> None:
    robot = Entity(get_microtaur_robot_cfg())
    model = robot.spec.compile()
    data = mujoco.MjData(model)

    if GRAVITY_OFF:
        model.opt.gravity[:] = 0.0

    print("\n=== Dynamic Microtaur actuator diagnostic ===")
    print("This test perturbs actuator controls and steps MuJoCo.")
    print("It is better than direct qpos perturbation for closed-chain legs.\n")

    print(f"ACTION_JOINT_NAMES = {ACTION_JOINT_NAMES}")
    print(f"FOOT_SITE_NAMES    = {FOOT_SITE_NAMES}")
    print(f"STAND_A            = {MICROTAUR_STAND_A_OFFSETS}")
    print(f"STAND_E            = {MICROTAUR_STAND_E_OFFSETS}")
    print(f"PERTURB            = {PERTURB} rad")
    print(f"SETTLE_STEPS       = {SETTLE_STEPS}")
    print(f"GRAVITY_OFF        = {GRAVITY_OFF}")

    if PRINT_ALL_ACTUATORS:
        print_debug(model)

    # These four actuator-control combinations identify common/differential effects.
    modes = [
        ("a_plus_e", +PERTURB, +PERTURB),
        ("a_minus_e", +PERTURB, -PERTURB),
        ("minus_a_plus_e", -PERTURB, +PERTURB),
        ("minus_a_minus_e", -PERTURB, -PERTURB),
    ]

    all_responses: list[LegResponse] = []

    for leg_idx in range(4):
        reset_to_settled_stand(model, data)
        base = foot_pos_body_frame(model, data, FOOT_SITE_NAMES[leg_idx])

        print(f"\n--- Leg {leg_idx + 1} ---")
        print(f"stand foot body-frame xyz = {base}")

        for mode_name, da, de in modes:
            p = perturb_leg_with_actuators(
                model=model,
                data=data,
                leg_idx=leg_idx,
                da=da,
                de=de,
            )
            dp = p - base

            response = LegResponse(
                leg_idx=leg_idx,
                mode_name=mode_name,
                da=da,
                de=de,
                dx=float(dp[0]),
                dy=float(dp[1]),
                dz=float(dp[2]),
            )
            all_responses.append(response)

            print(
                f"{mode_name:16s} "
                f"da={da:+.3f} de={de:+.3f} -> "
                f"dx={dp[0]:+.5f}, dy={dp[1]:+.5f}, dz={dp[2]:+.5f}"
            )

    print("\n=== Suggested interpretation ===")

    for leg_idx in range(4):
        leg_responses = [r for r in all_responses if r.leg_idx == leg_idx]

        # Fore/aft mode: largest absolute dx.
        best_fore_aft = max(leg_responses, key=lambda r: abs(r.dx))

        # Lift mode: largest positive dz.
        best_lift = max(leg_responses, key=lambda r: r.dz)

        # Push/down mode: most negative dz.
        best_down = min(leg_responses, key=lambda r: r.dz)

        print(f"\nLeg {leg_idx + 1}:")
        print(
            f"  strongest fore/aft mode: {best_fore_aft.mode_name:16s} "
            f"dx={best_fore_aft.dx:+.5f}, dz={best_fore_aft.dz:+.5f}"
        )
        print(
            f"  strongest lift mode:     {best_lift.mode_name:16s} "
            f"dx={best_lift.dx:+.5f}, dz={best_lift.dz:+.5f}"
        )
        print(
            f"  strongest push/down mode:{best_down.mode_name:16s} "
            f"dx={best_down.dx:+.5f}, dz={best_down.dz:+.5f}"
        )

    print("\n=== What to use this for ===")
    print("If a mode gives +dz, that mode lifts the foot.")
    print("If a mode gives -dz, that mode pushes the foot down/extends stance.")
    print("If a mode gives large +/-dx, that mode moves the foot fore/aft.")
    print("")
    print("For Minitaur-style mapping, we want an action basis like:")
    print("  motor_a = stand_a + s_a * swing + l_a * lift")
    print("  motor_e = stand_e + s_e * swing + l_e * lift")
    print("")
    print("The dynamic output tells us the correct signs for each leg.")


if __name__ == "__main__":
    main()