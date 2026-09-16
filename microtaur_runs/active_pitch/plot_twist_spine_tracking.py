"""Plot active-twist spine command vs yaw command vs actual spine angle.

Runs a trained teacher policy headless in MJLab while forcing a clean yaw-command
sequence. Records, for env 0 at every control step:

  - commanded yaw rate [rad/s]
  - policy-requested spine target [deg]
  - delayed/applied spine target [deg]  (saved to CSV for reference)
  - actual spine joint position [deg]

Outputs:
  - CSV with the raw control-step traces
  - PNG time-series plot
  - segment summary printed to the terminal

Example:
  python -m microtaur_velocity.plot_twist_spine_tracking \
    Mjlab-Velocity-Flat-microtaur_velocity \
    --checkpoint-file logs/rsl_rl/.../model_4499.pt \
    --cmd-x 0.14

The default yaw sequence is:
  0.00, +0.15, +0.25, 0.00, -0.15, -0.25, 0.00 rad/s
with each command held for 3 seconds at the environment's actual control rate.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from microtaur_velocity.distill_reliable.mjlab_utils import (
    get_initial_obs,
    make_env_and_teacher,
    step_env,
    unwrap_obs,
)


SPINE_JOINT_NAME = "spine_joint_act"
SPINE_ACTION_NAME = "spine_pos"
COMMAND_NAME = "twist"


def parse_yaw_sequence(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("--yaw-sequence must contain at least one value.")
    return values


def force_velocity_command(env_u, cmd_x: float, cmd_yaw: float) -> None:
    """Write the exact command the teacher should see this control step."""
    cmd = env_u.command_manager.get_command(COMMAND_NAME)
    if cmd is None:
        raise RuntimeError(f"Command '{COMMAND_NAME}' was not found.")

    cmd[:, 0] = float(cmd_x)
    cmd[:, 1] = 0.0
    cmd[:, 2] = float(cmd_yaw)

    # Prevent the command term from immediately replacing the forced command.
    term = env_u.command_manager.get_term(COMMAND_NAME)
    if hasattr(term, "time_left"):
        term.time_left.fill_(1.0e9)


def refresh_actor_obs(env, previous_obs=None):
    """Rebuild actor observations after forcing the command.

    This is important because the teacher observation contains the velocity
    command. Reusing an observation built before force_velocity_command() would
    make the teacher act on a stale yaw command.
    """
    if hasattr(env, "get_observations"):
        return unwrap_obs(env.get_observations())

    env_u = env.unwrapped
    if hasattr(env_u, "get_observations"):
        return unwrap_obs(env_u.get_observations())

    if previous_obs is not None:
        return previous_obs

    return unwrap_obs(env.reset())


def resolve_spine_joint_id(env_u) -> int:
    robot = env_u.scene["robot"]
    ids, names = robot.find_joints([SPINE_JOINT_NAME], preserve_order=True)
    if len(ids) != 1 or tuple(names) != (SPINE_JOINT_NAME,):
        raise RuntimeError(
            f"Expected exactly one joint named '{SPINE_JOINT_NAME}', got {tuple(names)}."
        )
    return int(ids[0])


def scalar_env0(tensor: torch.Tensor) -> float:
    """Read env 0 / scalar from a tensor regardless of [N] or [N,1] shape."""
    value = tensor[0]
    if value.numel() != 1:
        value = value.reshape(-1)[0]
    return float(value.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task_id",
        type=str,
        help="Registered MJLab Microtaur task id.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        required=True,
        help="Teacher RSL-RL checkpoint.",
    )
    parser.add_argument(
        "--cmd-x",
        type=float,
        default=0.14,
        help="Forward velocity command held during the diagnostic [m/s].",
    )
    parser.add_argument(
        "--yaw-sequence",
        type=str,
        default="0.0,0.15,0.25,0.0,-0.15,-0.25,0.0",
        help="Comma-separated yaw commands [rad/s].",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=3.0,
        help="Seconds to hold each yaw command.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of environments. Trace is recorded from env 0.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--action-clip",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="logs/twist_spine_diagnostic",
    )
    parser.add_argument(
        "--no-terminations",
        action="store_true",
        help="Disable task terminations if supported by your helper.",
    )
    args = parser.parse_args()

    yaw_sequence = parse_yaw_sequence(args.yaw_sequence)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env, _agent_cfg, teacher_policy = make_env_and_teacher(
        task_id=args.task_id,
        checkpoint_file=args.checkpoint_file,
        num_envs=args.num_envs,
        device=args.device,
        play=True,
        no_terminations=args.no_terminations,
    )
    env_u = env.unwrapped

    spine_joint_id = resolve_spine_joint_id(env_u)
    spine_term = env_u.action_manager.get_term(SPINE_ACTION_NAME)

    required_attrs = ("requested_targets", "applied_targets")
    missing = [name for name in required_attrs if not hasattr(spine_term, name)]
    if missing:
        raise RuntimeError(
            f"Action term '{SPINE_ACTION_NAME}' is missing {missing}. "
            "This script expects the active-twist direct spine action."
        )

    # Use the real policy/control period from the built environment.
    dt = float(getattr(env_u, "step_dt", 1.0 / 60.0))
    if dt <= 0.0:
        raise RuntimeError(f"Invalid env step_dt={dt}.")
    control_hz = 1.0 / dt
    segment_steps = max(1, int(round(float(args.segment_seconds) / dt)))

    print(f"[diag] control dt: {dt:.6f} s ({control_hz:.2f} Hz)")
    print(f"[diag] segment length: {segment_steps} steps ~= {segment_steps * dt:.2f} s")
    print(f"[diag] yaw sequence: {yaw_sequence}")
    print(f"[diag] cmd_x: {args.cmd_x:.3f} m/s")

    rows: list[dict[str, float | int]] = []
    actor_obs = get_initial_obs(env)

    global_step = 0
    try:
        for segment_idx, cmd_yaw in enumerate(yaw_sequence):
            for segment_step in range(segment_steps):
                with torch.no_grad():
                    # 1) Force command.
                    force_velocity_command(env_u, args.cmd_x, cmd_yaw)

                    # 2) Rebuild observations so teacher sees THIS command.
                    actor_obs = refresh_actor_obs(env, previous_obs=actor_obs)

                    # 3) Teacher action.
                    action = teacher_policy(actor_obs)
                    action = torch.clamp(
                        torch.nan_to_num(
                            action,
                            nan=0.0,
                            posinf=args.action_clip,
                            neginf=-args.action_clip,
                        ),
                        -args.action_clip,
                        args.action_clip,
                    )

                    # 4) Execute one control step. The action manager now updates
                    #    requested_targets and delayed/applied_targets.
                    actor_obs, _reward, _done, _info = step_env(env, action)

                    # Keep the command fixed even if the command manager tried to
                    # resample internally during env.step().
                    force_velocity_command(env_u, args.cmd_x, cmd_yaw)

                    robot = env_u.scene["robot"]

                    requested_rad = scalar_env0(spine_term.requested_targets)
                    applied_rad = scalar_env0(spine_term.applied_targets)
                    actual_rad = float(robot.data.joint_pos[0, spine_joint_id].item())

                    rows.append(
                        {
                            "step": global_step,
                            "time_s": global_step * dt,
                            "segment": segment_idx,
                            "segment_step": segment_step,
                            "cmd_x_m_s": float(args.cmd_x),
                            "cmd_yaw_rad_s": float(cmd_yaw),
                            "spine_requested_rad": requested_rad,
                            "spine_applied_rad": applied_rad,
                            "spine_actual_rad": actual_rad,
                            "spine_requested_deg": math.degrees(requested_rad),
                            "spine_applied_deg": math.degrees(applied_rad),
                            "spine_actual_deg": math.degrees(actual_rad),
                        }
                    )
                    global_step += 1
    finally:
        env.close()

    if not rows:
        raise RuntimeError("No samples were collected.")

    csv_path = out_dir / "twist_spine_tracking.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    time_s = np.asarray([float(r["time_s"]) for r in rows])
    cmd_yaw = np.asarray([float(r["cmd_yaw_rad_s"]) for r in rows])
    requested_deg = np.asarray([float(r["spine_requested_deg"]) for r in rows])
    applied_deg = np.asarray([float(r["spine_applied_deg"]) for r in rows])
    actual_deg = np.asarray([float(r["spine_actual_deg"]) for r in rows])

    # ------------------------------------------------------------------
    # Main plot requested:
    #   commanded spine position vs commanded yaw vs actual spine position
    # ------------------------------------------------------------------
    fig, ax_spine = plt.subplots(figsize=(13, 6))

    line_req, = ax_spine.plot(
        time_s,
        requested_deg,
        label="Spine commanded (requested)",
        linewidth=1.5,
    )
    line_actual, = ax_spine.plot(
        time_s,
        actual_deg,
        label="Spine actual",
        linewidth=1.5,
    )
    # Applied target is useful for debugging the 1-2 step delay. Keep it subtle
    # but visible; it is also saved separately in the CSV.
    line_applied, = ax_spine.plot(
        time_s,
        applied_deg,
        label="Spine target after delay",
        linewidth=1.0,
        alpha=0.65,
    )

    ax_spine.set_xlabel("Time [s]")
    ax_spine.set_ylabel("Spine position [deg]")
    ax_spine.axhline(0.0, linewidth=0.8)
    ax_spine.grid(True, alpha=0.25)

    ax_yaw = ax_spine.twinx()
    line_yaw, = ax_yaw.plot(
        time_s,
        cmd_yaw,
        label="Commanded yaw rate",
        linewidth=1.4,
        linestyle="--",
    )
    ax_yaw.set_ylabel("Commanded yaw rate [rad/s]")

    lines = [line_req, line_applied, line_actual, line_yaw]
    ax_spine.legend(lines, [line.get_label() for line in lines], loc="upper right")

    ax_spine.set_title(
        "Active-twist spine tracking vs commanded yaw\n"
        f"vx={args.cmd_x:.3f} m/s, control={control_hz:.1f} Hz"
    )

    fig.tight_layout()
    png_path = out_dir / "twist_spine_tracking.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    # Segment summary makes persistent command bias obvious.
    print("\n[diag] Per-yaw-segment means:")
    print(
        f"{'yaw cmd':>9} | {'spine req':>10} | {'spine applied':>13} | "
        f"{'spine actual':>12} | {'track err':>9}"
    )
    print("-" * 67)

    for segment_idx, yaw_value in enumerate(yaw_sequence):
        mask = np.asarray([int(r["segment"]) == segment_idx for r in rows])

        # Ignore the first 0.5 s of each segment when enough samples exist so
        # the summary reflects quasi-steady behavior rather than the transition.
        segment_indices = np.flatnonzero(mask)
        settle_steps = min(
            int(round(0.5 / dt)),
            max(0, len(segment_indices) // 3),
        )
        if settle_steps > 0:
            segment_indices = segment_indices[settle_steps:]

        req_mean = float(np.mean(requested_deg[segment_indices]))
        app_mean = float(np.mean(applied_deg[segment_indices]))
        act_mean = float(np.mean(actual_deg[segment_indices]))
        err_mean = float(np.mean(np.abs(applied_deg[segment_indices] - actual_deg[segment_indices])))

        print(
            f"{yaw_value:+9.3f} | "
            f"{req_mean:+9.2f}° | "
            f"{app_mean:+12.2f}° | "
            f"{act_mean:+11.2f}° | "
            f"{err_mean:8.2f}°"
        )

    print(f"\n[diag] CSV:  {csv_path}")
    print(f"[diag] Plot: {png_path}")
    print(
        "\nInterpretation:\n"
        "  - requested stays biased at yaw=0 -> policy itself prefers a twisted spine\n"
        "  - requested returns to 0 but actual does not -> actuator/dynamics/trunk loading issue\n"
        "  - requested follows yaw sign strongly -> policy is using twist as a steering strategy\n"
        "  - requested oscillates around 0 with actual lag -> dynamic gait coordination, not static bias"
    )


if __name__ == "__main__":
    main()
