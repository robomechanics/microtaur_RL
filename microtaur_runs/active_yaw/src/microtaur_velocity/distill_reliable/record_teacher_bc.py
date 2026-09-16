from __future__ import annotations

import argparse
from pathlib import Path

import torch

from microtaur_velocity.distill_reliable.dataset import DistillChunkWriter
from microtaur_velocity.distill_reliable.mjlab_utils import (
    flatten_done,
    get_action_dim,
    get_actor_obs_tensor,
    get_initial_obs,
    make_env_and_teacher,
    read_velocity_metrics,
    step_env,
)
from microtaur_velocity.distill_reliable.obs_prev_action import (
    AlignedStudentObsBuilder,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task_id")
    p.add_argument("--checkpoint-file", required=True)
    p.add_argument("--out-dir", default="logs/distill_reliable/bc_robust")
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--chunk-steps", type=int, default=500)
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--gait-freq-hz", type=float, default=1.45)
    p.add_argument("--action-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--clean-env", action="store_true")
    p.add_argument("--no-terminations", action="store_true")
    args = p.parse_args()

    robust = not args.clean_env
    env, _, teacher = make_env_and_teacher(
        args.task_id,
        args.checkpoint_file,
        args.num_envs,
        args.device,
        robust=robust,
        no_terminations=args.no_terminations,
    )
    env_u = env.unwrapped
    action_dim = get_action_dim(env_u)
    builder = AlignedStudentObsBuilder(
        env_u.num_envs,
        env_u.device,
        action_dim,
        args.gait_freq_hz,
    )
    writer = DistillChunkWriter(
        args.out_dir,
        "bc_teacher",
        args.chunk_steps,
        builder.layout,
        metadata={
            "mode": (
                "teacher_forced_bc_aligned_robust"
                if robust
                else "teacher_forced_bc_aligned_clean"
            ),
            "task_id": args.task_id,
            "checkpoint_file": str(Path(args.checkpoint_file)),
            "gait_freq_hz": args.gait_freq_hz,
            "num_envs": args.num_envs,
            "action_dim": action_dim,
            "obs_dim": builder.obs_dim,
            "robust_env": robust,
        },
        action_dim=action_dim,
    )

    # Keep the full TensorDict for the RSL-RL teacher.
    policy_obs = get_initial_obs(env)
    actor_tensor = get_actor_obs_tensor(policy_obs)

    print(
        f"[bc] robust={robust} action_dim={action_dim} "
        f"actor_obs_dim={actor_tensor.shape[-1]} "
        f"student_obs_dim={builder.obs_dim} layout={builder.layout}"
    )

    try:
        for step in range(args.steps):
            with torch.no_grad():
                # Student slice is extracted from the exact actor tensor inside
                # the same delayed/noisy TensorDict used by the teacher.
                actor_tensor = get_actor_obs_tensor(policy_obs)
                student_obs = builder.build(actor_tensor, env_u)

                teacher_action = torch.clamp(
                    torch.nan_to_num(
                        teacher(policy_obs),
                        nan=0.0,
                        posinf=args.action_clip,
                        neginf=-args.action_clip,
                    ),
                    -args.action_clip,
                    args.action_clip,
                )

                writer.record(
                    student_obs,
                    teacher_action,
                    executed_action=teacher_action,
                )

                policy_obs, _, done, _ = step_env(env, teacher_action)
                done = flatten_done(done)

            if args.log_every > 0 and step % args.log_every == 0:
                m = read_velocity_metrics(env_u)
                print(
                    f"[bc] {step}/{args.steps} "
                    f"vx={m.get('actual_forward_mean', 0):.3f} "
                    f"err={m.get('forward_error_abs_mean', 0):.3f} "
                    f"yaw_err={m.get('yaw_error_abs_mean', 0):.3f}"
                )
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
