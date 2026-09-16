"""Patch template for playing a distilled student inside an existing MJLab loop."""

from __future__ import annotations

import torch

from microtaur_velocity.distill.student_obs import DeployableStudentObsBuilder
from microtaur_velocity.distill.student_policy import StudentPolicy


def load_student_for_env(env, ckpt_path="logs/distill/student_policy_bc.pt", gait_freq_hz=1.45):
    ckpt = torch.load(ckpt_path, map_location=env.device)
    student = StudentPolicy(
        obs_dim=int(ckpt.get("obs_dim", 59)),
        action_dim=int(ckpt.get("action_dim", 8)),
        hidden_dims=tuple(ckpt.get("hidden_dims", (64, 64))),
    ).to(env.device)
    student.load_state_dict(ckpt["model_state_dict"])
    if "obs_mean" in ckpt and "obs_std" in ckpt:
        student.set_normalization(ckpt["obs_mean"].to(env.device), ckpt["obs_std"].to(env.device))
    student.eval()

    obs_builder = DeployableStudentObsBuilder(
        num_envs=env.num_envs,
        device=env.device,
        gait_freq_hz=gait_freq_hz,
        older_action_count=3,
        include_phase=True,
    )
    return student, obs_builder


def student_action(env, student, obs_builder):
    # Build obs BEFORE action is executed.
    with torch.no_grad():
        obs = obs_builder.build(env)
        action = student(obs)
        action = torch.clamp(torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    return action


# Correct rollout timing:
#
# student, obs_builder = load_student_for_env(env, "logs/distill/student_policy_bc.pt")
# obs, _ = env.reset()
# while True:
#     action = student_action(env, student, obs_builder)
#     obs, reward, terminated, truncated, info = env.step(action)
#
#     # Push AFTER env.step, because now this action has actually been executed.
#     obs_builder.push_action(action)
#
#     done = terminated | truncated
#     if done.ndim > 1:
#         done = done.reshape(done.shape[0], -1).any(dim=1)
#     env_ids = torch.nonzero(done, as_tuple=False).flatten()
#     if env_ids.numel() > 0:
#         obs_builder.reset(env_ids.to(env.device))
