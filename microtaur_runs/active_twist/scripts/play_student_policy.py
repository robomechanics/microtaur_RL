"""Play a distilled Microtaur student policy in MJLab.

Important fix vs the earlier script:
  - action history is pushed at the NEXT policy call, after the viewer/env had
    time to execute the previous action.
  - reset envs have their action history cleared from episode_length_buf.

This matches DistillRecorder timing:
  build obs -> action -> env.step(action) -> push action into history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from microtaur_velocity.distill.student_obs import DeployableStudentObsBuilder
from microtaur_velocity.distill.student_policy import StudentPolicy


@dataclass(frozen=True)
class PlayStudentConfig:
    task_id: str = "Mjlab-Velocity-Flat-microtaur_trot"
    student_checkpoint: str = "logs/distill_walk_small/student_policy_bc.pt"
    num_envs: int = 1
    device: str | None = None
    viewer: Literal["native", "viser"] = "native"
    gait_freq_hz: float = 1.45
    older_action_count: int = 3
    debug_every: int = 100


class StudentInferencePolicy:
    def __init__(self, env, checkpoint_path: str, gait_freq_hz: float, older_action_count: int, debug_every: int = 100):
        self.env = env
        self.device = env.unwrapped.device
        self.debug_every = int(debug_every)
        self.call_count = 0
        self.pending_executed_action: torch.Tensor | None = None

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        hidden_dims = tuple(ckpt.get("hidden_dims", (64, 64)))
        obs_dim = int(ckpt.get("obs_dim", 59))
        action_dim = int(ckpt.get("action_dim", 8))

        self.student = StudentPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
        ).to(self.device)

        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        self.student.load_state_dict(state_dict)
        self.student.eval()

        # Supports checkpoints where normalization is saved outside state_dict.
        if "obs_mean" in ckpt and "obs_std" in ckpt:
            self.student.set_normalization(
                ckpt["obs_mean"].to(self.device),
                ckpt["obs_std"].to(self.device),
            )

        self.obs_builder = DeployableStudentObsBuilder(
            num_envs=env.unwrapped.num_envs,
            device=self.device,
            gait_freq_hz=gait_freq_hz,
            older_action_count=older_action_count,
            include_phase=True,
        )

        print("[INFO] Loaded student policy")
        print(f"[INFO] checkpoint: {checkpoint_path}")
        print(f"[INFO] obs_dim: {obs_dim}")
        print(f"[INFO] action_dim: {action_dim}")
        print(f"[INFO] hidden_dims: {hidden_dims}")
        if "best_val_loss" in ckpt:
            print(f"[INFO] best_val_loss: {float(ckpt['best_val_loss']):.6f}")
        if "action_abs_mean" in ckpt:
            print(f"[INFO] train action_abs_mean: {float(ckpt['action_abs_mean']):.4f}")

    def _sync_action_history_after_previous_step(self) -> None:
        env_u = self.env.unwrapped

        # Previous returned action has now been executed by the viewer/env loop.
        # Push it now, before building the next observation.
        if self.pending_executed_action is not None:
            self.obs_builder.push_action(self.pending_executed_action)
            self.pending_executed_action = None

        # If env reset happened inside the viewer loop, clear the action memory
        # for those envs so old episode actions do not leak into reset states.
        self.obs_builder.reset_from_env_if_needed(env_u)

    def __call__(self, _rsl_obs) -> torch.Tensor:
        del _rsl_obs
        with torch.no_grad():
            self._sync_action_history_after_previous_step()

            env_u = self.env.unwrapped
            student_obs = self.obs_builder.build(env_u)
            action = self.student(student_obs)
            action = torch.clamp(
                torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0),
                -1.0,
                1.0,
            )

            # Do NOT push action here. The env has not executed it yet.
            self.pending_executed_action = action.detach()

            self.call_count += 1
            if self.debug_every > 0 and self.call_count % self.debug_every == 0:
                cmd = env_u.command_manager.get_command("twist")
                cmd_mean = float(cmd[:, 0].mean().item()) if cmd is not None else 0.0
                print(
                    f"[student] step={self.call_count} "
                    f"obs_abs={float(student_obs.abs().mean().item()):.4f} "
                    f"act_abs={float(action.abs().mean().item()):.4f} "
                    f"act_max={float(action.abs().max().item()):.4f} "
                    f"cmd_x_mean={cmd_mean:.3f}"
                )

            return action


def main():
    cfg = tyro.cli(PlayStudentConfig)
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(cfg.task_id, play=True)
    agent_cfg = load_rl_cfg(cfg.task_id)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.episode_length_s = int(1e9)
    env_cfg.observations["actor"].enable_corruption = False

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    policy = StudentInferencePolicy(
        env=env,
        checkpoint_path=cfg.student_checkpoint,
        gait_freq_hz=cfg.gait_freq_hz,
        older_action_count=cfg.older_action_count,
        debug_every=cfg.debug_every,
    )

    if cfg.viewer == "native":
        NativeMujocoViewer(env, policy).run()
    elif cfg.viewer == "viser":
        ViserPlayViewer(env, policy).run()
    else:
        raise RuntimeError(f"Unsupported viewer: {cfg.viewer}")

    env.close()


if __name__ == "__main__":
    main()
