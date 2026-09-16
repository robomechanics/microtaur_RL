from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import tyro

from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from microtaur_velocity.distill_reliable.eval_student import load_student
from microtaur_velocity.distill_reliable.mjlab_utils import (
    get_action_dim,
    get_actor_obs_tensor,
    make_env_only,
)
from microtaur_velocity.distill_reliable.obs_prev_action import (
    AlignedStudentObsBuilder,
)


@dataclass(frozen=True)
class PlayStudentConfig:
    task_id: str = "Mjlab-Velocity-Flat-microtaur_trot"
    student_checkpoint: str = (
        "logs/distill_reliable/students/student_64x64_robust_dagger2.pt"
    )
    num_envs: int = 1
    device: str | None = None
    viewer: Literal["native", "viser"] = "native"
    gait_freq_hz: float = 1.45
    debug_every: int = 100

    # Robust is the default. --clean-env is only an ablation/debug mode.
    clean_env: bool = False
    no_terminations: bool = False


class ViewerStudentPolicy:
    """Viewer-compatible aligned robust student policy.

    Do not reconstruct the policy input from fresh robot.data here.

    The viewer passes the environment's actor observation into __call__.
    That tensor already contains the same configured observation corruption and
    per-environment sensor delay as the environment. We slice the deployable
    student observation directly from that actor tensor, exactly as the robust
    BC and DAgger collectors do.

    This also means prev_action comes from the environment actor observation.
    It must NOT be maintained a second time in this viewer wrapper.
    """

    def __init__(
        self,
        env,
        checkpoint_path: str,
        gait_freq_hz: float,
        debug_every: int,
        robust: bool,
    ):
        self.env = env
        self.env_u = env.unwrapped
        self.device = torch.device(self.env_u.device)
        self.debug_every = int(debug_every)
        self.robust = bool(robust)
        self.call_count = 0

        self.action_dim = get_action_dim(self.env_u)
        self.student = load_student(checkpoint_path, self.device)
        self.obs_builder = AlignedStudentObsBuilder(
            num_envs=self.env_u.num_envs,
            device=self.env_u.device,
            action_dim=self.action_dim,
            gait_freq_hz=gait_freq_hz,
            include_phase=True,
        )

        if self.student.action_dim != self.action_dim:
            raise RuntimeError(
                "Student/environment action-width mismatch: "
                f"checkpoint={self.student.action_dim}, env={self.action_dim}."
            )
        if self.student.obs_dim != self.obs_builder.obs_dim:
            raise RuntimeError(
                "Student/environment observation-width mismatch: "
                f"checkpoint={self.student.obs_dim}, "
                f"expected={self.obs_builder.obs_dim}."
            )

        print("[INFO] Loaded robust Microtaur student")
        print(f"[INFO] checkpoint: {checkpoint_path}")
        print(f"[INFO] robust_env: {self.robust}")
        print(f"[INFO] obs_dim: {self.student.obs_dim}")
        print(f"[INFO] action_dim: {self.student.action_dim}")
        print(f"[INFO] hidden_dims: {self.student.hidden_dims}")
        print(f"[INFO] obs_layout: {self.obs_builder.layout}")

    def __call__(self, rsl_obs) -> torch.Tensor:
        with torch.no_grad():
            actor_obs = get_actor_obs_tensor(rsl_obs)
            if not isinstance(actor_obs, torch.Tensor):
                raise RuntimeError(
                    "Viewer did not provide a tensor actor observation; "
                    f"got {type(actor_obs).__name__}."
                )

            student_obs = self.obs_builder.build(actor_obs, self.env_u)
            action = self.student(student_obs)
            action = torch.clamp(
                torch.nan_to_num(
                    action,
                    nan=0.0,
                    posinf=1.0,
                    neginf=-1.0,
                ),
                -1.0,
                1.0,
            )

            self.call_count += 1
            if (
                self.debug_every > 0
                and self.call_count % self.debug_every == 0
            ):
                cmd = self.env_u.command_manager.get_command("twist")
                cmd_x = (
                    float(cmd[:, 0].mean().item())
                    if cmd is not None
                    else 0.0
                )
                cmd_yaw = (
                    float(cmd[:, 2].mean().item())
                    if cmd is not None
                    else 0.0
                )

                robot = self.env_u.scene["robot"]
                vx = float(
                    robot.data.root_link_lin_vel_b[:, 0].mean().item()
                )
                yaw = float(
                    robot.data.root_link_ang_vel_b[:, 2].mean().item()
                )

                print(
                    f"[student] step={self.call_count} "
                    f"robust={self.robust} "
                    f"vx={vx:.3f}/{cmd_x:.3f} "
                    f"yaw={yaw:.3f}/{cmd_yaw:.3f} "
                    f"obs_abs={float(student_obs.abs().mean()):.4f} "
                    f"act_abs={float(action.abs().mean()):.4f} "
                    f"act_max={float(action.abs().max()):.4f}"
                )

            return action


def main() -> None:
    cfg = tyro.cli(PlayStudentConfig)
    configure_torch_backends()

    device = cfg.device or (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    robust = not cfg.clean_env

    # Use the same helper as robust BC/DAgger/evaluation so the viewer does not
    # silently fall back to the old play=True + corruption-disabled setup.
    env, _agent_cfg = make_env_only(
        task_id=cfg.task_id,
        num_envs=cfg.num_envs,
        device=device,
        robust=robust,
        no_terminations=cfg.no_terminations,
    )

    policy = ViewerStudentPolicy(
        env=env,
        checkpoint_path=cfg.student_checkpoint,
        gait_freq_hz=cfg.gait_freq_hz,
        debug_every=cfg.debug_every,
        robust=robust,
    )

    try:
        if cfg.viewer == "native":
            NativeMujocoViewer(env, policy).run()
        elif cfg.viewer == "viser":
            ViserPlayViewer(env, policy).run()
        else:
            raise RuntimeError(f"Unsupported viewer: {cfg.viewer}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
