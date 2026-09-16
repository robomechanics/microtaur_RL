from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import torch

try:
    from microtaur_velocity.microtaur_constants import (
        LEG_JOINT_NAMES,
        MICROTAUR_STAND_A_OFFSETS,
        MICROTAUR_STAND_E_OFFSETS,
        OBS_JOINT_NAMES,
    )
except Exception:  # pragma: no cover
    LEG_JOINT_NAMES = tuple(f"joint_{i}" for i in range(8))
    OBS_JOINT_NAMES = LEG_JOINT_NAMES
    MICROTAUR_STAND_A_OFFSETS = (0.0, 0.0, 0.0, 0.0)
    MICROTAUR_STAND_E_OFFSETS = (0.0, 0.0, 0.0, 0.0)

ACTION_JOINT_NAMES = tuple(LEG_JOINT_NAMES)
OBS_FALLBACK_JOINT_NAMES = tuple(OBS_JOINT_NAMES)


def _episode_phase(env, num_envs: int, device: torch.device, gait_freq_hz: float):
    if hasattr(env, "episode_length_buf"):
        step_dt = float(getattr(env, "step_dt", getattr(env, "dt", 0.02)))
        t = env.episode_length_buf.to(device=device, dtype=torch.float32) * step_dt
    else:
        t = torch.zeros(num_envs, device=device, dtype=torch.float32)
    phase = 2.0 * math.pi * float(gait_freq_hz) * t
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)


@dataclass(frozen=True)
class AlignedObsSpec:
    action_dim: int
    include_phase: bool = True

    @property
    def has_spine(self) -> bool:
        return self.action_dim == 9

    @property
    def actor_obs_dim(self) -> int:
        # base_lin(3)+base_ang(3)+gravity(3)+leg_q(8)+leg_qd(8)
        # +last_action(A)+command(3)+optional spine q/qd(2)
        return 28 + self.action_dim + (2 if self.has_spine else 0)

    @property
    def student_obs_dim(self) -> int:
        # Drop unmeasurable base linear velocity (3); add gait phase (2).
        return self.actor_obs_dim - 3 + (2 if self.include_phase else 0)

    @property
    def layout(self) -> str:
        spine = "+spine_pos(1)+spine_vel(1)" if self.has_spine else ""
        phase = "+phase_sin_cos(2)" if self.include_phase else ""
        return (
            f"{self.student_obs_dim}D aligned: projected_gravity(3)+base_ang_vel(3)+"
            f"joint_pos_rel(8)+joint_vel(8)+prev_action({self.action_dim})+"
            f"command(3){spine}{phase}"
        )


class AlignedStudentObsBuilder:
    """Slice the deployable student input from the teacher actor observation.

    This is the key latency/noise fix. The teacher actor observation already
    contains the environment's per-env 1-2 step sensor delay and configured
    observation noise. Slicing from that exact tensor means the student sees
    the same sensor sample (same age, same noise realization) used to produce
    the teacher label.

    Supported actor layouts are the current flat Microtaur environments:
      rigid: 36D actor -> 35D student
      active spine: 39D actor -> 38D student
    The code fails loudly if the layout changes rather than silently training
    on mis-indexed observations.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        action_dim: int,
        gait_freq_hz: float = 1.45,
        include_phase: bool = True,
    ):
        if int(action_dim) not in (8, 9):
            raise ValueError(f"Expected Microtaur action_dim 8 or 9, got {action_dim}.")
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.gait_freq_hz = float(gait_freq_hz)
        self.spec = AlignedObsSpec(int(action_dim), bool(include_phase))

    @property
    def obs_dim(self) -> int:
        return self.spec.student_obs_dim

    @property
    def layout(self) -> str:
        return self.spec.layout

    def build(self, actor_obs: torch.Tensor, env) -> torch.Tensor:
        obs = actor_obs.to(device=self.device, dtype=torch.float32)
        if obs.ndim != 2:
            raise RuntimeError(f"Expected actor obs [N,D], got {tuple(obs.shape)}")
        if obs.shape[-1] != self.spec.actor_obs_dim:
            raise RuntimeError(
                "Actor observation layout changed. "
                f"Expected {self.spec.actor_obs_dim}D for action_dim={self.spec.action_dim}, "
                f"got {obs.shape[-1]}D. Do not guess slices; update the aligned layout."
            )

        a = self.spec.action_dim
        # Current env actor term order is fixed in env_cfgs.py / saved env.yaml.
        base_ang = obs[:, 3:6]
        gravity = obs[:, 6:9]
        joint_pos = obs[:, 9:17]
        joint_vel = obs[:, 17:25]
        prev_action = obs[:, 25 : 25 + a]
        command = obs[:, 25 + a : 28 + a]

        parts = [gravity, base_ang, joint_pos, joint_vel, prev_action, command]
        if self.spec.has_spine:
            spine_start = 28 + a
            parts.extend((obs[:, spine_start : spine_start + 1], obs[:, spine_start + 1 : spine_start + 2]))
        if self.spec.include_phase:
            parts.append(_episode_phase(env, self.num_envs, self.device, self.gait_freq_hz))

        student_obs = torch.cat(parts, dim=-1)
        student_obs = torch.nan_to_num(student_obs, nan=0.0, posinf=0.0, neginf=0.0)
        if student_obs.shape[-1] != self.obs_dim:
            raise RuntimeError(f"Expected student obs {self.obs_dim}D, got {student_obs.shape[-1]}D")
        return student_obs


# ---------------------------------------------------------------------------
# Fresh-state builder retained only for deployment/debugging.
# It should NOT be used for robust teacher labeling in simulation.
# ---------------------------------------------------------------------------

def _stand_joint_vector(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    stand = []
    for leg_idx in range(4):
        stand.append(float(MICROTAUR_STAND_A_OFFSETS[leg_idx]))
        stand.append(float(MICROTAUR_STAND_E_OFFSETS[leg_idx]))
    return torch.tensor(stand, device=device, dtype=dtype)


def _possible_joint_names(robot) -> Sequence[str] | None:
    for obj in (robot, getattr(robot, "data", None), getattr(robot, "cfg", None)):
        if obj is None:
            continue
        for attr in ("joint_names", "actuated_joint_names"):
            names = getattr(obj, attr, None)
            if names is not None:
                return tuple(str(x) for x in names)
    return None


@lru_cache(maxsize=32)
def _indices_for_names(all_names: tuple[str, ...], wanted_names: tuple[str, ...]):
    name_to_idx = {name: i for i, name in enumerate(all_names)}
    if all(name in name_to_idx for name in wanted_names):
        return tuple(name_to_idx[name] for name in wanted_names)
    return None


def _select_leg_joint_tensor(robot, tensor: torch.Tensor, wanted_names: tuple[str, ...]):
    all_names = _possible_joint_names(robot)
    if all_names is not None:
        indices = _indices_for_names(tuple(all_names), wanted_names)
        if indices is None and wanted_names != OBS_FALLBACK_JOINT_NAMES:
            indices = _indices_for_names(tuple(all_names), OBS_FALLBACK_JOINT_NAMES)
        if indices is not None:
            idx = torch.as_tensor(indices, device=tensor.device, dtype=torch.long)
            return tensor.index_select(1, idx)
    return tensor[:, :8]


class PrevActionObsBuilder:
    """Legacy fresh-state builder for hardware-side or clean-state debugging."""

    def __init__(self, num_envs, device, gait_freq_hz=1.45, action_dim=8, include_phase=True):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.gait_freq_hz = float(gait_freq_hz)
        self.action_dim = int(action_dim)
        self.include_phase = bool(include_phase)
        self.prev_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)

    @property
    def obs_dim(self):
        return 25 + self.action_dim + (2 if self.include_phase else 0) + (2 if self.action_dim == 9 else 0)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.prev_action.zero_()
        elif env_ids.numel() > 0:
            self.prev_action[env_ids.to(self.device, dtype=torch.long)] = 0.0

    def push_action(self, action):
        self.prev_action = torch.clamp(
            torch.nan_to_num(action[:, : self.action_dim].detach().to(self.device), nan=0.0), -1.0, 1.0
        )

    def build(self, env):
        robot = env.scene["robot"]
        command = env.command_manager.get_command("twist")
        projected_gravity = robot.data.projected_gravity_b.to(self.device, torch.float32)
        base_ang_vel = robot.data.root_link_ang_vel_b.to(self.device, torch.float32)
        joint_pos = _select_leg_joint_tensor(robot, robot.data.joint_pos.to(self.device), ACTION_JOINT_NAMES)
        joint_vel = _select_leg_joint_tensor(robot, robot.data.joint_vel.to(self.device), ACTION_JOINT_NAMES)
        joint_pos = joint_pos - _stand_joint_vector(self.device, torch.float32).unsqueeze(0)
        parts = [projected_gravity, base_ang_vel, joint_pos, joint_vel, self.prev_action, command[:, :3].to(self.device)]
        if self.action_dim == 9:
            # Fresh-state fallback: assume the spine is the first non-leg actuated joint.
            all_names = _possible_joint_names(robot) or ()
            spine_idx = next((i for i, n in enumerate(all_names) if "spine" in n), None)
            if spine_idx is None:
                raise RuntimeError("9D fresh builder could not locate spine joint.")
            parts += [robot.data.joint_pos[:, spine_idx:spine_idx+1].to(self.device), robot.data.joint_vel[:, spine_idx:spine_idx+1].to(self.device)]
        if self.include_phase:
            parts.append(_episode_phase(env, self.num_envs, self.device, self.gait_freq_hz))
        obs = torch.nan_to_num(torch.cat(parts, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
        if obs.shape[-1] != self.obs_dim:
            raise RuntimeError(f"Expected {self.obs_dim}D fresh student obs, got {obs.shape[-1]}D")
        return obs
