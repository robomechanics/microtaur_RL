from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


def cfg_to_dict(cfg):
    try:
        return asdict(cfg)
    except Exception:
        if hasattr(cfg, "to_dict"):
            return cfg.to_dict()
        return cfg


def _first_obs_from_reset(result):
    """Return only the observation object from reset-like APIs."""
    if isinstance(result, tuple):
        if not result:
            raise RuntimeError("Environment reset returned an empty tuple.")
        return result[0]
    return result


def get_actor_obs_tensor(obs: Any) -> torch.Tensor:
    """Extract the concatenated actor tensor from MJLab/RSL-RL observations.

    RslRlVecEnvWrapper returns a TensorDict with batch_size=[num_envs]. Its
    `.shape` is therefore `(num_envs,)`, which is NOT the actor feature shape.

    The teacher policy should continue receiving the full TensorDict. This
    helper is only for code that explicitly needs the deployable actor tensor,
    such as aligned student-observation slicing.
    """
    obs = _first_obs_from_reset(obs)

    if isinstance(obs, torch.Tensor):
        if obs.ndim != 2:
            raise RuntimeError(
                "Expected a 2D actor observation tensor [N,D], "
                f"got tensor shape {tuple(obs.shape)}."
            )
        return obs

    # TensorDict and mapping-like objects both expose keys()/getitem.
    keys = None
    if hasattr(obs, "keys") and callable(obs.keys):
        try:
            keys = list(obs.keys())
        except Exception:
            keys = None

    if keys is not None:
        # Current MJLab velocity environments use the "actor" observation group.
        # "policy" is retained for compatibility with older wrappers/configs.
        for key in ("actor", "policy"):
            if key in keys:
                value = obs[key]
                if isinstance(value, torch.Tensor):
                    if value.ndim != 2:
                        raise RuntimeError(
                            f"Observation group {key!r} must be [N,D], "
                            f"got {tuple(value.shape)}."
                        )
                    return value
                return get_actor_obs_tensor(value)

        # A single-key container is safe to unwrap recursively.
        if len(keys) == 1:
            return get_actor_obs_tensor(obs[keys[0]])

        raise RuntimeError(
            "Could not find an actor observation group in observation container. "
            f"Available keys: {keys}. Expected 'actor' (current MJLab) or 'policy'."
        )

    raise RuntimeError(
        "Unsupported observation object for aligned student extraction: "
        f"{type(obs).__name__}. Expected torch.Tensor, TensorDict, or mapping."
    )


# Backwards-compatible name for code that explicitly wants the actor tensor.
# IMPORTANT: get_initial_obs() and step_env() intentionally do NOT call this,
# because the RSL-RL teacher needs the full TensorDict.
def unwrap_obs(obs):
    return get_actor_obs_tensor(obs)


def get_initial_obs(env):
    """Get observations while preserving the full RSL-RL TensorDict."""
    if hasattr(env, "get_observations"):
        return _first_obs_from_reset(env.get_observations())
    return _first_obs_from_reset(env.reset())


def step_env(env, action: torch.Tensor):
    """Step while preserving the full observation object for teacher inference."""
    result = env.step(action)
    if len(result) == 4:
        obs, reward, done, info = result
        return obs, reward, done.bool(), info
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, (terminated | truncated).bool(), info
    raise RuntimeError(f"Unexpected env.step output length: {len(result)}")


def flatten_done(done: torch.Tensor) -> torch.Tensor:
    done = done.detach().bool()
    if done.ndim > 1:
        done = done.reshape(done.shape[0], -1).any(dim=1)
    return done


def get_action_dim(env_u) -> int:
    """Infer the policy action width without assuming rigid vs active spine."""
    manager = getattr(env_u, "action_manager", None)
    if manager is not None:
        for attr in ("total_action_dim", "action_dim"):
            value = getattr(manager, attr, None)
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    pass
        action = getattr(manager, "action", None)
        if isinstance(action, torch.Tensor) and action.ndim >= 2:
            return int(action.shape[-1])

    cfg = getattr(env_u, "cfg", None)
    actions = getattr(cfg, "actions", None)
    if actions is not None:
        try:
            return 9 if "spine_pos" in actions else 8
        except Exception:
            pass
    raise RuntimeError("Could not infer Microtaur action dimension.")


def _make_robust_distill_cfg(task_id: str, num_envs: int):
    """Training-realistic env locked to the teacher's final curriculum stage."""
    from mjlab.tasks.registry import load_env_cfg

    train_cfg = load_env_cfg(task_id, play=False)
    final_cfg = load_env_cfg(task_id, play=True)

    train_cfg.scene.num_envs = int(num_envs)

    # Preserve training-time sensor corruption/delay.
    if hasattr(train_cfg, "observations") and "actor" in train_cfg.observations:
        train_cfg.observations["actor"].enable_corruption = True

    # Preserve randomized IK reset states.
    events = getattr(train_cfg, "events", None)
    if events is not None and "ik_consistent_reset" in events:
        events["ik_consistent_reset"].params["randomize"] = True

    # Distill a final teacher at its final action/command authority rather than
    # restarting curriculum stage 0.
    train_cfg.actions = copy.deepcopy(final_cfg.actions)

    if "twist" in final_cfg.commands:
        train_cfg.commands["twist"] = copy.deepcopy(final_cfg.commands["twist"])

    final_curriculum = getattr(final_cfg, "curriculum", None)
    train_curriculum = getattr(train_cfg, "curriculum", None)
    if final_curriculum is not None and train_curriculum is not None:
        if "microtaur_command_ranges" in final_curriculum:
            train_curriculum["microtaur_command_ranges"] = copy.deepcopy(
                final_curriculum["microtaur_command_ranges"]
            )

    return train_cfg


def _make_clean_cfg(task_id: str, num_envs: int):
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(task_id, play=True)
    cfg.scene.num_envs = int(num_envs)
    if hasattr(cfg, "observations") and "actor" in cfg.observations:
        cfg.observations["actor"].enable_corruption = False
    return cfg


def make_env_and_teacher(
    task_id: str,
    checkpoint_file: str | Path,
    num_envs: int,
    device: str,
    robust: bool = True,
    no_terminations: bool = False,
):
    from rsl_rl.runners import OnPolicyRunner
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

    env_cfg = (
        _make_robust_distill_cfg(task_id, num_envs)
        if robust
        else _make_clean_cfg(task_id, num_envs)
    )
    agent_cfg = load_rl_cfg(task_id)

    if no_terminations:
        env_cfg.terminations = {}

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    runner = runner_cls(env, cfg_to_dict(agent_cfg), device=device)
    runner.load(
        str(checkpoint_file),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    teacher_policy = runner.get_inference_policy(device=device)
    return env, agent_cfg, teacher_policy


def make_env_only(
    task_id: str,
    num_envs: int,
    device: str,
    robust: bool = True,
    no_terminations: bool = False,
):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg

    env_cfg = (
        _make_robust_distill_cfg(task_id, num_envs)
        if robust
        else _make_clean_cfg(task_id, num_envs)
    )
    agent_cfg = load_rl_cfg(task_id)

    if no_terminations:
        env_cfg.terminations = {}

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    return env, agent_cfg


def read_velocity_metrics(env_u) -> dict:
    robot = env_u.scene["robot"]
    lin_vel_b = robot.data.root_link_lin_vel_b
    ang_vel_b = robot.data.root_link_ang_vel_b
    cmd = env_u.command_manager.get_command("twist")

    out = {
        "actual_forward_mean": float(lin_vel_b[:, 0].mean().item()),
        "lateral_abs_mean": float(lin_vel_b[:, 1].abs().mean().item()),
        "yaw_rate_abs_mean": float(ang_vel_b[:, 2].abs().mean().item()),
    }
    if cmd is not None:
        out["cmd_forward_mean"] = float(cmd[:, 0].mean().item())
        out["forward_error_abs_mean"] = float(
            (lin_vel_b[:, 0] - cmd[:, 0]).abs().mean().item()
        )
        out["yaw_error_abs_mean"] = float(
            (ang_vel_b[:, 2] - cmd[:, 2]).abs().mean().item()
        )
    return out
