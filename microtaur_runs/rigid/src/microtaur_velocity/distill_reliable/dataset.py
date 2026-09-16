from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch


def safe_action(action: torch.Tensor, action_dim: int | None = None) -> torch.Tensor:
    if action_dim is not None:
        action = action[:, : int(action_dim)]
    action = action.detach()
    return torch.clamp(torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)


def safe_obs(obs: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(obs.detach(), nan=0.0, posinf=0.0, neginf=0.0)


class DistillChunkWriter:
    def __init__(self, out_dir, prefix, chunk_steps, obs_layout, metadata=None, action_dim=None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = str(prefix)
        self.chunk_steps = int(chunk_steps)
        self.obs_layout = str(obs_layout)
        self.metadata = dict(metadata or {})
        self.action_dim = None if action_dim is None else int(action_dim)
        self.chunk_idx = 0
        self.step_count_in_chunk = 0
        self.total_record_calls = 0
        self.obs_chunks = []
        self.teacher_action_chunks = []
        self.executed_action_chunks = []
        (self.out_dir / f"{self.prefix}_metadata.json").write_text(
            json.dumps({"prefix": self.prefix, "obs_layout": self.obs_layout, **self.metadata}, indent=2),
            encoding="utf-8",
        )

    def record(self, student_obs, teacher_action, executed_action=None):
        if self.action_dim is None:
            self.action_dim = int(teacher_action.shape[-1])
        if teacher_action.shape[-1] != self.action_dim:
            raise RuntimeError(f"Teacher action width changed: expected {self.action_dim}, got {teacher_action.shape[-1]}")
        obs = safe_obs(student_obs).cpu()
        teacher = safe_action(teacher_action, self.action_dim).cpu()
        executed = teacher.clone() if executed_action is None else safe_action(executed_action, self.action_dim).cpu()
        self.obs_chunks.append(obs)
        self.teacher_action_chunks.append(teacher)
        self.executed_action_chunks.append(executed)
        self.step_count_in_chunk += 1
        self.total_record_calls += 1
        if self.step_count_in_chunk >= self.chunk_steps:
            self.flush()

    def flush(self):
        if not self.obs_chunks:
            return
        obs = torch.cat(self.obs_chunks, 0)
        teacher = torch.cat(self.teacher_action_chunks, 0)
        executed = torch.cat(self.executed_action_chunks, 0)
        payload = {
            "student_obs": obs,
            "teacher_action": teacher,
            "executed_action": executed,
            "obs_dim": int(obs.shape[-1]),
            "action_dim": int(teacher.shape[-1]),
            "student_obs_layout": self.obs_layout,
            "chunk_steps": self.chunk_steps,
            "chunk_idx": self.chunk_idx,
            "total_record_calls": self.total_record_calls,
            "obs_mean": obs.mean(0),
            "obs_std": obs.std(0).clamp_min(1e-4),
            "action_mean": teacher.mean(0),
            "action_std": teacher.std(0).clamp_min(1e-4),
            "action_abs_mean": teacher.abs().mean(),
            "executed_action_abs_mean": executed.abs().mean(),
            "metadata": self.metadata,
        }
        path = self.out_dir / f"{self.prefix}_chunk_{self.chunk_idx:04d}.pt"
        torch.save(payload, path)
        print(f"[distill] saved {path} obs={tuple(obs.shape)} actions={tuple(teacher.shape)}")
        self.obs_chunks.clear(); self.teacher_action_chunks.clear(); self.executed_action_chunks.clear()
        self.step_count_in_chunk = 0
        self.chunk_idx += 1

    def close(self):
        self.flush()


def find_chunk_files(dataset_dirs: Iterable[str | Path]) -> list[Path]:
    files = []
    for directory in dataset_dirs:
        d = Path(directory)
        files.extend(sorted(d.glob("*_chunk_*.pt")))
        files.extend(sorted(d.glob("teacher_distill*.pt")))
    out, seen = [], set()
    for f in files:
        r = f.resolve()
        if r not in seen:
            seen.add(r); out.append(f)
    return out


def load_dataset(dataset_dirs, max_samples=None, seed=0, require_obs_dim=None, require_action_dim=None):
    files = find_chunk_files(dataset_dirs)
    if not files:
        raise FileNotFoundError(f"No distillation chunks found in {list(dataset_dirs)}")
    obs_list, action_list, used = [], [], []
    expected_obs = require_obs_dim
    expected_action = require_action_dim
    for file in files:
        data = torch.load(file, map_location="cpu")
        obs = data["student_obs"].float()
        actions = data["teacher_action"].float()
        if expected_obs is not None and obs.shape[-1] != int(expected_obs):
            print(f"[skip] {file}: obs_dim={obs.shape[-1]} expected={expected_obs}"); continue
        if expected_action is not None and actions.shape[-1] != int(expected_action):
            print(f"[skip] {file}: action_dim={actions.shape[-1]} expected={expected_action}"); continue
        if obs_list and obs.shape[-1] != obs_list[0].shape[-1]:
            raise RuntimeError("Mixed observation dimensions in dataset; separate rigid and active-spine datasets.")
        if action_list and actions.shape[-1] != action_list[0].shape[-1]:
            raise RuntimeError("Mixed action dimensions in dataset; separate morphologies.")
        obs_list.append(obs); action_list.append(actions); used.append(file)
    if not obs_list:
        raise RuntimeError("No usable chunks after dimension filtering.")
    obs = torch.nan_to_num(torch.cat(obs_list, 0), nan=0.0, posinf=0.0, neginf=0.0)
    actions = torch.clamp(torch.nan_to_num(torch.cat(action_list, 0), nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    if max_samples and obs.shape[0] > max_samples:
        idx = torch.randperm(obs.shape[0], generator=torch.Generator().manual_seed(seed))[:max_samples]
        obs, actions = obs[idx], actions[idx]
    return obs, actions, used


def summarize_tensors(obs, actions):
    return {
        "samples": int(obs.shape[0]), "obs_dim": int(obs.shape[-1]), "action_dim": int(actions.shape[-1]),
        "obs_abs_mean": float(obs.abs().mean()), "action_abs_mean": float(actions.abs().mean()),
        "action_std_mean": float(actions.std(0).mean()),
        "action_abs_by_dim": [float(x) for x in actions.abs().mean(0)],
        "action_std_by_dim": [float(x) for x in actions.std(0)],
    }
