from __future__ import annotations

import argparse
from pathlib import Path

import torch

from microtaur_velocity.distill_reliable.dataset import find_chunk_files, load_dataset, summarize_tensors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dirs", type=str, nargs="+", default=["logs/distill_reliable/bc"])
    parser.add_argument("--require-obs-dim", type=int, default=None)
    parser.add_argument("--require-action-dim", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    files = find_chunk_files([Path(x) for x in args.dataset_dirs])
    print(f"Found {len(files)} chunk files")
    for f in files[:10]:
        data = torch.load(f, map_location="cpu")
        print(
            f"  {f} | obs={tuple(data['student_obs'].shape)} "
            f"action={tuple(data['teacher_action'].shape)} "
            f"layout={data.get('student_obs_layout', 'unknown')}"
        )
    if len(files) > 10:
        print(f"  ... {len(files) - 10} more")

    obs, actions, used = load_dataset(
        args.dataset_dirs,
        max_samples=args.max_samples,
        seed=0,
        require_obs_dim=args.require_obs_dim,
        require_action_dim=args.require_action_dim,
    )
    summary = summarize_tensors(obs, actions)

    print("\nDataset summary")
    for key, value in summary.items():
        if isinstance(value, list):
            print(f"  {key}: {[round(float(x), 4) for x in value]}")
        else:
            print(f"  {key}: {value}")

    if summary["obs_dim"] not in (35, 38):
        print(f"\n[WARN] Expected 35D rigid or 38D active-spine observation; got {summary['obs_dim']}D.")
    if summary["action_abs_mean"] < 0.03:
        print("\n[WARN] Teacher action magnitude is tiny. Student may learn to stand/freeze.")
    if summary["action_std_mean"] < 0.03:
        print("\n[WARN] Teacher action variation is tiny. Dataset may not contain a real gait.")

    print(f"\nUsable chunk files: {len(used)}")


if __name__ == "__main__":
    main()
