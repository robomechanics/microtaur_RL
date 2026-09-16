from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from microtaur_velocity.distill.student_policy import StudentPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="logs/distill/student_policy_bc.pt")
    parser.add_argument("--out", type=str, default="logs/distill/student_policy_weights.npz")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    student = StudentPolicy(
        obs_dim=int(ckpt.get("obs_dim", 59)),
        action_dim=int(ckpt.get("action_dim", 8)),
        hidden_dims=tuple(ckpt.get("hidden_dims", (64, 64))),
    )
    student.load_state_dict(ckpt["model_state_dict"])
    if "obs_mean" in ckpt and "obs_std" in ckpt:
        student.set_normalization(ckpt["obs_mean"], ckpt["obs_std"])
    student.eval()

    arrays = {
        "obs_mean": student.obs_mean.detach().cpu().numpy().astype(np.float32),
        "obs_std": student.obs_std.detach().cpu().numpy().astype(np.float32),
        "obs_dim": np.array([student.obs_dim], dtype=np.int32),
        "action_dim": np.array([student.action_dim], dtype=np.int32),
        "num_layers": np.array([len(student.hidden_dims) + 1], dtype=np.int32),
        "hidden_dims": np.array(student.hidden_dims, dtype=np.int32),
    }

    linear_idx = 0
    for module in student.net:
        if isinstance(module, torch.nn.Linear):
            arrays[f"W{linear_idx}"] = module.weight.detach().cpu().numpy().astype(np.float32)
            arrays[f"b{linear_idx}"] = module.bias.detach().cpu().numpy().astype(np.float32)
            linear_idx += 1

    np.savez(out_path, **arrays)
    print(f"Exported student weights to: {out_path}")
    print("For ESP32, convert these arrays to a C header or int8 model next.")


if __name__ == "__main__":
    main()
