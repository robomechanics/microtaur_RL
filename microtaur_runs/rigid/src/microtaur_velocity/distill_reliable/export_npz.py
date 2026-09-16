from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from microtaur_velocity.distill_reliable.policy import StudentPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, default="logs/distill_reliable/export/student_policy_weights.npz")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    student = StudentPolicy(
        obs_dim=int(ckpt.get("obs_dim", 35)),
        action_dim=int(ckpt.get("action_dim", 8)),
        hidden_dims=tuple(ckpt.get("hidden_dims", (64, 64))),
        activation=str(ckpt.get("activation", "elu")),
        output_tanh=True,
    )
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    student.load_state_dict(state_dict)
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

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)

    print(f"Exported NPZ: {out_path}")
    print(f"obs_dim={student.obs_dim}, action_dim={student.action_dim}, hidden_dims={student.hidden_dims}")
    print(f"parameters={student.num_parameters()}")


if __name__ == "__main__":
    main()
