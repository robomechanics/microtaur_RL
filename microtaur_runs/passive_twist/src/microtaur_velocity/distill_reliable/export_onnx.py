from __future__ import annotations

import argparse
from pathlib import Path

import torch
from collections import Counter

import onnx

from microtaur_velocity.distill_reliable.policy import StudentPolicy


class DeployableStudent(torch.nn.Module):
    """Includes observation normalization inside the ONNX graph."""

    def __init__(self, student: StudentPolicy):
        super().__init__()
        self.student = student

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.student(obs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")

    student = StudentPolicy(
        obs_dim=int(ckpt.get("obs_dim", 35)),
        action_dim=int(ckpt.get("action_dim", 8)),
        hidden_dims=tuple(ckpt.get("hidden_dims", (64, 64))),
        activation=str(ckpt.get("activation", "elu")),
        output_tanh=True,
    )

    state_dict = ckpt.get(
        "model_state_dict",
        ckpt.get("state_dict", ckpt),
    )
    student.load_state_dict(state_dict)

    if "obs_mean" in ckpt and "obs_std" in ckpt:
        student.set_normalization(
            ckpt["obs_mean"],
            ckpt["obs_std"],
        )

    student.eval()
    model = DeployableStudent(student).eval()

    dummy_obs = torch.zeros(1, student.obs_dim, dtype=torch.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_obs,
        str(out_path),
        input_names=["observation"],
        output_names=["action"],
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)

    op_counts = Counter(node.op_type for node in onnx_model.graph.node)

    print("\nONNX operators:")
    for op_type, count in sorted(op_counts.items()):
        print(f"  {op_type}: {count}")

    bad_ops = {
        op: op_counts[op]
        for op in ("IsInf", "IsNaN")
        if op_counts[op] > 0
    }

    if bad_ops:
        raise RuntimeError(
            f"Unsupported finite-check nodes found in ONNX: {bad_ops}"
        )

    print("PASS: no IsInf or IsNaN nodes found.")
    print(f"Exported ONNX model: {out_path}")
    print(f"Input shape:  [batch, {student.obs_dim}]")
    print(f"Output shape: [batch, {student.action_dim}]")


if __name__ == "__main__":
    main()