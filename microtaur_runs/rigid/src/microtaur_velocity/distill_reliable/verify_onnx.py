from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort
import torch

from microtaur_velocity.distill_reliable.eval_student import load_student


def _last_static_dim(shape) -> int | None:
    if not shape:
        return None
    value = shape[-1]
    return int(value) if isinstance(value, int) else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that an exported Microtaur ONNX student numerically matches "
            "the PyTorch checkpoint for either the 35D/8-action rigid policy or "
            "the 38D/9-action active-spine policy."
        )
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--num-tests", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    if args.num_tests <= 0:
        raise ValueError("--num-tests must be positive.")
    if args.atol <= 0.0:
        raise ValueError("--atol must be positive.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu")
    student = load_student(args.ckpt, device)
    student.eval()

    session = ort.InferenceSession(
        args.onnx,
        providers=["CPUExecutionProvider"],
    )

    if len(session.get_inputs()) != 1:
        raise RuntimeError(
            f"Expected exactly one ONNX input, found {len(session.get_inputs())}."
        )
    if len(session.get_outputs()) != 1:
        raise RuntimeError(
            f"Expected exactly one ONNX output, found {len(session.get_outputs())}."
        )

    onnx_input = session.get_inputs()[0]
    onnx_output = session.get_outputs()[0]
    input_name = onnx_input.name

    onnx_obs_dim = _last_static_dim(onnx_input.shape)
    onnx_action_dim = _last_static_dim(onnx_output.shape)

    if onnx_obs_dim is not None and onnx_obs_dim != student.obs_dim:
        raise RuntimeError(
            "ONNX/checkpoint observation width mismatch: "
            f"ONNX={onnx_obs_dim}, checkpoint={student.obs_dim}."
        )
    if onnx_action_dim is not None and onnx_action_dim != student.action_dim:
        raise RuntimeError(
            "ONNX/checkpoint action width mismatch: "
            f"ONNX={onnx_action_dim}, checkpoint={student.action_dim}."
        )

    if (student.obs_dim, student.action_dim) not in ((35, 8), (38, 9)):
        print(
            "[WARN] Checkpoint is not one of the current standard Microtaur "
            f"layouts: obs/action={student.obs_dim}/{student.action_dim}."
        )

    # Test around the actual training distribution rather than arbitrary raw
    # N(0,1) observations. StudentPolicy normalizes internally, so drawing
    # obs_mean + obs_std*N(0,1) exercises approximately unit-scale normalized
    # inputs without needlessly saturating the final tanh.
    obs_mean = student.obs_mean.detach().cpu().reshape(1, -1)
    obs_std = student.obs_std.detach().cpu().reshape(1, -1)

    max_error = 0.0
    mean_error_sum = 0.0
    nonfinite_count = 0

    # Include the normalization mean itself as a deterministic sanity case.
    test_observations = [obs_mean.clone()]
    for _ in range(max(0, args.num_tests - 1)):
        test_observations.append(
            obs_mean + obs_std * torch.randn(1, student.obs_dim)
        )

    for obs in test_observations:
        obs = obs.to(dtype=torch.float32)

        with torch.no_grad():
            torch_action = student(obs).cpu().numpy()

        onnx_action = session.run(
            None,
            {input_name: obs.numpy()},
        )[0]

        if not np.isfinite(torch_action).all() or not np.isfinite(onnx_action).all():
            nonfinite_count += 1
            continue

        if tuple(onnx_action.shape) != tuple(torch_action.shape):
            raise RuntimeError(
                "ONNX/PyTorch output-shape mismatch: "
                f"ONNX={onnx_action.shape}, PyTorch={torch_action.shape}."
            )

        error = np.abs(torch_action - onnx_action)
        max_error = max(max_error, float(error.max()))
        mean_error_sum += float(error.mean())

    if nonfinite_count:
        raise RuntimeError(
            f"Found non-finite PyTorch/ONNX outputs in {nonfinite_count} tests."
        )

    mean_error = mean_error_sum / len(test_observations)

    print("Microtaur ONNX verification")
    print(f"  checkpoint obs_dim:    {student.obs_dim}")
    print(f"  checkpoint action_dim: {student.action_dim}")
    print(f"  tests:                 {len(test_observations)}")
    print(f"  maximum abs error:     {max_error:.8f}")
    print(f"  mean abs error:        {mean_error:.8f}")
    print(f"  tolerance:             {args.atol:.8f}")

    if max_error > args.atol:
        raise RuntimeError(
            "FAIL: ONNX differs from PyTorch more than the requested tolerance."
        )

    print("PASS: ONNX closely matches the PyTorch student.")


if __name__ == "__main__":
    main()
