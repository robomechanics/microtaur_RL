from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _array_to_c(name: str, arr: np.ndarray) -> str:
    flat = arr.astype(np.float32).reshape(-1)
    values = ", ".join(f"{float(x):.9g}f" for x in flat)
    shape_comment = "x".join(str(x) for x in arr.shape)
    return f"// {name}: shape {shape_comment}\nstatic const float {name}[{flat.size}] = {{ {values} }};\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--out", type=str, default="logs/distill_reliable/export/student_policy_weights.h")
    parser.add_argument("--symbol-prefix", type=str, default="microtaur_student")
    args = parser.parse_args()

    data = np.load(args.npz)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    obs_dim = int(data["obs_dim"][0])
    action_dim = int(data["action_dim"][0])
    hidden_dims = [int(x) for x in data["hidden_dims"].tolist()]
    num_linear = int(data["num_layers"][0])

    lines = []
    guard = f"{args.symbol_prefix.upper()}_WEIGHTS_H"
    lines.append(f"#ifndef {guard}\n#define {guard}\n\n")
    lines.append("// Auto-generated from export_c_header.py\n")
    lines.append("// MLP inference order: normalize obs, Linear+ELU hidden layers, Linear+Tanh output.\n\n")
    lines.append(f"static const int {args.symbol_prefix}_obs_dim = {obs_dim};\n")
    lines.append(f"static const int {args.symbol_prefix}_action_dim = {action_dim};\n")
    lines.append(f"static const int {args.symbol_prefix}_num_linear_layers = {num_linear};\n")
    lines.append(f"static const int {args.symbol_prefix}_hidden_dims[{len(hidden_dims)}] = " + "{ " + ", ".join(map(str, hidden_dims)) + " };\n\n")

    for key in ["obs_mean", "obs_std"]:
        lines.append(_array_to_c(f"{args.symbol_prefix}_{key}", data[key]))

    for i in range(num_linear):
        lines.append(_array_to_c(f"{args.symbol_prefix}_W{i}", data[f"W{i}"]))
        lines.append(_array_to_c(f"{args.symbol_prefix}_b{i}", data[f"b{i}"]))

    lines.append(f"\n#endif  // {guard}\n")
    out_path.write_text("".join(lines), encoding="utf-8")

    print(f"Exported C header: {out_path}")
    print("This header contains weights only. Use your ESP32 code to run normalize -> dense/ELU -> dense/tanh.")


if __name__ == "__main__":
    main()
