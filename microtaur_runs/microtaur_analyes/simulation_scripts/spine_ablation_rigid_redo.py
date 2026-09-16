"""One-off: redo rigid's Stage 1 C0 cells that failed under the old in-process
batch driver's module-caching bug (see spine_ablation_batch.py docstring)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spine_ablation_batch import run_cell, OUT  # noqa: E402

import pandas as pd  # noqa: E402

cells = [("rigid", terr, "c0", 100.0, seed) for terr in ("flat", "steps") for seed in range(8)]
rows = []
for k, (variant, terrain, condition, alpha_pct, seed) in enumerate(cells):
    summary = run_cell(variant, terrain, condition, alpha_pct, seed)
    if summary.get("error"):
        print(f"FAILED {variant}/{terrain}/seed{seed}: {summary['error'][:300]}")
    rows.append(summary)
    print(f"{k+1}/{len(cells)} rigid {terrain} c0 seed{seed} done")
    pd.DataFrame(rows).to_csv(OUT / "stage1_rigid_redo.csv", index=False)
print("done")
