"""One-off: redo pitch's Stage 1 C1 cells with the fixed measure_natural_range
(was passing env.unwrapped to step_env instead of the wrapped env, skipping
the wrapper's action-history bookkeeping -- caught by the c3 alpha=100==C0
verification, which showed a 0.095 rad per-step divergence despite a
deceptively close aggregate distance). Pitch's C0 rows never call
measure_natural_range and are unaffected; only C1 needs redoing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spine_ablation_batch import run_cell, OUT  # noqa: E402

import pandas as pd  # noqa: E402

cells = [("pitch", terr, "c1", 100.0, seed) for terr in ("flat", "steps") for seed in range(8)]
rows = []
for k, (variant, terrain, condition, alpha_pct, seed) in enumerate(cells):
    summary = run_cell(variant, terrain, condition, alpha_pct, seed)
    if summary.get("error"):
        print(f"FAILED {variant}/{terrain}/seed{seed}: {summary['error'][:300]}")
    rows.append(summary)
    print(f"{k+1}/{len(cells)} pitch {terrain} c1 seed{seed} done")
    pd.DataFrame(rows).to_csv(OUT / "stage1_pitch_c1_redo.csv", index=False)
print("done")
