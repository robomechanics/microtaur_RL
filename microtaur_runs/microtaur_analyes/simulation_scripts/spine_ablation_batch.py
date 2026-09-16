"""Stage batch driver for spine_ablation.py.

Launches each run as a SEPARATE subprocess (like scripts/rollout_log.py's own
pattern), not an in-process loop. This is necessary, not just cautious:
use_variant() swaps env_cfgs.py's CONTENTS on disk, but Python caches an
already-imported `microtaur_velocity.env_cfgs` module in sys.modules for the
life of the interpreter -- swapping the file again within the same process
does not invalidate that cache. An earlier in-process version of this driver
ran pitch correctly (first variant imported, stayed cached and correct for
every pitch run) but silently would have been wrong for a same-action-shape
variant (roll/yaw, also 9 actions) switched to mid-process -- it only
surfaced loudly for rigid because rigid's checkpoint has a different action
count (8) than whatever spine variant's module happened to be cached,
producing a state_dict shape mismatch instead of a silent wrong-config run.
One process per run costs ~30-80s of startup/JIT overhead each, but that is
the price of not risking a silent correctness bug in a paired-comparison
experiment.

    python scripts/spine_ablation_batch.py --stage 1
    python scripts/spine_ablation_batch.py --stage 2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "rollouts" / "spine_ablation"
PY = sys.executable
VX = 0.20
SECONDS = 16.0
SEEDS = list(range(8))


def stage1():
    cells = []
    for terr in ("flat", "steps"):
        for seed in SEEDS:
            cells.append(("pitch", terr, "c0", 100.0, seed))
            cells.append(("pitch", terr, "c1", 100.0, seed))
            cells.append(("rigid", terr, "c0", 100.0, seed))
    return cells


def stage2():
    cells = []
    for terr in ("flat", "steps"):
        for seed in SEEDS:
            cells.append(("roll", terr, "c0", 100.0, seed))
            cells.append(("roll", terr, "c1", 100.0, seed))
            cells.append(("yaw", terr, "c0", 100.0, seed))
            cells.append(("yaw", terr, "c1", 100.0, seed))
    return cells


STAGES = {1: stage1, 2: stage2}


def run_cell(variant, terrain, condition, alpha_pct, seed):
    tag = f"{variant}_{terrain}_{condition}"
    if condition == "c3":
        tag += f"_{int(alpha_pct)}pct"
    tag += f"_seed{seed}"
    json_path = OUT / f"{tag}.json"

    cmd = [PY, str(ROOT / "scripts" / "spine_ablation.py"),
           "--variant", variant, "--terrain", terrain, "--condition", condition,
           "--seed", str(seed), "--vx", str(VX), "--seconds", str(SECONDS)]
    if condition == "c3":
        cmd += ["--alpha-pct", str(alpha_pct)]

    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    if proc.returncode != 0 or not json_path.exists():
        return dict(variant=variant, terrain=terrain, vx=VX, seed=seed, condition=condition,
                   alpha_pct=alpha_pct if condition == "c3" else float("nan"),
                   error=proc.stderr[-2000:] if proc.stderr else f"exit {proc.returncode}, no json")
    with open(json_path) as f:
        summary = json.load(f)
    summary["error"] = None
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=list(STAGES))
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    cells = STAGES[args.stage]()
    print(f"[batch] stage {args.stage}: {len(cells)} runs (subprocess-isolated)")
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.time()
    for k, (variant, terrain, condition, alpha_pct, seed) in enumerate(cells):
        t1 = time.time()
        summary = run_cell(variant, terrain, condition, alpha_pct, seed)
        if summary.get("error"):
            print(f"[batch] FAILED {variant}/{terrain}/{condition}/seed{seed}: {summary['error'][:300]}")
        rows.append(summary)
        dt = time.time() - t1
        print(f"[batch] {k+1}/{len(cells)} {variant:6s} {terrain:6s} {condition} "
              f"seed{seed}  ({dt:.0f}s, elapsed {time.time()-t0:.0f}s)")
        pd.DataFrame(rows).to_csv(
            OUT / (args.out_name or f"stage{args.stage}_summary.csv"), index=False)

    n_fail = sum(1 for r in rows if r.get("error"))
    print(f"[batch] stage {args.stage} done in {time.time()-t0:.0f}s, {n_fail} failed / {len(rows)}")


if __name__ == "__main__":
    main()
