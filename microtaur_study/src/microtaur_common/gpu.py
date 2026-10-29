"""Pick the simulation device for evaluation scripts on a shared GPU (framework-free: call it
before the Kit app starts).

The lab GPU is shared; when other jobs fill it, PhysX cannot allocate a GPU scene ("failed to
allocate memory ... Unable to create scene") and an eval script dies. Evaluations are small, so
CPU PhysX is a usable fallback (48 envs x 300 steps of the gait sweep: ~20 s).
"""

from __future__ import annotations

import subprocess
import sys

MIN_FREE_MIB = 3000  # a Teacher-Cur play env needs roughly 2-3 GB of GPU memory


def gpu_free_mib(index: int = 0) -> int | None:
  try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", f"--id={index}"],
                         capture_output=True, text=True, timeout=10).stdout.strip()
    return int(out.splitlines()[0])
  except Exception:  # noqa: BLE001  (no nvidia-smi, parse error: let the caller keep its device)
    return None


def pick_device(requested: str, min_free_mib: int = MIN_FREE_MIB) -> str:
  """`requested` unless the user did not pass --device and the GPU has less than min_free_mib free."""
  if "--device" in sys.argv or not requested.startswith("cuda"):
    return requested
  free = gpu_free_mib(int(requested.split(":")[1]) if ":" in requested else 0)
  if free is not None and free < min_free_mib:
    print(f"[gpu] only {free} MiB free on {requested} -> running on cpu", flush=True)
    return "cpu"
  return requested
