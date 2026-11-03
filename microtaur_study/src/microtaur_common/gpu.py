"""GPU budget for the Microtaur scripts on the shared lab GPU (framework-free: call it before the Kit app starts).

Two rules:
  - Fallback: evaluations are small, so when the GPU is full (PhysX: "failed to allocate memory ... Unable to create
    scene") they run with CPU PhysX instead (48 envs x 300 steps of the gait sweep: ~20 s).
  - Hard cap on OUR total GPU memory: Ben, 2026-11-01, "cap gpu ram at 5gb, hard cap until further notice", shared
    by all of this user's work on the machine (the spine and autofloat sessions together). The cap (MiB, as
    nvidia-smi / nvtop show it) is read from $MT_GPU_CAP_MIB or ~/.config/microtaur/gpu_cap_mib (no file and no
    variable: no cap). "Ours" = every GPU process of this user. A script that would push our total over the cap runs
    its physics on the CPU (pick_device), skips rendering (gpu_allows) or refuses to start a training
    (isaac_train.py). CPU physics still opens Kit's GPU context (220-360 MiB measured, also with CUDA hidden): when
    even that does not fit, pick_device exits (SKIPPED_GPU_CAP), so nothing Isaac runs beside a 4.9 GB training.
"""

from __future__ import annotations

import getpass
import os
import subprocess
import sys
from pathlib import Path

MIN_FREE_MIB = 3000  # a Teacher-Cur play env needs roughly 2-3 GB of GPU memory
KIT_CONTEXT_MIB = 400  # Kit's own GPU memory with CPU physics (220-360 MiB measured)
CAP_FILE = Path.home() / ".config" / "microtaur" / "gpu_cap_mib"


def _nvidia_smi(*query: str) -> list[str]:
  try:
    out = subprocess.run(["nvidia-smi", *query, "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         timeout=10).stdout
    return [ln.strip() for ln in out.splitlines() if ln.strip()]
  except Exception:  # noqa: BLE001  (no nvidia-smi: nothing to measure)
    return []


def gpu_free_mib(index: int = 0) -> int | None:
  rows = _nvidia_smi("--query-gpu=memory.free", f"--id={index}")
  try:
    return int(rows[0])
  except (IndexError, ValueError):
    return None


def gpu_cap_mib() -> int | None:
  raw = os.environ.get("MT_GPU_CAP_MIB") or (CAP_FILE.read_text().strip() if CAP_FILE.exists() else "")
  try:
    return int(raw) if raw else None
  except ValueError:
    return None


def our_gpu_mib() -> int:
  """GPU memory of all this user's processes (all GPUs): the cap is shared by every project of the user."""
  me, total = getpass.getuser(), 0
  for row in _nvidia_smi("--query-compute-apps=pid,used_memory"):
    try:
      pid, mib = (int(v) for v in row.split(","))
      if Path(f"/proc/{pid}").owner() == me:
        total += mib
    except (ValueError, OSError):
      continue
  return total


def gpu_allows(need_mib: int, what: str) -> bool:
  """False (with a message) when need_mib more would push our GPU memory over the cap."""
  cap = gpu_cap_mib()
  if cap is None:
    return True
  ours = our_gpu_mib()
  if ours + need_mib > cap:
    print(f"[gpu] cap {cap} MiB: ours {ours} + {what} ~{need_mib} would exceed it", flush=True)
    return False
  return True


def _cpu() -> str:
  if not gpu_allows(KIT_CONTEXT_MIB, "Kit's GPU context (also with CPU physics)"):
    print("SKIPPED_GPU_CAP", flush=True)
    raise SystemExit(3)
  return "cpu"


def pick_device(requested: str, min_free_mib: int = MIN_FREE_MIB) -> str:
  """`requested`, or "cpu" when the cap does not allow a GPU env, or (no --device given) the GPU is nearly full;
  exits when the cap does not even allow Kit's context."""
  if not requested.startswith("cuda"):
    return _cpu()
  if not gpu_allows(min_free_mib, "a GPU env"):
    return _cpu()  # the cap is hard: it also overrides an explicit --device cuda
  if "--device" in sys.argv:
    return requested
  free = gpu_free_mib(int(requested.split(":")[1]) if ":" in requested else 0)
  if free is not None and free < min_free_mib:
    print(f"[gpu] only {free} MiB free on {requested} -> running on cpu", flush=True)
    return _cpu()
  return requested
