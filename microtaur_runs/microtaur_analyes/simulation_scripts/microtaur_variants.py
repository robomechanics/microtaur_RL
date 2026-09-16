"""Registry of Microtaur morphology variants for this repo (microtaur_runs/).

Each variant lives in its own folder with its own `src/microtaur_velocity`
package, so switching a rollout to a variant means putting that folder's `src`
first on sys.path (ahead of the editable install of Microtaur_RL-main) and
setting its MICROTAUR_* environment variables -- before anything imports
`microtaur_velocity`. One variant per process.

The original version of this module swapped `*_env.txt` files into a single
repo; that layout does not exist here.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

RUNS_ROOT = Path(__file__).resolve().parents[2]  # microtaur_runs/
EXPERIMENT = "logs/rsl_rl/microtaur_minitaur_velocity"
TASK_ID = "Mjlab-Velocity-Rough-microtaur_velocity"


@dataclass(frozen=True)
class Variant:
  key: str
  folder: str                # microtaur_runs/<folder>
  mjcf_variant: str          # value for MICROTAUR_VARIANT
  revision: str              # expected ENV_CFG_REVISION after import
  action_dim: int            # policy action width (8 legs, +1 if active spine)
  spine: str                 # "" | "active"
  ckpt: str                  # default checkpoint for this variant
  clip_actions: float | None  # what the checkpoint was trained with
  env_extra: dict = field(default_factory=dict)


def _ckpt(folder: str, run: str) -> str:
  return str(RUNS_ROOT / folder / EXPERIMENT / run / "model_4499.pt")


# Rough-terrain policies trained with --agent.clip-actions 1.0 (2026-09-13).
VARIANTS: dict[str, Variant] = {
  "rigid": Variant(
    "rigid", "rigid", "rigid_microtaur",
    "2026-09-05-rigid-aligned-baseline-v1", 8, "",
    _ckpt("rigid", "2026-09-13_12-42-21_rigid_rough_with_flag"), 1.0,
    {"MICROTAUR_CONTROL_SPINE": "0"},
  ),
  "pitch": Variant(
    "pitch", "active_pitch", "active_pitch_microtaur",
    "2026-09-01-active-pitch-final-xml-v2", 9, "active",
    _ckpt("active_pitch", "2026-09-13_09-50-42_active_pitch_rough_with_flag"), 1.0,
    {"MICROTAUR_CONTROL_SPINE": "1"},
  ),
  "yaw": Variant(
    "yaw", "active_yaw", "active_yaw_microtaur",
    "2026-08-30-active-yaw-spine-residual-v1", 9, "active",
    _ckpt("active_yaw", "2026-09-13_08-41-15_active_yaw_rough_with_flag"), 1.0,
    {"MICROTAUR_CONTROL_SPINE": "1"},
  ),
  "roll": Variant(
    "roll", "active_twist", "active_twist_microtaur",
    "2026-09-05-active-twist-emergent-v4-mean-bias-fix", 9, "active",
    _ckpt("active_twist", "2026-09-13_13-41-21_active_twist_rough_with_flag"), 1.0,
    {"MICROTAUR_CONTROL_SPINE": "1"},
  ),
}


@contextlib.contextmanager
def use_variant(key: str, keep: bool = False):
  """Select a variant's package and environment for this process.

  Must wrap the block *before* any `import microtaur_velocity`. `keep` is
  accepted for compatibility; nothing on disk is modified.
  """
  del keep
  if key not in VARIANTS:
    raise KeyError(f"unknown variant {key!r}; choices: {list(VARIANTS)}")
  v = VARIANTS[key]
  src = RUNS_ROOT / v.folder / "src"
  if not (src / "microtaur_velocity" / "env_cfgs.py").exists():
    raise FileNotFoundError(f"variant package missing: {src}")
  if "microtaur_velocity" in sys.modules:
    loaded = Path(sys.modules["microtaur_velocity"].__file__).resolve()
    if src not in loaded.parents:
      raise RuntimeError(f"microtaur_velocity already imported from {loaded}; "
                         "run one variant per process")

  # Drop the ROS paths that leak into conda envs, then put this variant first.
  sys.path[:] = [p for p in sys.path if "/opt/ros/" not in p]
  sys.path.insert(0, str(src))
  os.environ["MICROTAUR_VARIANT"] = v.mjcf_variant
  os.environ.update(v.env_extra)
  print(f"[variant] {key}: MICROTAUR_VARIANT={v.mjcf_variant} "
        f"{' '.join(f'{k}={val}' for k, val in v.env_extra.items())}  package {src}")
  import microtaur_velocity.env_cfgs  # noqa: F401
  yield v


def verify_revision(expected: str) -> str:
  from microtaur_velocity.env_cfgs import ENV_CFG_REVISION

  if ENV_CFG_REVISION != expected:
    raise RuntimeError(
      f"ENV_CFG_REVISION mismatch: loaded {ENV_CFG_REVISION!r}, expected {expected!r}. "
      "Wrong env_cfgs.py is active for this variant."
    )
  return ENV_CFG_REVISION
