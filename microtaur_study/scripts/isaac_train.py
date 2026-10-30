"""Train a Microtaur IsaacLab task with IsaacLab's own rsl_rl train script.

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_train.py --task Microtaur-Isaac-Flat-v0 \
      --num_envs 2048 --headless [--max_iterations N] [...any IsaacLab train.py flag]

Registers the microtaur_isaac tasks, then runs
IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py unchanged (logging,
checkpoints, video, resume all come from it). Logs go to ./logs/rsl_rl/<experiment>.
IsaacLab is found from the installed isaaclab package (a source install: <IsaacLab>/source/isaaclab);
set ISAACLAB_DIR to point at another checkout.
"""

from __future__ import annotations

import importlib.util
import os
import runpy
import sys
from pathlib import Path

REL = Path("scripts") / "reinforcement_learning" / "rsl_rl" / "train.py"


def isaaclab_dir() -> Path:
  if os.environ.get("ISAACLAB_DIR"):
    return Path(os.environ["ISAACLAB_DIR"])
  spec = importlib.util.find_spec("isaaclab")  # <IsaacLab>/source/isaaclab/isaaclab/__init__.py
  if spec is not None and spec.origin is not None:
    root = Path(spec.origin).resolve().parents[3]
    if (root / REL).exists():
      return root
  sys.exit("isaac_train.py: IsaacLab's rsl_rl train.py not found; install IsaacLab from source "
           "(git clone, ./isaaclab.sh --install rsl_rl) or set ISAACLAB_DIR")


TRAIN = isaaclab_dir() / REL

sys.path.insert(0, str(TRAIN.parent))  # train.py imports its sibling cli_args
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import microtaur_isaac  # noqa: E402,F401  (registers the gym tasks)

sys.argv[0] = str(TRAIN)
runpy.run_path(str(TRAIN), run_name="__main__")
