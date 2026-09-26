"""Train a Microtaur IsaacLab task with IsaacLab's own rsl_rl train script.

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_train.py --task Microtaur-Isaac-Flat-v0 \
      --num_envs 2048 --headless [--max_iterations N] [...any IsaacLab train.py flag]

Registers the microtaur_isaac tasks, then runs
IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py unchanged (logging,
checkpoints, video, resume all come from it). Logs go to ./logs/rsl_rl/<experiment>.
Set ISAACLAB_DIR if IsaacLab is not at the default path below.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ISAACLAB = Path(os.environ.get("ISAACLAB_DIR", "/home/rml3/Documents/ben/spine/IsaacLab"))
TRAIN = ISAACLAB / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"

sys.path.insert(0, str(TRAIN.parent))  # train.py imports its sibling cli_args
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import microtaur_isaac  # noqa: E402,F401  (registers the gym tasks)

sys.argv[0] = str(TRAIN)
runpy.run_path(str(TRAIN), run_name="__main__")
