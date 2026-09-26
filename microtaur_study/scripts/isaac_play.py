"""Play a Microtaur IsaacLab policy with IsaacLab's own rsl_rl play script.

  # GUI (a machine with a display):
  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_play.py --task Microtaur-Isaac-Flat-Play-v0 \\
      --checkpoint checkpoints/<run>/model_<N>.pt [--num_envs 16] [--real-time]
  # headless, record an mp4 into <checkpoint dir>/videos/play:
  ... --headless --video --video_length 300

Registers the microtaur_isaac tasks, then runs
IsaacLab/scripts/reinforcement_learning/rsl_rl/play.py unchanged (it also exports
the policy to <checkpoint dir>/exported/policy.pt and policy.onnx). Set
ISAACLAB_DIR if IsaacLab is not at the default path below.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ISAACLAB = Path(os.environ.get("ISAACLAB_DIR", "/home/rml3/Documents/ben/spine/IsaacLab"))
PLAY = ISAACLAB / "scripts" / "reinforcement_learning" / "rsl_rl" / "play.py"

sys.path.insert(0, str(PLAY.parent))  # play.py imports its sibling cli_args
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import microtaur_isaac  # noqa: E402,F401  (registers the gym tasks)

sys.argv[0] = str(PLAY)
runpy.run_path(str(PLAY), run_name="__main__")
