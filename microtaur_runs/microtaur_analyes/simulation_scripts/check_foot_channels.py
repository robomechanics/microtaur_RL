"""Print the resolved foot-contact channel order for every variant.

No checkpoint, no rollout -- just builds each env and reads
`ContactSensor.primary_names`, then shows what `foot_index.leg_channels()`
makes of it. Run this after any model/spec change.

    python scripts/check_foot_channels.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MICROTAUR_USE_JOYSTICK_COMMANDS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from microtaur_variants import RUNS_ROOT, TASK_ID, VARIANTS, use_variant  # noqa: E402
from foot_index import LEG_CORNER, leg_channels  # noqa: E402

CHILD = r'''
import os, sys, json
os.environ.setdefault("MICROTAUR_USE_JOYSTICK_COMMANDS", "1"); sys.path.insert(0, os.environ["FOOT_CHANNEL_SCRIPTS_DIR"])
import mujoco
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from microtaur_velocity.microtaur_constants import get_spec, FOOT_SITE_NAMES
from foot_index import leg_channels, LEG_CORNER

cfg = load_env_cfg(os.environ["FOOT_CHANNEL_TASK"], play=True)
cfg.scene.num_envs = 1
m = get_spec().compile()
bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, m.nbody)]
if getattr(cfg.viewer, "body_name", None) not in bodies:
    cfg.viewer.body_name = bodies[0]
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
u = env.unwrapped
feet = u.scene["feet_ground_contact"]
names = list(feet.primary_names)
chan = leg_channels(feet)
r = u.scene["robot"]
sn = list(getattr(r, "site_names", []))
print("RESULT " + json.dumps(dict(
    primary_names=names,
    leg_to_channel={f"leg{i+1}": chan[i] for i in range(4)},
    channel_to_corner={str(chan[i]): LEG_CORNER[i+1] for i in range(4)},
    site_columns=[sn.index(s) for s in FOOT_SITE_NAMES],
    permuted=(chan != [0,1,2,3]),
)))
env.close()
'''


def main():
  import subprocess
  py = str(Path(sys.executable))
  out = {}
  for key in VARIANTS:
    # The child is a fresh interpreter: hand it this variant's package explicitly.
    v = VARIANTS[key]
    env = {**os.environ, **v.env_extra, "MICROTAUR_VARIANT": v.mjcf_variant,
           "PYTHONPATH": str(RUNS_ROOT / v.folder / "src"),
           "FOOT_CHANNEL_SCRIPTS_DIR": str(Path(__file__).resolve().parent),
           "FOOT_CHANNEL_TASK": TASK_ID}
    p = subprocess.run([py, "-c", CHILD], capture_output=True, text=True, timeout=600, env=env)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
      print(f"{key:6s} FAILED\n{p.stderr[-1500:]}")
      continue
    d = json.loads(line[len("RESULT "):])
    out[key] = d
    flag = "  <-- PERMUTED" if d["permuted"] else ""
    print(f"{key:6s} sites@{d['site_columns']}  channels {d['primary_names']}")
    print(f"       leg->chan {d['leg_to_channel']}   chan->corner {d['channel_to_corner']}{flag}")
  Path("rollouts").mkdir(exist_ok=True)
  Path("rollouts/foot_channels.json").write_text(json.dumps(out, indent=2))
  print("\nwrote rollouts/foot_channels.json")


if __name__ == "__main__":
  main()
