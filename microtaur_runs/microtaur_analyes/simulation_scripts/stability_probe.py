"""Short per-variant rollout that logs foot-site world positions + CoM ground
projection + per-foot contact, for support-polygon / static-stability analysis.

    python scripts/stability_probe.py                 # all 4 variants
    python scripts/stability_probe.py yaw roll        # subset

Writes rollouts/stability/<variant>.csv  (one row per 50 Hz control step).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MICROTAUR_USE_JOYSTICK_COMMANDS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from microtaur_variants import VARIANTS, use_variant  # noqa: E402

OUT = Path("rollouts/stability")
OUT.mkdir(parents=True, exist_ok=True)

CHILD = r'''
import os, sys, csv
os.environ.setdefault("MICROTAUR_USE_JOYSTICK_COMMANDS", "1"); sys.path.insert(0, "scripts")
import numpy as np, torch
from microtaur_velocity.distill_reliable.mjlab_utils import make_env_and_teacher, get_initial_obs, step_env
from microtaur_velocity.env_cfgs import set_joystick_twist_command

variant, ckpt, outpath, vx, seconds = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
torch.manual_seed(0); np.random.seed(0)
env, ag, pol = make_env_and_teacher("Mjlab-Velocity-Yaw-Flat-Microtaur", ckpt, 1, "cuda:0",
                                    robust=False, no_terminations=False)
u = env.unwrapped
try: u.seed(0)
except Exception: pass
r = u.scene["robot"]

FOOT_SITES = ["leg1_foot_site", "leg2_foot_site", "leg3_foot_site", "leg4_foot_site"]
site_names = list(getattr(r, "site_names", []))
sidx = [site_names.index(s) for s in FOOT_SITES]
feet = u.scene["feet_ground_contact"]
from foot_index import leg_channels
chan = leg_channels(feet)          # chan[j] = sensor channel carrying leg j+1
policy_dt = float(getattr(u, "step_dt", 0.02))
n = int(round(seconds / policy_dt))

obs = get_initial_obs(env); set_joystick_twist_command(env, vx, 0.0)
rows = []
for i in range(n):
    with torch.no_grad():
        a = torch.clamp(torch.nan_to_num(pol(obs)), -1.0, 1.0)
    obs, _, done, _ = step_env(env, a); set_joystick_twist_command(env, vx, 0.0)
    o = u.scene.env_origins[0].cpu().numpy()
    com = r.data.root_com_pos_w[0].cpu().numpy() - o
    base = r.data.root_link_pos_w[0].cpu().numpy() - o
    sp = r.data.site_pos_w[0].cpu().numpy()          # (nsite, 3)
    found = feet.data.found[0].float().cpu().numpy().ravel()[:4]
    force = feet.data.force[0].float().cpu().numpy()
    fmag = np.linalg.norm(force, axis=-1) if force.ndim == 2 else np.abs(force).ravel()
    row = dict(t=(i + 1) * policy_dt, done=int(bool(done.reshape(-1)[0])),
               com_x=com[0], com_y=com[1], com_z=com[2], base_z=base[2])
    for j in range(4):
        s = sp[sidx[j]] - o
        row[f"foot{j}_x"] = s[0]; row[f"foot{j}_y"] = s[1]; row[f"foot{j}_z"] = s[2]
        row[f"foot{j}_contact"] = int(found[chan[j]] > 0)
        row[f"foot{j}_force"] = float(fmag[chan[j]]) if chan[j] < len(fmag) else 0.0
    rows.append(row)
env.close()
with open(outpath, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"[{variant}] wrote {outpath}  ({len(rows)} rows, sites {FOOT_SITES} @ idx {sidx}, "
      f"contact channels {chan} from {list(feet.primary_names)})")
'''


def main():
  only = set(sys.argv[1:])
  py = str(Path(sys.executable))
  for key, v in VARIANTS.items():
    if only and key not in only:
      continue
    outp = OUT / f"{key}.csv"
    with use_variant(key):
      p = subprocess.run([py, "-c", CHILD, key, v.ckpt, str(outp), "0.14", "8.0"],
                         capture_output=True, text=True, timeout=600)
    tail = [l for l in p.stdout.splitlines() if l.startswith(f"[{key}]")]
    print(tail[0] if tail else f"[{key}] FAILED\n{p.stderr[-2500:]}")


if __name__ == "__main__":
  main()
