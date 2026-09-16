"""Smoke-test each morphology variant's env_cfgs.py against the installed mjlab.

No checkpoints needed. For every variant: swap in its env file, then in a fresh
subprocess import the package, build the play env, step a zero action 3x, and
report action terms / dims / root body / joint + actuator names.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from microtaur_variants import VARIANTS, use_variant  # noqa: E402

CHILD = r'''
import os, json, sys
os.environ.setdefault("MICROTAUR_USE_JOYSTICK_COMMANDS", "1")
import torch
import microtaur_velocity  # noqa: F401
from microtaur_velocity.env_cfgs import ENV_CFG_REVISION
from microtaur_velocity import microtaur_constants as mc
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

TASK = "Mjlab-Velocity-Yaw-Flat-Microtaur"
info = {"revision": ENV_CFG_REVISION, "MODEL_NAME": mc.MODEL_NAME,
        "ROOT_BODY": mc.ROOT_BODY, "HAS_ACTIVE_SPINE": mc.HAS_ACTIVE_SPINE,
        "ACTION_JOINT_NAMES": list(mc.ACTION_JOINT_NAMES),
        "OBS_JOINT_NAMES": list(mc.OBS_JOINT_NAMES)}
cfg = load_env_cfg(TASK, play=True)
agent = load_rl_cfg(TASK)
cfg.scene.num_envs = 1
info["action_terms"] = {k: getattr(v, "class_type", type(v)).__name__ for k, v in cfg.actions.items()}
info["obs_actor_terms"] = list(cfg.observations["actor"].terms.keys())
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
u = env.unwrapped
info["action_space_shape"] = list(u.action_space.shape)
info["actuator_names"] = list(u.scene["robot"].actuator_names)
info["joint_names"] = list(u.scene["robot"].joint_names)
info["action_manager_terms"] = {}
for name in u.action_manager.active_terms:
    t = u.action_manager.get_term(name)
    info["action_manager_terms"][name] = {
        "dim": int(getattr(t, "action_dim", -1)),
        "target_names": list(getattr(t, "_target_names", []) or []),
    }
obs = env.get_observations()
if isinstance(obs, tuple): obs = obs[0]
act = torch.zeros(u.action_space.shape, device=u.device)
for _ in range(3):
    env.step(act)
info["stepped_ok"] = True
env.close()
print("###JSON###" + json.dumps(info))
'''


def main():
  py = str(Path(sys.executable))
  only = set(sys.argv[1:])
  results = {}
  for key, v in VARIANTS.items():
    if only and key not in only:
      continue
    print(f"\n{'=' * 66}\n{key}  ({v.mjcf_variant})\n{'=' * 66}")
    try:
      with use_variant(key):
        proc = subprocess.run([py, "-c", CHILD], capture_output=True, text=True, timeout=600)
    except Exception as exc:  # noqa: BLE001
      results[key] = {"ok": False, "error": f"swap failed: {exc}"}
      print("  SWAP FAILED:", exc)
      continue
    out = proc.stdout
    tag = "###JSON###"
    if tag in out:
      import json

      info = json.loads(out.split(tag, 1)[1].splitlines()[0])
      results[key] = {"ok": True, **info}
      print(f"  rev            : {info['revision']}  (expect {v.revision})  "
            f"{'OK' if info['revision'] == v.revision else 'MISMATCH'}")
      print(f"  MODEL_NAME     : {info['MODEL_NAME']}")
      print(f"  ROOT_BODY      : {info['ROOT_BODY']}")
      print(f"  action space   : {info['action_space_shape']}   (registry says {v.action_dim})")
      print(f"  action terms   : {info['action_terms']}")
      print(f"  per-term       : {info['action_manager_terms']}")
      print(f"  actuator_names : {info['actuator_names']}")
      print(f"  obs actor terms: {info['obs_actor_terms']}")
      print(f"  stepped 3x     : {info.get('stepped_ok')}")
    else:
      results[key] = {"ok": False, "error": "no JSON", "stderr_tail": proc.stderr[-2500:]}
      print("  FAILED — stderr tail:\n" + proc.stderr[-2500:])

  print(f"\n{'=' * 66}\nSUMMARY\n{'=' * 66}")
  for k, r in results.items():
    if not r.get("ok"):
      print(f"  {k:6s} FAIL  {r.get('error')}")
    else:
      rev_ok = r["revision"] == VARIANTS[k].revision
      dim_ok = r["action_space_shape"] == [VARIANTS[k].action_dim] or \
               r["action_space_shape"] == [1, VARIANTS[k].action_dim]
      print(f"  {k:6s} OK    rev={'ok' if rev_ok else 'MISMATCH'}  "
            f"action_dim={r['action_space_shape']}{' ok' if dim_ok else ' CHECK'}  "
            f"terms={list(r['action_terms'])}")
  out_path = Path(__file__).resolve().parents[1] / "rollouts" / "variant_env_check.json"
  out_path.parent.mkdir(exist_ok=True)
  import json

  out_path.write_text(json.dumps(results, indent=2))
  print(f"\nwrote {out_path}")


if __name__ == "__main__":
  main()
