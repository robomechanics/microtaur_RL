"""Check microtaur_policy.py against a recording from the simulator (golden.npz).

  python test_golden.py            # numpy network
  python test_golden.py --onnx     # also run policy.onnx through onnxruntime

For each recorded sequence (it starts right after a reset) the raw robot signals are fed through
build_frame -> ObsHistory -> Policy -> ActionPipeline and compared, step by step, with what the
simulator computed: every observation term, the 705-D network input, the network output and the
safety-filter target. Run it on the robot computer after porting anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import microtaur_policy as mp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", action="store_true")
args = ap.parse_args()

g = np.load(HERE / "golden.npz")
policy = mp.Policy(HERE / "policy_weights.npz", HERE / "policy.onnx" if args.onnx else None)
assert int(g["history"]) == mp.HISTORY and abs(float(g["dt"]) - mp.DT) < 1e-9
TOL = {"frame": 1e-5, "obs": 1e-5, "action": 1e-4, "target": 1e-4}
ok = True
for seq in sorted({k.split("/")[0] for k in g.files if "/" in k}):
  S = lambda k: g[f"{seq}/{k}"]  # noqa: E731
  T = len(S("q"))
  hist, pipe = mp.ObsHistory(), mp.ActionPipeline()
  last = np.zeros(8)
  err = {"frame": 0.0, "obs": 0.0, "action": 0.0, "target": 0.0}
  worst_term = ("", 0.0)
  for t in range(T):
    fr = mp.build_frame(S("gravity")[t], S("ang_vel")[t], S("q")[t], S("qd")[t], last, S("command")[t])
    for name, _ in mp.TERMS:
      e = float(np.abs(np.clip(fr[name], -mp.OBS_CLIP, mp.OBS_CLIP) - S(f"frame/{name}")[t]).max())
      if e > worst_term[1]:
        worst_term = (f"{name} @ step {t}", e)
      err["frame"] = max(err["frame"], e)
    obs = hist.push(fr)
    err["obs"] = max(err["obs"], float(np.abs(obs - S("obs")[t]).max()))
    # feed the simulator's input to the network so a small input difference does not compound
    raw = policy(S("obs")[t].astype(np.float32))
    err["action"] = max(err["action"], float(np.abs(raw - S("raw_action")[t]).max()))
    _, target = pipe.step(S("raw_action")[t], S("q")[t], S("qd")[t])
    err["target"] = max(err["target"], float(np.abs(target - S("target_safe")[t]).max()))
    last = S("raw_action")[t]
  good = all(err[k] <= TOL[k] for k in err)
  ok &= good
  print(f"{seq:12s} {T:4d} steps | max abs err: frame {err['frame']:.1e} (worst {worst_term[0]}), obs {err['obs']:.1e}, "
        f"action {err['action']:.1e}, target {err['target']:.1e} -> {'PASS' if good else 'FAIL'}")
print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
