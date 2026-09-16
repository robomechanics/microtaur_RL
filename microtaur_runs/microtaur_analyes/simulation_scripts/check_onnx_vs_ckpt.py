"""Cross-check: model_4499.pt actor (mean action) vs the exported ONNX policy.

Builds the play env, steps a few times to reach a non-trivial state, then feeds
the exact same 36-d actor observation through both the rsl-rl checkpoint actor
and 2026-09-06_12-33-30.onnx. They should agree to ~1e-4.
"""

from __future__ import annotations

import os

os.environ.setdefault("MICROTAUR_VARIANT", "rigid_microtaur")
os.environ["MICROTAUR_USE_JOYSTICK_COMMANDS"] = "1"

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

REPO = Path(__file__).resolve().parents[1]
CKPT = REPO.parent / "rigid_flat_fixed_m077" / "model_4499.pt"
ONNX = REPO.parent / "rigid_flat_fixed_m077" / "2026-09-06_12-33-30.onnx"
TASK = "Mjlab-Velocity-Yaw-Flat-Microtaur"


def main():
  from microtaur_velocity.distill_reliable.mjlab_utils import (
    get_actor_obs_tensor,
    get_initial_obs,
    make_env_and_teacher,
    step_env,
  )
  from microtaur_velocity.env_cfgs import set_joystick_twist_command

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env, _cfg, policy = make_env_and_teacher(
    TASK, str(CKPT), num_envs=1, device=device, robust=False, no_terminations=False
  )
  obs = get_initial_obs(env)
  set_joystick_twist_command(env, 0.14, 0.0)
  for _ in range(40):
    with torch.no_grad():
      a = torch.clamp(torch.nan_to_num(policy(obs)), -1.0, 1.0)
    obs, _, _, _ = step_env(env, a)
    set_joystick_twist_command(env, 0.14, 0.0)

  actor_obs = get_actor_obs_tensor(obs).detach().float().cpu().numpy()  # (1, 36)
  print("actor obs shape:", actor_obs.shape)
  print("actor obs      :", np.array2string(actor_obs[0], precision=4, max_line_width=200))

  with torch.no_grad():
    a_ckpt = policy(obs).detach().float().cpu().numpy()[0]

  sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
  a_onnx = sess.run(None, {sess.get_inputs()[0].name: actor_obs})[0][0]

  env.close()

  diff = np.abs(a_ckpt - a_onnx)
  print("\n idx |    ckpt      |    onnx      |   |diff|")
  for i in range(8):
    print(f"  {i}  | {a_ckpt[i]:+12.6f} | {a_onnx[i]:+12.6f} | {diff[i]:.2e}")
  print(f"\nmax|diff| = {diff.max():.3e}   mean|diff| = {diff.mean():.3e}")
  ok = diff.max() < 1e-3
  print("PASS (max|diff| < 1e-3)" if ok else "MISMATCH")
  raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
  main()
