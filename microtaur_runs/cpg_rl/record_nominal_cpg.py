"""Record the fitted CPG walking open-loop (all actions zero) to mp4.

No policy is involved: the leg oscillators run at the values fitted from the
flat-terrain policy (MICROTAUR_CPG_FIT), so this shows what the CPG-RL policies
start from before any training. Frames stream to the file as they are rendered.

Usage (variant env vars + MICROTAUR_CPG=1 + MICROTAUR_CPG_FIT must be set):
    python record_nominal_cpg.py <out.mp4> [--seconds 25]
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MICROTAUR_USE_JOYSTICK_COMMANDS", "1")

import imageio.v2 as imageio
import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import microtaur_velocity  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-Velocity-Flat-microtaur_velocity"

# (duration s, forward m/s, yaw rad/s, caption)
SCRIPT = (
  (10.0, 0.15, 0.00, "forward 0.15 m/s"),
  (6.0, 0.12, 0.20, "forward 0.12 m/s + yaw +0.20 rad/s"),
  (6.0, 0.12, -0.20, "forward 0.12 m/s + yaw -0.20 rad/s"),
  (3.0, 0.00, 0.00, "stand"),
)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("out")
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  args = parser.parse_args()

  if os.environ.get("MICROTAUR_CPG", "0") in {"0", "", "false", "no"}:
    raise SystemExit("Set MICROTAUR_CPG=1 and MICROTAUR_CPG_FIT=<fit json>.")

  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.viewer.width = args.width
  env_cfg.viewer.height = args.height
  for sensor in env_cfg.scene.sensors or ():
    if hasattr(sensor, "debug_vis"):
      sensor.debug_vis = False

  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array")
  env.reset()
  action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device="cuda:0")
  command = env.command_manager.get_command("twist")
  term = env.command_manager.get_term("twist")

  fps = round(1.0 / env.step_dt)
  speeds, resets = [], 0
  with imageio.get_writer(args.out, fps=fps, codec="libx264", quality=8) as writer:
    for duration, vx, wz, caption in SCRIPT:
      print(f"[cpg-record] {caption}")
      for _ in range(int(duration / env.step_dt)):
        command[:, 0] = vx
        command[:, 1] = 0.0
        command[:, 2] = wz
        if hasattr(term, "time_left"):
          term.time_left.fill_(1.0e9)
        _, _, dones, _, _ = env.step(action)
        resets += int(dones.sum().item())
        if vx > 0:
          speeds.append(env.scene["robot"].data.root_link_lin_vel_b[0, 0].item())
        writer.append_data(env.render())
  env.close()
  print(
    f"[cpg-record] wrote {args.out} at {fps} fps, "
    f"mean forward speed while commanded {np.mean(speeds):.3f} m/s, resets: {resets}"
  )


if __name__ == "__main__":
  main()
