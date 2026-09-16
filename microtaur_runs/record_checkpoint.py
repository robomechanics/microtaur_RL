"""Record a scripted-command rollout of a trained rough-terrain checkpoint to mp4.

Recovered from the 2026-09-13 session (it produced videos/*_rough_with_flag_model_4499.mp4).

Usage (env must already select the variant, e.g. PYTHONPATH/MICROTAUR_VARIANT):
  MUJOCO_GL=egl MICROTAUR_USE_JOYSTICK_COMMANDS=1 python record_checkpoint.py \
      <checkpoint.pt> <out.mp4> [--clip-actions 1.0] [--terrain micro_random_rough] [--level 3]
"""

import argparse
import os
from dataclasses import asdict

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MICROTAUR_USE_JOYSTICK_COMMANDS", "1")

import imageio.v2 as imageio
import torch

import mjlab.tasks  # noqa: F401
import microtaur_velocity  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK = "Mjlab-Velocity-Rough-microtaur_velocity"

# (duration s, forward m/s, yaw rad/s, caption)
SCRIPT = (
  (8.0, 0.15, 0.00, "forward 0.15 m/s"),
  (6.0, 0.12, 0.20, "forward 0.12 m/s + yaw +0.20 rad/s"),
  (6.0, 0.12, -0.20, "forward 0.12 m/s + yaw -0.20 rad/s"),
  (5.0, 0.00, 0.00, "stand"),
)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint")
  parser.add_argument("out")
  parser.add_argument("--clip-actions", type=float, default=None)
  parser.add_argument("--terrain", default="micro_random_rough")
  parser.add_argument("--level", type=int, default=3)
  args = parser.parse_args()

  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.viewer.width = 1280
  env_cfg.viewer.height = 720
  for sensor in env_cfg.scene.sensors or ():
    if hasattr(sensor, "debug_vis"):
      sensor.debug_vis = False  # hide the height-scan ray markers
  # Show a single sub-terrain type so the robot does not land on a flat tile.
  generator = env_cfg.scene.terrain.terrain_generator
  generator.sub_terrains = {args.terrain: generator.sub_terrains[args.terrain]}
  generator.sub_terrains[args.terrain].proportion = 1.0

  agent_cfg = load_rl_cfg(TASK)
  agent_cfg.clip_actions = args.clip_actions

  device = "cuda:0"
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(TASK)(vec_env, asdict(agent_cfg), device=device)
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  terrain = env.scene.terrain
  terrain.terrain_levels[:] = args.level
  terrain.env_origins[:] = terrain.terrain_origins[args.level, 0]
  env.reset()
  print(f"[record] terrain {args.terrain} level {args.level} origin {terrain.env_origins[0].tolist()}")

  fps = round(1.0 / env.step_dt)
  command = env.command_manager.get_command("twist")
  term = env.command_manager.get_term("twist")
  obs = vec_env.get_observations()
  falls = 0

  with imageio.get_writer(args.out, fps=fps, codec="libx264", quality=8) as writer:
    for duration, vx, wz, caption in SCRIPT:
      print(f"[record] {caption}")
      for _ in range(int(duration / env.step_dt)):
        command[:, 0] = vx
        command[:, 1] = 0.0
        command[:, 2] = wz
        if hasattr(term, "time_left"):
          term.time_left.fill_(1.0e9)
        with torch.no_grad():
          actions = policy(obs)
        obs, _, dones, _ = vec_env.step(actions)
        falls += int(dones.sum().item())
        writer.append_data(env.render())
  env.close()
  print(f"[record] wrote {args.out} at {fps} fps, episode resets during clip: {falls}")


if __name__ == "__main__":
  main()
