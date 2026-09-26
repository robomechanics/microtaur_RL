"""Train the flat rigid task with a given final energy weight (energy-curriculum sweep).

  python scripts/train_energy_sweep.py --energy-weight -0.01 [--num-envs 2048] [--iters 1000]

The weight is 0 for the first ENERGY_WARMUP_ITERS iterations and then ramps to
the given value (see rewards.py). Logs go to <log-root>/microtaur_rigid_d0/.
"""

from __future__ import annotations

import argparse

from mjlab.scripts.train import TrainConfig, launch_training

from microtaur_rigid.env_cfg import make_env_cfg
from microtaur_rigid.rl_cfg import microtaur_velocity_ppo_runner_cfg

ap = argparse.ArgumentParser()
ap.add_argument("--energy-weight", type=float, required=True)
ap.add_argument("--num-envs", type=int, default=2048)
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--stage", type=int, default=0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--log-root", default="runs_local")
args = ap.parse_args()

env = make_env_cfg(stage=args.stage, energy_weight=args.energy_weight)
env.scene.num_envs = args.num_envs
agent = microtaur_velocity_ppo_runner_cfg()
agent.max_iterations = args.iters
agent.seed = args.seed
agent.logger = "tensorboard"
agent.run_name = f"energy_sweep_w{abs(args.energy_weight):g}"
launch_training("Microtaur-Rigid-Flat-S0", TrainConfig(env=env, agent=agent, log_root=args.log_root))
