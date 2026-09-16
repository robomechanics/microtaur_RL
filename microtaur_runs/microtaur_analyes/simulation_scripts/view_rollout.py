"""Watch a trained checkpoint on a task terrain in mjlab's live interactive
viewer -- free mouse-driven orbit/zoom, unlike render_terrain.py's fixed-camera
offscreen mp4s.

Determinism holds for a fixed (variant, seed, terrain, command): this
reproduces the exact rollout logged by rollout_log.py / scored in
rollouts/tidy/tidy_rollouts.csv, byte-for-byte.

    # pitch's best steps-waypoint run (highest progress AND lowest CoT, 0 resets)
    python scripts/view_rollout.py --variant pitch --terrain steps --vx 0.20 --seed 6 --lane-keep

    python scripts/view_rollout.py --variant rigid --terrain curb --vx 0.14 --seed 2 --lane-keep
    python scripts/view_rollout.py --variant yaw   --terrain weave --vx 0.12 --seed 0 --weave-drive

Close the viewer window or Ctrl+C to exit.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["MICROTAUR_USE_JOYSTICK_COMMANDS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from microtaur_variants import VARIANTS, use_variant  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


class SteeredPolicy:
    """Hold a fixed forward speed; yaw is either fixed or driven by a lane-keep
    / weave-slalom controller, exactly as render_terrain.py / rollout_log.py
    steer these terrains (the policy itself has no lateral-position feedback).
    """

    def __init__(self, env, learned_policy, vx, yaw, driver, set_joystick_twist_command):
        self.env = env
        self.learned_policy = learned_policy
        self.vx = vx
        self.yaw = yaw
        self.driver = driver
        self.set_joystick_twist_command = set_joystick_twist_command
        self._robot = env.unwrapped.scene["robot"]

    def _cmd_yaw(self):
        if self.driver is None:
            return self.yaw
        from weave_drive import yaw_from_quat_wxyz
        p = self._robot.data.root_link_pos_w[0].cpu().numpy()
        q = self._robot.data.root_link_quat_w[0].cpu().numpy()
        v = float(self._robot.data.root_link_lin_vel_b[0, 0])
        return self.driver.yaw_command(float(p[0]), float(p[1]), yaw_from_quat_wxyz(q),
                                        speed=max(v, 0.02))

    def __call__(self, stale_obs):
        del stale_obs  # mjlab's viewer computes obs before calling policy(); recompute below
        self.set_joystick_twist_command(self.env, self.vx, self._cmd_yaw())
        obs = self.env.get_observations()
        return self.learned_policy(obs)

    def reset(self) -> None:
        pass


def build(variant, ckpt, device, terrain, terrain_kw):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from rsl_rl.runners import OnPolicyRunner
    from microtaur_velocity.distill_reliable.mjlab_utils import cfg_to_dict
    from microtaur_velocity.microtaur_terrains import apply_terrain

    task = "Mjlab-Velocity-Yaw-Flat-Microtaur"
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    if "actor" in cfg.observations:
        cfg.observations["actor"].enable_corruption = False
    agent_cfg = load_rl_cfg(task)

    if terrain != "flat":
        apply_terrain(cfg, terrain, **terrain_kw)

    env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task) or OnPolicyRunner
    runner = runner_cls(env, cfg_to_dict(agent_cfg), device=device)
    runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
    return env, runner.get_inference_policy(device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS), default="pitch")
    ap.add_argument("--terrain", default="steps", help="flat | curb | steps | weave")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--vx", type=float, default=0.20)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--curb-height", type=float, default=None)
    ap.add_argument("--step-height", type=float, default=None)
    ap.add_argument("--tile-size", type=float, default=None)
    ap.add_argument("--pole-spacing", type=float, default=None)
    ap.add_argument("--num-poles", type=int, default=None)
    ap.add_argument("--lane-keep", action="store_true",
                    help="hold the spawn line (on curb: stay straddling the lip; "
                         "on steps: stay centred on the tile field -- this is the "
                         "'waypoint' mode in tidy_rollouts.csv)")
    ap.add_argument("--weave-drive", action="store_true",
                    help="steer the slalom through the weave poles")
    args = ap.parse_args()

    from mjlab.viewer import NativeMujocoViewer
    from microtaur_velocity.env_cfgs import set_joystick_twist_command

    kw = {}
    if args.terrain == "curb" and args.curb_height is not None:
        kw["curb_height"] = args.curb_height
    if args.terrain == "steps":
        if args.step_height is not None:
            kw["max_height"] = args.step_height
        if args.tile_size is not None:
            kw["tile_size"] = args.tile_size
    if args.terrain == "weave":
        if args.pole_spacing is not None:
            kw["spacing"] = args.pole_spacing
        if args.num_poles is not None:
            kw["num_poles"] = args.num_poles

    ckpt = Path(args.checkpoint) if args.checkpoint else Path(VARIANTS[args.variant].ckpt)
    if not ckpt.exists():
        ap.error(f"checkpoint not found: {ckpt}")

    # Seed exactly as rollout_log.py / render_terrain.py do, so this reproduces
    # the logged run with this (variant, seed, terrain, cmd) byte-for-byte.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = None
    with use_variant(args.variant):
        try:
            env, learned_policy = build(args.variant, ckpt, args.device, args.terrain, kw)
            u = env.unwrapped
            try:
                u.seed(args.seed)
            except Exception:  # noqa: BLE001
                pass

            driver = None
            if args.lane_keep or args.weave_drive:
                from weave_drive import WeaveDriver, first_pole_x
                o = u.scene.env_origins[0].detach().cpu().numpy()
                if args.weave_drive:
                    sp = float(kw.get("spacing", 0.10))
                    driver = WeaveDriver(x0=first_pole_x(u, sp), spacing=sp, y_center=float(o[1]))
                else:
                    driver = WeaveDriver(x0=float(o[0]), spacing=1.0, amplitude=0.0,
                                         y_center=float(o[1]))

            policy = SteeredPolicy(env, learned_policy, args.vx, args.yaw, driver,
                                   set_joystick_twist_command)

            print(f"[view] {args.variant} on {args.terrain}  cmd vx={args.vx:+.3f} "
                  f"yaw={args.yaw:+.2f}  seed={args.seed}  "
                  f"steering={'lane-keep' if args.lane_keep else 'weave-drive' if args.weave_drive else 'fixed'}")
            print("[view] Close the viewer window or Ctrl+C to exit.")
            NativeMujocoViewer(env, policy).run()
        finally:
            if env is not None:
                env.close()


if __name__ == "__main__":
    main()
