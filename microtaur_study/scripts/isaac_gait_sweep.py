"""Checkpoint gait-type sweep (pronk / bound / pace / trot check) on level-0 (flat) Teacher-Cur tiles under the t1 reward weights:
gait metrics (flight, 4-contact, pair correlations) + per-term reward rates (weighted, per s)."""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser(); ap.add_argument("--ckpts", nargs="+", required=True, help="label=path")
ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=240)
ap.add_argument("--set", nargs="*", default=[], help="reward weight overrides name=value (applied on top of t1)")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args(); args.headless = True
app = AppLauncher(args).app
import numpy as np, torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from microtaur_isaac.agents import MicrotaurTeacherPPORunnerCfg
from microtaur_isaac.env_cfg import MicrotaurTeacherCurPlayEnvCfg
from microtaur_isaac.policy_loader import load_policy, prepare_env_cfg
from microtaur_isaac.mdp.contact import foot_contact_timers
from microtaur_isaac.mdp.observations import feet_sensor_cfg
N = 192
cfg = MicrotaurTeacherCurPlayEnvCfg(); cfg.scene.num_envs = N
cfg.curriculum.command_ranges = None; cfg.commands.twist.resampling_time_range = (1e9, 1e9)
rw = cfg.rewards
T1 = {"action_rate": -0.08, "trot_gait": 1.0, "feet_air_time": 1.0, "flat_orientation_l2": -10.0, "ang_vel_xy_l2": -0.05,
      "feet_slide": -0.1, "heading_tracking": 2.0}
for kv in args.set:
  k, v = kv.split("="); T1[k] = float(v)
for k, v in T1.items(): getattr(rw, k).weight = v
rw.trot_gait.params["std"] = 0.1; rw.trot_gait.params["max_err"] = 0.2
rw.track_lin_vel_xy.params["sigma"] = 0.12; rw.feet_air_time.params["mode_time_s"] = 0.2
prepare_env_cfg(cfg, args.ckpts[0].split('=', 1)[1])  # all checkpoints must share the actor history
env = ManagerBasedRLEnv(cfg)
t = env.scene.terrain; t.terrain_levels[:] = 0; t.env_origins[:] = t.terrain_origins[0, t.terrain_types]
w = RslRlVecEnvWrapper(env)
cmd = env.command_manager.get_term("twist"); robot = env.scene["robot"]; rm = env.reward_manager
fc = feet_sensor_cfg(); fc.resolve(env.scene)
SPEEDS = torch.tensor([0.10, 0.20, 0.35], device=env.device)[torch.arange(N, device=env.device) % 3]
names = list(rm.active_terms)
def corr(a, b):
  a = a - a.mean(0); b = b - b.mean(0)
  return float(np.nanmean((a * b).mean(0) / (a.std(0) * b.std(0) + 1e-9)))
res = {}
for spec in args.ckpts:
  label, path = spec.split("=", 1)
  policy, policy_reset = load_policy(w, path)
  C, RW, Z, VX, RR, PR, D = [], [], [], [], [], [], []
  with torch.inference_mode():
    env.reset(); obs = w.get_observations()
    for k in range(60 + args.steps):
      cmd.vel_command_b[:, 0] = SPEEDS; cmd.vel_command_b[:, 1:] = 0.0
      obs, _, dones, _ = w.step(policy(obs))
      policy_reset(dones)
      if k < 60: continue
      C.append(foot_contact_timers(env, fc)[0].cpu().numpy().copy())
      RW.append(rm._step_reward.cpu().numpy().copy())
      Z.append(robot.data.root_link_pos_w[:, 2].cpu().numpy()); VX.append(robot.data.root_link_lin_vel_b[:, 0].cpu().numpy())
      av = robot.data.root_link_ang_vel_b; RR.append(av[:, 0].cpu().numpy()); PR.append(av[:, 1].cpu().numpy())
      D.append(dones.cpu().numpy())
  C, RW, Z, VX, RR, PR, D = map(np.stack, (C, RW, Z, VX, RR, PR, D))
  ok = ~D.any(0)
  c = C[:, ok].astype(float); nf = c.sum(2)
  r = {"n_ok": int(ok.sum()), "flight": float((nf == 0).mean()), "four": float((nf == 4).mean()),
       "diag": np.mean([corr(c[..., 0], c[..., 2]), corr(c[..., 1], c[..., 3])]),
       "same_end": np.mean([corr(c[..., 0], c[..., 1]), corr(c[..., 2], c[..., 3])]),
       "same_side": np.mean([corr(c[..., 0], c[..., 3]), corr(c[..., 1], c[..., 2])]),
       "stride_hz": float(((c[1:] > .5) & (c[:-1] < .5)).sum(0).mean() / ((c.shape[0] - 1) * env.step_dt)),
       "z_p2p_mm": float(np.mean(np.percentile(Z[:, ok], 95, 0) - np.percentile(Z[:, ok], 5, 0)) * 1e3),
       "roll_rate_rms": float(np.sqrt((RR[:, ok] ** 2).mean())), "pitch_rate_rms": float(np.sqrt((PR[:, ok] ** 2).mean())),
       "vx": {f"{s:.2f}": float(VX[:, ok & np.isclose(SPEEDS.cpu().numpy(), s)].mean()) for s in (0.10, 0.20, 0.35)},
       "reward_rate": {n: float(RW[:, ok, i].mean()) for i, n in enumerate(names)}}
  r["reward_rate"]["TOTAL"] = float(sum(v for k, v in r["reward_rate"].items() if k != "metrics"))
  res[label] = r
  print(f"{label:>10s}: flight {r['flight']:.2f} four {r['four']:.2f} diag {r['diag']:+.2f} same_end {r['same_end']:+.2f} same_side {r['same_side']:+.2f} "
        f"stride {r['stride_hz']:.1f} Hz z {r['z_p2p_mm']:.1f} mm roll/pitch rate rms {r['roll_rate_rms']:.2f}/{r['pitch_rate_rms']:.2f} vx {r['vx']}", flush=True)
  print("            " + " ".join(f"{n[:10]} {v:+.3f}" for n, v in r["reward_rate"].items() if n != "metrics"), flush=True)
json.dump(res, open(args.out, "w"), indent=1)
sys.stdout.flush(); os._exit(0)
