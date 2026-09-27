"""Evaluate a Microtaur teacher policy on the A/B/C terrain grid, per terrain type and level.

  OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_eval_terrain.py --checkpoint <model.pt> --out <dir> \
      [--num-envs 500] [--speed 0.20] [--seconds 20]

Teacher-Play config (no noise / randomisation / timeout, terrain curriculum off,
robots spread uniformly over all levels and the A/B/C columns). Every env walks
straight at --speed (C is straight anyway) for --seconds. Per env: progress along
its spawn heading until its first termination (or the end), whether and why it
terminated (falls and time-out truncations such as leaving the map counted apart),
mean forward speed, heading change and lateral drift from the spawn heading. Writes terrain_eval.json and prints a table of
type x level: n, fall rate, mean progress, fraction reaching half a tile (the
curriculum's promotion distance), mean speed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--num-envs", type=int, default=500)
ap.add_argument("--speed", type=float, default=0.20)
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--terrain-scale", type=float, default=None,
                help="B heights / C step multiplier (default: read terrain_scale from <ckpt dir>/params/env.yaml, else 1)")
ap.add_argument("--blind", action="store_true",
                help="blindfolded teacher: its height map reads flat ground at the tile origin height")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
LOG = open(out / "terrain_eval.log", "w")


def log(*a):
  LOG.write(" ".join(map(str, a)) + "\n"); LOG.flush()


import re  # noqa: E402

import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from microtaur_isaac import terrains as TR  # noqa: E402
from microtaur_isaac.agents import MicrotaurTeacherPPORunnerCfg  # noqa: E402
from microtaur_isaac.env_cfg import MicrotaurTeacherCurPlayEnvCfg, MicrotaurTeacherPlayEnvCfg  # noqa: E402
from microtaur_isaac.policy_loader import load_policy, prepare_env_cfg  # noqa: E402

y = Path(args.checkpoint).parent / "params" / "env.yaml"
if args.terrain_scale is None:
  m = re.search(r"^terrain_scale: ([0-9.eE+-]+)", y.read_text(), re.M) if y.exists() else None
  args.terrain_scale = float(m.group(1)) if m else 1.0
# Teacher-Cur runs (railed C lane, 7 levels + run-out row) are evaluated on their own terrain.
cur = y.exists() and re.search(r"^cur_terrain: true", y.read_text(), re.M) is not None
cfg = (MicrotaurTeacherCurPlayEnvCfg if cur else MicrotaurTeacherPlayEnvCfg)(terrain_scale=args.terrain_scale)
cfg.scene.num_envs = args.num_envs
cfg.curriculum.command_ranges = None
cfg.commands.twist.resampling_time_range = (1.0e9, 1.0e9)
m = re.search(r"action_scale_rad: ([0-9.eE+-]+)", y.read_text()) if y.exists() else None
if m:
  cfg.actions.joint_pos.action_scale_rad = float(m.group(1))
prepare_env_cfg(cfg, args.checkpoint)
env = ManagerBasedRLEnv(cfg)
wrapped = RslRlVecEnvWrapper(env)
policy, policy_reset = load_policy(wrapped, args.checkpoint)
if args.blind:
  om = env.observation_manager
  for group in om.active_terms:
    if "height_scan" in om.active_terms[group]:
      tc = om._group_obs_term_cfgs[group][om.active_terms[group].index("height_scan")]

      def _flat_scan(env, **kw):
        max_distance = kw.get("max_distance", 0.3)
        hits = env.scene.sensors[kw.get("sensor_name", "height_scanner")].data.ray_hits_w
        h = env.scene["robot"].data.root_link_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3]
        return torch.clamp(h, max=max_distance).expand(-1, hits.shape[1]).clone()

      tc.func = _flat_scan
      log(f"BLIND: {group}/height_scan replaced by flat ground at the env origin height")
    if "foot_height" in om.active_terms[group]:
      # foot_height measures against the height-scanner hits under each foot: without this the
      # blindfolded teacher would still see the local ground through 4 values (critic audit 2026-09-27)
      fc = om._group_obs_term_cfgs[group][om.active_terms[group].index("foot_height")]
      fc.params["scanner_name"] = None
      log(f"BLIND: {group}/foot_height measured against the env origin height instead of the scanner")

robot = env.scene["robot"]
terrain = env.scene.terrain
cmd = env.command_manager.get_term("twist")
types = TR.env_terrain_type_ids(terrain)  # 0 A, 1 B, 2 C
levels = terrain.terrain_levels.clone()
N = env.num_envs


def pin():
  cmd.vel_command_b[:, 0] = args.speed
  cmd.vel_command_b[:, 1:] = 0.0


pin()
obs = wrapped.get_observations()
d = robot.data
start = (d.root_link_pos_w[:, :2]).clone()
q = d.root_link_quat_w
heading = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
fwd = torch.stack((torch.cos(heading), torch.sin(heading)), dim=1)
alive = torch.ones(N, dtype=torch.bool, device=env.device)
progress = torch.zeros(N, device=env.device)
lateral = torch.zeros(N, device=env.device)
yaw0 = heading.clone()
dyaw = torch.zeros(N, device=env.device)
side = torch.stack((-fwd[:, 1], fwd[:, 0]), dim=1)  # left of the spawn heading
speed_sum = torch.zeros(N, device=env.device)
steps = torch.zeros(N, device=env.device)
cause = ["" for _ in range(N)]
tm = env.termination_manager
T = int(args.seconds / env.step_dt)
with torch.inference_mode():
  for t in range(T):
    pin()
    prev = d.root_link_pos_w[:, :2].clone()
    obs, _, dones, _ = wrapped.step(policy(obs))
    policy_reset(dones)
    now = torch.where(dones[:, None].bool(), prev, d.root_link_pos_w[:, :2])  # a reset moves the root: keep the last pose
    p = torch.sum((now - start) * fwd, dim=1)
    progress = torch.where(alive, p, progress)
    lateral = torch.where(alive, torch.sum((now - start) * side, dim=1), lateral)
    qn = d.root_link_quat_w
    yaw_now = torch.atan2(2 * (qn[:, 0] * qn[:, 3] + qn[:, 1] * qn[:, 2]), 1 - 2 * (qn[:, 2] ** 2 + qn[:, 3] ** 2))
    dy = torch.atan2(torch.sin(yaw_now - yaw0), torch.cos(yaw_now - yaw0))
    dyaw = torch.where(alive & ~dones.bool(), dy, dyaw)
    speed_sum += torch.where(alive, d.root_link_lin_vel_b[:, 0], torch.zeros_like(p))
    steps += alive.float()
    newly = alive & dones.bool()
    for i in torch.nonzero(newly).flatten().tolist():
      cause[i] = next((n for n in tm.active_terms if bool(tm.get_term(n)[i])), "?")
    alive &= ~dones.bool()

half_tile = TR.TILE_SIZE_M[0] / 2
names = ("A_flat", "B_blocks", "C_step")
rows = []
log(f"checkpoint {args.checkpoint}\n{'Teacher-Cur terrain; ' if cur else ''}terrain scale {args.terrain_scale}; straight {args.speed} m/s for {args.seconds} s; promotion distance (half a tile) {half_tile:.2f} m\n")
log(f"{'terrain':9s} {'lvl':>3s} {'n':>4s} {'fall':>6s} {'trunc':>6s} {'progress m':>11s} {'>= half tile':>12s} {'speed m/s':>10s} {'|dyaw| deg':>11s} {'|lateral| m':>12s}  causes")
for ty in range(3):
  for lv in range(int(levels.max()) + 1):
    sel = (types == ty) & (levels == lv)
    n = int(sel.sum())
    if n == 0:
      continue
    ended = (~alive & sel)
    trunc_names = {n for n in tm.active_terms if tm.get_term_cfg(n).time_out}
    is_trunc = torch.tensor([cause[i] in trunc_names for i in range(N)], device=env.device)
    fell = ended & ~is_trunc
    trunc = ended & is_trunc
    pr = progress[sel]
    sp = (speed_sum[sel] / steps[sel].clamp_min(1))
    cs = {}
    for i in torch.nonzero(ended).flatten().tolist():
      cs[cause[i]] = cs.get(cause[i], 0) + 1
    row = {"terrain": names[ty], "level": lv, "n": n, "fall_rate": float(fell.sum()) / n, "trunc_rate": float(trunc.sum()) / n,
           "abs_dyaw_deg": float(torch.rad2deg(dyaw[sel].abs()).mean()), "abs_lateral_m": float(lateral[sel].abs().mean()),
           "progress_m": float(pr.mean()),
           "reach_half_tile": float((pr >= half_tile).float().mean()), "speed_m_s": float(sp.mean()), "causes": cs}
    rows.append(row)
    log(f"{row['terrain']:9s} {lv:3d} {n:4d} {row['fall_rate']:6.2f} {row['trunc_rate']:6.2f} {row['progress_m']:11.2f} "
        f"{row['reach_half_tile']:12.2f} {row['speed_m_s']:10.3f} {row['abs_dyaw_deg']:11.1f} {row['abs_lateral_m']:12.2f}  {cs}")
(out / "terrain_eval.json").write_text(json.dumps({"checkpoint": args.checkpoint, "speed": args.speed, "rows": rows}, indent=1))
LOG.close()
os._exit(0)
