"""Classify gait WITHOUT trusting any name->corner lookup.

The trot/pace call hinges entirely on which contact channel is front-left vs
hind-left.  Swap those two and a pace reads as a trot.  So this script never
looks a corner up by name:

  * it records each foot SITE's position in the trunk frame every step,
  * labels each foot F/H from its own mean x and L/R from its own mean y,
  * derives stance two independent ways -- the contact sensor channel, and the
    foot's own height above its per-foot minimum (pure kinematics, no sensor),
  * and reports the diagonal-vs-lateral statistic for both.

It also prints the contact sensor's RESOLVED geom order next to the site order
so any mismatch between the two is visible rather than assumed.

    python scripts/gait_from_geometry.py --variant pitch --vx 0.15 --seconds 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["MICROTAUR_USE_JOYSTICK_COMMANDS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from microtaur_variants import VARIANTS, use_variant  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = Path("rollouts/gait_geom")


def build(ckpt, device, robust):
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from rsl_rl.runners import OnPolicyRunner
  from microtaur_velocity.distill_reliable.mjlab_utils import cfg_to_dict

  task = "Mjlab-Velocity-Yaw-Flat-Microtaur"
  if robust:
    from microtaur_velocity.distill_reliable.mjlab_utils import _make_robust_distill_cfg
    cfg = _make_robust_distill_cfg(task, 1)
  else:
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    if "actor" in cfg.observations:
      cfg.observations["actor"].enable_corruption = False
  agent_cfg = load_rl_cfg(task)

  import mujoco
  from microtaur_velocity.microtaur_constants import get_spec
  m = get_spec().compile()
  bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, m.nbody)]
  if getattr(cfg.viewer, "body_name", None) not in bodies:
    cfg.viewer.body_name = bodies[0]

  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task) or OnPolicyRunner
  runner = runner_cls(env, cfg_to_dict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  return env, runner.get_inference_policy(device=device)


def describe_contact_order(u, report):
  """Print whatever the contact sensor exposes about its own channel order."""
  import mujoco
  sensor = u.scene["feet_ground_contact"]
  mjm = u.sim.mj_model
  # The authoritative index->name map (mjlab ContactSensor.primary_names).
  pn = list(getattr(sensor, "primary_names", []) or [])
  report["primary_names"] = pn
  if pn:
    print(f"[geom] ContactSensor.primary_names = {pn}")
  found = {}
  for attr in ("geom_ids", "primary_ids", "_primary_ids", "primary_geom_ids",
               "_geom_ids", "ids", "_ids", "primary", "_primary"):
    val = getattr(sensor, attr, None)
    if val is None:
      continue
    try:
      arr = np.asarray(val.detach().cpu() if hasattr(val, "detach") else val).ravel()
    except Exception:  # noqa: BLE001
      continue
    if arr.dtype.kind not in "iu" or arr.size == 0 or arr.size > 32:
      continue
    names = []
    for gid in arr.tolist():
      nm = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
      names.append(nm if nm is not None else f"<geom {gid}>")
    found[attr] = names
  report["contact_sensor_channel_order"] = found
  print("[geom] contact sensor channel order, as resolved by mjlab:")
  if not found:
    print("       (sensor exposes no readable id list; relying on geometry only)")
  for k, v in found.items():
    print(f"       {k}: {v}")
  return found


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--variant", choices=list(VARIANTS), default="pitch")
  ap.add_argument("--checkpoint", default=None)
  ap.add_argument("--vx", type=float, default=0.15)
  ap.add_argument("--seconds", type=float, default=20.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--robust", action="store_true")
  ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  args = ap.parse_args()

  v = VARIANTS[args.variant]
  ckpt = Path(args.checkpoint) if args.checkpoint else Path(v.ckpt)
  OUT.mkdir(parents=True, exist_ok=True)
  report = {"variant": args.variant, "checkpoint": str(ckpt), "cmd_vx": args.vx,
            "robust": args.robust, "seed": args.seed}

  with use_variant(args.variant):
    import mujoco
    from microtaur_velocity.distill_reliable.mjlab_utils import get_initial_obs, step_env
    from microtaur_velocity.env_cfgs import set_joystick_twist_command
    from microtaur_velocity.microtaur_constants import FOOT_SITE_NAMES

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    env, policy = build(ckpt, args.device, args.robust)
    u = env.unwrapped
    robot = u.scene["robot"]
    feet = u.scene["feet_ground_contact"]
    mjm = u.sim.mj_model

    site_names = list(FOOT_SITE_NAMES)
    all_sites = [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(mjm.nsite)]
    # site_pos_w is indexed by the entity's own site ordering; find it.
    ent_sites = list(getattr(robot, "site_names", []) or all_sites)
    site_idx = [ent_sites.index(n) for n in site_names]
    report["foot_site_names"] = site_names
    report["entity_site_order"] = ent_sites
    print(f"[geom] foot sites {site_names} -> site_pos_w columns {site_idx}")
    describe_contact_order(u, report)

    dt = float(getattr(u, "step_dt", 0.02))
    n = int(round(args.seconds / dt))
    obs = get_initial_obs(env)
    set_joystick_twist_command(env, args.vx, 0.0)

    P, C, Q, R, V = [], [], [], [], []
    for i in range(n):
      with torch.no_grad():
        raw = torch.nan_to_num(policy(obs), nan=0.0, posinf=6.0, neginf=-6.0)
      obs, _, _done, _ = step_env(env, raw)
      set_joystick_twist_command(env, args.vx, 0.0)
      P.append(robot.data.site_pos_w[0, site_idx].detach().cpu().numpy().copy())
      C.append(feet.data.found[0].float().detach().cpu().numpy().ravel()[:4].copy())
      R.append(robot.data.root_link_pos_w[0].detach().cpu().numpy().copy())
      Q.append(robot.data.root_link_quat_w[0].detach().cpu().numpy().copy())
      V.append(float(robot.data.root_link_lin_vel_b[0, 0]))
      if i % 250 == 0:
        print(f"[geom] {i}/{n}  v_body_x={V[-1]:+.3f}")
    env.close()

  P = np.array(P); C = np.array(C); R = np.array(R); Q = np.array(Q); V = np.array(V)
  ss = slice(50, None)

  # ---- foot position in the trunk frame (yaw-only, so left/right is honest) --
  w, x, y, z = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
  yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  d = P - R[:, None, :]
  cy, sy = np.cos(-yaw)[:, None], np.sin(-yaw)[:, None]
  bx = cy * d[:, :, 0] - sy * d[:, :, 1]
  by = sy * d[:, :, 0] + cy * d[:, :, 1]
  bz = P[:, :, 2]

  mx, my = bx[ss].mean(axis=0), by[ss].mean(axis=0)
  corner = ["%s%s" % ("F" if mx[i] > mx.mean() else "H",
                      "L" if my[i] > my.mean() else "R") for i in range(4)]
  report["measured_trunk_frame_xy_m"] = {
    site_names[i]: dict(x=round(float(mx[i]), 4), y=round(float(my[i]), 4), corner=corner[i])
    for i in range(4)}
  print("\n[geom] foot positions measured in the trunk frame (mean over the run):")
  for i in range(4):
    print(f"       ch{i} {site_names[i]:16s} x={mx[i]:+.4f} y={my[i]:+.4f}  -> {corner[i]}")
  if sorted(corner) != ["FL", "FR", "HL", "HR"]:
    print(f"[geom] WARNING: corners not unique: {corner}")

  # ---- stance, two independent ways ----------------------------------------
  zmin = bz[ss].min(axis=0); zmax = bz[ss].max(axis=0)
  kin = (bz < zmin + 0.25 * (zmax - zmin)).astype(float)   # lowest quarter of travel
  report["foot_z_travel_mm"] = {site_names[i]: round(float((zmax - zmin)[i]) * 1000, 2)
                                for i in range(4)}

  def stat(S, label):
    idx = {corner[i]: i for i in range(4)}
    def cr(a, b):
      p = S[ss, idx[a]] - S[ss, idx[a]].mean(); q = S[ss, idx[b]] - S[ss, idx[b]].mean()
      return float((p * q).mean() / (p.std() * q.std() + 1e-9))
    diag = float(np.mean([cr("FL", "HR"), cr("FR", "HL")]))
    lat = float(np.mean([cr("FL", "HL"), cr("FR", "HR")]))
    fore = float(np.mean([cr("FL", "FR"), cr("HL", "HR")]))
    best = max([("TROT", diag), ("PACE", lat), ("BOUND", fore)], key=lambda t: t[1])
    print(f"       {label:22s} diag(FL-HR,FR-HL)={diag:+.3f}  "
          f"lat(FL-HL,FR-HR)={lat:+.3f}  fore(FL-FR,HL-HR)={fore:+.3f}  -> {best[0]}")
    return dict(diag=diag, lateral=lat, fore_aft=fore, verdict=best[0],
                duty={corner[i]: round(float(S[ss, i].mean()), 3) for i in range(4)})

  print("\n[geom] pairwise contact correlation, corners taken from the geometry above:")
  report["by_contact_sensor"] = stat(C, "contact sensor")
  report["by_foot_height"] = stat(kin, "foot height (kinematic)")
  report["mean_v_body_x"] = float(V[ss].mean())
  print(f"\n[geom] mean v_body_x = {V[ss].mean():+.3f} m/s   "
        f"foot z travel (mm) = {[round(float(t)*1000,1) for t in (zmax-zmin)]}")

  tag = f"{args.variant}_vx{args.vx:.2f}{'_robust' if args.robust else ''}_seed{args.seed}"
  np.savez_compressed(OUT / f"{tag}.npz", bx=bx, by=by, bz=bz, contact=C, kin=kin,
                      v_body_x=V, corner=np.array(corner), sites=np.array(site_names))
  (OUT / f"{tag}.json").write_text(json.dumps(report, indent=2))
  print(f"[geom] wrote {OUT / (tag + '.json')} and .npz")


if __name__ == "__main__":
  main()
