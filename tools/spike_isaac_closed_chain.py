"""IsaacLab / PhysX closed-chain feasibility spike for the rigid Microtaur.

1. Builds a USD directly from the patched, compiled MuJoCo model exported by
   `tools/spike_mujoco_reference.py --export-model model.json` (so masses,
   inertias, joint frames and closure sites are identical to the MuJoCo
   reference). The four five-bar loops are closed with PhysX loop-closure joints
   (`physics:excludeFromArticulation = true`).
2. Spawns N robots on a plane in the IK stand pose, holds the 8 motors with
   IsaacLab's DCMotor (same KP / effort / saturation / velocity limits as
   training), runs for --seconds, and reports closing-site gaps, joint jitter,
   root drift, NaNs and env-steps/s.

Usage (isaaclab conda env):
  python tools/spike_isaac_closed_chain.py --model model.json --num_envs 512
      [--seconds 20] [--loop revolute|spherical2] [--pos_iters 4] [--vel_iters 0]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--usd", default="")
ap.add_argument("--num_envs", type=int, default=1)
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--throughput_seconds", type=float, default=10.0)
ap.add_argument("--loop", choices=("revolute", "spherical2"), default="revolute")
ap.add_argument("--pos_iters", type=int, default=4)
ap.add_argument("--vel_iters", type=int, default=0)
ap.add_argument("--kd", type=float, default=0.045)
ap.add_argument("--dt", type=float, default=0.005)
ap.add_argument("--decimation", type=int, default=4)
ap.add_argument("--jitter_envs", type=int, default=16)
ap.add_argument("--motion", choices=("stand", "trot", "drop"), default="stand")
ap.add_argument("--drop_height", type=float, default=0.010)
ap.add_argument("--out", default="")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app

# Kit swallows stdout; write everything to a log file as well.
_log = open(args.out.replace(".json", ".txt") if args.out else os.devnull, "w")


def log(*a):
  s = " ".join(str(x) for x in a)
  _log.write(s + "\n")
  _log.flush()
  print(s, flush=True)


import numpy as np
import torch
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

M = json.load(open(args.model))
BODY = {b["name"]: b for b in M["bodies"]}
ROOT = "battery"


def _q(qwxyz, cls=Gf.Quatf):
  w, x, y, z = qwxyz
  vec = Gf.Vec3f if cls is Gf.Quatf else Gf.Vec3d
  return cls(w, vec(x, y, z))


def build_usd(path: str) -> None:
  stage = Usd.Stage.CreateNew(path)
  UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
  UsdGeom.SetStageMetersPerUnit(stage, 1.0)
  UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
  root = UsdGeom.Xform.Define(stage, "/microtaur")
  stage.SetDefaultPrim(root.GetPrim())
  UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

  mats = {}

  def material(mu: float):
    if mu not in mats:
      mat = UsdShade.Material.Define(stage, f"/microtaur/materials/mu_{int(round(mu * 100))}")
      api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
      api.CreateStaticFrictionAttr(mu)
      api.CreateDynamicFrictionAttr(mu)
      api.CreateRestitutionAttr(0.0)
      mats[mu] = mat
    return mats[mu]

  for b in M["bodies"]:
    xf = UsdGeom.Xform.Define(stage, f"/microtaur/{b['name']}")
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*b["world_pos"]))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(_q(b["world_quat"], Gf.Quatd))
    prim = xf.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mapi = UsdPhysics.MassAPI.Apply(prim)
    mapi.CreateMassAttr(b["mass"])
    mapi.CreateCenterOfMassAttr(Gf.Vec3f(*b["ipos"]))
    mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*b["inertia"]))
    mapi.CreatePrincipalAxesAttr(_q(b["iquat"]))

  for g in M["geoms"]:
    p = f"/microtaur/{g['body']}/{g['name']}"
    if g["type"] == "sphere":
      geom = UsdGeom.Sphere.Define(stage, p)
      geom.CreateRadiusAttr(g["size"][0])
    else:
      geom = UsdGeom.Cube.Define(stage, p)
      geom.CreateSizeAttr(2.0)
    geom.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*g["pos"]))
    geom.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(_q(g["quat"], Gf.Quatd))
    if g["type"] == "box":
      geom.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*g["size"]))
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(
      material(g["friction"]), UsdShade.Tokens.weakerThanDescendants, "physics"
    )

  for j in M["joints"]:
    assert j["axis"] == [0.0, 0.0, 1.0], j
    jt = UsdPhysics.RevoluteJoint.Define(stage, f"/microtaur/joints/{j['name']}")
    jt.CreateBody0Rel().SetTargets([f"/microtaur/{j['parent']}"])
    jt.CreateBody1Rel().SetTargets([f"/microtaur/{j['child']}"])
    c = BODY[j["child"]]
    jt.CreateLocalPos0Attr(Gf.Vec3f(*c["pos"]))
    jt.CreateLocalRot0Attr(_q(c["quat"]))
    jt.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    jt.CreateLocalRot1Attr(Gf.Quatf(1.0))
    jt.CreateAxisAttr("Z")
    if j["limited"]:
      jt.CreateLowerLimitAttr(math.degrees(j["range"][0]))
      jt.CreateUpperLimitAttr(math.degrees(j["range"][1]))
    drive = UsdPhysics.DriveAPI.Apply(jt.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(0.0)

  # Each leg has two MuJoCo <connect> constraints whose site pairs lie on the
  # same z axis, i.e. together they form a revolute closure. "revolute" uses one
  # PhysX revolute loop joint per leg (5 constrained DoF); "spherical2" mirrors
  # the MJCF literally with two spherical joints per leg (6 rows, 1 redundant).
  loops = M["loops"]
  if args.loop == "revolute":
    loops = [l for l in loops if not l["name"].endswith("_z")]
  for l in loops:
    cls = UsdPhysics.RevoluteJoint if args.loop == "revolute" else UsdPhysics.SphericalJoint
    jt = cls.Define(stage, f"/microtaur/loops/{l['name']}")
    jt.CreateBody0Rel().SetTargets([f"/microtaur/{l['body0']}"])
    jt.CreateBody1Rel().SetTargets([f"/microtaur/{l['body1']}"])
    jt.CreateLocalPos0Attr(Gf.Vec3f(*l["pos0"]))
    jt.CreateLocalPos1Attr(Gf.Vec3f(*l["pos1"]))
    jt.CreateLocalRot0Attr(Gf.Quatf(1.0))
    jt.CreateLocalRot1Attr(Gf.Quatf(1.0))
    jt.CreateAxisAttr("Z")
    jt.CreateExcludeFromArticulationAttr(True)
  stage.GetRootLayer().Save()


usd_path = args.usd or os.path.join(os.path.dirname(os.path.abspath(args.model)), f"microtaur_rigid_{args.loop}.usda")
build_usd(usd_path)
log(f"USD written: {usd_path}")

MOTORS = [n for n in M["stand_q"] if n.endswith("_act")]
PASSIVE = [n for n in M["stand_q"] if not n.endswith("_act")]

ROBOT_CFG = ArticulationCfg(
  prim_path="{ENV_REGEX_NS}/Robot",
  spawn=sim_utils.UsdFileCfg(
    usd_path=usd_path,
    activate_contact_sensors=False,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=1.0),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
      enabled_self_collisions=False,
      solver_position_iteration_count=args.pos_iters,
      solver_velocity_iteration_count=args.vel_iters,
    ),
    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
  ),
  init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, M["nominal_root_z"]),
    joint_pos=dict(M["stand_q"]),
    joint_vel={".*": 0.0},
  ),
  actuators={
    "motors": DCMotorCfg(
      joint_names_expr=MOTORS,
      stiffness=1.0,
      damping=args.kd,
      effort_limit=0.60 * 0.215,
      saturation_effort=0.215,
      velocity_limit=0.80 * 40.11,
      armature=2e-4,
      friction=0.0,
    ),
    "passive": ImplicitActuatorCfg(
      joint_names_expr=PASSIVE,
      stiffness=0.0,
      damping=1e-4,
      armature=1e-6,
      friction=0.0,
    ),
  },
)


@configclass
class SceneCfg(InteractiveSceneCfg):
  ground = AssetBaseCfg(
    prim_path="/World/ground",
    spawn=sim_utils.GroundPlaneCfg(
      physics_material=sim_utils.RigidBodyMaterialCfg(
        static_friction=1.0, dynamic_friction=1.0, friction_combine_mode="max"
      )
    ),
  )
  robot: ArticulationCfg = ROBOT_CFG

sim = sim_utils.SimulationContext(
  sim_utils.SimulationCfg(dt=args.dt, render_interval=args.decimation, device="cuda:0")
)
scene = InteractiveScene(SceneCfg(num_envs=args.num_envs, env_spacing=0.5, replicate_physics=True))
sim.reset()
robot: Articulation = scene["robot"]
dev = robot.device
N = args.num_envs

log(f"num_envs {N}  loop {args.loop}  pos_iters {args.pos_iters}  vel_iters {args.vel_iters}  kd {args.kd}  dt {args.dt}  decimation {args.decimation}")
log(f"joints ({robot.num_joints}): {robot.joint_names}")
log(f"bodies ({robot.num_bodies}): {robot.body_names}")
mass = robot.root_physx_view.get_masses()[0]
log(f"PhysX total mass {float(mass.sum()):.10f} kg (MuJoCo {M['total_mass']:.10f})")
inert = robot.root_physx_view.get_inertias()[0].reshape(-1, 3, 3)
for i, bn in enumerate(robot.body_names):
  b = BODY[bn]
  log(f"  {bn:30s} mass {float(mass[i]):.9f} (mj {b['mass']:.9f})  eig(I) {np.sort(np.linalg.eigvalsh(inert[i].cpu().numpy()))}  mj {np.sort(b['inertia'])}")

# Closure site pairs (all 8, including the *_z ones, as in MuJoCo).
bidx = {n: i for i, n in enumerate(robot.body_names)}
L = M["loops"]
b0 = torch.tensor([bidx[l["body0"]] for l in L], device=dev)
b1 = torch.tensor([bidx[l["body1"]] for l in L], device=dev)
p0 = torch.tensor([l["pos0"] for l in L], device=dev, dtype=torch.float32)
p1 = torch.tensor([l["pos1"] for l in L], device=dev, dtype=torch.float32)


def closure_gaps() -> torch.Tensor:
  pose = robot.data.body_link_pose_w  # [N, B, 7] pos + quat(wxyz)
  s0 = pose[:, b0, :3] + quat_apply(pose[:, b0, 3:7], p0.expand(N, -1, -1))
  s1 = pose[:, b1, :3] + quat_apply(pose[:, b1, 3:7], p1.expand(N, -1, -1))
  return torch.linalg.norm(s0 - s1, dim=-1)  # [N, 8]


jorder = [robot.joint_names.index(n) for n in M["stand_q"]]
target = robot.data.default_joint_pos.clone()
motor_cols = torch.tensor([robot.joint_names.index(n) for n in MOTORS], device=dev)
SIGNS = (1.0, -1.0, -1.0, 1.0)  # MINITAUR_SWING_SIGNS
TROT_COMMON, TROT_DIFF, TROT_HZ = 0.25, 0.15, 2.0


def motion_target(t: float) -> torch.Tensor:
  """Same trot as spike_mujoco_reference.motion_delta (MOTORS order: leg1 a, leg1 e, ...)."""
  if args.motion != "trot":
    return target
  delta = torch.zeros(8, device=dev)
  for leg in range(4):
    ph = 2 * math.pi * TROT_HZ * t + (0.0 if leg in (0, 2) else math.pi)
    common, diff = TROT_COMMON * math.sin(ph), TROT_DIFF * math.cos(ph)
    delta[2 * leg] = SIGNS[leg] * (common - diff)
    delta[2 * leg + 1] = SIGNS[leg] * (common + diff)
  out = target.clone()
  out[:, motor_cols] += delta
  return out


def reset_all():
  rs = robot.data.default_root_state.clone()
  rs[:, :3] += scene.env_origins
  if args.motion == "drop":
    rs[:, 2] += args.drop_height
  robot.write_root_state_to_sim(rs)
  robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
  scene.reset()


reset_all()
robot.update(0.0)
g0 = closure_gaps()
log(f"t=0 closure gaps: max {float(g0.max()):.3e} m")

n_phys = int(round(args.seconds / args.dt))
J = min(args.jitter_envs, N)
gap_hist = torch.zeros(n_phys, N, len(L), device=dev)
q_hist = torch.zeros(n_phys, J, 16, device=dev)
rootz = torch.zeros(n_phys, N, device=dev)
tilt = torch.zeros(n_phys, N, device=dev)
xy0 = None
nan_step = None
torch.cuda.synchronize()
t0 = time.perf_counter()
for k in range(n_phys):
  if k % args.decimation == 0:
    robot.set_joint_position_target(motion_target(k * args.dt))
  scene.write_data_to_sim()
  sim.step(render=False)
  scene.update(args.dt)
  gap_hist[k] = closure_gaps()
  q_hist[k] = robot.data.joint_pos[:J][:, jorder]
  rootz[k] = robot.data.root_link_pos_w[:, 2] - scene.env_origins[:, 2]
  pg = robot.data.projected_gravity_b
  tilt[k] = torch.rad2deg(torch.acos(torch.clamp(-pg[:, 2], -1.0, 1.0)))
  if nan_step is None and not torch.isfinite(robot.data.root_state_w).all():
    nan_step = k
torch.cuda.synchronize()
t_measure = time.perf_counter() - t0
root_xy = (robot.data.root_link_pos_w[:, :2] - scene.env_origins[:, :2]).cpu()
# Statistics on CPU: the GPU may be shared and nearly full.
gap_hist, q_hist, rootz, tilt = gap_hist.cpu(), q_hist.cpu(), rootz.cpu(), tilt.cpu()

# ---- statistics over the settled second half ----
h = slice(n_phys // 2, n_phys)
g = gap_hist[h]
finite = torch.isfinite(g)
gf = g[finite]
q_all = gap_hist.flatten()
q_all = q_all[torch.isfinite(q_all)]


def pct(x, p):
  x = x.flatten()
  if x.numel() > 16_000_000:
    x = x[torch.randint(0, x.numel(), (16_000_000,), device=x.device)]
  return float(torch.quantile(x.float(), p))


per_env_max = torch.nan_to_num(g, nan=1.0).amax(dim=(0, 2))
bad_1mm = int((per_env_max > 1e-3).sum())
bad_5mm = int((per_env_max > 5e-3).sum())
fell = int(((rootz[h] < 0.05217) | ~torch.isfinite(rootz[h])).any(dim=0).sum())

# joint jitter: residual of a centred 0.1 s moving average (content above ~10 Hz)
w = max(3, int(round(0.1 / args.dt)) | 1)
qh = q_hist[h].permute(1, 2, 0).reshape(-1, 1, q_hist[h].shape[0])
lo = torch.nn.functional.conv1d(qh, torch.ones(1, 1, w) / w, padding=w // 2)
res = (qh - lo)[..., w:-w].reshape(J, 16, -1)
hf = torch.sqrt((res**2).mean(dim=-1))  # [J, 16]

rz = rootz[h]
fell_full = int(((rootz < 0.05217) | ~torch.isfinite(rootz)).any(dim=0).sum())
res_out = {
  "motion": args.motion, "full_gap_mean": float(q_all.mean()), "full_gap_p95": pct(q_all, 0.95),
  "full_gap_p99": pct(q_all, 0.99), "full_gap_max": float(q_all.max()), "rootz_min": float(torch.nan_to_num(rootz, nan=0.0).min()),
  "fell_or_nan_envs_fullrun": fell_full,
  "frac_samples_gt_0p5mm": float((q_all > 5e-4).float().mean()), "frac_samples_gt_1mm": float((q_all > 1e-3).float().mean()),
  "frac_samples_gt_2mm": float((q_all > 2e-3).float().mean()),
  "t_of_max_s": float(int(torch.nan_to_num(gap_hist, nan=0.0).amax(dim=(1, 2)).argmax()) * args.dt),
  "per_step_max_after_1s_max": float(torch.nan_to_num(gap_hist[int(1.0 / args.dt):], nan=0.0).max()),
  "per_pair_max": torch.nan_to_num(gap_hist, nan=0.0).amax(dim=(0, 1)).tolist(),
  "envs_fullrun_gap_gt_1mm": int((torch.nan_to_num(gap_hist, nan=1.0).amax(dim=(0, 2)) > 1e-3).sum()),
  "num_envs": N, "loop": args.loop, "pos_iters": args.pos_iters, "vel_iters": args.vel_iters, "kd": args.kd,
  "gap_mean": float(gf.mean()), "gap_p95": pct(gf, 0.95), "gap_max": float(gf.max()),
  "gap_max_fullrun": float(q_all.max()),
  "envs_gap_gt_1mm": bad_1mm, "envs_gap_gt_5mm": bad_5mm, "nonfinite_gap_samples": int((~finite).sum()),
  "hf_rms_max": float(hf.max()), "hf_rms_mean": float(hf.mean()),
  "hf_rms_act_mean": float(hf[:, [i for i, n in enumerate(M["stand_q"]) if n.endswith("_act")]].mean()),
  "rootz_mean": float(rz.mean()), "rootz_std_across_envs": float(rz.mean(0).std()) if N > 1 else 0.0,
  "rootz_drift_mean": float((rz[-1] - rz[0]).mean()), "rootz_drift_absmax": float((rz[-1] - rz[0]).abs().max()),
  "tilt_final_mean_deg": float(tilt[-1].mean()), "tilt_final_max_deg": float(tilt[-1].max()),
  "xy_final_max": float(torch.linalg.norm(root_xy, dim=-1).max()),
  "fell_or_nan_envs": fell, "nan_step": nan_step,
  "q_mean": q_hist[h].mean(dim=(0, 1)).tolist(),
  "measure_wall_s": t_measure,
}

# ---- throughput: pure control loop, no metric collection ----
reset_all()
n_ctrl = int(round(args.throughput_seconds / (args.dt * args.decimation)))
for _ in range(10):  # warm-up
  robot.set_joint_position_target(target)
  for _ in range(args.decimation):
    scene.write_data_to_sim(); sim.step(render=False); scene.update(args.dt)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(n_ctrl):
  robot.set_joint_position_target(target)
  for _ in range(args.decimation):
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(args.dt)
torch.cuda.synchronize()
wall = time.perf_counter() - t0
res_out["env_steps_per_s"] = N * n_ctrl / wall
res_out["phys_steps_per_s_per_env"] = n_ctrl * args.decimation / wall
res_out["realtime_factor_per_env"] = n_ctrl * args.decimation * args.dt / wall

log(json.dumps({k: v for k, v in res_out.items() if k != "q_mean"}, indent=1))
log("settled joint means vs stand:")
for i, n in enumerate(M["stand_q"]):
  log(f"  {n:18s} {res_out['q_mean'][i]:+.5f}  stand {M['stand_q'][n]:+.5f}")
if args.out:
  json.dump(res_out, open(args.out, "w"), indent=1)
_log.close()
# SimulationApp.close() hangs on this Isaac Sim 5.0 install; outputs are flushed.
os._exit(0)
