"""Build the rigid Microtaur USD directly from the MJCF (the single source of truth).

The XML is compiled with MuJoCo (plus the motor-range patch shared with mjlab),
so masses, inertias, joint frames, collision primitives and closure sites are
the MuJoCo values exactly. Each leg's two MJCF <connect> constraints share one
axis, so one PhysX revolute loop-closure joint per leg (excluded from the
articulation tree) closes the five-bar; the closed-chain spike measured this at
pos_iters 16, dt 2.5 ms. Bodies are authored at the IK stand pose. Visual meshes
are not included (collision primitives only).

Requires pxr (call after the Isaac app is launched) and mujoco.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from microtaur_common.kinematics import FULL_JOINT_NAMES
from microtaur_common.reset import NOMINAL_FULL_Q
from microtaur_common.robot_constants import XML_PATH, motor_joint_ranges


def compile_model() -> mujoco.MjModel:
  spec = mujoco.MjSpec.from_file(str(XML_PATH))
  for name, (lo, hi) in motor_joint_ranges().items():
    j = spec.joint(name)
    j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    j.range[:] = (lo, hi)
  return spec.compile()


def build_usd(out_path: str) -> dict:
  from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

  m = compile_model()
  d = mujoco.MjData(m)
  d.qpos[3] = 1.0  # root at origin, identity orientation
  full = NOMINAL_FULL_Q.reshape(16)
  for name, q in zip(FULL_JOINT_NAMES, full):
    d.qpos[m.joint(name).qposadr[0]] = q
  mujoco.mj_kinematics(m, d)

  def qf(w):
    return Gf.Quatf(float(w[0]), Gf.Vec3f(*map(float, w[1:])))

  def qd(w):
    return Gf.Quatd(float(w[0]), Gf.Vec3d(*map(float, w[1:])))

  stage = Usd.Stage.CreateNew(out_path)
  UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
  UsdGeom.SetStageMetersPerUnit(stage, 1.0)
  UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
  root = UsdGeom.Xform.Define(stage, "/microtaur")
  stage.SetDefaultPrim(root.GetPrim())
  UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

  mats = {}

  def material(mu):
    if mu not in mats:
      mat = UsdShade.Material.Define(stage, f"/microtaur/materials/mu_{int(round(mu * 100))}")
      api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
      api.CreateStaticFrictionAttr(mu)
      api.CreateDynamicFrictionAttr(mu)
      api.CreateRestitutionAttr(0.0)
      mats[mu] = mat
    return mats[mu]

  body_names = [m.body(b).name for b in range(1, m.nbody)]
  for b in range(1, m.nbody):
    xf = UsdGeom.Xform.Define(stage, f"/microtaur/{m.body(b).name}")
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, d.xpos[b])))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(qd(d.xquat[b]))
    UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
    mapi = UsdPhysics.MassAPI.Apply(xf.GetPrim())
    mapi.CreateMassAttr(float(m.body_mass[b]))
    mapi.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, m.body_ipos[b])))
    mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, m.body_inertia[b])))
    mapi.CreatePrincipalAxesAttr(qf(m.body_iquat[b]))

  for g in range(m.ngeom):
    if m.geom_bodyid[g] == 0 or not (m.geom_contype[g] or m.geom_conaffinity[g]):
      continue
    gtype = int(m.geom_type[g])
    path = f"/microtaur/{m.body(m.geom_bodyid[g]).name}/{m.geom(g).name}"
    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
      geom = UsdGeom.Sphere.Define(stage, path)
      geom.CreateRadiusAttr(float(m.geom_size[g][0]))
    elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
      geom = UsdGeom.Cube.Define(stage, path)
      geom.CreateSizeAttr(2.0)
    else:
      raise ValueError(f"unsupported collision geom type {gtype} ({m.geom(g).name})")
    geom.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, m.geom_pos[g])))
    geom.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(qd(m.geom_quat[g]))
    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
      geom.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, m.geom_size[g])))
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(
      material(float(m.geom_friction[g][0])), UsdShade.Tokens.weakerThanDescendants, "physics"
    )

  for j in range(m.njnt):
    if int(m.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE):
      continue
    assert np.allclose(m.jnt_axis[j], (0, 0, 1)) and np.allclose(m.jnt_pos[j], 0), m.joint(j).name
    child = int(m.jnt_bodyid[j])
    jt = UsdPhysics.RevoluteJoint.Define(stage, f"/microtaur/joints/{m.joint(j).name}")
    jt.CreateBody0Rel().SetTargets([f"/microtaur/{m.body(int(m.body_parentid[child])).name}"])
    jt.CreateBody1Rel().SetTargets([f"/microtaur/{m.body(child).name}"])
    jt.CreateLocalPos0Attr(Gf.Vec3f(*map(float, m.body_pos[child])))
    jt.CreateLocalRot0Attr(qf(m.body_quat[child]))
    jt.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    jt.CreateLocalRot1Attr(Gf.Quatf(1.0))
    jt.CreateAxisAttr("Z")
    if m.jnt_limited[j]:
      jt.CreateLowerLimitAttr(math.degrees(float(m.jnt_range[j][0])))
      jt.CreateUpperLimitAttr(math.degrees(float(m.jnt_range[j][1])))
    drive = UsdPhysics.DriveAPI.Apply(jt.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(0.0)

  # Loop closures: the first <connect> of each leg (the *_z twin shares its axis).
  loops = []
  for i in range(m.neq):
    s1, s2 = int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
    name = m.site(s1).name
    if name.endswith("_z"):
      continue
    jt = UsdPhysics.RevoluteJoint.Define(stage, f"/microtaur/loops/{name}")
    jt.CreateBody0Rel().SetTargets([f"/microtaur/{m.body(int(m.site_bodyid[s1])).name}"])
    jt.CreateBody1Rel().SetTargets([f"/microtaur/{m.body(int(m.site_bodyid[s2])).name}"])
    jt.CreateLocalPos0Attr(Gf.Vec3f(*map(float, m.site_pos[s1])))
    jt.CreateLocalPos1Attr(Gf.Vec3f(*map(float, m.site_pos[s2])))
    jt.CreateLocalRot0Attr(Gf.Quatf(1.0))
    jt.CreateLocalRot1Attr(Gf.Quatf(1.0))
    jt.CreateAxisAttr("Z")
    jt.CreateExcludeFromArticulationAttr(True)
    loops.append(name)
  stage.GetRootLayer().Save()
  return {"bodies": body_names, "total_mass_kg": float(m.body_mass.sum()), "loops": loops,
          "stand_full_q": dict(zip(FULL_JOINT_NAMES, map(float, full)))}
