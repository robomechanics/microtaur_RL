"""Rewrite the leg-link <inertial> elements of the active-yaw MJCF from the rigid model's.

  python scripts/fix_yaw_leg_inertials.py [--check]

The yaw robot's legs are the rigid robot's parts (same masses, principal inertias, link lengths). In
assets/active_yaw_microtaur/robot_2imu.xml several link bodies have a rotated body frame (quat) whose
<inertial> was not rotated with it: at the stand pose the COM of leg 2's spacer / lower link, leg 3's
lower link and leg 4's spacer / lower link sits 15-62 mm away from where the rigid model (all four legs
consistent) puts it, and their inertia tensors point the wrong way. Each of the 16 leg-link bodies gets
the rigid model's COM and inertia tensor, expressed relative to its own leg's motor-pivot midpoint in
the root frame (spine 0), converted into the yaw body's frame (pos + fullinertia, mass unchanged).
--check only reports the differences.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from microtaur_common.kinematics import FULL_JOINT_NAMES  # noqa: E402
from microtaur_common.reset import NOMINAL_FULL_Q  # noqa: E402
from microtaur_common.robot_constants import YAW_ROOT_BODY, YAW_XML_PATH  # noqa: E402
from microtaur_isaac.usd import compile_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true")
args = ap.parse_args()


def stand(m):
  d = mujoco.MjData(m)
  free = next(j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
  d.qpos[m.jnt_qposadr[free] + 3] = 1.0
  for n, q in zip(FULL_JOINT_NAMES, NOMINAL_FULL_Q.reshape(16)):
    d.qpos[m.jnt_qposadr[m.joint(n).id]] = q
  mujoco.mj_forward(m, d)
  return d


def leg_links(m, d, root):
  """{body name: (COM rel. hip in root frame, inertia tensor in root frame)} for the 16 leg-link bodies."""
  R0 = d.xmat[m.body(root).id].reshape(3, 3)
  out = {}
  for leg in range(1, 5):
    ja, je = m.joint(f"leg{leg}_a_joint_act").id, m.joint(f"leg{leg}_e_joint_act").id
    mid = 0.5 * (d.xanchor[ja] + d.xanchor[je])
    for j in ("a_joint_act", "e_joint_act", "b_joint", "d_joint"):
      b = int(m.jnt_bodyid[m.joint(f"leg{leg}_{j}").id])
      Ri = d.ximat[b].reshape(3, 3)
      out[(leg, j)] = (m.body(b).name, R0.T @ (d.xipos[b] - mid), R0.T @ Ri @ np.diag(m.body_inertia[b]) @ Ri.T @ R0, mid, b)
  return out


mr, my = compile_model(), compile_model(YAW_XML_PATH)
dr, dy = stand(mr), stand(my)
rig, yaw = leg_links(mr, dr, "battery"), leg_links(my, dy, YAW_ROOT_BODY)
R0y = dy.xmat[my.body(YAW_ROOT_BODY).id].reshape(3, 3)
text = Path(YAW_XML_PATH).read_bytes().decode()  # keep the file's CRLF line ends
worst = 0.0
for key, (name, c_hip, I_root, mid, b) in yaw.items():
  _, c_ref, I_ref, _, _ = rig[key]
  dc, dI = float(np.abs(c_hip - c_ref).max()), float(np.abs(I_root - I_ref).max())
  worst = max(worst, dc)
  print(f"leg{key[0]} {key[1][0]} {name:30s} COM - rigid {1e3 * dc:6.2f} mm, |I - rigid| {dI:.2e} kg m2")
  if args.check or (dc < 1e-6 and dI < 1e-12):
    continue
  Rb, pb = dy.xmat[b].reshape(3, 3), dy.xpos[b]
  ipos = Rb.T @ (mid + R0y @ c_ref - pb)
  Ib = Rb.T @ (R0y @ I_ref @ R0y.T) @ Rb
  fmt = lambda v: " ".join(f"{x:.10g}" for x in v)  # noqa: E731
  pat = re.compile(r'(<body name="' + re.escape(name) + r'"[^>]*>\s*(?:<[^>]*/>\s*)*?)<inertial ([^>]*)/>')
  m_ = pat.search(text)
  assert m_, name
  attrs = dict(re.findall(r'(\w+)="([^"]*)"', m_.group(2)))
  new = (f'<inertial pos="{fmt(ipos)}" mass="{attrs["mass"]}" '
         f'fullinertia="{fmt((Ib[0, 0], Ib[1, 1], Ib[2, 2], Ib[0, 1], Ib[0, 2], Ib[1, 2]))}"/>')
  text = text[:m_.start(2) - len("<inertial ")] + new + text[m_.end():]
if not args.check and worst > 1e-6:
  Path(YAW_XML_PATH).write_bytes(text.encode())
  my2 = compile_model(YAW_XML_PATH)
  dy2 = stand(my2)
  after = leg_links(my2, dy2, YAW_ROOT_BODY)
  e_c = max(float(np.abs(after[k][1] - rig[k][1]).max()) for k in after)
  e_I = max(float(np.abs(after[k][2] - rig[k][2]).max()) for k in after)
  print(f"written {YAW_XML_PATH}: max |COM - rigid| {1e3 * e_c:.2e} mm, |I - rigid| {e_I:.1e} kg m2; "
        f"total mass {my.body_subtreemass[0]:.6f} -> {my2.body_subtreemass[0]:.6f} kg")
