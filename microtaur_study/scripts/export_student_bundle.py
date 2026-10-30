"""Build the on-robot deployment bundle for a v2 (47-D "student" group) MLP student.

  OMNI_KIT_ACCEPT_EULA=YES python scripts/export_student_bundle.py --checkpoint <model.pt> --out <dir>

Writes to <dir>:
  policy.onnx          obs [1, 705] float32 -> actions [1, 8] (normalisation included, unclipped)
  policy_weights.npz   the same network as numpy arrays (W0, b0, ..., obs_mean, obs_std, obs_eps)
  golden.npz           two recorded sequences from the simulator (Teacher-Cur-Student-Play, no noise):
                       per policy step the raw robot signals, each observation term's current frame,
                       the 705-D network input, the network output and the safety-filter target
  microtaur_policy.py, test_golden.py, README.md   (copied from microtaur_study/deploy/unoq)
and <dir>.zip. The ONNX and numpy networks are checked against torch here; test_golden.py
checks the numpy pipeline against the simulator recording (run it on the robot computer too).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from isaaclab.app import AppLauncher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

from microtaur_common.robot_constants import LEG_JOINT_NAMES  # noqa: E402
from microtaur_isaac import terrains as TR  # noqa: E402
from microtaur_isaac.env_cfg import MicrotaurTeacherCurStudentPlayEnvCfg  # noqa: E402
from microtaur_isaac.mdp.observations import STUDENT_OBS_DIM, STUDENT_TERMS  # noqa: E402
from microtaur_isaac.policy_loader import load_policy, policy_history, prepare_env_cfg, student_group  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
assert student_group(args.checkpoint) == "student", "expects a v2 (47-D student group) checkpoint"
H = policy_history(args.checkpoint)
OBS = STUDENT_OBS_DIM * H

# ---------------------------------------------------------------- network -> npz + onnx
sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
lin = sorted({int(k.split(".")[1]) for k in sd if k.startswith("student.") and k.endswith(".weight")})
if "student_obs_normalizer._mean" in sd:
  mean, std = sd["student_obs_normalizer._mean"][0].float(), sd["student_obs_normalizer._std"][0].float()
else:
  mean, std = torch.zeros(OBS), torch.ones(OBS)
EPS = 1e-2  # rsl_rl EmpiricalNormalization default
npz = {"obs_mean": mean.numpy(), "obs_std": std.numpy(), "obs_eps": np.float32(EPS), "n_layers": np.int64(len(lin))}
for i, k in enumerate(lin):
  npz[f"W{i}"] = sd[f"student.{k}.weight"].float().numpy()
  npz[f"b{i}"] = sd[f"student.{k}.bias"].float().numpy()
np.savez(out / "policy_weights.npz", **npz)


class Net(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.register_buffer("mean", mean.clone())
    self.register_buffer("den", std.clone() + EPS)
    layers = []
    for i, k in enumerate(lin):
      m = torch.nn.Linear(sd[f"student.{k}.weight"].shape[1], sd[f"student.{k}.weight"].shape[0])
      m.weight.data.copy_(sd[f"student.{k}.weight"]); m.bias.data.copy_(sd[f"student.{k}.bias"])
      layers.append(m)
      if i < len(lin) - 1:
        layers.append(torch.nn.ELU())
    self.mlp = torch.nn.Sequential(*layers)

  def forward(self, obs):
    return self.mlp((obs - self.mean) / self.den)


net = Net().eval()
torch.onnx.export(net, torch.zeros(1, OBS), str(out / "policy.onnx"), input_names=["obs"], output_names=["actions"],
                  opset_version=17, dynamo=False)
import onnx  # noqa: E402
from onnx.reference import ReferenceEvaluator  # noqa: E402

onnx.checker.check_model(onnx.load(str(out / "policy.onnx")))
ref = ReferenceEvaluator(str(out / "policy.onnx"))

# ---------------------------------------------------------------- golden recording
cfg = MicrotaurTeacherCurStudentPlayEnvCfg()
cfg.scene.num_envs = 10  # env i on column i: 0-1 A, 2-6 B, 7-9 C
cfg.curriculum.command_ranges = None
cfg.commands.twist.resampling_time_range = (1.0e9, 1.0e9)
prepare_env_cfg(cfg, args.checkpoint)
env = ManagerBasedRLEnv(cfg)
t = env.scene.terrain
t.terrain_levels[:] = torch.tensor([0, 0, 0, 0, 0, 0, 0, 3, 3, 3], device=env.device)
t.env_origins[:] = t.terrain_origins[t.terrain_levels, t.terrain_types]
w = RslRlVecEnvWrapper(env)
act, _ = load_policy(w, args.checkpoint)
robot = env.scene["robot"]
mids = [robot.joint_names.index(n) for n in LEG_JOINT_NAMES]
core = env.action_manager.get_term("joint_pos").core
cmd = env.command_manager.get_term("twist")
SEQ = {"A_flat": 0, "C_step_lvl3": 8}
assert TR.env_terrain_type_ids(t)[0] == 0 and TR.env_terrain_type_ids(t)[8] == 2
sizes = dict(zip(STUDENT_TERMS, [3, 3, 3, 3, 8, 8, 8, 8, 3]))
rec = {k: [] for k in ("obs", "raw_action", "q", "qd", "gravity", "ang_vel", "command", "target_safe", "done")}
frames = {n: [] for n in STUDENT_TERMS}
with torch.inference_mode():
  obs = w.get_observations()
  for k in range(args.steps):
    c = torch.zeros(env.num_envs, 3, device=env.device)
    c[:, 0] = 0.20
    if k >= args.steps // 2:  # second half: turn left on A (C is straight-only)
      c[0, 2] = 0.20
    obs_cmd = cmd.command.clone()  # the command inside this observation (built at the end of the last step)
    cmd.vel_command_b[:] = c  # takes effect in the next observation
    o = obs["student"]
    d = robot.data
    rec["obs"].append(o.cpu().numpy())
    rec["q"].append(d.joint_pos[:, mids].cpu().numpy()); rec["qd"].append(d.joint_vel[:, mids].cpu().numpy())
    rec["gravity"].append(d.projected_gravity_b.cpu().numpy()); rec["ang_vel"].append(d.root_link_ang_vel_b.cpu().numpy())
    rec["command"].append(obs_cmd.cpu().numpy())
    off = 0
    for n in STUDENT_TERMS:  # current frame = the last of the term's H frames (term-major layout)
      frames[n].append(o[:, off + (H - 1) * sizes[n]: off + H * sizes[n]].cpu().numpy())
      off += H * sizes[n]
    a = act(obs)
    rec["raw_action"].append(a.cpu().numpy())
    obs, _, dones, _ = w.step(a)
    rec["target_safe"].append(core.safe.cpu().numpy())
    rec["done"].append(dones.cpu().numpy())
R = {k: np.stack(v) for k, v in rec.items()}
F = {n: np.stack(v) for n, v in frames.items()}
golden = {"dt": np.float64(env.step_dt), "history": np.int64(H), "terms": np.array(STUDENT_TERMS)}
for name, e in SEQ.items():
  n_ok = int(np.argmax(R["done"][:, e])) if R["done"][:, e].any() else args.steps
  for k, v in R.items():
    if k != "done":
      golden[f"{name}/{k}"] = v[:n_ok, e].astype(np.float64)
  for n, v in F.items():
    golden[f"{name}/frame/{n}"] = v[:n_ok, e].astype(np.float64)
  print(f"golden {name}: {n_ok} steps (env {e})", flush=True)
np.savez_compressed(out / "golden.npz", **golden)

# ---------------------------------------------------------------- network checks vs torch
o = torch.tensor(golden["A_flat/obs"][:50], dtype=torch.float32)
with torch.no_grad():
  a_t = net(o).numpy()
a_sim = golden["A_flat/raw_action"][:50]
a_onnx = np.concatenate([ref.run(None, {"obs": o[i:i + 1].numpy()})[0] for i in range(len(o))])
x = (o.numpy() - npz["obs_mean"]) / (npz["obs_std"] + EPS)
for i in range(len(lin)):
  x = x @ npz[f"W{i}"].T + npz[f"b{i}"]
  if i < len(lin) - 1:
    x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
print(f"max |torch export - sim policy| {np.abs(a_t - a_sim).max():.2e}, |onnx - torch| {np.abs(a_onnx - a_t).max():.2e}, "
      f"|numpy - torch| {np.abs(x - a_t).max():.2e}", flush=True)

src = ROOT / "deploy" / "unoq"
for f in ("microtaur_policy.py", "test_golden.py", "README.md"):
  shutil.copy(src / f, out / f)
(out / "SOURCE.txt").write_text(f"checkpoint: {Path(args.checkpoint).resolve()}\nobs: {STUDENT_OBS_DIM} x {H} = {OBS}\n"
                               f"layers: {[npz[f'W{i}'].shape for i in range(len(lin))]}\n")
shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
print(f"BUNDLE_OK {out} and {out}.zip", flush=True)
sys.stdout.flush()
os._exit(0)
