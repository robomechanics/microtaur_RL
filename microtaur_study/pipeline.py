#!/usr/bin/env python
"""One-command Microtaur training pipeline (IsaacLab 2.3 / Isaac Sim 5.1; see the README for the install).

  python microtaur_study/pipeline.py --robot yaw --name y3          # teacher -> eval -> student -> eval (~8-10 h)
  python microtaur_study/pipeline.py --robot rigid --name r1         # same for the rigid robot (+ UNO Q bundle)
  python microtaur_study/pipeline.py --robot yaw --smoke             # install check: a few iterations of every step
  python microtaur_study/pipeline.py --stage teacher --robot yaw --name y3 [--iters 3000]
  python microtaur_study/pipeline.py --stage student --teacher-ckpt runs/logs/rsl_rl/.../model_2999.pt
  python microtaur_study/pipeline.py --stage eval --checkpoint runs/logs/rsl_rl/.../model_2999.pt
  python microtaur_study/pipeline.py --stage video --checkpoint <model.pt>

Every step runs as its own Python process (Isaac Sim holds one environment per process). Outputs, under --runs
(default ./runs):
  runs/logs/rsl_rl/microtaur_isaac_teacher/<date>_<run>/   checkpoints model_N.pt, params/ (env / agent cfg),
                                                           tensorboard events (tensorboard --logdir runs/logs)
  runs/<name>/progress.md                                  early-checkpoint gait checks while the teacher trains
  runs/<name>/<label>/                                     eval of one checkpoint: gait.json, terrain_eval.log,
                                                           c_posture.txt, turn_legs_*.txt, videos/*.mp4, crop png
  runs/<name>/summary.md                                   the numbers of every evaluated checkpoint + video paths
The recipe (reward weights, command curriculum, student settings) is the one of the best runs so far (t4 / v4c,
y1); change it in TEACHER_RECIPE / STUDENT_RECIPE below or append Hydra overrides with --override.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
PY = sys.executable
EXPERIMENT = "microtaur_isaac_teacher"  # teacher and student runs both log here (the student loads its teacher by run name)

TASKS = {
  "rigid": ("Microtaur-Isaac-Teacher-Cur-v0", "Microtaur-Isaac-Teacher-Cur-Student-v0"),
  "yaw": ("Microtaur-Isaac-Yaw-Teacher-Cur-v0", "Microtaur-Isaac-Yaw-Teacher-Cur-Student-v0"),
}
TEACHER_AGENT = "rsl_rl_sym_cfg_entry_point"  # PPO + left/right symmetry augmentation
STUDENT_AGENT = "rsl_rl_student_v2_small_cfg_entry_point"  # MLP student, 15-frame history, distillation
STAGES_CMD = ("[{step:0,lin_vel_x:[0.10,0.20],ang_vel_z:[-0.25,0.25],standing:0.0},"
              "{step:4000,lin_vel_x:[0.10,0.30],ang_vel_z:[-0.25,0.25],standing:0.0},"
              "{step:10000,lin_vel_x:[0.10,0.35],ang_vel_z:[-0.25,0.25],standing:0.0}]")
TEACHER_RECIPE = [  # t4 / y1 (Hydra overrides of the env cfg)
  "env.rewards.action_rate.weight=-0.08", "env.rewards.trot_gait.weight=2.0", "env.rewards.trot_gait.params.std=0.1",
  "env.rewards.trot_gait.params.max_err=0.2", "env.rewards.track_lin_vel_xy.params.sigma=0.07",
  "env.rewards.feet_air_time.weight=1.0", "env.rewards.feet_air_time.params.mode_time_s=0.2",
  "env.rewards.flat_orientation_l2.weight=-2.5", "env.rewards.ang_vel_xy_l2.weight=0.0", "env.rewards.feet_slide.weight=-0.1",
  "env.rewards.heading_tracking.weight=1.0", "env.commands.twist.rel_straight_envs=0.3",
  f"env.curriculum.command_ranges.params.stages={STAGES_CMD}", "env.commands.twist.ranges.ang_vel_z=[-0.25,0.25]",
  "env.curriculum.reward_schedule.params.stages=['9600|flat_orientation_l2.weight=-10.0|track_lin_vel_xy.sigma=0.12']",
]
STUDENT_RECIPE = [  # v4c / y1s
  "env.c_reverse_frac=0.5", "env.commands.twist.rel_straight_envs=0.3", "env.commands.twist.ranges.ang_vel_z=[-0.25,0.25]",
  "env.observations.student.history_length=15", "env.events.init_terrain_levels.params.level_range=[1,2]",
  "env.curriculum.terrain_levels.params.min_level=1", "env.curriculum.terrain_levels.params.steps_per_level=3000",
  "env.rewards.action_rate.weight=-0.08", "env.rewards.trot_gait.weight=2.0", "env.rewards.trot_gait.params.std=0.1",
  "env.rewards.trot_gait.params.max_err=0.2", "env.rewards.track_lin_vel_xy.params.sigma=0.12",
  "env.rewards.feet_air_time.weight=1.0", "env.rewards.feet_air_time.params.mode_time_s=0.2",
  "env.rewards.flat_orientation_l2.weight=-10.0", "env.rewards.feet_slide.weight=-0.1", "env.rewards.heading_tracking.weight=1.0",
  "env.curriculum.command_ranges.params.stages=[{step:0,lin_vel_x:[0.10,0.35],ang_vel_z:[-0.25,0.25],standing:0.0},"
  "{step:1,lin_vel_x:[0.10,0.35],ang_vel_z:[-0.25,0.25],standing:0.0},{step:2,lin_vel_x:[0.10,0.35],ang_vel_z:[-0.25,0.25],standing:0.0}]",
]
CHECK_AT = (50, 100, 200, 300, 500, 1000, 1500, 2000, 2500)  # early gait checks while the teacher trains
# terrain videos: (env index in the play scene, name, camera eye, look-at), relative to the robot
TERRAIN_VIDEOS = [(0, "A_flat", (0.30, -0.45, 0.18), (0, 0, 0.04)), (5, "B_lvl6", (0.30, -0.45, 0.25), (0, 0, 0.02)),
                  (9, "C_lvl6", (-0.45, -0.18, 0.32), (0.25, 0, 0.0))]


def log(msg: str, f: Path | None = None):
  line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
  print(line, flush=True)
  if f is not None:
    with open(f, "a") as fh:
      fh.write(line + "\n")


def env_vars() -> dict:
  return {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}


def gpu_free_mib() -> int | None:
  try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=20).stdout.split("\n")[0]
    total, used = (int(v) for v in out.split(","))
    return total - used
  except Exception:  # noqa: BLE001
    return None


def run(cmd: list, logfile: Path, cwd: Path, timeout: float) -> int:
  """One step in its own process; Kit can hang on exit, so the process group is killed on timeout."""
  logfile.parent.mkdir(parents=True, exist_ok=True)
  with open(logfile, "w") as fh:
    p = subprocess.Popen([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT, cwd=cwd, env=env_vars(), start_new_session=True)
    try:
      return p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
      os.killpg(p.pid, signal.SIGKILL)
      return -9


def step(cmd: list, logfile: Path, cwd: Path, timeout: float, ok_file: Path) -> bool:
  """run(); on a failure (a full shared GPU: CUDA / Warp / cusolver out of memory) once more with CPU physics."""
  if run(cmd, logfile, cwd, timeout) == 0 and ok_file.exists():
    return True
  if "--device" in [str(c) for c in cmd]:
    return False
  log(f"  {Path(str(cmd[1])).name} failed (see {logfile.name}); retrying with --device cpu")
  return run([*cmd, "--device", "cpu"], logfile.with_name(logfile.stem + "_cpu.log"), cwd, timeout * 2) == 0 and ok_file.exists()


class Pipeline:
  def __init__(self, a):
    self.a = a
    self.runs = Path(a.runs).resolve()
    self.logs = self.runs / "logs" / "rsl_rl" / EXPERIMENT
    self.runs.mkdir(parents=True, exist_ok=True)
    self.smoke = a.smoke

  # ------------------------------------------------------------------ training
  def train(self, task: str, run_name: str, iters: int, agent: str, overrides: list, extra: list, out: Path,
            watch: bool) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    trainlog = out / f"train_{run_name}.log"
    cmd = [PY, SCRIPTS / "isaac_train.py", "--task", task, "--num_envs", self.a.num_envs, "--headless",
           "--max_iterations", iters, "--run_name", run_name, "--agent", agent, "--seed", self.a.seed, *overrides, *extra]
    log(f"train {run_name}: {task}, {iters} it, {self.a.num_envs} envs -> {trainlog}", out / "progress.md")
    fh = open(trainlog, "w")
    t0 = time.time()
    p = subprocess.Popen([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT, cwd=self.runs, env=env_vars(),
                         start_new_session=True)
    last = f"model_{iters - 1}.pt"
    run_dir, checked = None, set()
    try:
      while True:
        time.sleep(15)
        if run_dir is None:
          found = sorted(d for d in self.logs.glob(f"*_{run_name}") if d.stat().st_mtime >= t0 - 5)
          run_dir = found[-1] if found else None
          if run_dir is not None:
            log(f"run dir {run_dir}", out / "progress.md")
        text = trainlog.read_text(errors="ignore")
        if "Traceback" in text:
          raise RuntimeError(f"training crashed, see {trainlog}:\n" + text[text.find("Traceback"):][-2000:])
        if run_dir is not None and (run_dir / last).exists():
          time.sleep(5)
          break
        if p.poll() is not None:
          raise RuntimeError(f"training exited (code {p.returncode}) before {last}; see {trainlog}")
        if watch and run_dir is not None:
          for n in CHECK_AT:
            ck = run_dir / f"model_{n}.pt"
            if n < iters - 1 and n not in checked and ck.exists():
              checked.add(n)
              time.sleep(3)
              g = self.gait(ck, f"{run_name}_{n}", out / "early_checks")
              it = re.findall(r"Learning iteration (\d+)/", text)
              log(f"checkpoint {n}: {g.get('line', 'gait check failed')} | training at {it[-1] if it else '?'}", out / "progress.md")
              if n >= 150 and g.get("pronk"):
                raise RuntimeError(f"pronk locked in at model_{n} (same_end, same_side > 0.2, flight > 0.15): stopped; "
                                   "look at the frames in early_checks/ before relaunching")
    finally:
      if p.poll() is None:  # Kit's SimulationApp.close() can hang after training
        os.killpg(p.pid, signal.SIGTERM)
        time.sleep(5)
        if p.poll() is None:
          os.killpg(p.pid, signal.SIGKILL)
      fh.close()
    log(f"train {run_name} done: {run_dir / last}", out / "progress.md")
    return run_dir / last

  # ------------------------------------------------------------------ evaluation
  def gait(self, ck: Path, label: str, out: Path) -> dict:
    """Flat gait sweep (gait type, stride, speed at 0.10 / 0.20 / 0.35) + a side-view crop of 12 steps."""
    out.mkdir(parents=True, exist_ok=True)
    j = out / f"gait_{label}.json"
    ok = step([PY, SCRIPTS / "isaac_gait_sweep.py", "--ckpts", f"{label}={ck}", "--out", j], out / f"gait_{label}.log", self.runs, 1500, j)
    res = {}
    if ok:
      r = list(json.loads(j.read_text()).values())[0]
      res = dict(r)
      res["pronk"] = r["same_end"] > 0.2 and r["same_side"] > 0.2 and r["flight"] > 0.15
      res["line"] = (f"flight {r['flight']:.2f} four {r['four']:.2f} diag {r['diag']:+.2f} same_end {r['same_end']:+.2f} "
                     f"same_side {r['same_side']:+.2f} stride {r['stride_hz']:.1f} Hz")
    self.video(ck, out / f"{label}_flat_side.mp4", env=0, steps=150, eye=(0.05, -0.40, 0.10), lookat=(0.05, 0, 0.03), crop=True)
    return res

  def video(self, ck: Path, mp4: Path, env: int, steps: int, eye, lookat, crop=False, tour=False) -> bool:
    free = gpu_free_mib()
    dev = ["--device", "cpu"] if free is not None and free < 4000 else []  # physics on the CPU when the GPU is full
    cmd = [PY, SCRIPTS / "isaac_render_policy.py", "--checkpoint", ck, "--env", env, "--steps", steps, "--out", mp4,
           "--eye", *eye, "--lookat", *lookat] + (["--tour"] if tour else []) + dev
    ok = step(cmd, mp4.with_suffix(".log"), self.runs, 1500, mp4)
    if ok and crop:
      import imageio
      import numpy as np
      from PIL import Image, ImageDraw
      fr = [f for f in imageio.get_reader(mp4)]
      if len(fr) >= 69:
        H, W = fr[0].shape[:2]
        tiles = []
        for k in range(57, 69):
          im = Image.fromarray(fr[k][int(H * .35):int(H * .95), int(W * .08):int(W * .92)])
          ImageDraw.Draw(im).text((8, 8), f"step {k}", fill=(255, 0, 0))
          tiles.append(np.asarray(im))
        Image.fromarray(np.concatenate([np.concatenate(tiles[j:j + 4], 1) for j in range(0, 12, 4)], 0)).resize((1800, 690)).save(
          mp4.with_name(mp4.stem + "_crop.png"))
    return ok

  def evaluate(self, ck: Path, label: str, out: Path) -> Path:
    """Gait, terrain (A / B / C, all levels), C step posture, turning + virtual legs, videos; summary appended."""
    d = out / label
    vid = d / "videos"
    vid.mkdir(parents=True, exist_ok=True)
    log(f"eval {label}: {ck} -> {d}", out / "progress.md")
    g = self.gait(ck, label, vid)
    n_terrain = 40 if self.smoke else 500
    step([PY, SCRIPTS / "isaac_eval_terrain.py", "--checkpoint", ck, "--out", d, "--num-envs", n_terrain], d / "terrain_eval.stdout",
         self.runs, 3600, d / "terrain_eval.json")
    step([PY, SCRIPTS / "isaac_c_posture.py", "--checkpoint", ck, "--out", d / "c_posture"], d / "c_posture.log", self.runs, 1800,
         d / "c_posture" / "c_posture.txt")
    step([PY, SCRIPTS / "isaac_turn_legs.py", "--checkpoint", ck, "--label", label, "--out", d], d / "turn_legs.log", self.runs, 1800,
         d / f"turn_legs_{label}.txt")
    steps = 40 if self.smoke else 420
    for env, name, eye, look in TERRAIN_VIDEOS:
      self.video(ck, vid / f"{label}_{name}.mp4", env=env, steps=steps, eye=eye, lookat=look)
    self.video(ck, vid / f"{label}_turn_tour.mp4", env=0, steps=0, eye=(0.0, -0.55, 0.45), lookat=(0, 0, 0), tour=True)
    self.video(ck, vid / f"{label}_turn_tour_top.mp4", env=0, steps=0, eye=(0.0, -0.02, 0.60), lookat=(0, 0, 0), tour=True)
    self.summarize(label, ck, d, g, out)
    log(f"eval {label} done: {out / 'summary.md'}", out / "progress.md")
    return d

  def summarize(self, label: str, ck: Path, d: Path, g: dict, out: Path):
    read = lambda p: p.read_text(errors="ignore").strip() if p.exists() else "(missing: see the .log next to it)"  # noqa: E731
    vx = g.get("vx", {})
    lines = [f"## {label}", f"checkpoint `{ck}`", "",
             f"**Gait (flat):** {g.get('line', 'gait sweep failed')}",
             "**Speed at command 0.10 / 0.20 / 0.35 m/s:** " + " / ".join(f"{float(vx[k]):.3f}" for k in sorted(vx)) if vx else "",
             "", "**Terrain (straight 0.20 m/s, 20 s):**", "```", read(d / "terrain_eval.log"), "```",
             "**C step posture:**", "```", read(d / "c_posture" / "c_posture.txt"), "```",
             "**Turning + virtual legs:**", "```", read(d / f"turn_legs_{label}.txt"), "```",
             "**Videos:** " + ", ".join(f"`{p}`" for p in sorted((d / "videos").glob("*.mp4"))), ""]
    with open(out / "summary.md", "a") as fh:
      fh.write("\n".join(lines) + "\n")

  # ------------------------------------------------------------------ stages
  def teacher(self, name: str) -> Path:
    iters = 3 if self.smoke else self.a.iters
    return self.train(TASKS[self.a.robot][0], name, iters, TEACHER_AGENT, TEACHER_RECIPE + self.a.override, [],
                      self.runs / name, watch=not self.smoke)

  def student(self, name: str, teacher_ck: Path) -> Path:
    teacher_ck = teacher_ck.resolve()
    if teacher_ck.parent.parent != self.logs:
      raise SystemExit(f"the teacher run must be under {self.logs} (the student loads it by run name); copy its folder there")
    iters = 2 if self.smoke else self.a.student_iters
    task = TASKS[self.robot_of(teacher_ck)][1]
    return self.train(task, f"{name}_student", iters, STUDENT_AGENT, STUDENT_RECIPE + self.a.override_student,
                      ["--load_run", teacher_ck.parent.name, "--checkpoint", teacher_ck.name], self.runs / name, watch=False)

  @staticmethod
  def robot_of(ck: Path) -> str:
    y = (ck.parent / "params" / "env.yaml")
    return "yaw" if y.exists() and re.search(r"^variant: yaw$", y.read_text(), re.M) else "rigid"

  def bundle(self, student_ck: Path, out: Path):
    if self.robot_of(student_ck) != "rigid":
      log("UNO Q bundle: rigid robot only for now (yaw: spine motor ID / calibration not defined) -> skipped", out / "progress.md")
      return
    b = out / "unoq_bundle"
    rc = run([PY, SCRIPTS / "export_student_bundle.py", "--checkpoint", student_ck, "--out", b], out / "bundle.log", self.runs, 1800)
    log(f"UNO Q bundle {'written to ' + str(b) if rc == 0 else 'FAILED, see ' + str(out / 'bundle.log')}", out / "progress.md")


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--stage", default="all", choices=("all", "teacher", "student", "eval", "video"))
  ap.add_argument("--robot", default="yaw", choices=("yaw", "rigid"))
  ap.add_argument("--name", default=None, help="run name (default: <robot>_<date>)")
  ap.add_argument("--runs", default="runs", help="output root (default ./runs)")
  ap.add_argument("--iters", type=int, default=3000, help="teacher PPO iterations")
  ap.add_argument("--student-iters", type=int, default=300, help="student distillation iterations (loss flat after ~300)")
  ap.add_argument("--num-envs", type=int, default=2048)
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--checkpoint", help="--stage eval / video: the model_N.pt to evaluate")
  ap.add_argument("--teacher-ckpt", help="--stage student: the teacher model_N.pt to distill")
  ap.add_argument("--override", nargs="*", default=[], help="extra Hydra overrides for the teacher, e.g. env.rewards.trot_gait.weight=1.5")
  ap.add_argument("--override-student", nargs="*", default=[], help="extra Hydra overrides for the student")
  ap.add_argument("--smoke", action="store_true", help="install check: 3 teacher / 2 student iterations, 256 envs, short evals")
  a = ap.parse_args()
  if a.smoke:
    a.num_envs = min(a.num_envs, 256)
  name = a.name or f"{a.robot}_{time.strftime('%m%d_%H%M')}" + ("_smoke" if a.smoke else "")
  P = Pipeline(a)
  out = P.runs / name
  out.mkdir(parents=True, exist_ok=True)
  log(f"pipeline {a.stage} ({a.robot}) -> {out}", out / "progress.md")
  if a.stage in ("eval", "video"):
    ck = Path(a.checkpoint).resolve()
    label = f"{ck.parent.name.split('_', 2)[-1]}_{ck.stem.split('_')[-1]}"
    if a.stage == "eval":
      P.evaluate(ck, label, out)
    else:
      vid = out / label / "videos"
      for env, nm, eye, look in TERRAIN_VIDEOS:
        P.video(ck, vid / f"{label}_{nm}.mp4", env=env, steps=420, eye=eye, lookat=look)
      P.video(ck, vid / f"{label}_turn_tour.mp4", env=0, steps=0, eye=(0.0, -0.55, 0.45), lookat=(0, 0, 0), tour=True)
      P.video(ck, vid / f"{label}_turn_tour_top.mp4", env=0, steps=0, eye=(0.0, -0.02, 0.60), lookat=(0, 0, 0), tour=True)
      P.gait(ck, label, vid)
    return
  teacher_ck = Path(a.teacher_ckpt) if a.teacher_ckpt else None
  if a.stage in ("all", "teacher"):
    teacher_ck = P.teacher(name)
    P.evaluate(teacher_ck, f"{name}_teacher", out)
  if a.stage in ("all", "student"):
    if teacher_ck is None:
      raise SystemExit("--stage student needs --teacher-ckpt")
    student_ck = P.student(name, teacher_ck)
    P.evaluate(student_ck, f"{name}_student", out)
    P.bundle(student_ck, out)
  log(f"DONE: {out / 'summary.md'}", out / "progress.md")


if __name__ == "__main__":
  main()
