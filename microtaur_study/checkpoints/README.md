# IsaacLab checkpoints (rigid Microtaur)

Committed so a policy can be viewed on another machine. `*.pt` is git-ignored
repo-wide; files here are added with `git add -f`.

| folder | task | notes |
|---|---|---|
| `isaac_flat_pilot2/` | `Microtaur-Isaac-Flat-v0` | first IsaacLab pilot, 2048 envs, D1 reward (tracking + Spot trot GaitReward), 28.6 Hz. `params/` holds the exact env / agent config; the tfevents file opens in TensorBoard. |

`isaac_flat_pilot2/model_150.pt` is an early snapshot (iteration 150 of 1000):
it stands and does not fall, but walks only ~0.07 m/s against 0.10-0.20 m/s
commands.

## View a policy in the Isaac Sim GUI

Needs an NVIDIA GPU with a display, Isaac Sim 5.1 and IsaacLab v2.3.0 (conda
env `spine`: `SPINE_ROOT=<dir holding IsaacLab/ and microtaur_RL/> bash env/setup_env_spine.sh`).

```bash
cd microtaur_RL/microtaur_study
conda activate spine
ISAACLAB_DIR=<path to IsaacLab> OMNI_KIT_ACCEPT_EULA=YES \
  python scripts/isaac_play.py --task Microtaur-Isaac-Flat-Play-v0 \
  --checkpoint checkpoints/isaac_flat_pilot2/model_150.pt --real-time
```

The Play task runs 16 robots with the visual-mesh USD
(`assets/rigid_microtaur/microtaur_rigid_visual.usd`), no noise or
randomisation and no episode timeout, commands from the final curriculum stage
(0.10-0.35 m/s, yaw +-0.25 rad/s), camera following env 0. Add
`--headless --video --video_length 300` to record an mp4 into
`checkpoints/<run>/videos/play/` instead. play.py also exports the policy to
`checkpoints/<run>/exported/policy.{pt,onnx}`.

## Gait numbers and a MuJoCo replay (no GUI needed)

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/isaac_eval_gait.py \
  --checkpoint checkpoints/isaac_flat_pilot2/model_150.pt --out /tmp/eval
DISPLAY=:0 MUJOCO_GL=glfw python scripts/isaac_render_traj.py --traj /tmp/eval/traj_0.20.npz
```

`isaac_render_traj.py` needs only mujoco, numpy, matplotlib and imageio (no
Isaac), so a `traj_<v>.npz` copied from a GPU machine can be rendered anywhere.
