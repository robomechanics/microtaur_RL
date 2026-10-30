# Microtaur training kit

Reinforcement-learning walking controllers for the **Microtaur** quadruped in IsaacLab: the **rigid** robot (8 leg
servos) and the **active-yaw** robot (8 leg servos + a yaw "spine" servo between the front and rear halves). One
command trains a terrain-aware **teacher** with PPO and distils it into a blind, deployable **student**. Both are
then evaluated (gait, terrain, side step, turning) and rendered to videos.

This branch (`train-kit`) holds only what training needs. It is generated from the development branch
`spine-study-isaac` by `microtaur_study/scripts/make_train_kit.sh`.

## 1. Install (once)

Tested with: Ubuntu, NVIDIA RTX 5080 (16 GB), Python 3.11, Isaac Sim 5.1.0, IsaacLab 2.3.0, rsl-rl-lib 3.0.1, torch 2.7.0+cu128.
Training uses about 5 GB of GPU memory at 2048 environments. Plan for about 40 GB of disk (Isaac Sim + caches).

```bash
conda create -n microtaur python=3.11 -y
conda activate microtaur
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com

git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && git checkout v2.3.0 && ./isaaclab.sh --install rsl_rl && cd ..   # isaaclab + rsl-rl-lib 3.0.1

pip install mujoco==3.14.0 "imageio[ffmpeg]" pillow onnx onnxruntime            # model tools, videos, deployment export

git clone -b train-kit --single-branch https://github.com/robomechanics/microtaur_RL.git microtaur_train
cd microtaur_train
```

IsaacLab has to be installed **from source**, as above. The pipeline runs IsaacLab's own `scripts/reinforcement_learning/rsl_rl/train.py`
and finds it through the installed `isaaclab` package. If your IsaacLab checkout lives elsewhere, set `ISAACLAB_DIR=/path/to/IsaacLab`.
The Isaac Sim EULA is accepted automatically (`OMNI_KIT_ACCEPT_EULA=YES`). The first start of Isaac Sim compiles shaders and takes a few minutes.

## 2. Check the install (about 40 min)

```bash
python microtaur_study/pipeline.py --robot yaw --smoke
```

This runs every step for a few iterations with 256 environments: teacher training, its evaluation and videos, then
the student. It must end with `DONE: runs/<name>/summary.md`. Open `runs/<name>/summary.md` and the videos it lists.
The robot will not walk yet, because it has only trained for 3 iterations.

## 3. Train

```bash
python microtaur_study/pipeline.py --robot yaw --name y3          # active-yaw robot
python microtaur_study/pipeline.py --robot rigid --name r5        # rigid robot (also writes the UNO Q deployment bundle)
```

Run it inside `tmux`, or as `nohup python ... > y3.out 2>&1 &`. It takes about 8-10 h on one RTX 5080:

| step | time |
|---|---|
| teacher, 3000 iterations | 4-5 h |
| teacher evaluation + videos | about 20 min |
| student, 300 iterations | about 1 h |
| student evaluation + videos | about 20 min |

While the teacher trains, `runs/<name>/progress.md` gets a gait check at iterations 50, 100, 200, 300, 500, 1000, …, 2500.
Each check records the gait type, stride frequency and speed, plus a side-view picture of 12 steps in
`runs/<name>/early_checks/*_crop.png`. The healthy gait is a diagonal **trot** (`diag +1.00 same_end -1.00 same_side -1.00`,
no flight). If the policy locks into a **pronk** (all legs together: `same_end` and `same_side` > 0.2 with flight),
the pipeline stops the run by itself.

Single steps:

```bash
python microtaur_study/pipeline.py --stage teacher --robot yaw --name y3
python microtaur_study/pipeline.py --stage student --name y3 --teacher-ckpt runs/logs/rsl_rl/microtaur_isaac_teacher/<date>_y3/model_2999.pt
python microtaur_study/pipeline.py --stage eval  --name y3 --checkpoint <any model_N.pt>
python microtaur_study/pipeline.py --stage video --name y3 --checkpoint <any model_N.pt>
```

To change the recipe, append Hydra overrides. The defaults are `TEACHER_RECIPE` / `STUDENT_RECIPE` in `microtaur_study/pipeline.py`:

```bash
python microtaur_study/pipeline.py --robot yaw --name y3b --override env.rewards.trot_gait.weight=1.5 --seed 7
```

## 4. Results to send back

- `runs/<name>/summary.md` and `runs/<name>/progress.md`.
- `runs/<name>/<label>/videos/*.mp4`:
  - flat side view;
  - flat, random blocks (B level 6) and side step (C level 6);
  - a turn tour, which shows the command, the measured yaw rate and the spine angle.
- The run folders `runs/logs/rsl_rl/microtaur_isaac_teacher/<date>_<name>/` and `<date>_<name>_student/`.
  The last `model_N.pt` and `params/` are enough; the other checkpoints are optional.

How to read the evaluation:

| file | what it shows |
|---|---|
| `terrain_eval.log` | per terrain and level: falls; whether the robot covered half a tile (3.15 m) in 20 s at 0.20 m/s; speed and heading drift |
| `c_posture/c_posture.txt` | on the side step (one side raised by up to 36.8 mm): body roll, and how much shorter the stance legs on the raised side are (target: about the step height, roll about 0) |
| `turn_legs_<label>.txt` | measured yaw rate for ±0.25 rad/s commands (turning is still weak: 20-35 % of the command); spine angle in turns; virtual leg length per leg while trotting |

## 5. Where things are

| path | content |
|---|---|
| `microtaur_study/pipeline.py` | the one-command pipeline (recipes at the top) |
| `microtaur_study/src/microtaur_isaac/` | the IsaacLab task: env cfg, MDP terms (`mdp/`), robot cfg, terrains |
| `microtaur_study/src/microtaur_common/` | simulator-free pieces: kinematics, reset, reward maths, robot constants |
| `microtaur_study/assets/` | MJCF + meshes + USD of both robots |
| `microtaur_study/scripts/` | training entry, evaluation and rendering scripts the pipeline calls |
| `microtaur_study/deploy/unoq/` | template of the Arduino UNO Q deployment bundle (rigid robot) |
| `microtaur_study/tests/check_isaac_yaw_env.py` | checks of the yaw environment (`--part train` / `--part play`) |

Gym task ids: `Microtaur-Isaac-{Teacher-Cur, Teacher-Cur-Student, Yaw-Teacher-Cur, Yaw-Teacher-Cur-Student}-v0` (+ `-Play-v0`).

## Troubleshooting

- **The GPU is shared or full.** The evaluation scripts fall back to CPU physics when less than 3-4 GB of GPU memory is free. They are slower then, but the numbers stay valid.
- **The first video frame is black.** That is the renderer warming up.
- **Kit prints a lot and sometimes hangs on exit.** Every step runs in its own process with a timeout, and the training process is stopped once its last checkpoint exists.
- **Isaac Sim prints `GLFW initialization failed` / no display warnings.** Everything runs headless, so these are expected.
