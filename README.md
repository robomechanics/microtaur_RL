# Microtaur RL

Reinforcement-learning locomotion for **Microtaur**, a ~0.5 kg quadruped with planar five-bar legs driven by eight XL330 servos and an Arduino UNO Q. The repository covers several body variants (rigid body, passive twist spine, active pitch / yaw / twist spine), trained in simulation with PPO.

## What is on `main`

| Folder | Contents |
|---|---|
| [`Microtaur_RL-main/`](Microtaur_RL-main/) | The mjlab task package `microtaur_velocity` (velocity-tracking PPO) and the robot MJCF models for every variant (`microtaur_xmls/`). |
| [`microtaur_runs/`](microtaur_runs/) | One run folder per variant (`rigid/`, `passive_twist/`, `active_pitch/`, `active_yaw/`, `active_twist/`) and an analysis package (`microtaur_analyes/`). |
| [`microtaur_runs/cpg_rl/`](microtaur_runs/cpg_rl/) | Open-loop CPG gaits (sine waves on one shared clock) fitted to each variant's RL policy and compared against it in simulation, with plots and a comparison video. See its [README](microtaur_runs/cpg_rl/README.md). |

## Branches

| Branch | Contents |
|---|---|
| `main` | The mjlab packages, run folders and CPG results above. |
| `cpg-rl` | Same history as `main`. |
| [`spine-study-isaac`](https://github.com/robomechanics/microtaur_RL/tree/spine-study-isaac) | The rigid Microtaur ported to IsaacLab, terrain teacher and blind student policies, and a deployment bundle for the UNO Q (details below). |

### `spine-study-isaac`

Work in [`microtaur_study/`](https://github.com/robomechanics/microtaur_RL/tree/spine-study-isaac/microtaur_study) on that branch (IsaacLab 2.3 / Isaac Sim 5.1, rsl_rl). Everything is simulation; nothing has run on hardware yet.

- **IsaacLab port of the rigid Microtaur.** The closed-chain five-bar legs are simulated in PhysX. The task code (rewards, actions, kinematics) is shared with mjlab through `src/microtaur_common`.
- **Terrain teacher.** A policy with a privileged height map, trained with a terrain curriculum on flat ground, random blocks and a side-step lane. It walks a diagonal trot and keeps the body level on a 37 mm side step by shortening the legs on the raised side.
- **Blind students.** Distilled from the teacher (DAgger) using only sensors the robot has: IMU, motor encoders and five-bar forward kinematics.
- **UNO Q deployment.** [`microtaur_study/deploy/unoq/`](https://github.com/robomechanics/microtaur_RL/tree/spine-study-isaac/microtaur_study/deploy/unoq) has a numpy-only reference of the full control step (observation, history, network, safety filter, servo ticks), a test that replays a simulator recording through it, and a hardware bring-up guide. Exported policy files are not committed; `microtaur_study/scripts/export_student_bundle.py` regenerates them from a checkpoint.
- **Write-ups** in [`docs/`](https://github.com/robomechanics/microtaur_RL/tree/spine-study-isaac/docs). The [tuning log](https://github.com/robomechanics/microtaur_RL/blob/spine-study-isaac/docs/ISAAC_TUNING_LOG_2026-09-26.md) records every training run, including how a pronking gait was traced to the reward set with checkpoint sweeps and ablations. [`TERRAIN_DESIGN.md`](https://github.com/robomechanics/microtaur_RL/blob/spine-study-isaac/docs/TERRAIN_DESIGN.md) describes the terrains.
