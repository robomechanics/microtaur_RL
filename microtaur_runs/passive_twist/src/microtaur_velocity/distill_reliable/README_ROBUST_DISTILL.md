# Robust Microtaur distillation pipeline

This is the latency/domain-randomization-aware replacement for the earlier
`distill_reliable` workflow.

The main rule is:

> During simulation distillation, build the deployable student observation from
> the exact actor observation seen by the teacher. Do not rebuild it from fresh
> `robot.data`.

That preserves the teacher environment's observation noise and 1-2-step sensor
delay while keeping teacher labels and student inputs temporally aligned.

## Supported morphologies

The pipeline no longer assumes every robot has eight actions.

| Robot type | Student observation | Student action |
|---|---:|---:|
| rigid | 35D | 8D |
| pitch spine | 38D | 9D |
| yaw spine | 38D | 9D |
| twist spine | 38D | 9D |

### Rigid 35D layout

```text
projected_gravity       3
base_ang_vel            3
leg_joint_pos_rel       8
leg_joint_vel           8
prev_action             8
command                 3
phase_sin_cos           2
--------------------------
total                   35
```

### Active-spine 38D layout

```text
projected_gravity       3
base_ang_vel            3
leg_joint_pos_rel       8
leg_joint_vel           8
prev_action             9
command                 3
spine_pos               1
spine_vel               1
phase_sin_cos           2
--------------------------
total                   38
```

`base_lin_vel` remains excluded because it is not a directly deployable
measurement on the physical robot.

## What robust mode preserves

Robust collection/evaluation starts from the training environment (`play=False`)
and keeps:

- configured actor observation corruption
- configured 1-2 control-step sensor delay
- configured action delay
- randomized IK-consistent resets
- startup/domain-randomization events such as friction, COM, and encoder bias
- normal episode termination behavior unless explicitly disabled

It then copies the final play-stage command/action configuration so a final
teacher checkpoint is not accidentally distilled at curriculum stage 0.

Robust mode is the default.

`--clean-env` is only for an ablation against the old clean evaluation setup.

## Important alignment rule

Teacher and student must see the same delayed/noisy sensor realization for each
supervised label:

```text
randomized dynamics
       |
       v
environment actor observation
(delayed + noisy)
       |
       +-----------> teacher ----------> teacher action label
       |
       +-----------> deployable slice -> student input
```

Do not independently add another random delay while constructing the student
input. That would create label ambiguity.

## Install

Copy the files in this folder over:

```text
src/microtaur_velocity/distill_reliable/
```

Then from the project root:

```powershell
$env:PYTHONPATH="$pwd\src"
```

## Stage 0 - choose the final teacher

Use the teacher checkpoint you actually intend to deploy/distill, normally the
final checkpoint from the completed command curriculum.

Example:

```powershell
$Task = "YOUR_REGISTERED_MICROTAUR_TASK"
$Teacher = "logs\rsl_rl\YOUR_EXPERIMENT\YOUR_RUN\model_4499.pt"
```

## Stage 1 - collect robust teacher-forced BC data

```powershell
python -m microtaur_velocity.distill_reliable.record_teacher_bc `
  $Task `
  --checkpoint-file $Teacher `
  --out-dir logs\distill_reliable\bc_robust `
  --steps 100000 `
  --num-envs 2048
```

This is robust by default.

For a clean ablation only:

```powershell
python -m microtaur_velocity.distill_reliable.record_teacher_bc `
  $Task `
  --checkpoint-file $Teacher `
  --out-dir logs\distill_reliable\bc_clean_ablation `
  --steps 100000 `
  --num-envs 2048 `
  --clean-env
```

## Stage 2 - audit the dataset

```powershell
python -m microtaur_velocity.distill_reliable.audit_dataset `
  --dataset-dirs logs\distill_reliable\bc_robust
```

Expected dimensions are inferred from the data:

```text
rigid:        obs_dim=35, action_dim=8
active spine: obs_dim=38, action_dim=9
```

Do not aggregate datasets from different morphologies into one student.

## Stage 3 - train the first robust BC student

```powershell
python -m microtaur_velocity.distill_reliable.train_bc `
  --dataset-dirs logs\distill_reliable\bc_robust `
  --out logs\distill_reliable\students\student_64x64_robust_bc.pt `
  --hidden-dims 64 64 `
  --epochs 80 `
  --loss huber
```

Leave `--obs-noise-std` at its default `0` initially. The rollout dataset
already contains the environment's real observation corruption; adding
independent normalized noise is a separate augmentation, not a replacement for
domain randomization or latency.

## Stage 4 - compare teacher and student in robust closed loop

Teacher:

```powershell
python -m microtaur_velocity.distill_reliable.eval_teacher `
  $Task `
  --checkpoint-file $Teacher `
  --steps 5000 `
  --num-envs 256 `
  --out-json logs\distill_reliable\eval\teacher_robust.json
```

Student:

```powershell
python -m microtaur_velocity.distill_reliable.eval_student `
  $Task `
  --student-checkpoint logs\distill_reliable\students\student_64x64_robust_bc.pt `
  --steps 5000 `
  --num-envs 256 `
  --out-json logs\distill_reliable\eval\student_robust_bc.json
```

Repeat both with `--clean-env` to separate nominal imitation quality from
robustness loss.

## Stage 5 - robust DAgger round 1

```powershell
python -m microtaur_velocity.distill_reliable.collect_dagger `
  $Task `
  --teacher-checkpoint $Teacher `
  --student-checkpoint logs\distill_reliable\students\student_64x64_robust_bc.pt `
  --out-dir logs\distill_reliable\dagger_robust_beta050 `
  --beta 0.50 `
  --steps 50000 `
  --num-envs 1024
```

Retrain:

```powershell
python -m microtaur_velocity.distill_reliable.train_bc `
  --dataset-dirs `
    logs\distill_reliable\bc_robust `
    logs\distill_reliable\dagger_robust_beta050 `
  --out logs\distill_reliable\students\student_64x64_robust_dagger1.pt `
  --hidden-dims 64 64 `
  --epochs 80 `
  --loss huber
```

## Stage 6 - robust DAgger round 2

```powershell
python -m microtaur_velocity.distill_reliable.collect_dagger `
  $Task `
  --teacher-checkpoint $Teacher `
  --student-checkpoint logs\distill_reliable\students\student_64x64_robust_dagger1.pt `
  --out-dir logs\distill_reliable\dagger_robust_beta025 `
  --beta 0.25 `
  --steps 50000 `
  --num-envs 1024
```

Retrain:

```powershell
python -m microtaur_velocity.distill_reliable.train_bc `
  --dataset-dirs `
    logs\distill_reliable\bc_robust `
    logs\distill_reliable\dagger_robust_beta050 `
    logs\distill_reliable\dagger_robust_beta025 `
  --out logs\distill_reliable\students\student_64x64_robust_dagger2.pt `
  --hidden-dims 64 64 `
  --epochs 80 `
  --loss huber
```

Optional beta 0.10 can be added after this if closed-loop performance is still
improving.

## Stage 7 - view the student

Robust viewer:

```powershell
python -m microtaur_velocity.distill_reliable.play_student `
  --task-id $Task `
  --student-checkpoint logs\distill_reliable\students\student_64x64_robust_dagger2.pt `
  --viewer native `
  --num-envs 1 `
  --debug-every 50
```

Clean ablation viewer:

```powershell
python -m microtaur_velocity.distill_reliable.play_student `
  --task-id $Task `
  --student-checkpoint logs\distill_reliable\students\student_64x64_robust_dagger2.pt `
  --viewer native `
  --num-envs 1 `
  --clean-env
```

The viewer now slices the student observation from the actor observation passed
by MJLab. It does not reconstruct fresh sensors or maintain a second
`prev_action` state.

## Stage 8 - export

ONNX:

```powershell
python -m microtaur_velocity.distill_reliable.export_onnx `
  --ckpt logs\distill_reliable\students\student_64x64_robust_dagger2.pt `
  --out logs\distill_reliable\export\student_64x64_robust_dagger2.onnx
```

Verify ONNX against PyTorch:

```powershell
python -m microtaur_velocity.distill_reliable.verify_onnx `
  --ckpt logs\distill_reliable\students\student_64x64_robust_dagger2.pt `
  --onnx logs\distill_reliable\export\student_64x64_robust_dagger2.onnx `
  --num-tests 1000
```

The verifier reads `obs_dim` and `action_dim` from the checkpoint, so it works
for both 35D/8D and 38D/9D policies.

NPZ:

```powershell
python -m microtaur_velocity.distill_reliable.export_npz `
  --ckpt logs\distill_reliable\students\student_64x64_robust_dagger2.pt `
  --out logs\distill_reliable\export\student_64x64_robust_dagger2.npz
```

C header:

```powershell
python -m microtaur_velocity.distill_reliable.export_c_header `
  --npz logs\distill_reliable\export\student_64x64_robust_dagger2.npz `
  --out logs\distill_reliable\export\student_64x64_robust_dagger2_weights.h
```

## Deployment timing

Do not add an extra artificial 1-2-step delay on hardware merely because the
simulation trained with that delay. The physical IMU, Dynamixel status packets,
communications, and control loop already introduce latency.

Feed the student the latest sensor measurements actually available at the
control tick plus the known previous commanded action. Measure hardware timing
separately before deciding whether any additional delay model is needed.

## Final acceptance test

For each morphology, compare:

```text
teacher clean
teacher robust
student clean
student robust
```

Track at minimum:

- forward velocity error
- yaw error/correlation
- reset/fall rate
- action magnitude
- per-command yaw response

A low BC validation loss alone is not a transfer criterion.
