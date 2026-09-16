# Reliable Microtaur distillation pipeline

This package distills your large PPO velocity policy into a smaller deployable student using:

1. **Behavior cloning** from teacher-forced rollouts.
2. **Closed-loop evaluation** of the student.
3. **DAgger-style data aggregation**: roll out the student/mixed policy, query the teacher on those student-visited states, retrain on the aggregated dataset.
4. **NPZ/C-header export** for ESP32-S3 conversion.

The student observation is 35D:

```text
projected_gravity_b      3
base_ang_vel_b           3
joint_pos_rel            8
joint_vel                8
prev_action t-1          8
command                  3
phase_sin_cos            2
---------------------------
total                   35
```

It excludes:

```text
height scan
foot contact sensors
contact force sensors
older action history t-2/t-3/t-4
```

It keeps `prev_action t-1` because that is cheap on ESP32-S3 and makes gait imitation much less ambiguous.

## Install

From project root:

```powershell
Expand-Archive .\microtaur_distill_reliable.zip -DestinationPath .\microtaur_distill_reliable_unpacked -Force
Copy-Item .\microtaur_distill_reliable_unpacked\src\microtaur_velocity\distill_reliable .\src\microtaur_velocity\ -Recurse -Force
$env:PYTHONPATH="$pwd\src"
```

Linux/macOS:

```bash
unzip -o microtaur_distill_reliable.zip -d microtaur_distill_reliable_unpacked
cp -r microtaur_distill_reliable_unpacked/src/microtaur_velocity/distill_reliable src/microtaur_velocity/
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

## Stage 0: choose teacher checkpoint

Use a straight-walking teacher checkpoint that passes:

```text
actual_forward_mean > 0.16
forward_error_abs_mean < 0.06
yaw_rate_abs_mean < 0.10
fell_over = 0
illegal_contact = 0
```

For your current run, keep both an iteration ~2000 checkpoint and the final 3000 checkpoint if possible.

## Stage 1: collect teacher-forced BC dataset

```powershell
python -m microtaur_velocity.distill_reliable.record_teacher_bc `
  Mjlab-Velocity-Flat-microtaur_trot `
  --checkpoint-file logs\rsl_rl\YOUR_EXPERIMENT\YOUR_RUN\model_3000.pt `
  --out-dir logs\distill_reliable\bc_final3000 `
  --steps 1000 `
  --num-envs 2048
```

For a real dataset, use more steps:

```powershell
python -m microtaur_velocity.distill_reliable.record_teacher_bc `
  Mjlab-Velocity-Flat-microtaur_trot `
  --checkpoint-file logs\rsl_rl\YOUR_EXPERIMENT\YOUR_RUN\model_3000.pt `
  --out-dir logs\distill_reliable\bc_final3000 `
  --steps 100000 `
  --num-envs 2048
```

Samples = `steps * num_envs`.

## Stage 2: audit dataset

```powershell
python -m microtaur_velocity.distill_reliable.audit_dataset `
  --dataset-dirs logs\distill_reliable\bc_final3000
```

You want:

```text
obs_dim = 35
action_abs_mean not tiny
action_std_mean not tiny
```

Red flags:

```text
obs_dim = 59  -> old history recorder got mixed in
obs_dim = 27  -> no-prev-action version got mixed in
action_abs_mean < 0.03 -> teacher was basically standing
```

## Stage 3: train first BC student

First train slightly bigger than final:

```powershell
python -m microtaur_velocity.distill_reliable.train_bc `
  --dataset-dirs logs\distill_reliable\bc_final3000 `
  --out logs\distill_reliable\students\student_128x64_bc.pt `
  --hidden-dims 128 64 `
  --epochs 80 `
  --loss huber
```

Then train the deployable size:

```powershell
python -m microtaur_velocity.distill_reliable.train_bc `
  --dataset-dirs logs\distill_reliable\bc_final3000 `
  --out logs\distill_reliable\students\student_64x64_bc.pt `
  --hidden-dims 64 64 `
  --epochs 80 `
  --loss huber
```

During training, do not only look at validation loss. Watch:

```text
action_abs_mean from dataset
pred_abs from student
```

If dataset `action_abs_mean` is 0.18 but student `pred_abs` is 0.02, the student will probably freeze.

## Stage 4: closed-loop eval

```powershell
python -m microtaur_velocity.distill_reliable.eval_student `
  Mjlab-Velocity-Flat-microtaur_trot `
  --student-checkpoint logs\distill_reliable\students\student_64x64_bc.pt `
  --steps 5000 `
  --num-envs 256 `
  --out-json logs\distill_reliable\eval\student_64x64_bc.json
```

Pass target:

```text
actual_forward_mean > 0.16
forward_error_abs_mean < 0.06
yaw_rate_abs_mean < 0.10
action_abs_mean not tiny
reset_count low
```

## Stage 5: DAgger round 1

Roll out a mixture of teacher and student actions, but label each state with the teacher action.

```powershell
python -m microtaur_velocity.distill_reliable.collect_dagger `
  Mjlab-Velocity-Flat-microtaur_trot `
  --teacher-checkpoint logs\rsl_rl\YOUR_EXPERIMENT\YOUR_RUN\model_3000.pt `
  --student-checkpoint logs\distill_reliable\students\student_64x64_bc.pt `
  --out-dir logs\distill_reliable\dagger_beta050 `
  --beta 0.50 `
  --steps 50000 `
  --num-envs 1024
```

Retrain on aggregated data:

```powershell
python -m microtaur_velocity.distill_reliable.train_bc `
  --dataset-dirs logs\distill_reliable\bc_final3000 logs\distill_reliable\dagger_beta050 `
  --out logs\distill_reliable\students\student_64x64_dagger1.pt `
  --hidden-dims 64 64 `
  --epochs 80 `
  --loss huber
```

## Stage 6: DAgger round 2

```powershell
python -m microtaur_velocity.distill_reliable.collect_dagger `
  Mjlab-Velocity-Flat-microtaur_trot `
  --teacher-checkpoint logs\rsl_rl\YOUR_EXPERIMENT\YOUR_RUN\model_3000.pt `
  --student-checkpoint logs\distill_reliable\students\student_64x64_dagger1.pt `
  --out-dir logs\distill_reliable\dagger_beta025 `
  --beta 0.25 `
  --steps 50000 `
  --num-envs 1024
```

Retrain:

```powershell
python -m microtaur_velocity.distill_reliable.train_bc `
  --dataset-dirs logs\distill_reliable\bc_final3000 logs\distill_reliable\dagger_beta050 logs\distill_reliable\dagger_beta025 `
  --out logs\distill_reliable\students\student_64x64_dagger2.pt `
  --hidden-dims 64 64 `
  --epochs 80 `
  --loss huber
```

Optional round 3:

```powershell
python -m microtaur_velocity.distill_reliable.collect_dagger `
  Mjlab-Velocity-Flat-microtaur_trot `
  --teacher-checkpoint logs\rsl_rl\YOUR_EXPERIMENT\YOUR_RUN\model_3000.pt `
  --student-checkpoint logs\distill_reliable\students\student_64x64_dagger2.pt `
  --out-dir logs\distill_reliable\dagger_beta010 `
  --beta 0.10 `
  --steps 50000 `
  --num-envs 1024
```

## Stage 7: view in sim

```powershell
python -m microtaur_velocity.distill_reliable.play_student `
  Mjlab-Velocity-Flat-microtaur_trot `
  --student-checkpoint logs\distill_reliable\students\student_64x64_dagger2.pt `
  --viewer native `
  --num-envs 1 `
  --debug-every 50
```

## Stage 8: export

```powershell
python -m microtaur_velocity.distill_reliable.export_npz `
  --ckpt logs\distill_reliable\students\student_64x64_dagger2.pt `
  --out logs\distill_reliable\export\student_64x64_dagger2.npz
```

Optional C header:

```powershell
python -m microtaur_velocity.distill_reliable.export_c_header `
  --npz logs\distill_reliable\export\student_64x64_dagger2.npz `
  --out logs\distill_reliable\export\student_64x64_dagger2_weights.h
```

## Expected parameter counts

Approximate parameter counts:

```text
35 -> 128 -> 64 -> 8 : about 13k parameters
35 -> 64  -> 64 -> 8 : about 7k parameters
35 -> 48  -> 48 -> 8 : about 4.5k parameters
```

For ESP32-S3, start with 64x64 after it works in sim. Only compress to 48x48 after DAgger passes.
