# Robust Microtaur distillation command sequence (PowerShell)

Set the registered task and final teacher checkpoint:

```powershell
$Task = "YOUR_REGISTERED_MICROTAUR_TASK"
$Teacher = "logs\rsl_rl\YOUR_EXPERIMENT\YOUR_RUN\model_4499.pt"
$BC = "logs\distill_reliable\bc_robust"
$Students = "logs\distill_reliable\students"
$Eval = "logs\distill_reliable\eval"
$Export = "logs\distill_reliable\export"

# 1) Robust teacher-forced BC collection.
python -m microtaur_velocity.distill_reliable.record_teacher_bc `
  $Task `
  --checkpoint-file $Teacher `
  --out-dir $BC `
  --steps 100000 `
  --num-envs 2048

# 2) Audit dimensions/data.
python -m microtaur_velocity.distill_reliable.audit_dataset `
  --dataset-dirs $BC

# 3) Train first robust BC student.
python -m microtaur_velocity.distill_reliable.train_bc `
  --dataset-dirs $BC `
  --out "$Students\student_64x64_robust_bc.pt" `
  --hidden-dims 64 64 `
  --epochs 80 `
  --loss huber

# 4) Robust teacher/student baseline.
python -m microtaur_velocity.distill_reliable.eval_teacher `
  $Task `
  --checkpoint-file $Teacher `
  --steps 5000 `
  --num-envs 256 `
  --out-json "$Eval\teacher_robust.json"

python -m microtaur_velocity.distill_reliable.eval_student `
  $Task `
  --student-checkpoint "$Students\student_64x64_robust_bc.pt" `
  --steps 5000 `
  --num-envs 256 `
  --out-json "$Eval\student_robust_bc.json"

# Optional clean ablations.
python -m microtaur_velocity.distill_reliable.eval_teacher `
  $Task `
  --checkpoint-file $Teacher `
  --steps 5000 `
  --num-envs 256 `
  --clean-env `
  --out-json "$Eval\teacher_clean.json"

python -m microtaur_velocity.distill_reliable.eval_student `
  $Task `
  --student-checkpoint "$Students\student_64x64_robust_bc.pt" `
  --steps 5000 `
  --num-envs 256 `
  --clean-env `
  --out-json "$Eval\student_clean_bc.json"

# 5) Robust DAgger beta 0.50.
python -m microtaur_velocity.distill_reliable.collect_dagger `
  $Task `
  --teacher-checkpoint $Teacher `
  --student-checkpoint "$Students\student_64x64_robust_bc.pt" `
  --out-dir logs\distill_reliable\dagger_robust_beta050 `
  --beta 0.50 `
  --steps 50000 `
  --num-envs 1024

python -m microtaur_velocity.distill_reliable.train_bc `
  --dataset-dirs `
    $BC `
    logs\distill_reliable\dagger_robust_beta050 `
  --out "$Students\student_64x64_robust_dagger1.pt" `
  --hidden-dims 64 64 `
  --epochs 80 `
  --loss huber

# 6) Robust DAgger beta 0.25.
python -m microtaur_velocity.distill_reliable.collect_dagger `
  $Task `
  --teacher-checkpoint $Teacher `
  --student-checkpoint "$Students\student_64x64_robust_dagger1.pt" `
  --out-dir logs\distill_reliable\dagger_robust_beta025 `
  --beta 0.25 `
  --steps 50000 `
  --num-envs 1024

python -m microtaur_velocity.distill_reliable.train_bc `
  --dataset-dirs `
    $BC `
    logs\distill_reliable\dagger_robust_beta050 `
    logs\distill_reliable\dagger_robust_beta025 `
  --out "$Students\student_64x64_robust_dagger2.pt" `
  --hidden-dims 64 64 `
  --epochs 80 `
  --loss huber

# 7) View robust student. Tyro dataclass arguments use --task-id.
python -m microtaur_velocity.distill_reliable.play_student `
  --task-id $Task `
  --student-checkpoint "$Students\student_64x64_robust_dagger2.pt" `
  --viewer native `
  --num-envs 1 `
  --debug-every 50

# Optional clean viewer.
python -m microtaur_velocity.distill_reliable.play_student `
  --task-id $Task `
  --student-checkpoint "$Students\student_64x64_robust_dagger2.pt" `
  --viewer native `
  --num-envs 1 `
  --clean-env

# 8) Export ONNX and verify numerically.
python -m microtaur_velocity.distill_reliable.export_onnx `
  --ckpt "$Students\student_64x64_robust_dagger2.pt" `
  --out "$Export\student_64x64_robust_dagger2.onnx"

python -m microtaur_velocity.distill_reliable.verify_onnx `
  --ckpt "$Students\student_64x64_robust_dagger2.pt" `
  --onnx "$Export\student_64x64_robust_dagger2.onnx" `
  --num-tests 1000

# 9) Optional NPZ/C-header export.
python -m microtaur_velocity.distill_reliable.export_npz `
  --ckpt "$Students\student_64x64_robust_dagger2.pt" `
  --out "$Export\student_64x64_robust_dagger2.npz"

python -m microtaur_velocity.distill_reliable.export_c_header `
  --npz "$Export\student_64x64_robust_dagger2.npz" `
  --out "$Export\student_64x64_robust_dagger2_weights.h"
```

Expected dataset/checkpoint widths:

```text
rigid:        35 observations, 8 actions
pitch/yaw/twist active spine: 38 observations, 9 actions
```

Robust mode is the default for collection, DAgger, evaluation, and the viewer.
Use `--clean-env` only for controlled clean-environment comparisons.
