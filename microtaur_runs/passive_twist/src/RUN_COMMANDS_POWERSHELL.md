# Reliable distillation command sequence

Replace:

```text
Mjlab-Velocity-Flat-microtaur_trot
logs\rsl_rl\YOUR_EXPERIMENT\YOUR_RUN\model_3000.pt
```

with your actual task and teacher checkpoint.

```powershell
$Task = "Mjlab-Velocity-Flat-microtaur_trot"
$Teacher = "logs\rsl_rl\YOUR_EXPERIMENT\YOUR_RUN\model_3000.pt"

python -m microtaur_velocity.distill_reliable.record_teacher_bc `
  $Task `
  --checkpoint-file $Teacher `
  --out-dir logs\distill_reliable\bc_final3000 `
  --steps 100000 `
  --num-envs 2048

python -m microtaur_velocity.distill_reliable.audit_dataset `
  --dataset-dirs logs\distill_reliable\bc_final3000

python -m microtaur_velocity.distill_reliable.train_bc `
  --dataset-dirs logs\distill_reliable\bc_final3000 `
  --out logs\distill_reliable\students\student_64x64_bc.pt `
  --hidden-dims 64 64 `
  --epochs 80 `
  --loss huber

python -m microtaur_velocity.distill_reliable.eval_student `
  $Task `
  --student-checkpoint logs\distill_reliable\students\student_64x64_bc.pt `
  --steps 5000 `
  --num-envs 256 `
  --out-json logs\distill_reliable\eval\student_64x64_bc.json

python -m microtaur_velocity.distill_reliable.collect_dagger `
  $Task `
  --teacher-checkpoint $Teacher `
  --student-checkpoint logs\distill_reliable\students\student_64x64_bc.pt `
  --out-dir logs\distill_reliable\dagger_beta050 `
  --beta 0.50 `
  --steps 50000 `
  --num-envs 1024

python -m microtaur_velocity.distill_reliable.train_bc `
  --dataset-dirs logs\distill_reliable\bc_final3000 logs\distill_reliable\dagger_beta050 `
  --out logs\distill_reliable\students\student_64x64_dagger1.pt `
  --hidden-dims 64 64 `
  --epochs 80 `
  --loss huber

python -m microtaur_velocity.distill_reliable.collect_dagger `
  $Task `
  --teacher-checkpoint $Teacher `
  --student-checkpoint logs\distill_reliable\students\student_64x64_dagger1.pt `
  --out-dir logs\distill_reliable\dagger_beta025 `
  --beta 0.25 `
  --steps 50000 `
  --num-envs 1024

python -m microtaur_velocity.distill_reliable.train_bc `
  --dataset-dirs logs\distill_reliable\bc_final3000 logs\distill_reliable\dagger_beta050 logs\distill_reliable\dagger_beta025 `
  --out logs\distill_reliable\students\student_64x64_dagger2.pt `
  --hidden-dims 64 64 `
  --epochs 80 `
  --loss huber

python -m microtaur_velocity.distill_reliable.play_student `
  $Task `
  --student-checkpoint logs\distill_reliable\students\student_64x64_dagger2.pt `
  --viewer native `
  --num-envs 1 `
  --debug-every 50

python -m microtaur_velocity.distill_reliable.export_npz `
  --ckpt logs\distill_reliable\students\student_64x64_dagger2.pt `
  --out logs\distill_reliable\export\student_64x64_dagger2.npz

python -m microtaur_velocity.distill_reliable.export_c_header `
  --npz logs\distill_reliable\export\student_64x64_dagger2.npz `
  --out logs\distill_reliable\export\student_64x64_dagger2_weights.h
```
