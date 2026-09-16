"""Robust Microtaur policy-distillation pipeline.

The package supports both current deployment layouts:

* rigid Microtaur:       35D student observation -> 8 actions
* active-spine variants: 38D student observation -> 9 actions

Pipeline:
  1. robust teacher-forced BC collection using the training-time sensor
     corruption/delay, randomized resets, domain randomization, and action delay
     while locking commands/action authority to the final curriculum stage
  2. BC student training
  3. closed-loop robust and clean-ablation evaluation
  4. robust DAgger collection on student-visited states
  5. retraining on aggregated robust data
  6. ONNX/NPZ/C-header export and numerical export verification

Simulation distillation uses aligned student observations sliced directly from
the teacher actor observation. Fresh robot.data reconstruction is retained only
for deployment/debugging paths where no teacher actor tensor exists.
"""
