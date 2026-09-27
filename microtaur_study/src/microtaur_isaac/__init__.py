"""Rigid Microtaur velocity task for IsaacLab v2.3 / Isaac Sim 5.1 (conda env `spine`).

The task logic that does not depend on a simulator lives in microtaur_common
and is shared with the mjlab environment (microtaur_rigid); this package is the
IsaacLab glue. Importing it must not import isaaclab (gym entry points are
strings), so it is safe before the app is launched.

Interface contract between modules (keep in sync):

  scene names
    robot            Articulation, prim "{ENV_REGEX_NS}/Robot"
    contact_forces   ContactSensor on "{ENV_REGEX_NS}/Robot/.*", track_air_time=True
    height_scanner   RayCaster on the root body (teacher / critic height map; optional)
    terrain          TerrainImporter

  bodies (USD prim names = MJCF body names)
    root             "battery" (both body collision boxes live on it)
    feet             FOOT_BODY_NAMES below, in leg order 1..4 (RR, RL, FL, FR);
                     each carries one 6.2 mm foot sphere at FOOT_OFFSET_IN_BODY_M[k]

  joints             canonical order LEG_JOINT_NAMES (microtaur_common.robot_constants);
                     IsaacLab may resolve them in another order: always map by name

  action term        "joint_pos": mdp.actions.MicrotaurWalkAction, exposes .core
                     (microtaur_common.walk_action.WalkActionCore: requested, safe,
                     applied, filter_correction, filter_blend, stand)

  encoder bias       env.microtaur_encoder_bias: tensor [num_envs, 8], canonical
                     order, rad; created by the startup event mdp.events.encoder_bias
                     (zeros if the event is absent). Observation adds it to the actor's
                     joint_pos; the action term subtracts it from the joint target.

  command            "twist": UniformVelocityCommandCfg, [vx, vy=0, wz]

  terrain types      sub-terrain names "A_flat", "B_blocks", "C_step" (terrains.py);
                     env_cfg reads the per-env type to spawn C at the course start
                     heading +x and to zero yaw commands on C
"""

FOOT_BODY_NAMES = (
  "leglink2_v2_with_leg_down",
  "leglink2_v2_with_leg_down_2",
  "leglink2_v2_with_leg_down_3",
  "leglink2_v2_with_leg_down_4",
)
# Foot sphere centre (= MJCF foot site) in its body frame, leg order 1..4.
FOOT_OFFSET_IN_BODY_M = (
  (-0.017059, -0.081192, -0.0015),
  (-0.017059, 0.081192, -0.0015),
  (0.08273, 0.006242, -0.0015),
  (0.08273, -0.006242, -0.0015),
)
ROOT_BODY_NAME = "battery"
TERRAIN_TYPES = ("A_flat", "B_blocks", "C_step")

import gymnasium as gym

# Microtaur-Isaac-{Flat,Rough,Teacher}-v0 train; -Play-v0 is the same task with
# the play config (16 envs, no noise / randomisation / timeout, camera on env 0).
for _name, _cfg in (
  ("Flat", "MicrotaurFlatEnvCfg"), ("Rough", "MicrotaurRoughEnvCfg"), ("Teacher", "MicrotaurTeacherEnvCfg"),
  ("Flat-Play", "MicrotaurFlatPlayEnvCfg"), ("Rough-Play", "MicrotaurRoughPlayEnvCfg"),
  ("Teacher-Play", "MicrotaurTeacherPlayEnvCfg"),
  ("Teacher-Hard", "MicrotaurTeacherHardEnvCfg"), ("Teacher-Hard-Play", "MicrotaurTeacherHardPlayEnvCfg"),
  ("Teacher-Hard2x", "MicrotaurTeacherHard2xEnvCfg"), ("Teacher-Hard2x-Play", "MicrotaurTeacherHard2xPlayEnvCfg"),
  ("Teacher-Cur", "MicrotaurTeacherCurEnvCfg"), ("Teacher-Cur-Play", "MicrotaurTeacherCurPlayEnvCfg"),
):
  gym.register(
    id=f"Microtaur-Isaac-{_name}-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
      "env_cfg_entry_point": f"microtaur_isaac.env_cfg:{_cfg}",
      "rsl_rl_cfg_entry_point": "microtaur_isaac.agents:"
      + ("MicrotaurTeacherPPORunnerCfg" if _name.startswith("Teacher") else "MicrotaurPPORunnerCfg"),
      # left/right symmetry augmentation: train.py --agent rsl_rl_sym_cfg_entry_point
      "rsl_rl_sym_cfg_entry_point": "microtaur_isaac.agents:"
      + ("MicrotaurTeacherPPORunnerSymCfg" if _name.startswith("Teacher") else "MicrotaurPPORunnerSymCfg"),
      # student distillation from a teacher checkpoint: --agent rsl_rl_student_{mlp,gru}_cfg_entry_point
      "rsl_rl_student_mlp_cfg_entry_point": "microtaur_isaac.agents:MicrotaurStudentMLPRunnerCfg",
      "rsl_rl_student_gru_cfg_entry_point": "microtaur_isaac.agents:MicrotaurStudentGRURunnerCfg",
    },
  )
