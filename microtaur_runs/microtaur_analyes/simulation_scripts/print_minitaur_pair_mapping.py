"""Print the Minitaur-style pair-action mapping without launching MuJoCo.

Run from project root:
  python scripts/print_minitaur_pair_mapping.py

Action order:
  [leg1_swing, leg1_extend, leg2_swing, leg2_extend,
   leg3_swing, leg3_extend, leg4_swing, leg4_extend]
"""

from __future__ import annotations

LEG_JOINT_NAMES = (
  "leg1_a_joint_act", "leg1_e_joint_act",
  "leg2_a_joint_act", "leg2_e_joint_act",
  "leg3_a_joint_act", "leg3_e_joint_act",
  "leg4_a_joint_act", "leg4_e_joint_act",
)

SWING_SCALE = 0.22
EXTENSION_SCALE = 0.26

def map_action(action):
  out = []
  for i in range(4):
    swing = action[2*i] * SWING_SCALE
    extension = action[2*i + 1] * EXTENSION_SCALE
    out.extend([swing + extension, swing - extension])
  return dict(zip(LEG_JOINT_NAMES, out))

for name, action in {
  "zero": [0,0,0,0,0,0,0,0],
  "all_extend": [0,1,0,1,0,1,0,1],
  "all_contract": [0,-1,0,-1,0,-1,0,-1],
  "all_swing_forward": [1,0,1,0,1,0,1,0],
  "diagonal_trot_A": [1,0.5,-1,-0.5,-1,-0.5,1,0.5],
  "diagonal_trot_B": [-1,-0.5,1,0.5,1,0.5,-1,-0.5],
}.items():
  print(f"\n{name}")
  for joint, val in map_action(action).items():
    print(f"  {joint:18s} {val:+.3f}")
