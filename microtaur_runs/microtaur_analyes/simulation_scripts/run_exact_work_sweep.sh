#!/usr/bin/env bash
# Re-run every published rollout with --substep-log, for exact joint work.
#
# The original logger sampled tau*qd once per 20 ms policy step. The gait is
# phase-locked to those ticks, so the sample lands at the same point of every
# control interval (just before the next target update, when the servo has
# mostly caught up) and the time-average under-reads mechanical power: on pitch
# / flat / 0.14 / seed 3 it gave 0.75 W of positive work against 1.33 W exact,
# and flipped the sign of the spine's net work.
#
# Rollouts are deterministic and the substep hook is read-only (verified: the
# re-run reproduces the logged trajectory bit-for-bit), so this changes nothing
# but the energy columns. Same terrains, speeds, seeds and durations as the
# published 8-seed set. Spine variants first -- they answer the spine-function
# question -- then rigid for the CoT comparison.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
SPEEDS="0.08 0.14 0.20"
SEEDS="0 1 2 3 4 5 6 7"
SECS=16

run () {                       # run <variant> <terrain> [extra flags...]
  local v="$1" t="$2"; shift 2
  echo "##### $v / $t $*  ($(date +%H:%M:%S)) #####"
  $PY scripts/rollout_log.py --variant "$v" --terrain "$t" --substep-log \
      --vx $SPEEDS --seeds $SEEDS --sim-seconds $SECS "$@" 2>&1 \
    | grep --line-buffered -E "^\[rollout_log\]|v_body_x=|Traceback|Error"
}

for v in ${VARIANTS:-pitch yaw roll rigid}; do
  run "$v" flat
  run "$v" curb  --curb-height 0.015
  run "$v" weave --weave-drive --pole-spacing 0.80 --num-poles 5
  run "$v" steps --step-height 0.010 --lane-keep
  run "$v" steps --step-height 0.010
done
echo "##### EXACT-WORK SWEEP COMPLETE $(date +%H:%M:%S) #####"
