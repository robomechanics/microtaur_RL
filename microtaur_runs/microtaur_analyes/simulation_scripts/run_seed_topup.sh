#!/usr/bin/env bash
# Top the remaining terrain cells up from 3 seeds to 8 (seeds 3-7).
#
# Ordered by how provisional each column is, so the most valuable data lands
# first: steps (largest seed spread -- pitch's peak-speed result reversed when
# topped up), then curb (bimodal for yaw), then weave and flat (both tight).
#
# Already done, and skipped here: pitch/steps and yaw/curb.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
SPEEDS="0.08 0.14 0.20"
SEEDS="3 4 5 6 7"
SECS=16

run () {                       # run <variant> <terrain> [extra flags...]
  local v="$1" t="$2"; shift 2
  echo "##### $v / $t  ($(date +%H:%M:%S)) #####"
  $PY scripts/rollout_log.py --variant "$v" --terrain "$t" \
      --vx $SPEEDS --seeds $SEEDS --sim-seconds $SECS "$@" 2>&1 \
    | grep -E "^\[rollout_log\]|v_body_x=|Traceback|Error"
}

for v in rigid yaw roll;        do run "$v" steps --step-height 0.010; done
for v in rigid pitch roll;      do run "$v" curb  --curb-height 0.015; done
for v in rigid pitch yaw roll;  do run "$v" weave --weave-drive --pole-spacing 0.80 --num-poles 5; done
for v in rigid pitch yaw roll;  do run "$v" flat; done

echo "##### SEED TOPUP COMPLETE $(date +%H:%M:%S) #####"
