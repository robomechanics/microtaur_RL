#!/usr/bin/env bash
# Terrain test suite for the rough-terrain `*_with_flag` checkpoints (Linux, this repo).
#
# Same protocol as run_exact_work_sweep.sh (the published 8-seed set): flat, curb
# 15 mm (open loop), weave 0.80 m x 5 poles (weave-drive), steps 10 mm (lane-keep and
# open loop), at 0.08 / 0.14 / 0.20 m/s, 16 s, with exact joint work. Variants run in
# parallel (one process each; a 1-robot rollout is CPU-bound).
#
#   ./simulation_scripts/run_terrain_suite.sh              # all four variants
#   VARIANTS="pitch yaw" SEEDS="0 1 2" ./simulation_scripts/run_terrain_suite.sh
#
# Output: rollouts/<variant>[_<terrain>]_<stamp>/ under microtaur_analyes/, and
# rollouts/suite_<stamp>/<variant>.log with progress.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=/home/naomio/anaconda3/envs/microtaur/bin/python
SPEEDS="0.08 0.14 0.20"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7}"
SECS=16
LOG_DIR=rollouts/suite_$(date +%Y%m%d_%H%M%S)
mkdir -p "$LOG_DIR"

run () {                       # run <variant> <terrain> [extra flags...]
  local v="$1" t="$2"; shift 2
  echo "##### $v / $t $*  ($(date +%H:%M:%S)) #####"
  env -u PYTHONPATH MUJOCO_GL=egl "$PY" -W ignore simulation_scripts/rollout_log.py \
      --variant "$v" --terrain "$t" --substep-log \
      --vx $SPEEDS --seeds $SEEDS --sim-seconds $SECS "$@" 2>&1 \
    | grep --line-buffered -E "^\[rollout_log\]|v_body_x=|Traceback|Error"
}

suite () {                     # suite <variant>
  local v="$1"
  run "$v" flat
  run "$v" curb  --curb-height 0.015
  run "$v" weave --weave-drive --pole-spacing 0.80 --num-poles 5
  run "$v" steps --step-height 0.010 --lane-keep
  run "$v" steps --step-height 0.010
  echo "##### $v SUITE COMPLETE $(date +%H:%M:%S) #####"
}

echo "terrain suite: variants=${VARIANTS:-rigid pitch yaw roll} seeds=$SEEDS logs=$LOG_DIR"
pids=()
for v in ${VARIANTS:-rigid pitch yaw roll}; do
  suite "$v" > "$LOG_DIR/$v.log" 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
echo "ALL SUITES COMPLETE $(date +%H:%M:%S)" | tee "$LOG_DIR/done.txt"
