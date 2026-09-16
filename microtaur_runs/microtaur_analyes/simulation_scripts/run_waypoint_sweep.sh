#!/usr/bin/env bash
# Waypoint test on the step field: steer every variant down the centreline.
#
# The open-loop step results were confounded -- the robots drift, and the one
# that drifts most (rigid at peak) wandered off the 1.2 m-wide field, got reset
# by out_of_terrain_bounds, and restarted on the easy flat lead-in. Here a
# pure-pursuit controller holds each robot on the line to a waypoint at the far
# end of the field, so everyone stays on the tiles and the comparison becomes
# "how far down the course do you get, how straight, and at what energy per
# metre actually gained".
set -uo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
# Seed-matched render of a logged run that LEFT the field (rigid, steps, 0.20,
# seed 1: reached y=-1.30 m and was reset at t=9.74 s). If the HUD reports that
# reset at ~9.7 s, the render reproduces the logged run exactly.
echo "##### VIDEO rigid open-loop seed 1 (logged: left field, reset t=9.74) #####"
$PY scripts/render_terrain.py --variant rigid --terrain steps --vx 0.20 --seed 1     --seconds 16 --settle 2 --azimuth 180 --elevation -72 --distance 2.4     --tag rigid_steps_openloop_seed1_top     --label "STEPS 10mm | rigid | open loop | seed 1 | cmd 0.20" 2>&1 | grep -E "^\[terrain\]|Traceback"
for v in rigid pitch yaw roll; do
  echo "##### waypoint / steps / $v  ($(date +%H:%M:%S)) #####"
  $PY scripts/rollout_log.py --variant "$v" --terrain steps --step-height 0.010 --lane-keep \
      --vx 0.08 0.14 0.20 --seeds 0 1 2 3 4 5 6 7 --sim-seconds 16 2>&1 \
    | grep -E "^\[rollout_log\]|v_body_x=|Traceback|Error"
done
echo "##### WAYPOINT SWEEP COMPLETE $(date +%H:%M:%S) #####"
