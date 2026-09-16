#!/usr/bin/env bash
# Terrain sweep: one variant across flat / curb / steps / weave at slow, medium
# and peak commanded speed.
#
#   ./scripts/run_terrain_sweep.sh [variant] [seeds...]
#
# Speeds are the endpoints and midpoint of the trained command range
# (set_joystick_twist_command clamps to 0.08-0.20 m/s).
#
# Design notes:
#  * flat is included as the reference arm -- the pre-2026-09-09 flat baseline
#    came from the clipped prev-action path and is not comparable.
#  * the weave uses ONE pole pitch for all three speeds, otherwise speed and
#    course difficulty are confounded. Minimum trackable pitch grows as
#    sqrt(speed) (0.44 m at 0.08, 0.69 m at 0.20), so 0.80 m is used: it demands
#    0.185 rad/s at peak against the 0.25 rad/s command clamp.
#  * the curb runs OPEN LOOP. With the prev-action fix the robot holds the lip
#    unaided; --lane-keep was only ever needed to counter a harness artifact.
#  * 16 s not 18 s: at 0.20 m/s that is 3.2 m, which keeps the robot inside the
#    4.0 m course instead of running out onto the flat border.
set -euo pipefail
cd "$(dirname "$0")/.."

VARIANT="${1:-rigid}"
shift || true
SEEDS="${*:-0 1 2}"
PY=./.venv/Scripts/python.exe
SPEEDS="0.08 0.14 0.20"
SECS=16

echo "=== terrain sweep: variant=$VARIANT  speeds=$SPEEDS  seeds=$SEEDS  ${SECS}s ==="

run () {
  local name="$1"; shift
  echo ""
  echo "--- $name ---"
  $PY scripts/rollout_log.py --variant "$VARIANT" --vx $SPEEDS --seeds $SEEDS \
      --sim-seconds $SECS "$@" 2>&1 | grep -E "^\[rollout_log\]|v_body_x=|Traceback|Error"
}

run "flat (reference)"      --terrain flat
run "curb 15 mm"            --terrain curb  --curb-height 0.015
run "steps 10 mm"           --terrain steps --step-height 0.010
run "weave 0.80 m pitch"    --terrain weave --weave-drive --pole-spacing 0.80 --num-poles 5

echo ""
echo "=== sweep complete ==="
ls -dt rollouts/${VARIANT}_* | head -4
