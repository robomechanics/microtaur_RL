#!/usr/bin/env bash
# Full 4-variant rollout sweep: rigid + pitch + yaw + roll.
# Forward speed sweep (7 speeds x 3 seeds) + yaw command sweep (+/-0.2 x 3 seeds),
# 18 s each. Run from the repo root. Progress goes to rollouts/run_all.log.
set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
LOG=rollouts/run_all.log
mkdir -p rollouts
echo "=== run_all_sweeps start $(date) ===" | tee -a "$LOG"

VX="0.08 0.10 0.12 0.14 0.16 0.18 0.20"

for V in rigid pitch yaw roll; do
  echo "" | tee -a "$LOG"
  echo "########## $V : forward sweep  $(date) ##########" | tee -a "$LOG"
  $PY scripts/rollout_log.py --variant "$V" \
      --vx $VX --yaw 0.0 --seeds 0 1 2 --sim-seconds 18 --settle-seconds 1.0 \
      2>&1 | tee -a "$LOG"

  echo "########## $V : yaw sweep  $(date) ##########" | tee -a "$LOG"
  $PY scripts/rollout_log.py --variant "$V" \
      --vx 0.12 --yaw -0.2 0.2 --seeds 0 1 2 --sim-seconds 18 --settle-seconds 1.0 \
      2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "########## comparison  $(date) ##########" | tee -a "$LOG"
$PY scripts/compare_variants.py 2>&1 | tee -a "$LOG"
echo "=== run_all_sweeps done $(date) ===" | tee -a "$LOG"
