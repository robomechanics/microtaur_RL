#!/usr/bin/env bash
# Resume: run only the variants still missing. rigid + pitch already complete.
set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
LOG=rollouts/resume.log
VX="0.08 0.10 0.12 0.14 0.16 0.18 0.20"
echo "=== resume start $(date) ===" | tee -a "$LOG"

have_fwd () {  # $1 = variant key ; true if a *_ dir with >=21 csv + summary exists
  for d in rollouts/$1_*; do
    [ -d "$d" ] || continue
    [ "$(ls "$d"/rollout_*.csv 2>/dev/null | wc -l)" -ge 21 ] && [ -f "$d/summary.json" ] && return 0
  done
  return 1
}
have_yaw () {  # true if a *_ dir with exactly 6 csv + summary exists
  for d in rollouts/$1_*; do
    [ -d "$d" ] || continue
    n=$(ls "$d"/rollout_*.csv 2>/dev/null | wc -l)
    [ "$n" -ge 6 ] && [ "$n" -lt 21 ] && [ -f "$d/summary.json" ] && return 0
  done
  return 1
}

for V in yaw roll; do
  if have_fwd "$V"; then echo "## $V forward: already done, skip" | tee -a "$LOG"
  else
    echo "########## $V : forward sweep  $(date) ##########" | tee -a "$LOG"
    $PY scripts/rollout_log.py --variant "$V" --vx $VX --yaw 0.0 --seeds 0 1 2 \
        --sim-seconds 18 --settle-seconds 1.0 2>&1 | tee -a "$LOG"
  fi
  if have_yaw "$V"; then echo "## $V yaw: already done, skip" | tee -a "$LOG"
  else
    echo "########## $V : yaw sweep  $(date) ##########" | tee -a "$LOG"
    $PY scripts/rollout_log.py --variant "$V" --vx 0.12 --yaw -0.2 0.2 --seeds 0 1 2 \
        --sim-seconds 18 --settle-seconds 1.0 2>&1 | tee -a "$LOG"
  fi
done

echo "########## comparison  $(date) ##########" | tee -a "$LOG"
$PY scripts/compare_variants.py 2>&1 | tee -a "$LOG"
echo "=== resume done $(date) ===" | tee -a "$LOG"
