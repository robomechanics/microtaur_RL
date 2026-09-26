#!/usr/bin/env bash
# Train and evaluate numeric variants one after another (one run at a time is
# the fastest per variant on this GPU: parallel runs share it without adding
# throughput).
#
#   bash scripts/isaac_variant_queue.sh <queue file> [--iters 600]
#
# Queue file: one variant per line, "<run_name> <hydra override> ...", '#' comments.
# For each: scripts/isaac_train_variant.sh, then scripts/isaac_eval_round.sh on the
# last checkpoint into figures/isaac_eval/<run_name>_it<N>. Prints
# "VARIANT_DONE <run_name> <eval dir>" or "VARIANT_FAILED <run_name>" per line.
set -uo pipefail
set -f  # overrides contain [ ] { }: no globbing when they are word-split
Q=$1; shift
ITERS=600
[[ ${1:-} == --iters ]] && ITERS=$2
HERE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=/home/rml3/Documents/ben/spine/runs_local/isaac
FIG=/home/rml3/Documents/ben/spine/figures/isaac_eval
grep -vE '^\s*(#|$)' "$Q" | while read -r NAME OVR; do
  # shellcheck disable=SC2086
  if ! bash "$HERE/scripts/isaac_train_variant.sh" "$NAME" --iters "$ITERS" $OVR > "$ROOT/$NAME.driver" 2>&1; then
    echo "VARIANT_FAILED $NAME (train) $(tail -3 "$ROOT/$NAME.driver" | tr '\n' ' ')"; continue
  fi
  CK=$(ls "$ROOT"/logs/rsl_rl/*/*_"$NAME"/model_$((ITERS - 1)).pt | head -1)
  OUT="$FIG/${NAME}_it$((ITERS - 1))"
  if DISPLAY=${DISPLAY:-:1} bash "$HERE/scripts/isaac_eval_round.sh" "$CK" "$OUT" > "$ROOT/$NAME.eval" 2>&1; then
    cp -f "$OUT/play.mp4" "/home/rml3/Documents/ben/spine/figures/isaac_play/${NAME}_isaac.mp4" 2>/dev/null
    cp -f "$OUT/traj_0.20.mp4" "/home/rml3/Documents/ben/spine/figures/isaac_play/${NAME}_mujoco_0.20.mp4" 2>/dev/null
    echo "VARIANT_DONE $NAME $OUT"
  else
    echo "VARIANT_FAILED $NAME (eval) $(tail -3 "$ROOT/$NAME.eval" | tr '\n' ' ')"
  fi
done
echo "QUEUE_FINISHED $Q"
