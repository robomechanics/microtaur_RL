#!/usr/bin/env bash
# Train and evaluate numeric variants one after another (one run at a time is
# the fastest per variant on this GPU: parallel runs share it without adding
# throughput).
#
#   bash scripts/isaac_variant_queue.sh <queue file> [--iters 600]
#
# Queue file: one variant per line, "<run_name> <hydra override> ...", '#' comments.
# For each: scripts/isaac_train_variant.sh, then scripts/isaac_eval_round.sh on the
# last checkpoint into figures/isaac_eval/flat/<run_name>_it<N>. Prints
# "VARIANT_DONE <run_name> <eval dir>" or "VARIANT_FAILED <run_name>" per line.
# A variant whose final checkpoint already exists is only evaluated.
set -uo pipefail
Q=$1; shift
ITERS=600
[[ ${1:-} == --iters ]] && ITERS=$2
HERE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=/home/rml3/Documents/ben/spine/runs_local/isaac
FIG=/home/rml3/Documents/ben/spine/figures/isaac_eval/flat
PLAYFIG=/home/rml3/Documents/ben/spine/figures/isaac_play/flat
grep -vE '^\s*(#|$)' "$Q" | while read -r NAME OVR; do
  set -f; read -r -a ARGS <<< "$OVR"; set +f  # overrides contain [ ] { }: split without globbing
  CK=$(ls "$ROOT"/logs/rsl_rl/*/*_"$NAME"/model_$((ITERS - 1)).pt 2>/dev/null | head -1)
  if [[ -z $CK ]]; then
    if ! bash "$HERE/scripts/isaac_train_variant.sh" "$NAME" --iters "$ITERS" "${ARGS[@]}" > "$ROOT/$NAME.driver" 2>&1; then
      echo "VARIANT_FAILED $NAME (train) $(tail -3 "$ROOT/$NAME.driver" | tr '\n' ' ')"; continue
    fi
    CK=$(ls "$ROOT"/logs/rsl_rl/*/*_"$NAME"/model_$((ITERS - 1)).pt 2>/dev/null | head -1)
  fi
  [[ -z $CK ]] && { echo "VARIANT_FAILED $NAME (no checkpoint)"; continue; }
  OUT="$FIG/${NAME}_it$((ITERS - 1))"
  if DISPLAY=${DISPLAY:-:1} bash "$HERE/scripts/isaac_eval_round.sh" "$CK" "$OUT" > "$ROOT/$NAME.eval" 2>&1; then
    cp -f "$OUT/play.mp4" "$PLAYFIG/${NAME}_isaac.mp4" 2>/dev/null
    cp -f "$OUT/traj_0.20.mp4" "$PLAYFIG/${NAME}_mujoco_0.20.mp4" 2>/dev/null
    echo "VARIANT_DONE $NAME $OUT"
  else
    echo "VARIANT_FAILED $NAME (eval) $(tail -3 "$ROOT/$NAME.eval" | tr '\n' ' ')"
  fi
done
echo "QUEUE_FINISHED $Q"
