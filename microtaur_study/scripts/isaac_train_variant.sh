#!/usr/bin/env bash
# Train one numeric variant of a Microtaur IsaacLab task and stop when the last
# checkpoint is written (Kit's SimulationApp.close can hang after training).
#
#   bash scripts/isaac_train_variant.sh <run_name> [--task T] [--envs N] [--iters I] [hydra overrides ...]
#   e.g. bash scripts/isaac_train_variant.sh r2a env.rewards.trot_gait.weight=1.0 env.rewards.trot_gait.params.std=0.1
#
# Logs: /home/rml3/Documents/ben/spine/runs_local/isaac/<run_name>.log,
# checkpoints: .../logs/rsl_rl/<experiment>/<date>_<run_name>/ (params/env.yaml records the overrides).
set -uo pipefail
RUN=$1; shift
TASK=Microtaur-Isaac-Flat-v0; ENVS=2048; ITERS=1000
while [[ $# -gt 0 && $1 == --* ]]; do
  case $1 in
    --task) TASK=$2; shift 2;;
    --envs) ENVS=$2; shift 2;;
    --iters) ITERS=$2; shift 2;;
    *) echo "unknown option $1"; exit 2;;
  esac
done
ROOT=/home/rml3/Documents/ben/spine/runs_local/isaac
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT"; cd "$ROOT"
LAST=model_$((ITERS - 1)).pt
echo "$(date '+%F %T') $RUN task=$TASK envs=$ENVS iters=$ITERS overrides: $*" >> "$ROOT/variants.txt"
OMNI_KIT_ACCEPT_EULA=YES /home/rml3/anaconda3/envs/spine/bin/python "$HERE/scripts/isaac_train.py" \
  --task "$TASK" --num_envs "$ENVS" --headless --max_iterations "$ITERS" --run_name "$RUN" "$@" > "$RUN.log" 2>&1 &
P=$!
until ls logs/rsl_rl/*/*_"$RUN"/"$LAST" >/dev/null 2>&1 || ! kill -0 $P 2>/dev/null || grep -q Traceback "$RUN.log"; do sleep 20; done
sleep 5; kill $P 2>/dev/null; sleep 3; kill -9 $P 2>/dev/null
D=$(ls -d logs/rsl_rl/*/*_"$RUN" 2>/dev/null | head -1)
if grep -q Traceback "$RUN.log"; then echo "$RUN FAILED"; grep -A8 Traceback "$RUN.log" | tail -10; exit 1; fi
echo "$RUN done: $ROOT/$D/$LAST"
grep -E "Learning iteration|forward_speed|track_lin_vel_xy|track_ang_vel_z|trot_gait" "$RUN.log" | tail -5
