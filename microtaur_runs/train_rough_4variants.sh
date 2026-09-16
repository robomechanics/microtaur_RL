#!/usr/bin/env bash
# Rough-terrain retraining for four Microtaur variants in two parallel GPU queues.
#
#   queue 1: active_twist (active roll) -> rigid
#   queue 2: active_pitch -> active_yaw
#
# Launch (survives closing the terminal):
#   nohup ./train_rough_4variants.sh > /dev/null 2>&1 &
# Progress:  tail -f rough_logs_<stamp>/status.txt
# Stop all:  kill <script pid>   (the trap stops the training processes too)
#
# Reduced njmax/nconmax cut GPU memory per run from ~9 GB to ~3.3 GB. Measured peak
# usage was 83 constraint rows (limit 256) and ~2.6 contacts per env (limit 64);
# if a buffer ever overflows, mujoco_warp prints "overflow - please increase" and
# that run is stopped.

set -uo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
TRAIN_BIN=/home/naomio/anaconda3/envs/microtaur/bin/train

ITERATIONS=${ITERATIONS:-4500}
NUM_ENVS=2048
LOG_ROOT=${LOG_ROOT:-logs/rsl_rl}   # relative paths resolve inside each variant folder
MIN_FREE_MIB=3072                   # always leave 3 GB for other GPU work
RUN_MIB=3500                        # measured ~3.3 GB per spine run, ~2.3 GB for rigid
POLL_S=30

STATUS_DIR=${STATUS_DIR:-$ROOT/rough_logs_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$STATUS_DIR"
exec > >(tee -a "$STATUS_DIR/status.txt") 2>&1
trap 'echo "[$(date +%T)] stopping all runs"; kill 0' INT TERM

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

run_variant() {  # <variant folder> <MICROTAUR_VARIANT> <MICROTAUR_CONTROL_SPINE>
  local variant=$1 model=$2 control_spine=$3
  local log=$STATUS_DIR/$variant.log

  until [ "$(free_mib)" -ge $((MIN_FREE_MIB + RUN_MIB)) ]; do
    echo "[$(date +%T)] $variant waiting for GPU memory ($(free_mib) MiB free)"
    sleep "$POLL_S"
  done

  echo "[$(date +%T)] START $variant ($ITERATIONS iterations, log: $log)"
  # exec so $pid is the training process itself. PYTHONPATH is replaced, which
  # also drops the ROS paths that leak into conda envs.
  (
    cd "$ROOT/$variant" && exec env \
      PYTHONPATH="$ROOT/$variant/src" \
      MICROTAUR_VARIANT="$model" \
      MICROTAUR_CONTROL_SPINE="$control_spine" \
      CUDA_VISIBLE_DEVICES=0 \
      "$TRAIN_BIN" Mjlab-Velocity-Rough-microtaur_velocity \
        --env.scene.num-envs "$NUM_ENVS" \
        --env.sim.njmax 256 \
        --env.sim.nconmax 64 \
        --agent.max-iterations "$ITERATIONS" \
        --agent.logger tensorboard \
        --agent.run-name "${variant}_rough" \
        --log-root "$LOG_ROOT"
  ) > "$log" 2>&1 &
  local pid=$!

  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_S"
    if grep -aq "overflow - please increase" "$log"; then
      echo "[$(date +%T)] ERROR $variant: simulator buffer overflow, stopping run"
      grep -a "overflow - please increase" "$log" | sort | uniq -c
      kill "$pid"
      break
    fi
    local free
    free=$(free_mib)
    if [ "$free" -lt "$MIN_FREE_MIB" ]; then
      echo "[$(date +%T)] WARNING only $free MiB GPU memory free"
    fi
  done

  wait "$pid"
  local status=$?
  local last_iteration
  last_iteration=$(grep -a "Learning iteration" "$log" | tail -1 | tr -s ' ')
  echo "[$(date +%T)] END $variant exit=$status |${last_iteration}"
}

queue_1() {
  run_variant active_twist active_twist_microtaur 1
  run_variant rigid rigid_microtaur 0
}

queue_2() {
  sleep 120  # let queue 1 allocate first so the free-memory check is meaningful
  run_variant active_pitch active_pitch_microtaur 1
  run_variant active_yaw active_yaw_microtaur 1
}

echo "[$(date +%T)] Microtaur rough-terrain retraining: $ITERATIONS iterations, status in $STATUS_DIR"
queue_1 &
queue_1_pid=$!
queue_2 &
queue_2_pid=$!
# Wait on the queues only; a bare `wait` would also wait on the status.txt tee.
wait "$queue_1_pid" "$queue_2_pid"
echo "[$(date +%T)] ALL DONE"
