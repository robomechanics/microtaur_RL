#!/usr/bin/env bash
# CPG-RL training for the four Microtaur variants, all four runs in parallel.
#
# Each policy commands four leg oscillators (swing gain, lift gain, frequency per leg,
# 12 actions) instead of eight direct motor targets. The oscillators are initialized
# from the fit of each variant's flat-terrain model_4499 gait (fit_<variant>.json,
# produced by fit_cpg_from_flat.py); with zero actions that nominal CPG already walks
# (rigid: 0.138 m/s against a 0.15 m/s command, no falls).
#
# Launch:   nohup ./train_cpg_4variants.sh > /dev/null 2>&1 &
# Progress: tail -f cpg_logs_<stamp>/status.txt

set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
CPG_DIR=$ROOT/cpg_rl
TRAIN_BIN=/home/naomio/anaconda3/envs/microtaur/bin/train
TASK=${TASK:-Mjlab-Velocity-Flat-microtaur_velocity}
ITERATIONS=${ITERATIONS:-4500}
NUM_ENVS=${NUM_ENVS:-2048}

STATUS_DIR=${STATUS_DIR:-$CPG_DIR/cpg_logs_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$STATUS_DIR"
exec > >(tee -a "$STATUS_DIR/status.txt") 2>&1
trap 'echo "[$(date +%T)] stopping all runs"; kill 0' INT TERM

run() {  # <variant folder> <MICROTAUR_VARIANT> <MICROTAUR_CONTROL_SPINE> <fit name>
  local variant=$1 model=$2 control_spine=$3 fit=$4
  local name=${variant}_cpg
  local log=$STATUS_DIR/$name.log
  echo "[$(date +%T)] START $name (fit: fit_$fit.json, log: $log)"
  (
    cd "$ROOT/$variant" && exec env \
      PYTHONPATH="$ROOT/$variant/src" \
      MICROTAUR_VARIANT="$model" \
      MICROTAUR_CONTROL_SPINE="$control_spine" \
      MICROTAUR_CURRICULUM_START_STEP=0 \
      MICROTAUR_CPG=1 \
      MICROTAUR_CPG_FIT="$CPG_DIR/fit_$fit.json" \
      MUJOCO_GL=egl \
      CUDA_VISIBLE_DEVICES=0 \
      "$TRAIN_BIN" "$TASK" \
        --env.scene.num-envs "$NUM_ENVS" \
        --env.sim.njmax 256 \
        --env.sim.nconmax 64 \
        --agent.max-iterations "$ITERATIONS" \
        --agent.logger tensorboard \
        --agent.clip-actions 1.0 \
        --agent.run-name "$name"
  ) > "$log" 2>&1 &
  pids+=("$!")
}

pids=()
run rigid rigid_microtaur 0 rigid
run active_twist active_twist_microtaur 1 roll
run active_pitch active_pitch_microtaur 1 pitch
run active_yaw active_yaw_microtaur 1 yaw

echo "[$(date +%T)] four CPG runs started ($ITERATIONS iterations, $NUM_ENVS envs each)"
for pid in "${pids[@]}"; do wait "$pid"; done

for log in "$STATUS_DIR"/*_cpg.log; do
  last=$(grep -a "Learning iteration" "$log" | tail -1 | tr -s ' ')
  if grep -aq "Traceback" "$log"; then
    echo "[$(date +%T)] CRASH $(basename "$log" .log) |$last"
    grep -a "Error\|Assertion" "$log" | head -3
  else
    echo "[$(date +%T)] END $(basename "$log" .log) |$last"
  fi
done
echo "[$(date +%T)] ALL DONE"
