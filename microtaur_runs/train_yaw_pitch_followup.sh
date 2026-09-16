#!/usr/bin/env bash
# Follow-up rough-terrain runs, one at a time (a single run saturates the GPU when the
# CPU is idle, so running two in parallel gives no extra throughput).
#
#   ./train_yaw_pitch_followup.sh with_flag
#       active_yaw then active_pitch from scratch with --agent.clip-actions 1.0
#       (run names *_rough_with_flag)
#   ./train_yaw_pitch_followup.sh rigid_roll_with_flag
#       rigid then active_twist (active roll) from scratch with --agent.clip-actions 1.0
#   ./train_yaw_pitch_followup.sh from_flat
#       all four variants warm-started from their flat-terrain model_4499.pt with
#       --agent.clip-actions 1.0 (run names *_rough_from_flat). The warm-start
#       checkpoints (<variant>/logs/.../*_flat_warmstart/model_0.pt) are built by
#       make_flat_warmstart.py: height-scan inputs get zero weights, iter resets to 0.
#   ./train_yaw_pitch_followup.sh yaw_as_is <350|1400>
#       resume active_yaw without action clipping from that checkpoint to 4500
#
# If another instance is already training, this one waits for it to finish first.
# A crash is logged and the queue moves on to the next run.
#
# Launch:   nohup ./train_yaw_pitch_followup.sh with_flag > /dev/null 2>&1 &
# Progress: tail -f followup_logs_<stamp>/status.txt
#
# Resuming: MJLab >= 1.6 stores common_step_counter in each checkpoint and restores
# it, so MICROTAUR_CURRICULUM_START_STEP must stay 0 (a nonzero value double counts).

set -uo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
TRAIN_BIN=/home/naomio/anaconda3/envs/microtaur/bin/train
TOTAL_ITERATIONS=4500

declare -A YAW_RESUME_RUN=(
  [350]=2026-09-12_23-31-07_active_yaw_rough
  [1400]=2026-09-13_03-45-20_active_yaw_rough_continued
)

STATUS_DIR=${STATUS_DIR:-$ROOT/followup_logs_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$STATUS_DIR"
exec > >(tee -a "$STATUS_DIR/status.txt") 2>&1

while pgrep -f "bin/train Mjlab-Velocity-Rough" > /dev/null; do
  echo "[$(date +%T)] another training run is active, waiting"
  sleep 60
done

run() {  # <variant folder> <MICROTAUR_VARIANT> <MICROTAUR_CONTROL_SPINE> <run name> <train args...>
  local variant=$1 model=$2 control_spine=$3 run_name=$4
  shift 4
  local log=$STATUS_DIR/$run_name.log
  echo "[$(date +%T)] START $run_name (log: $log)"
  (
    cd "$ROOT/$variant" && exec env \
      PYTHONPATH="$ROOT/$variant/src" \
      MICROTAUR_VARIANT="$model" \
      MICROTAUR_CONTROL_SPINE="$control_spine" \
      MICROTAUR_CURRICULUM_START_STEP=0 \
      CUDA_VISIBLE_DEVICES=0 \
      "$TRAIN_BIN" Mjlab-Velocity-Rough-microtaur_velocity \
        --env.scene.num-envs 2048 \
        --env.sim.njmax 256 \
        --env.sim.nconmax 64 \
        --agent.logger tensorboard \
        --agent.run-name "$run_name" \
        "$@"
  ) > "$log" 2>&1
  local status=$?
  local last
  last=$(grep -a "Learning iteration" "$log" | tail -1 | tr -s ' ')
  if [ $status -ne 0 ] || grep -aq "Traceback" "$log"; then
    echo "[$(date +%T)] CRASH $run_name exit=$status |$last"
    grep -a "Assertion\|Error" "$log" | grep -v "Warp CUDA error" | head -3
  else
    echo "[$(date +%T)] END $run_name |$last"
  fi
}

case "${1:-}" in
  with_flag)
    run active_yaw active_yaw_microtaur 1 active_yaw_rough_with_flag \
      --agent.max-iterations "$TOTAL_ITERATIONS" --agent.clip-actions 1.0
    run active_pitch active_pitch_microtaur 1 active_pitch_rough_with_flag \
      --agent.max-iterations "$TOTAL_ITERATIONS" --agent.clip-actions 1.0
    ;;
  rigid_roll_with_flag)
    run rigid rigid_microtaur 0 rigid_rough_with_flag \
      --agent.max-iterations "$TOTAL_ITERATIONS" --agent.clip-actions 1.0
    run active_twist active_twist_microtaur 1 active_twist_rough_with_flag \
      --agent.max-iterations "$TOTAL_ITERATIONS" --agent.clip-actions 1.0
    ;;
  from_flat)
    for spec in rigid:rigid_microtaur:0:2026-09-06_12-33-30 \
                active_twist:active_twist_microtaur:1:2026-09-05_18-26-09 \
                active_pitch:active_pitch_microtaur:1:2026-09-04_11-52-40 \
                active_yaw:active_yaw_microtaur:1:2026-09-04_13-55-36; do
      IFS=: read -r variant model control_spine stamp <<< "$spec"
      run "$variant" "$model" "$control_spine" "${variant}_rough_from_flat" \
        --agent.max-iterations "$TOTAL_ITERATIONS" --agent.clip-actions 1.0 \
        --agent.resume True \
        --agent.load-run "${stamp}_${variant}_flat_warmstart" \
        --agent.load-checkpoint model_0.pt
    done
    ;;
  yaw_as_is)
    iteration=${2:?usage: $0 yaw_as_is <350|1400>}
    load_run=${YAW_RESUME_RUN[$iteration]:?no resume run for iteration $iteration}
    run active_yaw active_yaw_microtaur 1 "active_yaw_rough_resume_${iteration}" \
      --agent.max-iterations $((TOTAL_ITERATIONS - iteration)) \
      --agent.resume True \
      --agent.load-run "$load_run" \
      --agent.load-checkpoint "model_${iteration}.pt"
    ;;
  *)
    echo "usage: $0 with_flag | rigid_roll_with_flag | from_flat | yaw_as_is <350|1400>"
    exit 2
    ;;
esac

echo "[$(date +%T)] ALL DONE ($1)"
