#!/bin/bash
# One Microtaur job on PSC Bridges-2 (1 GPU, GPU-shared) in the shared Isaac Lab image. Submit through psc_submit.sh
# (it sets the GPU, CPUs, walltime and job name):
#
#   pipeline <name> <robot> [seed=N] [iters=N] [Hydra overrides...]  teacher + student, pipeline.py --train-only (no cameras):
#                                                 checkpoints land in $MT_RUNS/logs/rsl_rl/microtaur_isaac_teacher/;
#                                                 evaluate them on rml3 with pipeline.py --stage eval
#   preflight                                     versions, a --smoke --train-only run of both stages (yaw, 256 envs),
#                                                 and 30 teacher iterations at 2048 envs for s/iter -> preflight.txt
#
# Kit's caches and /tmp are bound to per-job node-local dirs (two jobs on one node never share Kit's kvdb lock); the
# repo is bound read-only, the runs root read-write. Success is judged from the files written, not from exit codes
# (Kit can exit 0 after a traceback).
#SBATCH --partition=GPU-shared
#SBATCH --output=%x_%j.out
set -uo pipefail
source "${MT_REPO:?export MT_REPO (psc_submit.sh does)}/microtaur_study/cluster/psc_env.sh"
SING="$(command -v apptainer || command -v singularity)"
MODE="${1:?usage: psc_job.sbatch pipeline <name> <robot> [overrides...] | preflight}"; shift
JOB="${LOCAL:-${TMPDIR:-/tmp}}/mt_${SLURM_JOB_ID}"
mkdir -p "$JOB"/{local,cache,data,logs,outputs,tmp,home_cache,home_nv,home_logs} "$MT_RUNS" "$MT_KIT_CACHE" "$MT_PYDEPS"
[ -d "$MT_KIT_CACHE" ] && cp -r "$MT_KIT_CACHE/." "$JOB/cache/" 2>/dev/null
CLEAN=0
on_exit() {
  [ "$CLEAN" = 1 ] && rsync -a "$JOB/cache/" "$MT_KIT_CACHE/" 2>/dev/null   # shader cache only after a clean run
  rm -rf "$JOB"
}
trap on_exit EXIT

run() {  # run <log> <command...> inside the image
  local log=$1; shift
  "$SING" exec --nv \
    --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y --env OMNI_KIT_ACCEPT_EULA=YES --env TMPDIR=/tmp \
    --env ISAACLAB_DIR=/workspace/isaaclab --env PYTHONPATH=/workspace/mt_pydeps \
    --bind "$JOB/local:$HOME/.local,$JOB/cache:/isaac-sim/kit/cache,$JOB/data:/isaac-sim/kit/data" \
    --bind "$JOB/logs:/workspace/isaaclab/logs,$JOB/outputs:/workspace/isaaclab/outputs" \
    --bind "$JOB/home_cache:$HOME/.cache,$JOB/home_nv:$HOME/.nv,$JOB/home_logs:$HOME/.nvidia-omniverse,$JOB/tmp:/tmp" \
    --bind "$MT_REPO:/workspace/microtaur_RL:ro,$MT_RUNS:/workspace/runs,$MT_PYDEPS:/workspace/mt_pydeps:ro" \
    --cwd /workspace/microtaur_RL "$MT_SIF" "$@" > "$log" 2>&1
}
PIPE=(/isaac-sim/python.sh microtaur_study/pipeline.py --runs /workspace/runs --train-only)
echo "[mt] $(date -Is) job $SLURM_JOB_ID on $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) $MODE $*"

if [ "$MODE" = preflight ]; then
  OUT="$MT_RUNS/preflight_${SLURM_JOB_ID}"; mkdir -p "$OUT"
  run "$OUT/versions.log" /isaac-sim/python.sh -c "import importlib.metadata as m, torch; print({p: m.version(p) for p in ('isaaclab', 'rsl-rl-lib', 'torch', 'numpy', 'trimesh')}, torch.cuda.get_device_name(0))"
  run "$OUT/smoke.log" "${PIPE[@]}" --robot yaw --name "pf${SLURM_JOB_ID}_smoke" --smoke
  run "$OUT/timing.log" "${PIPE[@]}" --robot yaw --name "pf${SLURM_JOB_ID}_t30" --stage teacher --iters 30
  L="$MT_RUNS/logs/rsl_rl/microtaur_isaac_teacher"
  {
    echo "versions: $(tail -1 "$OUT/versions.log")"
    s1=$(ls -d "$L"/*_pf${SLURM_JOB_ID}_smoke 2>/dev/null | head -1); s2=$(ls -d "$L"/*_pf${SLURM_JOB_ID}_smoke_student 2>/dev/null | head -1)
    [ -f "$s1/model_2.pt" ] && [ -f "$s2/model_1.pt" ] && echo "smoke: PASS (teacher model_2.pt, student model_1.pt)" \
      || echo "smoke: FAIL, see $OUT/smoke.log and $MT_RUNS/pf${SLURM_JOB_ID}_smoke/"
    t=$(ls -d "$L"/*_pf${SLURM_JOB_ID}_t30 2>/dev/null | head -1)
    if [ -f "$t/model_29.pt" ]; then
      a=$(stat -c %Y "$t/model_0.pt"); b=$(stat -c %Y "$t/model_29.pt")
      echo "timing: 29 iterations in $((b - a)) s -> $(awk -v d=$((b - a)) 'BEGIN { printf "%.2f", d / 29 }') s/iter (yaw, 2048 envs)"
    else echo "timing: FAIL, see $OUT/timing.log and $MT_RUNS/pf${SLURM_JOB_ID}_t30/"; fi
  } > "$OUT/preflight.txt"
  cat "$OUT/preflight.txt"
  grep -q "smoke: PASS" "$OUT/preflight.txt" && grep -q "s/iter" "$OUT/preflight.txt" && { CLEAN=1; exit 0; }
  exit 1   # non-zero: jobs submitted with --dependency=afterok:<this job> are cancelled
fi

[ "$MODE" = pipeline ] || { echo "[mt] unknown mode $MODE"; exit 2; }
NAME="${1:?name}"; ROBOT="${2:?robot}"; shift 2
mkdir -p "$MT_RUNS/$NAME"
{ echo "job=$SLURM_JOB_ID host=$(hostname) start=$(date -Is) name=$NAME robot=$ROBOT"
  printf 'override: %s\n' "$@"
  echo "repo=$(git -C "$MT_REPO" --no-pager log --oneline -1)"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; } > "$MT_RUNS/$NAME/job_${SLURM_JOB_ID}.txt"
ov=(); extra=()   # seed=N -> --seed N; iters=N -> --iters N (teacher iterations); everything else is a Hydra override
for x in "$@"; do case "$x" in seed=*) extra+=(--seed "${x#seed=}") ;; iters=*) extra+=(--iters "${x#iters=}") ;; *) ov+=("$x") ;; esac; done
[ ${#ov[@]} -gt 0 ] && ov=(--override "${ov[@]}")
run "$MT_RUNS/$NAME/pipeline_${SLURM_JOB_ID}.out" "${PIPE[@]}" --robot "$ROBOT" --name "$NAME" "${extra[@]}" "${ov[@]}"
if grep -q "DONE:" "$MT_RUNS/$NAME/progress.md" 2>/dev/null; then CLEAN=1; echo "[mt] $(date -Is) $NAME done"
else echo "[mt] $(date -Is) $NAME FAILED: tail of $MT_RUNS/$NAME/progress.md"; tail -3 "$MT_RUNS/$NAME/progress.md"; exit 1; fi
