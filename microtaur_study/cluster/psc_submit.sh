#!/bin/bash
# Submit Microtaur jobs on PSC Bridges-2 (login node). Prints the sbatch lines and the SU bound; submits only with
# CONFIRM_SU=yes. The allocation is the user's default account (never printed).
#
#   bash microtaur_study/cluster/psc_submit.sh preflight                      # ~30 min, <= 1 SU
#   bash microtaur_study/cluster/psc_submit.sh plan <plan.txt> [-t 06:00:00]  # one job per plan line
#   CONFIRM_SU=yes bash microtaur_study/cluster/psc_submit.sh ...
#   MT_DEPEND=afterok:<preflight job id> ... plan <plan.txt>                 # run only if the preflight passes
#
# Plan file: one run per line, "<name> <yaw|rigid> [seed=N] [Hydra override ...]" (overrides without spaces);
# "= <override ...>" lines set overrides added to every later run line; # comments. A run whose $MT_RUNS/<name>
# exists is skipped (no double submission).
set -euo pipefail
export MT_REPO="${MT_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$MT_REPO/microtaur_study/cluster/psc_env.sh"
MODE="${1:?usage: psc_submit.sh preflight | plan <plan.txt> [-t HH:MM:SS]}"; shift
WALL=06:00:00
if [ "$MODE" = plan ]; then
  PLAN="$(readlink -f "${1:?plan file}")"; shift
  [ "${1:-}" = -t ] && { WALL=$2; shift 2; }
fi
acct="$(sacctmgr -nP show user "$USER" format=DefaultAccount 2>/dev/null | head -1 || true)"
A=(); [ -n "$acct" ] && A=(-A "$acct")
RES=(--gpus="$MT_GPU:1" --cpus-per-task="$MT_CPUS")
SB=(sbatch --parsable --export=ALL,MT_REPO="$MT_REPO" "${RES[@]}")
[ -n "${MT_DEPEND:-}" ] && SB+=(--dependency="$MT_DEPEND" --kill-on-invalid-dep=yes)
mkdir -p "$MT_RUNS/slurm"
cd "$MT_PROJECT"
submit() {  # submit <sbatch args...>
  if [ "${CONFIRM_SU:-}" = yes ]; then
    id=$("${SB[@]}" "${A[@]}" -o "$MT_RUNS/slurm/%x_%j.out" "$@")
    echo "submitted $id: $*"
  else
    echo "dry-run: sbatch ${RES[*]} ${MT_DEPEND:+--dependency=$MT_DEPEND} ${A:+-A <default account>} $*"
  fi
}
su_per_h=1; case "$MT_GPU" in h100*) su_per_h=2 ;; esac
if [ "$MODE" = preflight ]; then
  submit -J mt_preflight -t 01:00:00 "$MT_REPO/microtaur_study/cluster/psc_job.sbatch" preflight
  echo "SU bound: 1 h x $su_per_h = $su_per_h SU (typically ~0.5)"
  exit 0
fi
[ "$MODE" = plan ] || { echo "unknown mode $MODE"; exit 2; }
h=$(awk -F: '{ printf "%.2f", $1 + $2 / 60 + $3 / 3600 }' <<< "$WALL")
n=0
common=""
set -f   # overrides hold [ ] { }: no globbing when they are split into words
while read -r name robot overrides; do
  case "$name" in ""|\#*) continue ;; "=") common="$robot $overrides"; continue ;; esac
  [ -d "$MT_RUNS/$name" ] && { echo "skip $name: $MT_RUNS/$name exists"; continue; }
  # shellcheck disable=SC2086  # overrides are whitespace-separated Hydra args without spaces
  submit -J "mt_$name" -t "$WALL" "$MT_REPO/microtaur_study/cluster/psc_job.sbatch" pipeline "$name" "$robot" $common $overrides
  n=$((n + 1))
done < "$PLAN"
echo "$n jobs x <= $WALL x $su_per_h SU/h = <= $(awk -v n=$n -v h=$h -v s=$su_per_h 'BEGIN { printf "%.0f", n * h * s }') SU (charged by the hours actually used)"
