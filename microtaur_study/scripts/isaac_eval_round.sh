#!/usr/bin/env bash
# One tuning-loop evaluation of a checkpoint: gait statistics, pass/fail, MuJoCo
# replays, and an Isaac Play video, all into one folder.
#
#   bash scripts/isaac_eval_round.sh <checkpoint.pt> <out dir>
#
# out dir gets: eval.log summary.json criteria.json criteria.txt gait_*.png
# traj_*.npz traj_*.mp4 traj_*_frames.png play.mp4 (Isaac GUI view, visual USD).
set -euo pipefail
CK=$(realpath "$1")
OUT=$(realpath -m "$2")
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=/home/rml3/anaconda3/envs/spine/bin/python
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p "$OUT"
cd "$HERE"

echo "== GPU"; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | while IFS=, read -r p m; do
  echo "  $p $m $(ps -o user=,cmd= -p "$p" | cut -c1-80)"; done

echo "== gait statistics"
$PY scripts/isaac_eval_gait.py --checkpoint "$CK" --out "$OUT" > "$OUT/eval_gait.stdout" 2>&1 \
  || { echo "isaac_eval_gait.py failed:"; tail -30 "$OUT/eval_gait.stdout"; exit 1; }
$PY scripts/isaac_eval_criteria.py "$OUT" | tee "$OUT/criteria.txt"

echo "== MuJoCo replays"
for T in "$OUT"/traj_0.20.npz "$OUT"/traj_0.35.npz "$OUT"/traj_0.20_w+0.25.npz "$OUT"/traj_0.20_w-0.25.npz; do
  [ -f "$T" ] && DISPLAY=${DISPLAY:-:1} MUJOCO_GL=glfw $PY scripts/isaac_render_traj.py --traj "$T" | tail -1
done

echo "== Isaac Play video"
PLAYDIR="$OUT/play_ckpt"; mkdir -p "$PLAYDIR"; cp "$CK" "$PLAYDIR/"
(cd "$OUT" && $PY "$HERE/scripts/isaac_play.py" --task Microtaur-Isaac-Flat-Play-v0 \
  --checkpoint "$PLAYDIR/$(basename "$CK")" --headless --video --video_length 300 > "$OUT/play.stdout" 2>&1) \
  || { echo "play failed:"; tail -30 "$OUT/play.stdout"; }
V=$(ls "$PLAYDIR"/videos/play/*.mp4 2>/dev/null | head -1 || true)
[ -n "$V" ] && cp "$V" "$OUT/play.mp4" && echo "wrote $OUT/play.mp4"
rm -f "$PLAYDIR/$(basename "$CK")"
echo "== done: $OUT"
