#!/usr/bin/env bash
# Build / refresh the `train-kit` branch: only what training needs (pipeline, IsaacLab task, assets, the scripts the
# pipeline calls, the deployment template), with TRAIN_KIT_README.md as its README. Run from anywhere in the repo:
#
#   bash microtaur_study/scripts/make_train_kit.sh          # commit on train-kit (history kept), no push
#   git push origin train-kit
#
# The branch has no common history with the development branches (orphan), so `git clone -b train-kit
# --single-branch` downloads only the kit.
set -euo pipefail
REPO=$(git rev-parse --show-toplevel)
SRC_SHA=$(git -C "$REPO" rev-parse --short HEAD)
SRC_BRANCH=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)
WT=$(mktemp -d)/train-kit
MS=microtaur_study
FILES=(
  "$MS/pipeline.py"
  "$MS/scripts/isaac_train.py"
  "$MS/scripts/isaac_gait_sweep.py"
  "$MS/scripts/isaac_eval_terrain.py"
  "$MS/scripts/isaac_c_posture.py"
  "$MS/scripts/isaac_turn_legs.py"
  "$MS/scripts/isaac_render_policy.py"
  "$MS/scripts/isaac_student_teacher_c.py"
  "$MS/scripts/isaac_torque_saturation.py"
  "$MS/scripts/export_student_bundle.py"
  "$MS/scripts/make_train_kit.sh"
  "$MS/tests/check_isaac_yaw_env.py"
)
DIRS=("$MS/src/microtaur_common" "$MS/src/microtaur_isaac" "$MS/assets/rigid_microtaur" "$MS/assets/active_yaw_microtaur" "$MS/deploy/unoq")

if git -C "$REPO" show-ref --verify --quiet refs/heads/train-kit; then
  git -C "$REPO" worktree add -q "$WT" train-kit
else
  git -C "$REPO" worktree add -q --detach "$WT"
  git -C "$WT" checkout -q --orphan train-kit
fi
trap 'git -C "$REPO" worktree remove --force "$WT"' EXIT
git -C "$WT" read-tree --empty  # start from an empty index; only the kit files are added back
find "$WT" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +

# committed content of HEAD only (no uncommitted edits, caches or local outputs)
mapfile -t TRACKED < <(git -C "$REPO" ls-tree -r --name-only HEAD -- "${FILES[@]}" "${DIRS[@]}")
for f in "${TRACKED[@]}"; do
  mkdir -p "$WT/$(dirname "$f")"
  git -C "$REPO" cat-file blob "HEAD:$f" > "$WT/$f"
done
chmod +x "$WT/$MS/scripts/make_train_kit.sh"
git -C "$REPO" cat-file blob "HEAD:$MS/TRAIN_KIT_README.md" > "$WT/README.md"
cat > "$WT/.gitignore" <<'EOF'
runs/
__pycache__/
*.pyc
*.pt
*.onnx
*.npz
EOF
git -C "$WT" add -- README.md .gitignore "${TRACKED[@]}"
if git -C "$WT" diff --cached --quiet; then
  echo "train-kit: no change"
else
  git -C "$WT" commit -q -m "train-kit from $SRC_BRANCH $SRC_SHA"
  echo "train-kit: $(git -C "$WT" log --oneline -1)  ($(git -C "$WT" ls-files | wc -l) files)"
fi
