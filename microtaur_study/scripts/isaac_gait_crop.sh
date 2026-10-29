#!/usr/bin/env bash
# isaac_gait_crop.sh <label> <checkpoint> [out_dir]: flat (A level 0) side-view video of the policy and a crop of
# 12 consecutive policy steps of the legs (<out_dir>/<label>_flat_side_crop.png), for checking the gait type by eye
# (trot: near front and rear legs in opposite phase; pronk / pace: same shape in every frame).
set -u
L=$1; CK=$2
O=${3:-/home/rml3/Documents/ben/spine/figures/isaac_play/pronk_diag}
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=/home/rml3/anaconda3/envs/spine/bin/python
export OMNI_KIT_ACCEPT_EULA=YES
mkdir -p "$O"
cd "$HERE"
timeout 900 $PY scripts/isaac_render_policy.py --checkpoint "$CK" --env 0 --steps 150 --out "$O/${L}_flat_side.mp4" \
  --eye 0.05 -0.40 0.10 --lookat 0.05 0 0.03 > "$O/${L}_flat_side.log" 2>&1
grep -E "VIDEO_OK|Traceback" "$O/${L}_flat_side.log" | head -1
[ -f "$O/${L}_flat_side.mp4" ] || exit 1
$PY - "$O/${L}_flat_side.mp4" "$O/${L}_flat_side_crop.png" <<'EOF'
import sys, imageio, numpy as np
from PIL import Image, ImageDraw
fr = [f for f in imageio.get_reader(sys.argv[1])]; H, W = fr[0].shape[:2]
tiles = []
for k in range(57, 69):
  im = Image.fromarray(fr[k][int(H * .35):int(H * .95), int(W * .08):int(W * .92)]); ImageDraw.Draw(im).text((8, 8), f"step {k}", fill=(255, 0, 0))
  tiles.append(np.asarray(im))
Image.fromarray(np.concatenate([np.concatenate(tiles[j:j + 4], 1) for j in range(0, 12, 4)], 0)).resize((1800, 690)).save(sys.argv[2])
print("CROP_OK", sys.argv[2])
EOF
