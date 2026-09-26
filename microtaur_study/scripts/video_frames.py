"""Consecutive frames of an mp4 as one grid image (for checking motion frame by frame).

  python scripts/video_frames.py <video.mp4> <out.png> [--start-s 2.0] [--n 20] [--every 1] [--cols 5]
"""

from __future__ import annotations

import argparse
import math

import imageio.v2 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("out")
ap.add_argument("--start-s", type=float, default=2.0)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--every", type=int, default=1)
ap.add_argument("--cols", type=int, default=5)
args = ap.parse_args()

r = iio.get_reader(args.video)
fps = float(r.get_meta_data().get("fps", 30.0))
first = int(args.start_s * fps)
idx = [first + k * args.every for k in range(args.n)]
frames = [r.get_data(i) for i in idx if i < r.count_frames()]
rows = math.ceil(len(frames) / args.cols)
fig, axs = plt.subplots(rows, args.cols, figsize=(args.cols * 4.0, rows * 2.4))
for i, ax in enumerate(np.ravel(axs)):
  ax.axis("off")
  if i < len(frames):
    ax.imshow(frames[i]); ax.set_title(f"t = {idx[i] / fps:.3f} s", fontsize=8)
fig.suptitle(f"{args.video.split('/')[-1]}: consecutive frames every {args.every / fps * 1e3:.0f} ms")
fig.savefig(args.out, dpi=80, bbox_inches="tight")
print("wrote", args.out)
