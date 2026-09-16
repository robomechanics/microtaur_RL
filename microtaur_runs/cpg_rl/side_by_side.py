"""Put rollout videos next to each other, each with a label, frame-synced from t = 0.

  python side_by_side.py pitch_match/side_by_side.mp4 "RL=pitch_match/video_rl_d1.mp4" "CPG=pitch_match/video_cpg_d1_k3.mp4"

Videos come from openloop_spine_cpg.py --video (same camera, 50 fps); each panel is scaled to
640x360 and the output is as long as the shortest input.
"""

import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PANEL = (640, 360)


def main() -> None:
  out = sys.argv[1]
  labels, readers = [], []
  for arg in sys.argv[2:]:
    label, path = arg.split("=", 1)
    labels.append(label)
    readers.append(imageio.get_reader(path))
  fps = readers[0].get_meta_data().get("fps", 50)
  try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
  except OSError:
    font = ImageFont.load_default()
  writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=7, macro_block_size=8)
  frames = 0
  for group in zip(*readers):
    panels = []
    for label, frame in zip(labels, group):
      image = Image.fromarray(frame).resize(PANEL)
      draw = ImageDraw.Draw(image)
      box = draw.textbbox((12, 10), label, font=font)
      draw.rectangle((box[0] - 6, box[1] - 4, box[2] + 6, box[3] + 4), fill=(252, 252, 251))
      draw.text((12, 10), label, font=font, fill=(11, 11, 11))
      panels.append(np.asarray(image))
    writer.append_data(np.hstack(panels))
    frames += 1
  writer.close()
  print(f"[side-by-side] wrote {out}: {frames} frames, {len(labels)} panels")


if __name__ == "__main__":
  main()
