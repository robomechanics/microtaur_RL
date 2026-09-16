"""Tile rollout videos into one grid: one row per robot, one column per controller, all synced from t = 0.

  python grid_video.py results/00_all_robots_rl_vs_cpg.mp4 \
      --columns "RL (neural network)" "Final CPG (sine waves only)" "Old CPG" \
      --row "Pitch=pitch_match/video_rl_d1.mp4,pitch_match/video_cpg_final.mp4,pitch_match/video_cpg_old.mp4" \
      --row "Yaw=yaw_match/video_rl_d1.mp4,yaw_match/video_cpg_final.mp4,yaw_match/video_cpg_old.mp4"

Videos come from openloop_spine_cpg.py --video (same camera, 50 fps). Each cell is scaled to 640x360; the output
is as long as the shortest input.
"""

import argparse

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CELL = (640, 360)
HEADER = 48
LIGHT, DARK = (252, 252, 251), (11, 11, 11)


def load_font(size: int):
  try:
    return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
  except OSError:
    return ImageFont.load_default()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("out")
  parser.add_argument("--columns", nargs="+", required=True, help="column titles, left to right")
  parser.add_argument("--row", action="append", required=True, help="label=video,video,... (one video per column)")
  args = parser.parse_args()

  rows = []
  for spec in args.row:
    label, paths = spec.split("=", 1)
    readers = [imageio.get_reader(path) for path in paths.split(",")]
    if len(readers) != len(args.columns):
      raise SystemExit(f"row {label}: {len(readers)} videos for {len(args.columns)} columns")
    rows.append((label, readers))
  fps = rows[0][1][0].get_meta_data().get("fps", 50)
  width = CELL[0] * len(args.columns)

  header = Image.new("RGB", (width, HEADER), LIGHT)
  draw = ImageDraw.Draw(header)
  title_font, label_font = load_font(24), load_font(26)
  for c, title in enumerate(args.columns):
    box = draw.textbbox((0, 0), title, font=title_font)
    x = c * CELL[0] + (CELL[0] - (box[2] - box[0])) // 2
    draw.text((x, (HEADER - (box[3] - box[1])) // 2 - box[1]), title, font=title_font, fill=DARK)
  header = np.asarray(header)

  writer = imageio.get_writer(args.out, fps=fps, codec="libx264", quality=7, macro_block_size=16)
  frames = [[iter(reader) for reader in readers] for _, readers in rows]
  count = 0
  while True:
    try:
      tiles = [header]
      for (label, _), streams in zip(rows, frames):
        row = Image.fromarray(np.hstack([np.asarray(Image.fromarray(next(s)).resize(CELL)) for s in streams]))
        draw = ImageDraw.Draw(row)
        for c in range(1, len(args.columns)):  # thin separators between columns
          draw.rectangle((c * CELL[0] - 1, 0, c * CELL[0] + 1, CELL[1]), fill=LIGHT)
        draw.rectangle((0, 0, width, 1), fill=LIGHT)  # separator above each row
        box = draw.textbbox((14, 12), label, font=label_font)
        draw.rectangle((box[0] - 8, box[1] - 6, box[2] + 8, box[3] + 6), fill=DARK)
        draw.text((14, 12), label, font=label_font, fill=LIGHT)
        tiles.append(np.asarray(row))
      writer.append_data(np.vstack(tiles))
      count += 1
    except StopIteration:
      break
  writer.close()
  print(f"[grid] wrote {args.out}: {count} frames, {len(rows)} rows x {len(args.columns)} columns, "
        f"{width}x{HEADER + CELL[1] * len(rows)} px")


if __name__ == "__main__":
  main()
