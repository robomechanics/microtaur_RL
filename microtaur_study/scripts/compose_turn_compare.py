"""Side-by-side turn video: two renders of the same command schedule (e.g. spine free / spine locked) on top, and
below them the paths against the ideal path of the commands, the yaw rates and the spine angles, with a moving cursor.
Colours follow the two header bars (first video blue, second orange); the ideal path and the command are dashed grey.

  python scripts/compose_turn_compare.py --videos free.mp4 locked.mp4 --labels "spine free" "spine locked at 0" \
      --title "y3 teacher, S tour" --out compare.mp4 [--paths-out paths.mp4]

Each video needs the trace written next to it by isaac_render_policy.py --trace (<video>.npz). Paths are drawn in
each robot's start frame (start at 0, heading +x). The ideal path integrates the commanded forward speed and yaw
rate from the same start. --paths-out also writes the path panel alone as its own video.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

W, TOP_H, BOTTOM_H, HEADER_H = 1280, 360, 376, 32  # total height 768 (a multiple of 16 for the encoder)
INK, MUTED, IDEAL = "#222222", "#666666", "#555555"
COLORS = ("#2a78d6", "#eb6834")  # categorical slots 1, 2: first and second video


def start_frame(d: dict) -> tuple[np.ndarray, np.ndarray]:
  """Root path rotated / shifted so that it starts at 0 heading +x."""
  c, s = np.cos(-d["yaw"][0]), np.sin(-d["yaw"][0])
  dx, dy = d["x"] - d["x"][0], d["y"] - d["y"][0]
  return c * dx - s * dy, s * dx + c * dy


def ideal_path(d: dict) -> tuple[np.ndarray, np.ndarray]:
  """Integrate the commanded forward speed and yaw rate (body frame) from 0, heading +x."""
  dt = float(d["t"][1] - d["t"][0])
  th = np.concatenate([[0.0], np.cumsum(d["cmd_wz"][:-1] * dt)])
  x = np.concatenate([[0.0], np.cumsum(d["cmd_vx"][:-1] * np.cos(th[:-1] + 0.5 * d["cmd_wz"][:-1] * dt) * dt)])
  y = np.concatenate([[0.0], np.cumsum(d["cmd_vx"][:-1] * np.sin(th[:-1] + 0.5 * d["cmd_wz"][:-1] * dt) * dt)])
  return x, y


def panel(traces: list[dict], labels: list[str]):
  """Static bottom panel and the pixel-coordinate functions of its three axes."""
  dpi = 100
  fig = plt.figure(figsize=(W / dpi, BOTTOM_H / dpi), dpi=dpi)
  gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], hspace=0.55, wspace=0.18,
                        left=0.06, right=0.98, top=0.9, bottom=0.13)
  ax_p, ax_w, ax_s = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
  t = traces[0]["t"]
  ix, iy = ideal_path(traces[0])
  peak = 0.0  # largest 1 s mean yaw rate
  ax_p.plot(ix, iy, color=IDEAL, lw=1.6, ls="--", label="ideal (commands integrated)")
  for d, lab, col in zip(traces, labels, COLORS):
    ax_p.plot(*start_frame(d), color=col, lw=1.0, alpha=0.25)
    ax_p.plot([], [], color=col, lw=2.2, label=lab)
    k = max(1, round(1.0 / float(d["t"][1] - d["t"][0])))  # 1 s moving mean over the stride-rate wobble
    ax_w.plot(d["t"], d["wz"], color=col, lw=0.5, alpha=0.2)
    smooth = np.convolve(d["wz"], np.ones(k) / k, mode="same")
    peak = max(peak, float(np.abs(smooth).max()))
    ax_w.plot(d["t"], smooth, color=col, lw=1.6)
    ax_s.plot(d["t"], np.degrees(d["spine"]), color=col, lw=1.0)
  ax_w.plot(t, traces[0]["cmd_wz"], color=IDEAL, lw=1.2, ls="--", label="command")
  lim = max(1.6 * max(0.1, float(np.abs(traces[0]["cmd_wz"]).max())), 1.25 * peak)
  ax_w.set_ylim(-lim, lim)  # per-step spikes are clipped; the 1 s means stay inside
  ax_p.plot(0, 0, "o", color=INK, ms=4)
  ax_p.set_aspect("equal", adjustable="datalim")
  ax_p.set_title("path from the start (start frame: heading +x)", fontsize=9, color=INK)
  ax_p.set_xlabel("x (m)", fontsize=8)
  ax_p.set_ylabel("y (m)", fontsize=8)
  ax_p.legend(fontsize=7, loc="upper center", frameon=False)
  ax_w.set_title("yaw rate (rad/s): 1 s mean, per step faint", fontsize=9, color=INK)
  ax_w.legend(fontsize=7, loc="upper right", frameon=False)
  ax_s.set_title("spine angle (deg, + = bent to the left)", fontsize=9, color=INK)
  ax_s.set_xlabel("t (s)", fontsize=8)
  for a in (ax_p, ax_w, ax_s):
    a.tick_params(labelsize=7, colors=MUTED)
    a.grid(color="#e6e6e6", lw=0.6)
    for side in ("top", "right"):
      a.spines[side].set_visible(False)
  for a in (ax_w, ax_s):
    a.axhline(0, color=MUTED, lw=0.5)
    a.set_xlim(t[0], t[-1])
  fig.canvas.draw()
  bg = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()

  def to_px(ax):
    def f(x, y):
      p = ax.transData.transform(np.column_stack([np.atleast_1d(x), np.atleast_1d(y)]))
      return np.column_stack([p[:, 0], BOTTOM_H - p[:, 1]])
    return f

  ylim_w, ylim_s = ax_w.get_ylim(), ax_s.get_ylim()
  plt.close(fig)
  return bg, to_px(ax_p), to_px(ax_w), to_px(ax_s), ylim_w, ylim_s


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--videos", nargs=2, required=True)
  ap.add_argument("--labels", nargs=2, required=True)
  ap.add_argument("--title", default="")
  ap.add_argument("--out", required=True)
  ap.add_argument("--paths-out", default=None)
  a = ap.parse_args()
  traces = [dict(np.load(Path(v).with_suffix(".npz"))) for v in a.videos]
  readers = [imageio.get_reader(v) for v in a.videos]
  n = min([r.count_frames() for r in readers] + [len(tr["t"]) for tr in traces])
  dt = float(traces[0]["t"][1] - traces[0]["t"][0])
  fps = round(1.0 / dt)
  writer = imageio.get_writer(a.out, fps=fps, quality=8)
  writer_paths = imageio.get_writer(a.paths_out, fps=fps, quality=8) if a.paths_out else None
  bg, px_p, px_w, px_s, ylim_w, ylim_s = panel(traces, a.labels)
  paths = [np.column_stack(start_frame(d)) for d in traces]
  ideal = np.column_stack(ideal_path(traces[0]))
  font = ImageFont.load_default(size=18)
  small = ImageFont.load_default(size=14)
  for k, pair in enumerate(zip(*readers)):
    if k == n:
      break
    im = Image.new("RGB", (W, HEADER_H + TOP_H + BOTTOM_H), "white")
    for i, f in enumerate(pair):
      im.paste(Image.fromarray(f).resize((W // 2, TOP_H)), (i * W // 2, HEADER_H))
    dr = ImageDraw.Draw(im)
    for i, lab in enumerate(a.labels):
      dr.rectangle((i * W // 2, 0, (i + 1) * W // 2 - 1, HEADER_H - 1), fill=COLORS[i])
      dr.text((i * W // 2 + 10, 6), lab, fill="white", font=font)
    if a.title:
      tw = dr.textlength(a.title, font=small)
      dr.text((W - tw - 10, HEADER_H + TOP_H - 22), a.title, fill="white", font=small)
    bottom = Image.fromarray(bg)
    db = ImageDraw.Draw(bottom)
    db.ellipse(_dot(px_p(*ideal[k])[0], 4), fill=IDEAL)
    for p, col in zip(paths, COLORS):
      pts = px_p(p[:k + 1, 0], p[:k + 1, 1])
      if len(pts) > 1:
        db.line([tuple(q) for q in pts], fill=col, width=3)
      db.ellipse(_dot(pts[-1], 5), fill=col, outline="white")
    tk = k * dt
    for px, ylim in ((px_w, ylim_w), (px_s, ylim_s)):
      x0, y0 = px(tk, ylim[0])[0]
      _, y1 = px(tk, ylim[1])[0]
      db.line([(x0, y0), (x0, y1)], fill=INK, width=1)
    im.paste(bottom, (0, HEADER_H + TOP_H))
    writer.append_data(np.asarray(im))
    if writer_paths:
      writer_paths.append_data(np.asarray(bottom.resize((W, BOTTOM_H + 8))))  # 384 = a multiple of 16
  writer.close()
  if writer_paths:
    writer_paths.close()
  print("COMPOSE_OK", a.out, f"{n} frames at {fps} fps", flush=True)


def _dot(p, r: float) -> tuple[float, float, float, float]:
  return (p[0] - r, p[1] - r, p[0] + r, p[1] + r)


if __name__ == "__main__":
  main()
