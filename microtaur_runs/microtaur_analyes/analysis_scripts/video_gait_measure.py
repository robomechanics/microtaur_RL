"""RETIRED — DO NOT USE FOR GAIT CLASSIFICATION.  Kept only for reference.

This statistic is NOT diagnostic of gait. The silhouette centroid moves with
camera projection and with which legs happen to occlude the body, not with
ground contact. It was written to corroborate a contact-timing result that was
itself wrong (see TODO.md §2), and it duly "confirmed" it -- two measurements
that share no ground truth can agree while both being meaningless.

Classify gait from ground contact instead:
  scripts/gait_from_geometry.py   -- corners from measured trunk-frame foot XY,
                                     stance from the contact sensor AND from
                                     foot height independently.

Original docstring follows.

Measure gait rhythm from a rendered video, with no access to the simulator.

Segments the (bright, desaturated) robot from the (saturated blue) floor, then
tracks the silhouette's horizontal centroid (body sway) and vertical centroid
(bounce) per frame and takes their spectra.

A PACE swings the two same-side legs together, so the body sways left-right once
per stride: sway power concentrates at the stride frequency f.
A TROT swings diagonal pairs, so lateral forces cancel: sway is weak, and the
bounce sits at 2f.

The discriminator reported is  sway_power(f) / bounce_power(2f).

    python scripts/video_gait_measure.py A.mp4 [B.mp4 ...] --start 8 --dur 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def robot_mask(frame):
  """Bright + desaturated = the white robot; the floor is saturated blue."""
  f = frame.astype(np.float32) / 255.0
  mx = f.max(axis=2); mn = f.min(axis=2)
  sat = (mx - mn) / np.maximum(mx, 1e-6)
  return (mx > 0.55) & (sat < 0.18)


def analyse(path, start, dur, fps_hint=50.0, crop_top=0.10):
  import imageio.v2 as imageio

  rd = imageio.get_reader(str(path))
  meta = rd.get_meta_data()
  fps = float(meta.get("fps", fps_hint))
  i0 = int(round(start * fps)); n = int(round(dur * fps))
  cx, cy, area = [], [], []
  for i, frame in enumerate(rd):
    if i < i0:
      continue
    if i >= i0 + n:
      break
    h = frame.shape[0]
    fr = frame[int(h * crop_top):]          # drop any HUD banner
    m = robot_mask(fr)
    if m.sum() < 200:
      cx.append(np.nan); cy.append(np.nan); area.append(0); continue
    ys, xs = np.nonzero(m)
    cx.append(xs.mean()); cy.append(ys.mean()); area.append(m.sum())
  rd.close()
  cx = np.array(cx, float); cy = np.array(cy, float)
  ok = np.isfinite(cx) & np.isfinite(cy)
  cx, cy = cx[ok], cy[ok]
  if len(cx) < 64:
    return None
  cx = cx - np.polyval(np.polyfit(np.arange(len(cx)), cx, 1), np.arange(len(cx)))
  cy = cy - np.polyval(np.polyfit(np.arange(len(cy)), cy, 1), np.arange(len(cy)))
  w = np.hanning(len(cx))
  fx = np.fft.rfft(cx * w); fy = np.fft.rfft(cy * w)
  freqs = np.fft.rfftfreq(len(cx), d=1.0 / fps)
  band = (freqs > 1.0) & (freqs < 8.0)
  # stride frequency = dominant peak of the vertical (bounce) signal / 2,
  # or the dominant sway peak; take the strongest peak across both as a seed.
  py = np.abs(fy) ** 2; px = np.abs(fx) ** 2
  f_bounce = freqs[band][np.argmax(py[band])]
  f_sway = freqs[band][np.argmax(px[band])]

  def power_at(p, f, tol=0.35):
    sel = np.abs(freqs - f) <= tol
    return float(p[sel].sum()) if sel.any() else 0.0

  # assume stride f: the sway peak in a pace IS the stride frequency and the
  # bounce sits at 2x it; in a trot the bounce peak is the stride's 2x too.
  f_stride = f_sway if abs(f_bounce - 2 * f_sway) < 0.8 else f_bounce / 2.0
  ratio = power_at(px, f_stride) / max(power_at(py, 2 * f_stride), 1e-9)
  return dict(path=Path(path).name, fps=fps, n=len(cx),
              f_sway=f_sway, f_bounce=f_bounce, f_stride=f_stride,
              sway_rms=float(cx.std()), bounce_rms=float(cy.std()),
              sway_at_f=power_at(px, f_stride), bounce_at_2f=power_at(py, 2 * f_stride),
              ratio=ratio)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("videos", nargs="+")
  ap.add_argument("--start", type=float, default=8.0)
  ap.add_argument("--dur", type=float, default=12.0)
  args = ap.parse_args()
  rows = []
  for v in args.videos:
    r = analyse(v, args.start, args.dur)
    if r is None:
      print(f"{v}: too few usable frames"); continue
    rows.append(r)
    print(f"{r['path']:44s} fps={r['fps']:.0f} n={r['n']:4d}  "
          f"f_stride={r['f_stride']:.2f}Hz  sway_rms={r['sway_rms']:.2f}px  "
          f"bounce_rms={r['bounce_rms']:.2f}px  sway(f)/bounce(2f)={r['ratio']:.3f}")
  if len(rows) > 1:
    print("\nhigher sway(f)/bounce(2f) => more lateral rocking => more pace-like")


if __name__ == "__main__":
  main()
