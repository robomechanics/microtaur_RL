"""Recreate the microtaur `curb` and `steps` terrains as solid, printable STLs.

Geometry mirrors src/microtaur_velocity/microtaur_terrains.py (CurbTerrainCfg,
RandomStepsTerrainCfg) exactly at the feature scale -- curb height, tile size,
number of discrete levels -- but the sim's full course is 4.0 x 1.2 m, which
doesn't fit a desktop printer bed. `--length` / `--width` crop that patch to
something printable; everything else (curb_height, tile_size, num_levels,
max_height) defaults to the sim's own numbers, so the print is a faithful
section of the simulated ground, not a rescaled model of it.

Pure Python + numpy (no bpy / Blender, no external STL library needed):

    python scripts/export_print_terrains.py --terrain curb  --out curb.stl
    python scripts/export_print_terrains.py --terrain steps --out steps.stl --seed 0

Everything is authored directly in millimetres (1 STL unit = 1 mm, which is
what every slicer and MeshLab assume by default), and each output is checked
for closed-manifold, positive-volume geometry before being written, so it
should open in MeshLab with 0 non-manifold edges / 0 holes.
"""

import argparse
import random
import struct
import sys

import numpy as np

# ---- sim defaults, in mm (see microtaur_terrains.py) ----------------------
CURB_HEIGHT_MM = 15.0
TILE_SIZE_MM = 80.0
MAX_STEP_HEIGHT_MM = 10.0
NUM_LEVELS = 5
FLAT_START_MM = 0.0        # sim default 350 mm; 0 here so a small print isn't all flat
BASE_THICKNESS_MM = 3.0    # solid backing slab under the feature, for print adhesion

# 12 triangles per box, outward-facing, consistent CCW winding when viewed
# from outside (required for a valid, non-inverted STL solid).
_BOX_FACES = [
    (0, 3, 2, 1), (4, 5, 6, 7),  # -z, +z
    (0, 1, 5, 4), (1, 2, 6, 5),  # -y, +x
    (2, 3, 7, 6), (3, 0, 4, 7),  # +y, -x
]


def box_triangles(center, half):
    cx, cy, cz = center
    hx, hy, hz = half
    signs = [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
             (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    verts = [(cx + sx * hx, cy + sy * hy, cz + sz * hz) for sx, sy, sz in signs]
    tris = []
    for a, b, c, d in _BOX_FACES:
        tris.append((verts[a], verts[b], verts[c]))
        tris.append((verts[a], verts[c], verts[d]))
    return tris


def write_binary_stl(triangles, path):
    with open(path, "wb") as f:
        f.write(b"\0" * 80)
        f.write(struct.pack("<I", len(triangles)))
        for v0, v1, v2 in triangles:
            n = np.cross(np.subtract(v1, v0), np.subtract(v2, v0))
            norm = np.linalg.norm(n)
            n = n / norm if norm > 0 else n
            f.write(struct.pack("<3f", *n))
            for v in (v0, v1, v2):
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))


def _check_one_solid(triangles):
    """Closed-2-manifold + positive-volume check for a single solid piece
    (every directed edge must appear once, its reverse exactly once)."""
    edges = {}
    vol = 0.0
    for v0, v1, v2 in triangles:
        v0a, v1a, v2a = np.array(v0), np.array(v1), np.array(v2)
        vol += np.dot(v0a, np.cross(v1a, v2a)) / 6.0
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            key = (tuple(round(c, 4) for c in a), tuple(round(c, 4) for c in b))
            edges[key] = edges.get(key, 0) + 1
    dup_fwd = {k: v for k, v in edges.items() if v != 1}
    unmatched = sum(1 for k in edges if (k[1], k[0]) not in edges)
    return vol, len(dup_fwd), unmatched


def check_solid(parts, name):
    """Check each solid piece (base slab, kerb / each tile) independently --
    pieces that only touch along a shared face/edge are still each valid,
    separate watertight solids, so checking them jointly would misreport a
    shared boundary edge as a defect."""
    ok = True
    total_vol = 0.0
    for tris in parts:
        vol, dup, unmatched = _check_one_solid(tris)
        total_vol += vol
        if dup or unmatched or vol <= 0:
            ok = False
    n_tris = sum(len(t) for t in parts)
    print(f"[{name}] parts={len(parts)}  triangles={n_tris}  "
          f"total_volume={total_vol:.1f} mm^3  "
          f"-> {'OK, every part watertight manifold' if ok else 'PROBLEM'}")
    return ok


def build_curb(length_mm, width_mm, curb_height_mm=CURB_HEIGHT_MM,
               base_thickness_mm=BASE_THICKNESS_MM, lead_in_mm=0.0,
               edge_offset_mm=0.0):
    """Kerb edge on the patch centreline (y = width/2): the robot's +y feet
    walk on the kerb top, -y feet on the road -- matches CurbTerrainCfg's
    default spawn state (straddling, not mounting)."""
    parts = [box_triangles((length_mm / 2, width_mm / 2, -base_thickness_mm / 2),
                            (length_mm / 2, width_mm / 2, base_thickness_mm / 2))]

    y_edge = width_mm / 2 + edge_offset_mm
    x0, x1 = lead_in_mm, length_mm
    if curb_height_mm > 0 and width_mm > y_edge and x1 > x0:
        parts.append(box_triangles(((x0 + x1) / 2, (y_edge + width_mm) / 2, curb_height_mm / 2),
                                    ((x1 - x0) / 2, (width_mm - y_edge) / 2, curb_height_mm / 2)))
    return parts


def build_steps(length_mm, width_mm, tile_mm=TILE_SIZE_MM,
                 max_height_mm=MAX_STEP_HEIGHT_MM, num_levels=NUM_LEVELS,
                 flat_start_mm=FLAT_START_MM, base_thickness_mm=BASE_THICKNESS_MM,
                 seed=0):
    """Discrete-height tile field: sharp risers, not smooth noise -- what
    actually stresses foot placement (see RandomStepsTerrainCfg)."""
    parts = [box_triangles((length_mm / 2, width_mm / 2, -base_thickness_mm / 2),
                            (length_mm / 2, width_mm / 2, base_thickness_mm / 2))]

    levels = [i * max_height_mm / (num_levels - 1) for i in range(num_levels)] \
        if num_levels > 1 else [max_height_mm]
    rng = random.Random(seed)

    nx = max(0, int((length_mm - flat_start_mm) // tile_mm))
    ny = max(0, int(width_mm // tile_mm))
    if nx < 2 or ny < 2:
        print(f"WARNING: grid is only {nx} x {ny} tiles -- with ny<2 every column is a "
              f"full-width slab (looks like a few giant steps, not a field). Increase "
              f"--width/--length or shrink --tile-size so both nx and ny are >= 3.")
    for i in range(nx):
        for j in range(ny):
            h = rng.choice(levels)
            if h <= 0.0:
                continue
            cx = flat_start_mm + (i + 0.5) * tile_mm
            cy = (j + 0.5) * tile_mm
            parts.append(box_triangles((cx, cy, h / 2), (tile_mm / 2, tile_mm / 2, h / 2)))
    return parts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--terrain", choices=["curb", "steps"], required=True)
    p.add_argument("--out", default=None, help="output .stl path")
    p.add_argument("--length", type=float, default=300.0, help="mm, along travel dir")
    p.add_argument("--width", type=float, default=240.0, help="mm, across travel dir")
    p.add_argument("--curb-height", type=float, default=CURB_HEIGHT_MM)
    p.add_argument("--lead-in", type=float, default=0.0,
                   help="mm of flat road before the kerb (0 = kerb runs full length)")
    p.add_argument("--tile-size", type=float, default=TILE_SIZE_MM)
    p.add_argument("--step-height", type=float, default=MAX_STEP_HEIGHT_MM)
    p.add_argument("--num-levels", type=int, default=NUM_LEVELS)
    p.add_argument("--flat-start", type=float, default=FLAT_START_MM)
    p.add_argument("--base-thickness", type=float, default=BASE_THICKNESS_MM)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    out = args.out or f"{args.terrain}.stl"

    if args.terrain == "curb":
        parts = build_curb(args.length, args.width, args.curb_height,
                            args.base_thickness, args.lead_in)
    else:
        parts = build_steps(args.length, args.width, args.tile_size,
                             args.step_height, args.num_levels,
                             args.flat_start, args.base_thickness, args.seed)

    ok = check_solid(parts, args.terrain)
    write_binary_stl([t for part in parts for t in part], out)
    print(f"wrote {out}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
