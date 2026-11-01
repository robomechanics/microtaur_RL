# Real-world test terrains (1:1)

These are 1:1 copies of the simulated Teacher-Cur terrains: B (random blocks) and C (side step), levels 1 to 6. Each piece is a 2000 mm × 1000 mm closed solid with a flat bottom. All units are millimetres.

## Coordinates

- x runs 0 to 2000 mm. This is the robot's walking direction.
- y runs 0 to 1000 mm. +y is the robot's left.
- z is measured up from the bottom of the piece.

## Files

| file | content |
|---|---|
| `B_lv<l>.3mf`, `B_lv<l>.stl` | B, level l. Use the 3MF if you can (see the STL note below). |
| `C_lv<l>.3mf`, `C_lv<l>.stl` | C, level l. |
| `B_lv<l>_cells.csv` | Every B cell: its x/y edges, its top height above the bottom, and its height relative to the simulated ground. Use this to build the piece from blocks instead of printing it. |
| `pieces.json` | The numbers in the tables below. |
| `preview.png` | Top view of the six B pieces. |
| `export_real_terrain.py` | The script that writes everything in this folder. |

To regenerate the folder:

```
python microtaur_study/cad/real_terrain/export_real_terrain.py
```

This needs numpy, trimesh and matplotlib. It does not need Isaac Sim or a GPU.

## Pieces

**B: random blocks.**
- Each piece is a window of the simulated B tile. It starts 0.25 m behind the tile centre (the robot's spawn point), runs 2 m forward, and spans ±0.5 m sideways.
- Cells are 70 mm squares. The window cuts the cells on its edges: the first column is 40 mm long, and the two side rows are 10 mm wide.
- All six levels use the same pattern. Heights scale with the level, by l/6.
- Every piece is as thin as possible: its lowest cell is 3 mm thick. The simulated ground level therefore sits at a different height in each piece.

| piece | simulated ground above the bottom | cell tops | relative to the simulated ground |
|---|---|---|---|
| B_lv1 | 5.5 mm | 3.0 to 8.9 mm | −2.5 to +3.4 mm |
| B_lv2 | 8.1 mm | 3.0 to 14.9 mm | −5.1 to +6.8 mm |
| B_lv3 | 10.6 mm | 3.0 to 20.8 mm | −7.6 to +10.2 mm |
| B_lv4 | 13.1 mm | 3.0 to 26.7 mm | −10.1 to +13.6 mm |
| B_lv5 | 15.7 mm | 3.0 to 32.6 mm | −12.6 to +17.0 mm |
| B_lv6 | 18.2 mm | 3.0 to 38.6 mm | −15.2 to +20.4 mm |

**C: side step.**
- Each piece is the middle 2 m of the 4 m raised section, centred on the step edge.
- The right half (y < 500) is at the simulated ground: 3 mm thick.
- The left half (y > 500) is raised by the step height.
- There are no guard rails.

| piece | C_lv1 | C_lv2 | C_lv3 | C_lv4 | C_lv5 | C_lv6 |
|---|---|---|---|---|---|---|
| step | 6.1 mm | 12.3 mm | 18.4 mm | 24.5 mm | 30.7 mm | 36.8 mm |

**Robot start.** Put the robot on the piece, at x ≈ 250 mm, facing +x. On C, it straddles the step edge at y = 500. In the simulation the robot never steps up from the floor onto a piece, so don't start it on the floor.

## Notes

- **Volume.** Solid volumes run from about 11 L (B_lv1) to 43 L (C_lv6). A 2 m × 1 m piece is bigger than any printer bed, so cut it in the slicer, or build it from boards or blocks using the CSV.
- **STL vs 3MF.** Where four B cells meet diagonally (two high, two low), the two high blocks touch only along a vertical edge. The 3MF files give each block its own copy of that edge, so every edge belongs to exactly two triangles. STL cannot store this, so slicers report those edges as "non-manifold edges" (133 of them on B_lv6). This is the real geometry; letting the slicer repair it is harmless.
- **Sources.** The B heights come from `microtaur_common.terrain_maps.gaussian_blocks`, with the same arguments the simulator uses (`microtaur_isaac.terrains.block_cell_heights`, ×2 for the curriculum terrain, level l → difficulty l/6). The C step and the tile constants are read from `microtaur_isaac/terrains.py`. The training code was only read, not changed.
