"""Millimetre-scale rough terrain (unchanged from upstream).

On this ladder all 2026-09 policies plateaued at level 1.06-1.40 of 4, so it
does not separate morphologies; it is kept as-is until it is re-scaled.
"""

from __future__ import annotations

import mjlab.terrains as tg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

MICRO_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(4.0, 4.0),
  border_width=8.0,
  num_rows=4,
  num_cols=8,
  difficulty_range=(0.0, 1.0),
  sub_terrains={
    "flat": tg.BoxFlatTerrainCfg(proportion=0.40),
    "tiny_pyramid_stairs": tg.BoxPyramidStairsTerrainCfg(
      proportion=0.20, step_height_range=(0.005, 0.01), step_width=0.20, platform_width=1.0, border_width=0.25,
    ),
    "tiny_pyramid_stairs_inv": tg.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.10, step_height_range=(0.00, 0.015), step_width=0.20, platform_width=1.0, border_width=0.25,
    ),
    "micro_random_rough": tg.HfRandomUniformTerrainCfg(
      proportion=0.20, noise_range=(-0.008, 0.008), noise_step=0.004,
      horizontal_scale=0.15, vertical_scale=0.001, border_width=0.25,
    ),
    "micro_wave": tg.HfWaveTerrainCfg(
      proportion=0.10, amplitude_range=(0.0, 0.009), num_waves=5,
      horizontal_scale=0.15, vertical_scale=0.001, border_width=0.25,
    ),
  },
  add_lights=True,
)
