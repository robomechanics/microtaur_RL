"""Resolve the foot-contact sensor's channel order.

`feet_ground_contact` is built from a *pattern* (`FOOT_GEOM_NAMES`), and mjlab
expands patterns through `Entity.find_geoms`, which returns matches in MuJoCo
geom-id order -- NOT in the order the tuple was written. On the microtaur models
the leg elements are not laid out leg1..leg4, so the contact channels come back
permuted (pitch, for example, gives leg1, leg2, leg4, leg3).

Reading channel i as "leg i+1" swaps two feet, and swapping one diagonal pair
for its lateral partner turns a trot into an apparent pace. Always go through
`leg_channels()` instead of assuming the order.

`ContactSensor.primary_names` is the authoritative index->name map.
"""

from __future__ import annotations

import re

# leg -> corner, verified against microtaur_kinematics.HIP_MIDPOINTS_ROOT_M and
# the compiled MJCF site positions for all four variants (and re-confirmed from
# measured trunk-frame foot positions by scripts/gait_from_geometry.py).
LEG_CORNER = {1: "HR", 2: "HL", 3: "FL", 4: "FR"}


def leg_channels(sensor, n_legs: int = 4) -> list[int]:
  """Return `chan[i]` = the sensor channel carrying leg ``i+1``.

  So ``found[chan[i]]`` is the contact flag for ``leg{i+1}_foot_site``.
  Falls back to the identity order only if the sensor exposes no names.
  """
  names = list(getattr(sensor, "primary_names", []) or [])
  if not names:
    return list(range(n_legs))
  chan: list[int] = []
  for leg in range(1, n_legs + 1):
    # NB: not `\b` -- these names are like "leg1_foot_collision" and `_` is a
    # word character, so `leg1\b` never matches. Require a non-digit instead.
    hits = [i for i, n in enumerate(names) if re.search(rf"leg0*{leg}(?!\d)", n)]
    if len(hits) != 1:
      raise ValueError(
        f"cannot place leg{leg} in contact-sensor channels {names!r} (matches {hits})"
      )
    chan.append(hits[0])
  return chan


def corner_of_channel(sensor, n_legs: int = 4) -> dict[int, str]:
  """channel index -> 'FL'/'FR'/'HL'/'HR'."""
  chan = leg_channels(sensor, n_legs)
  return {chan[i]: LEG_CORNER[i + 1] for i in range(n_legs)}
