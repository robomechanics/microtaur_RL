"""Contact and height sensors.

mjlab's base velocity config ships raycast placeholders with empty frames;
they are all rebuilt here.
"""

from __future__ import annotations

from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sensor.raycast_sensor import RingPatternCfg
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensorCfg

from .robot import BODY_COLLISION_GEOM_NAMES, FOOT_GEOM_NAMES, FOOT_SITE_NAMES, ROOT_BODY

FOOT_CONTACT = "feet_ground_contact"
BODY_CONTACT = "nonfoot_ground_touch"


def _contact(name: str, geoms: tuple[str, ...], air_time: bool) -> ContactSensorCfg:
  return ContactSensorCfg(
    name=name,
    primary=ContactMatch(mode="geom", pattern=geoms, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=air_time,
  )


def configure_sensors(cfg, play: bool, rough: bool) -> None:
  keep = [
    s for s in (cfg.scene.sensors or ())
    if not isinstance(s, RayCastSensorCfg) and s.name not in {FOOT_CONTACT, BODY_CONTACT}
  ]
  # Critic-only foot height (the actor never sees it).
  keep.append(
    TerrainHeightSensorCfg(
      name="foot_height_scan",
      frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITE_NAMES),
      pattern=RingPatternCfg.single_ring(radius=0.008, num_samples=4, include_center=True, direction=(0.0, 0.0, -1.0)),
      ray_alignment="world",
      max_distance=1.0,
      exclude_parent_body=True,
      include_geom_groups=(0,),
      debug_vis=play,
    )
  )
  if rough:
    # Sensor must be named terrain_scan: mjlab's height_scan observation looks it up.
    keep.append(
      RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name=ROOT_BODY, entity="robot"),
        pattern=GridPatternCfg(size=(0.40, 0.30), resolution=0.05, direction=(0.0, 0.0, -1.0)),
        ray_alignment="yaw",
        max_distance=0.30,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=play,
      )
    )
  keep.append(_contact(FOOT_CONTACT, FOOT_GEOM_NAMES, air_time=True))
  keep.append(_contact(BODY_CONTACT, BODY_COLLISION_GEOM_NAMES, air_time=False))
  cfg.scene.sensors = tuple(keep)
