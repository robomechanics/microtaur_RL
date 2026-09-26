"""IsaacLab environment cfgs for the rigid Microtaur: flat, rough (A/B/C), teacher.

Physics: 2.5 ms step (closed chain needs it), decimation 14 -> 35 ms policy
period (28.6 Hz, task_params.POLICY_DT_S). Everything task-specific comes from
microtaur_common (shared with mjlab) or the mdp modules.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as il_mdp
from isaaclab.managers import CurriculumTermCfg, EventTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from microtaur_common.sim2real import PUSH_FLAT_SCALE, PUSH_INTERVAL_S, PUSH_VELOCITY_RANGE, STAGES
from microtaur_common.task_params import EPISODE_S, POLICY_DT_S

from . import ROOT_BODY_NAME
from .mdp import commands as C
from .mdp import events as E
from .mdp import observations as O
from .mdp import rewards as R
from .mdp import terminations as T
from .mdp.actions import MicrotaurWalkActionCfg
from .robot import MICROTAUR_RIGID_CFG

PHYSICS_DT_S = 0.0025
DECIMATION = round(POLICY_DT_S / PHYSICS_DT_S)  # 14

# -----------------------------------------------------------------------------
# Spawn (terrain aware)
# -----------------------------------------------------------------------------

B_SPAWN_MARGIN_M = (0.4, 0.3)  # keep the footprint inside the tile
FOOTPRINT_HALF_M = 0.13  # conservative half-size of the foot polygon at any heading
_B_CELLS: dict[int, np.ndarray] = {}


def _b_cells(level: int):
  from . import terrains as TR
  if level not in _B_CELLS:
    _B_CELLS[level] = TR.block_cell_heights(TR.TILE_SIZE_M, level / (TR.NUM_LEVELS - 1))
  return _B_CELLS[level]


def terrain_spawn(env, ids, xy, yaw):
  """A: sampled offset. B: anywhere on the fixed map, root lifted by the highest
  cell under the footprint. C: course start, heading +x."""
  terrain = env.scene.terrain
  if getattr(terrain, "terrain_types", None) is None:
    return xy, yaw, torch.zeros_like(yaw)
  from . import terrains as TR
  types = TR.env_terrain_type_ids(terrain)[ids]
  levels = terrain.terrain_levels[ids]
  xy, yaw, ground = xy.clone(), yaw.clone(), torch.zeros_like(yaw)
  is_c, is_b = types == 2, types == 1
  xy[is_c] = torch.tensor(TR.C_SPAWN_OFFSET_XY, device=xy.device, dtype=xy.dtype)
  yaw[is_c] = TR.C_SPAWN_YAW_RAD
  if torch.any(is_b):
    lx, ly = TR.TILE_SIZE_M
    nb = int(is_b.sum())
    hx, hy = 0.5 * lx - B_SPAWN_MARGIN_M[0], 0.5 * ly - B_SPAWN_MARGIN_M[1]
    rand = torch.rand(nb, 2, device=xy.device, dtype=xy.dtype) * 2.0 - 1.0
    xy[is_b] = rand * torch.tensor((hx, hy), device=xy.device, dtype=xy.dtype)
    c = TR.CELL_M
    g = []
    for k, (p, lev) in enumerate(zip(xy[is_b].cpu().numpy(), levels[is_b].cpu().numpy())):
      h = _b_cells(int(lev))
      i0, i1 = (int(math.floor((p[0] + 0.5 * lx + s * FOOTPRINT_HALF_M) / c)) for s in (-1, 1))
      j0, j1 = (int(math.floor((p[1] + 0.5 * ly + s * FOOTPRINT_HALF_M) / c)) for s in (-1, 1))
      g.append(float(h[max(i0, 0):i1 + 1, max(j0, 0):j1 + 1].max()))
    ground[is_b] = torch.tensor(g, device=xy.device, dtype=xy.dtype)
  return xy, yaw, ground


def terrain_levels_from_spawn(env, env_ids, command_name: str = "twist") -> torch.Tensor:
  """IsaacLab's terrain_levels_vel, measured from the spawn point.

  Promote if the robot walked farther than half a tile length, demote if it
  walked less than half the commanded distance (as IsaacLab / mjlab). IsaacLab's
  version measures from the env origin and reads a "base_velocity" command.
  """
  terrain = env.scene.terrain
  robot = env.scene["robot"]
  # The first env.reset() runs the curriculum before any spawn event, with the
  # robots still at their clone-grid positions; that is not a walked distance.
  spawn = getattr(env, "microtaur_spawn_xy", None)
  if spawn is None:
    return torch.mean(terrain.terrain_levels.float())
  distance = torch.norm(robot.data.root_pos_w[env_ids, :2] - spawn[env_ids], dim=1)
  command = env.command_manager.get_command(command_name)
  move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
  move_down = (distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5) & ~move_up
  terrain.update_env_origins(env_ids, move_up, move_down)
  return torch.mean(terrain.terrain_levels.float())


def terrain_flags(env, env_ids) -> None:
  """Startup: env.microtaur_zero_yaw_mask = True on terrain C (straight commands only)."""
  terrain = env.scene.terrain
  mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  if getattr(terrain, "terrain_types", None) is not None:
    from . import terrains as TR
    mask = TR.env_terrain_type_ids(terrain) == 2
  setattr(env, C.ZERO_YAW_MASK_ATTR, mask)


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------


def _contact_sensor() -> ContactSensorCfg:
  # force_threshold: IsaacLab's 1.0 N default is about one foot's stance load.
  # history_length >= decimation so air/contact times update every substep.
  return ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=DECIMATION, track_air_time=True,
                          force_threshold=O.CONTACT_FORCE_THRESHOLD_N)


def _height_scanner(resolution: float, size: tuple[float, float]) -> RayCasterCfg:
  return RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/" + ROOT_BODY_NAME,
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=resolution, size=list(size)),
    max_distance=O.HEIGHT_SCAN_MAX_DISTANCE_M + 0.2,
    mesh_prim_paths=["/World/ground"],
  )


@configclass
class MicrotaurSceneCfg(InteractiveSceneCfg):
  terrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
      friction_combine_mode="max", restitution_combine_mode="multiply", static_friction=1.0, dynamic_friction=1.0,
    ),
  )
  robot = MICROTAUR_RIGID_CFG
  contact_forces = _contact_sensor()
  light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(intensity=3000.0))


# -----------------------------------------------------------------------------
# Managers
# -----------------------------------------------------------------------------


def _events(stage, play: bool, rough: bool) -> dict[str, EventTermCfg]:
  ev = {
    "terrain_flags": EventTermCfg(func=terrain_flags, mode="startup"),
    "ik_consistent_reset": EventTermCfg(func=E.ik_consistent_reset, mode="reset",
                                        params={"randomize": not play, "spawn_fn": terrain_spawn}),
  }
  if not play:
    ev["encoder_bias"] = EventTermCfg(func=E.encoder_bias, mode="startup", params={"bias_rad": stage.encoder_bias_rad})
    if stage.root_com_xy_m is not None:
      xy, z = stage.root_com_xy_m, stage.root_com_z_m
      ev["base_com"] = EventTermCfg(
        func=il_mdp.randomize_rigid_body_com, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=ROOT_BODY_NAME),
                "com_range": {"x": (-xy, xy), "y": (-xy, xy), "z": (-z, z)}},
      )
    if rough or stage.push:
      k = 1.0 if rough else PUSH_FLAT_SCALE
      ev["push_robot"] = EventTermCfg(
        func=il_mdp.push_by_setting_velocity, mode="interval", interval_range_s=PUSH_INTERVAL_S,
        params={"velocity_range": {a: (-k * v, k * v) for a, v in PUSH_VELOCITY_RANGE.items()}},
      )
  return ev


@configclass
class MicrotaurFlatEnvCfg(ManagerBasedRLEnvCfg):
  stage: int = 0
  rough: bool = False
  teacher: bool = False
  play: bool = False
  scene: MicrotaurSceneCfg = MicrotaurSceneCfg(num_envs=2048, env_spacing=0.5)

  def __post_init__(self):
    s2r = STAGES[self.stage]
    self.decimation = DECIMATION
    self.episode_length_s = EPISODE_S
    self.sim.dt = PHYSICS_DT_S
    self.sim.render_interval = DECIMATION
    self.sim.physics_material = self.scene.terrain.physics_material
    self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

    if self.rough:
      from .terrains import MICROTAUR_TERRAINS_CFG
      self.scene.terrain.terrain_type = "generator"
      self.scene.terrain.terrain_generator = MICROTAUR_TERRAINS_CFG
      self.scene.terrain.max_init_terrain_level = 0
    if self.rough or self.teacher:
      # 12 x 9 = 108 values at 35 mm (two samples per 70 mm cell) for the teacher.
      self.scene.height_scanner = _height_scanner(0.035, (0.385, 0.28)) if self.teacher else \
        _height_scanner(0.05, (0.40, 0.30))

    self.observations = O.make_observations_cfg(s2r, rough=self.rough and not self.teacher,
                                                teacher=self.teacher, play=self.play)
    self.actions = _Actions()
    self.actions.joint_pos = MicrotaurWalkActionCfg(
      target_gain_range=(1.0, 1.0) if self.play else s2r.action_gain_range,
      target_bias_rad=0.0 if self.play else s2r.action_bias_rad,
    )
    self.commands = _Commands()
    self.commands.twist = C.make_twist_command_cfg(play=self.play)
    self.rewards = R.make_rewards_cfg()
    self.terminations = T.make_terminations_cfg()
    self.events = _Events()
    for k, v in _events(s2r, self.play, self.rough).items():
      setattr(self.events, k, v)
    self.curriculum = _Curriculum()
    self.curriculum.command_ranges = C.make_command_curriculum_term(play=self.play)
    self.curriculum.energy_weight = R.make_energy_curriculum_term()
    if self.rough:
      self.curriculum.terrain_levels = CurriculumTermCfg(func=terrain_levels_from_spawn)
    if self.play:
      self.episode_length_s = 1.0e9


@configclass
class _Actions:
  joint_pos: MicrotaurWalkActionCfg = MicrotaurWalkActionCfg()


@configclass
class _Commands:
  twist: object = None


@configclass
class _Events:
  pass


@configclass
class _Curriculum:
  command_ranges: object = None
  energy_weight: object = None


@configclass
class MicrotaurRoughEnvCfg(MicrotaurFlatEnvCfg):
  rough: bool = True


@configclass
class MicrotaurTeacherEnvCfg(MicrotaurFlatEnvCfg):
  """Rough terrain; the actor gets the privileged teacher group (critic set + 35 mm height map)."""
  rough: bool = True
  teacher: bool = True
