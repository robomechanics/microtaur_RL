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
from .robot import MICROTAUR_RIGID_CFG, VISUAL_USD_PATH

PHYSICS_DT_S = 0.0025
DECIMATION = round(POLICY_DT_S / PHYSICS_DT_S)  # 14

# -----------------------------------------------------------------------------
# Spawn (terrain aware)
# -----------------------------------------------------------------------------

B_SPAWN_MARGIN_M = (0.4, 0.3)  # keep the footprint inside the tile
FOOTPRINT_HALF_M = 0.13  # conservative half-size of the foot polygon at any heading
_B_CELLS: dict[int, np.ndarray] = {}


def _b_cells(level: int, scale: float = 1.0, num_levels: int | None = None):
  from . import terrains as TR
  n = num_levels or TR.NUM_LEVELS
  key = (level, scale, n)
  if key not in _B_CELLS:
    _B_CELLS[key] = scale * TR.block_cell_heights(TR.TILE_SIZE_M, min(level, n - 1) / (n - 1))
  return _B_CELLS[key]


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
    scale = float(terrain.cfg.terrain_generator.sub_terrains["B_blocks"].height_scale)
    n_lv = TR.terrain_num_levels(terrain)
    g = []
    for k, (p, lev) in enumerate(zip(xy[is_b].cpu().numpy(), levels[is_b].cpu().numpy())):
      h = _b_cells(int(lev), scale, n_lv)
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


def terrain_levels_progress(env, env_ids, command_name: str = "twist", promote_frac: float = 0.8,
                            demote_frac: float = 0.4, steps_per_level: int = 300 * 32) -> torch.Tensor:
  """Terrain curriculum on commanded distance (Teacher-Cur).

  Per episode the command term integrates the commanded path length and the body's
  progress along the commanded direction. ratio = progress / (mean commanded speed x
  full episode length), so a robot that falls early scores low. Promote one level if
  ratio >= promote_frac, demote if < demote_frac. Levels above
  common_step_counter // steps_per_level are locked (300 iterations x 32 steps per
  level: only level 0 for the first 300 iterations, the top level from 300 x (L - 1)).
  Robots at the unlocked cap stay there. Run-out rows past the last level are never
  assigned."""
  from . import terrains as TR
  terrain = env.scene.terrain
  if getattr(env, "microtaur_spawn_xy", None) is None:
    return torch.mean(terrain.terrain_levels.float())
  n_lv = TR.terrain_num_levels(terrain)
  cap = min(n_lv - 1, int(env.common_step_counter) // steps_per_level)
  term = env.command_manager.get_term(command_name)
  elapsed = env.episode_length_buf[env_ids].float().clamp_min(1.0) * env.step_dt
  expected = term.cmd_distance[env_ids] / elapsed * env.max_episode_length_s
  ratio = term.cmd_progress[env_ids] / expected.clamp_min(1e-3)
  lv = terrain.terrain_levels[env_ids]
  up = (ratio >= promote_frac) & (lv < cap)
  down = ratio < demote_frac
  lv = torch.clamp(lv + up.long() - down.long(), 0, cap)
  terrain.terrain_levels[env_ids] = lv
  terrain.env_origins[env_ids] = terrain.terrain_origins[lv, terrain.terrain_types[env_ids]]
  return torch.mean(terrain.terrain_levels.float())


def hard_terrain_placement(env, env_ids) -> None:
  """Startup (Teacher-Hard): every env on a B or C column (A skipped, round-robin over
  the B / C columns, so 5:3 like the generator) at the hardest level, for the whole
  run. Must run before terrain_flags (the zero-yaw mask reads the terrain types)."""
  from . import terrains as TR
  terrain = env.scene.terrain
  cols = [c for c, t in enumerate(TR.COLUMN_TERRAIN_TYPES) if t != "A_flat"]
  n = env.num_envs
  terrain.terrain_types[:] = torch.tensor(cols, device=terrain.terrain_types.device)[torch.arange(n) % len(cols)]
  terrain.terrain_levels[:] = TR.terrain_num_levels(terrain) - 1
  terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]


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
  terrain_scale: float = 1.0  # rough only: B heights and the C step x this (terrains.scaled_terrains_cfg)
  cur_terrain: bool = False  # rough only: 7-level railed-C terrain + run-out row, commanded-distance curriculum
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
      from . import terrains as TR
      from .terrains import MICROTAUR_TERRAINS_CFG
      self.scene.terrain.terrain_type = "generator"
      if self.cur_terrain:
        self.scene.terrain.terrain_generator = TR.curriculum_terrains_cfg(self.terrain_scale)
      else:
        self.scene.terrain.terrain_generator = (MICROTAUR_TERRAINS_CFG if self.terrain_scale == 1.0 else
                                                TR.scaled_terrains_cfg(self.terrain_scale))
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
      self.curriculum.terrain_levels = CurriculumTermCfg(
        func=terrain_levels_progress if self.cur_terrain else terrain_levels_from_spawn)
    if self.play:
      self.episode_length_s = 1.0e9
      if self.rough:
        # Spread robots over all levels and keep them there.
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None
      self.scene.robot = self.scene.robot.replace(
        spawn=self.scene.robot.spawn.replace(usd_path=str(VISUAL_USD_PATH)))
      # GUI camera follows env 0's robot from the front-left side.
      self.viewer.origin_type = "asset_root"
      self.viewer.asset_name = "robot"
      self.viewer.env_index = 0
      self.viewer.eye = (0.30, -0.45, 0.18)
      self.viewer.lookat = (0.0, 0.0, 0.04)


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
  reward_schedule: object = None


@configclass
class MicrotaurRoughEnvCfg(MicrotaurFlatEnvCfg):
  rough: bool = True


@configclass
class MicrotaurTeacherEnvCfg(MicrotaurFlatEnvCfg):
  """Rough terrain; the actor gets the privileged teacher group (critic set + 35 mm height map)."""
  rough: bool = True
  teacher: bool = True


@configclass
class MicrotaurFlatPlayEnvCfg(MicrotaurFlatEnvCfg):
  play: bool = True
  scene: MicrotaurSceneCfg = MicrotaurSceneCfg(num_envs=16, env_spacing=0.5)


@configclass
class MicrotaurRoughPlayEnvCfg(MicrotaurRoughEnvCfg):
  play: bool = True
  scene: MicrotaurSceneCfg = MicrotaurSceneCfg(num_envs=16, env_spacing=0.5)


@configclass
class MicrotaurTeacherPlayEnvCfg(MicrotaurTeacherEnvCfg):
  play: bool = True
  scene: MicrotaurSceneCfg = MicrotaurSceneCfg(num_envs=16, env_spacing=0.5)


@configclass
class MicrotaurTeacherHardEnvCfg(MicrotaurTeacherEnvCfg):
  """No terrain curriculum: every robot on the hardest B or C tiles from iteration 0
  (side experiment: can the teacher learn the hard terrain directly?)."""

  def __post_init__(self):
    super().__post_init__()
    self.curriculum.terrain_levels = None
    events = _Events()
    setattr(events, "hard_terrain_placement", EventTermCfg(func=hard_terrain_placement, mode="startup"))
    for k, v in self.events.to_dict().items():
      if v is not None:
        setattr(events, k, getattr(self.events, k))
    self.events = events


@configclass
class MicrotaurTeacherHardPlayEnvCfg(MicrotaurTeacherHardEnvCfg):
  play: bool = True
  scene: MicrotaurSceneCfg = MicrotaurSceneCfg(num_envs=16, env_spacing=0.5)


@configclass
class MicrotaurTeacherHard2xEnvCfg(MicrotaurTeacherHardEnvCfg):
  """Teacher-Hard on terrain x2: every robot on level-4 B (-15.2 / +21.6 mm, sigma 7.6 mm)
  or C (36.8 mm step), i.e. 100% of the stance-phase budget, no terrain curriculum."""
  terrain_scale: float = 2.0


@configclass
class MicrotaurTeacherHard2xPlayEnvCfg(MicrotaurTeacherHardPlayEnvCfg):
  terrain_scale: float = 2.0


@configclass
class MicrotaurTeacherCurEnvCfg(MicrotaurTeacherEnvCfg):
  """Teacher with the curriculum terrain: 7 levels up to x2 (B -15.2 / +21.6 mm, C 36.8 mm),
  a run-out row, C as a railed lane, promotion on commanded distance, one level unlocked
  per 300 iterations."""
  cur_terrain: bool = True
  terrain_scale: float = 2.0

  def __post_init__(self):
    super().__post_init__()
    # The C guard rails guide, they do not end the episode (a centred trot with +-0.1 rad
    # yaw wiggle brushes a rail at 0.12 m); only the body landing on the ground terminates.
    self.terminations.illegal_contact.params["vertical_only"] = True
    # Staged reward weights / params, see R.reward_schedule. One inert stage by default: IsaacLab's
    # cfg update rejects a Hydra list override whose length differs from the default.
    self.curriculum.reward_schedule = CurriculumTermCfg(
      func=R.reward_schedule, params={"stages": ["off"]})


@configclass
class MicrotaurTeacherCurPlayEnvCfg(MicrotaurTeacherCurEnvCfg):
  play: bool = True
  scene: MicrotaurSceneCfg = MicrotaurSceneCfg(num_envs=16, env_spacing=0.5)
