# What is missing to run the rigid Microtaur in IsaacLab

2026-09-23 · branch `spine-study-isaac` · **scope: `rigid_microtaur` only**

Every fact below was checked against the files on 2026-09-23, and corrected
against the newer upstream tree on 2026-09-25. Line references are to
`Microtaur_RL-main/` unless stated otherwise.

⚠ **This repository's own rigid configuration is the old one.** Its
`env_cfgs.py` is `ENV_CFG_REVISION = "2026-09-05-rigid-aligned-baseline-v1"`
(2291 lines) and its `robot_modified.xml` is the 72,002-byte version with the
pre-correction mass model. The configuration to port is in
`github.com/aryan-chandra-cmu/Microtaur_RL`, whose `main` is at **`5880025`**
("rigid sim2real hardware char and stand pos", 09-21):
`src/microtaur_velocity/env_cfgs.py`, 2529 lines,
`ENV_CFG_REVISION = "2026-09-20-rigid-hw-calibrated-sim2real-v3-observation-fix"`.
Relative to `eef9f92` that commit only relocates files under `src/` and
`microtaur_xmls/` — env and model are identical — and adds a 799-line
`sim2real/README.md` plus hardware-characterisation scripts, among them
`quantify_closure_error_test25.py`, which by its name measures closure error and
may already contain a hardware criterion for §0. That repository is read-only:
`log` / `diff` / `show` only, no commits, pushes or branch checkouts.

Scope note: this covers the rigid variant only. The three spine variants
(`active_twist`, `active_pitch`, `active_yaw`) add a spine action term, spine
reward terms, a second trunk body and an extra observation block; none of that
is in this document.

---

## 0. Do this before writing any environment code

**Picking rigid does not avoid the closed-kinematic-loop problem.** Each of the
four legs is modelled as two open 2-link chains stitched together by MuJoCo
equality constraints, and rigid has all eight of them —
`microtaur_xmls/rigid_microtaur/robot_modified.xml`, `<equality>` block:

```xml
<!-- Firmer closed-chain constraints; tune solver iterations in env_cfg separately if needed. -->
<connect site1="closing_leg1_c_joint_1"   site2="closing_leg1_c_joint_2"   solref="0.01 1" solimp="0.95 0.99 0.001 0.5 2"/>
<connect site1="closing_leg1_c_joint_1_z" site2="closing_leg1_c_joint_2_z" solref="0.01 1" solimp="0.95 0.99 0.001 0.5 2"/>
...  (the same pair for leg2, leg3, leg4 — eight in total)
```

The kinematic tree is **16 hinges + one `<freejoint name="battery_freejoint">`**:
eight actuated (`leg{1-4}_{a,e}_joint_act`) and eight passive (`leg{1-4}_{b,d}_joint`).
The passive eight exist only to be closed by those constraints.

PhysX articulations are reduced-coordinate trees and cannot express a closed
loop natively. Both routes have real costs:

- **Loop-closure D6 joints.** PhysX supports them, but they are solved as
  constraints rather than in reduced coordinates, so they are softer and can
  drift or jitter. Eight of them, on a robot 10 cm long, over
  millimetre-scale terrain features, at a couple of thousand parallel
  environments. Note the MJCF deliberately uses stiff settings
  (`solimp="0.95 0.99 0.001 0.5 2"`) and the env raises
  `cfg.sim.njmax = 1024`, `nconmax = 256`, `ccd_iterations = 50` — the current
  setup already needs help to keep these constraints clean in MuJoCo, which is
  the engine that handles them natively.
- **Serialise each leg** into a 2-DoF chain, computing the foot position from
  the closed-form FK that already exists (`src/microtaur_velocity/microtaur_kinematics.py`).
  Fast, stable, PhysX-native. But a five-bar's effective inertia and torque
  transmission ratio vary with configuration, so both become approximations;
  the passive-link `armature = 1e-6` and `frictionloss = 1e-3` stop meaning
  anything; and the identified root mass / COM values no longer describe the
  same mechanism. The sim-to-real identification would have to be redone.

**So the first task is a feasibility spike, not a port.** Get one leg, or one
robot, standing in IsaacLab with loop-closure joints, and measure constraint
drift and jitter at the real scale and the real environment count. If that does
not hold up, everything downstream changes, and finding out first is much
cheaper than finding out after porting the reward function.

### 0b. Spike result, 2026-09-25 — feasible, at a solver cost

The spike was run. Loop closure in PhysX **does** hold up, but only with

- **`pos_iters = 16`**, and
- **`dt = 0.0025 s`** — half MuJoCo's 0.005 s

so the physics cost per simulated second is at least double, before counting the
extra position iterations. That belongs in the compute budget: the throughput
number to plan against is the one measured under these settings, not a default.

Also established:

- Standing at zero action gives root z = **0.06706 m**, matching the MuJoCo
  reference. The conversion is faithful at least in the static pose.
- The model the spike used differs from the 09-20 XML by at most **5e-9 m**, so
  its stand / trot / drop conclusions carry over to the new model unchanged.
- **`frictionloss` does map, on Isaac Sim 5.0** (corrected 2026-09-25; an
  earlier version of this list said it did not). IsaacLab documents actuator
  `friction` as a unitless coefficient on Isaac Sim 4.5 but as an effort from
  5.0, and `tools/spike_isaac_joint_friction.py` measures it on the installed
  5.0: with `friction = 0.010` a joint holds under 0.009 N·m and slips at
  0.011; with `0.020` it holds up to 0.015 and slips fully at 0.021 (partly at
  0.019). ⚠ Further corrected on Isaac Sim 5.1 (CPU and GPU agree): those
  thresholds are only the *static* friction; IsaacLab's `friction=F` leaves
  `dynamic_friction` at 0, so once a joint slips it has no friction at all.
  Setting `friction=F, dynamic_friction=F, viscous_friction=0` reproduces
  MuJoCo's constant Coulomb `frictionloss` (holds below F, accelerates in
  proportion to tau - F above it), and the XML values carry over directly
  (1.0e-2 N·m actuated, 1.0e-3 passive).
- Contact-model differences between the two engines still need calibrating, and
  the contact sensor still has to be rewritten (§3).
- The environment builds and runs: actor 33-dim, critic 60-dim, total mass
  0.540 kg — after working around the reset bug below.

⚠ **Open: the spike used `kd = 0.045`, but the 09-20 configuration sets
`kd = 0`.** With less joint damping the loop-closure constraint spikes and the
jitter can both grow, so stand / trot / drop need re-running at `kd = 0` before
the feasibility verdict is final. This is the next thing to do.

⚠ **`quantify_closure_error_test25.py`**, added to the upstream tree in `5880025`
alongside a 799-line `sim2real/README.md`, may already contain a *hardware*
measurement of closure error. If it does, it supplies the acceptance threshold
this section currently lacks — read it before choosing one.

### 0c. The reset bug is in the upstream tree too

`microtaur_ik_consistent_reset` calls `set_joint_position_target` with
pre-expanded indices, which mjlab >= 1.6 rejects:

```
RuntimeError: shape mismatch: value tensor of shape [N, 1]
cannot be broadcast to indexing result of shape [N, 1, 1]
```

Upstream has **not** fixed this. The fix is two `.unsqueeze()` calls removed, and
it is already written in this repository's per-variant run snapshots:

> *"MJLab >= 1.6 outer-indexes 1-D env and joint IDs internally, so pass them
> unexpanded; pre-expanding them to [N, 1] and [1, 8] breaks broadcasting."*

Until it is fixed, anything that builds an environment needs a runtime shim.

---

## 1. Assets — nothing exists yet

**There is no USD and no URDF anywhere in this project.** Only MJCF plus STL.
For rigid:

| | |
|---|---|
| model | `microtaur_xmls/rigid_microtaur/robot_modified.xml`, 72,002 bytes |
| meshes | 13 STL in `assets/robot/` (`meshdir="assets/robot"`), 2.6 MB total, all references relative, no absolute paths |
| scene | `scene.xml`, 930 bytes |
| sites | `imu_site`, `leg{1-4}_foot_site`, plus 16 `closing_leg*` sites used only by the equality constraints |
| collision geoms | `leg{1-4}_foot_collision`, `rl_body_collision`, `rl_battery_collision` |
| **`<actuator>` block** | **none — zero actuator elements in the XML** |

That last row matters: **the entire actuator model lives in Python**, applied at
config-construction time. The XML's `<default><position kp="1.0" dampratio="0.045"/>`
class is never instantiated.

Two consequences for the conversion:

**(a) There is no URDF intermediate to lean on.** Either write one, or use a
direct MJCF import if the Isaac Sim version on the machine has one, and then
verify the result rather than trusting it — mass, inertia, joint axes, joint
limits and the collision/visual split all need checking against the MJCF.

**(b) Only ONE model change has to be relocated: the actuated joint range.**

> Corrected 2026-09-25. An earlier version of this section listed armature,
> frictionloss, damping, mass, COM and inertia as Python-side patches that had to
> be moved. That was wrong, and checking the XML settles it.

The XML already carries all of those per joint:

```xml
<joint name="leg1_a_joint_act" limited="true" range="-0.500000 1.400000"
       frictionloss="0.010" armature="0.0002" damping="0"/>
<joint name="leg1_b_joint"    frictionloss="0.001" armature="0.000001" damping="0.0001"/>
```

`_make_xml_validated_spec` re-sets armature / frictionloss / damping to exactly
the values the XML already has, so for those fields it is a **no-op**. Verified by
reading the compiled model with `mujoco`: actuated joints are armature 2e-4,
frictionloss 1.0e-2, damping 0; passive joints are 1e-6, 1.0e-3, 1e-4.

| target | field | XML has | Python sets | net effect |
|---|---|---|---|---|
| 8 × `*_joint_act` | `range` | ±0.95 rad about stand (`-0.5 … 1.4`, mirrored by side) | stand ± **0.75 rad** → `-0.3 … 1.2` | **narrowed — the only real change** |
| 8 × `*_joint_act` | `armature` / `frictionloss` / `damping` | 2e-4 / 1.0e-2 / 0 | the same | no-op |
| 8 × `leg{1-4}_{b,d}_joint` | `armature` / `frictionloss` / `damping` | 1e-6 / 1.0e-3 / 1e-4 | the same | no-op |
| body `battery` | mass / COM / inertia | **already the hardware-corrected values** | the same | no-op on this XML |

Two things worth noticing in that table:

- **The XML's own range is ±0.95 rad about the stand pose.** So
  `MICROTAUR_JOINT_HALF_RANGE_RAD = 0.95`, which is what the *pitch* variant uses,
  means "do not narrow the mechanical range at all". Rigid, roll and yaw narrow it
  to ±0.75. That is a cleaner reading of the per-variant difference than treating
  pitch as an outlier.
- **The mass patch is a no-op only on the new XML.** The 09-20 XML already contains
  the hardware-corrected root inertial, so the port needs no mass work at all.
  On the older 09-05 XML the same patch is load-bearing — see §1c.

What is *not* in the XML: the entire actuator model (kp, kd, effort and velocity
limits) — there are **zero `<actuator>` elements**, so a converter reading the XML
alone produces a robot with no actuators. See §4.

## 1c. Which XML, and the mass difference

The new and old rigid XMLs differ by **exactly one line**, the root `<inertial>`
of the `battery` body:

| | old (09-05) | **new (09-20)** |
|---|---|---|
| file size | 72,002 B | **72,099 B** |
| **total model mass** | 0.465334 kg | **0.540000 kg** |
| root (`battery`) mass | 0.391549 kg | **0.466215 kg** |
| root COM x | **+0.0196 m** | **−0.0008 m** |
| root COM y | +7.0e-07 | −7.2e-04 |
| `fullinertia` | — | scaled by **1.1907×** |
| `neq` (loop closures) | 8 | 8 — unchanged |

Both numbers are from loading each file with `mujoco` and summing `body_mass`, not
from the comments. The new total, **0.540000 kg, is exactly the measured robot
mass**, so that is the acceptance criterion for the USD conversion; the old one is
13.8% light.

The COM shift along x is **20.4 mm**, against a 170 mm fore-aft foot spacing — 12%
of the body length. It changes the static load distribution, and it is the change
that matches the measured 52.99 / 47.01% fore/aft support split. The old model was
markedly nose-heavy.

⚠ `microtaur_xmls/rigid_microtaur/robot_modified_rigid_sim2real.xml` is
**byte-identical** to `robot_modified.xml` in the new tree. It is a copy, not a
second variant; do not convert both.

---

## 2. Terrain — the easiest part of the port, plus two families that do not exist yet

### 2a. The existing rough terrain maps almost one-to-one

`src/microtaur_velocity/microtaur_config.py` defines `MICRO_ROUGH_TERRAINS_CFG`
using `mjlab.terrains`. mjlab's terrain modules are themselves adapted from
`isaaclab.terrains`, so the mapping is close to a find-and-replace:

| sub-terrain | mjlab class | IsaacLab counterpart | params |
|---|---|---|---|
| `flat` | `BoxFlatTerrainCfg` | `MeshPlaneTerrainCfg` | proportion 0.40 |
| `tiny_pyramid_stairs` | `BoxPyramidStairsTerrainCfg` | `MeshPyramidStairsTerrainCfg` — same field names | 0.20, step_height 0.005–0.01, step_width 0.20, platform_width 1.0, border 0.25 |
| `tiny_pyramid_stairs_inv` | `BoxInvertedPyramidStairsTerrainCfg` | `MeshInvertedPyramidStairsTerrainCfg` — same field names | 0.10, step_height 0.00–0.015 |
| `micro_random_rough` | `HfRandomUniformTerrainCfg` | **identical class name** | 0.20, noise ±0.008, noise_step 0.004, h_scale 0.15, v_scale 0.001 |
| `micro_wave` | `HfWaveTerrainCfg` | **identical class name** | 0.10, amplitude 0–0.009, num_waves 5 |

`TerrainGeneratorCfg` carries `size=(4.0, 4.0)`, `border_width=8.0`,
`num_rows=4`, `num_cols=8`, `difficulty_range=(0.0, 1.0)`, and all of those
fields exist in `isaaclab.terrains.TerrainGeneratorCfg`. The only field with no
counterpart is **`add_lights=True`** (mjlab-only; IsaacLab lighting is a
separate `AssetBaseCfg`).

`terrain_levels` → `mdp.terrain_levels_vel` exists in both. Promotion is
"walked further than `size[0] / 2`" = 2.0 m; demotion is "walked less than half
the commanded distance". The env sets `max_init_terrain_level = 1`.

`MICRO_ROUGH_TERRAINS_CFG_MEDIUM` (same five classes, larger amplitudes) is
defined and referenced nowhere — port it or drop it.

⚠ One thing to fix while re-expressing it, independent of framework: on this
ladder all four variants plateaued at terrain level **1.06–1.40 out of 4**, i.e.
at the floor, so it has no power to separate anything. Re-scale so the current
plateau sits near level 2 of 4.

### 2b. The two families the study needs do not exist in either framework

`tools/terrain_geometry.py` on this branch generates them as **pure-numpy
heightfields** — no simulator, no framework — so the only Isaac-specific work is
the wrapper that turns a heightfield into a registered sub-terrain
(`SubTerrainBaseCfg` returning a trimesh). Both are zero-mean by construction,
because a net grade puts potential-energy change into the numerator of cost of
transport and shifts the posture that `body_height` / `standing_pose` /
`upright` are written against.

- **`band_limited_random`** — anisotropic band-limited random field. Takes a
  ridge `orientation_deg` and a `wavelength_m`, which is what lets a map set
  have *decorrelated* geometric content (see §2c).
- **`alternating_lateral_steps`** — one side of the path raised, the other
  lowered, sign flipping every `wavelength_m`, flat approach and exit.
  Alternating rather than one long step, because the roll demand of a step lives
  in its entry and exit transients and the sustained middle contributes nothing.

Amplitude is normalised on the **height range inside a body footprint**, not on
global RMS, and expressed as a fraction of the legs' ~25 mm of vertical travel.
Normalising RMS is the wrong scale: a band-limited Gaussian field has
peak-to-peak around 7× its RMS, so "RMS = full leg travel" produces 170+ mm
features for a robot 70 mm tall. With the footprint normalisation,
`difficulty = 1.0` means "on the hardest 5% of the map the legs are at the edge
of their travel", which is checkable and mechanically meaningful.

### 2c. Measured: which terrain actually demands which axis

`tools/terrain_content.py` measures a heightfield's differential-rotation
content per axis. Running it at wavelength 0.34 m, difficulty 1.0:

| ridge orientation | twist | curvature | twist / curvature |
|---|---|---|---|
| 0° (ridges across the path) | 0.033 | **0.148** | 0.22 |
| 45° | **0.117** | 0.060 | 1.96 |
| 60° | 0.117 | 0.028 | 4.16 |
| 90° (ridges along the path) | 0.069 | 0.008 | 8.87 |

**Twist peaks at oblique orientations, around 45–60°, not at 90°.** Ridges
running exactly along the path give a lateral profile identical at every x, so
the front and rear axles see the same roll demand and the differential is zero —
that is the "sustained left-high / right-low" case, which a rigid trunk handles
by rolling as a whole. So the terrain that demands a roll spine is *oblique*,
not lateral. The ratio sweeps 0.22 → 8.87 across orientation, which is ample
range for a regression.

Randomising orientation and wavelength across a 24-map set gives regressor
correlations of −0.401 (twist vs curvature) — identifiable. An isotropic,
single-scale map set looks varied and is not.

⚠ **A heightfield cannot supply an independent yaw/path regressor.** Measured
across the same 24 maps, the path proxy correlates −0.786 with curvature,
because on open ground both are essentially functions of ridge orientation.
Real yaw demand needs features a heightfield does not have — walls, pillars,
gaps — and also a different command protocol, since the robot has to be
commanded along a turning trajectory rather than straight. Fit the regression on
(twist, curvature) only, and treat yaw as needing its own terrain family if it
is wanted at all.

---

## 3. Sensors

| sensor | current | IsaacLab | verdict |
|---|---|---|---|
| `terrain_scan` | `RayCastSensorCfg` + `GridPatternCfg`, size (0.40, 0.30) m at 0.05 m → **9 × 7 = 63 rays**, `ray_alignment="yaw"`, `max_distance=0.30`, frame = root body, `include_geom_groups=(0,)` | `RayCasterCfg` + `patterns.GridPatternCfg` with `attach_yaw_only=True` | **close to 1:1** |
| `foot_height_scan` | `TerrainHeightSensorCfg` + `RingPatternCfg.single_ring(radius=0.008, num_samples=4, include_center=True)` attached to four **MuJoCo sites** | none — IsaacLab has no site concept, no ring pattern, and no `TerrainHeightSensorCfg` | **no counterpart.** It is critic-only, so the cheapest answer is to drop it and drop the `foot_height` critic observation with it |
| `feet_ground_contact` | `ContactSensorCfg` with `primary=ContactMatch(mode="geom", pattern=FOOT_GEOM_NAMES)`, `secondary=ContactMatch(mode="body", pattern="terrain")`, `fields=("found","force")`, `reduce="netforce"`, `track_air_time=True` | `ContactSensorCfg(prim_path=..., filter_prim_paths_expr=[...])` giving `net_forces_w` / `force_matrix_w` / `current_air_time` | **rewrite.** There is no `found` boolean and no `reduce="netforce"`; derive contact as `norm(net_forces_w) > threshold`. Geom-level matching becomes prim/body-level in USD |
| `nonfoot_ground_touch` | same, on `rl_body_collision` and `rl_battery_collision` | same as above | **rewrite** |

Contact sensing is the widest of these: it drives three reward sub-terms
(`feet_air_time`, `moving_contact_pattern`, `undesired_contacts`), the
`illegal_contact` termination, and the hand-rolled air-time state machine in the
reward. IsaacLab's `data.current_air_time` may let that state machine be deleted.

Also delete on port: `cfg.sim.njmax`, `nconmax`, `mujoco.ccd_iterations`,
`contact_sensor_maxmatch` — all MuJoCo-only.

---

## 4. Actuator and environment mechanics

**The actuator maps almost exactly.** mjlab's `DcMotorActuatorCfg` and
IsaacLab's `DCMotorCfg` are the same model (explicit PD plus a torque–speed
clip), and the values carry over unchanged:

| field | value | note |
|---|---|---|
| `stiffness` (KP) | 1.0 | |
| `damping` (KD) | **0.0** | changed from 0.045 on 2026-09-20; whether that is right has never been validated |
| `effort_limit` | 0.129 N·m | 0.60 × stall |
| `saturation_effort` | 0.215 N·m | XL330-M077 stall at 5.0 V |
| `velocity_limit` | 32.086 rad/s | 0.80 × no-load 40.11 rad/s |
| `armature` | 2e-4 kg·m² | |
| `frictionloss` → `friction` | 0.010 N·m | field renamed |

**`encoder_bias` has no counterpart and will disappear silently.** mjlab
threads a per-environment `data.encoder_bias` through **both** sides:

- observation — `joint_pos_rel(biased=True)` reads `data.joint_pos_biased`
- action — `JointPositionAction.apply_actions` subtracts it before calling
  `set_joint_position_target`

and the 09-20 config asserts hard that the actor's `joint_pos` is `biased=True`
while the critic's is not. In IsaacLab this needs a custom observation term, a
custom startup event, and a custom `apply_actions`. Porting term-by-term will
drop it without any error, and the only symptom is a quietly worse sim-to-real
transfer.

**`decimation`** — the existing code reads

```python
if hasattr(cfg.sim, "dt") and hasattr(cfg, "decimation"):
  cfg.decimation = max(1, int(round((1.0 / 60.0) / float(cfg.sim.dt))))
```

In mjlab this guard is **always False**, because `SimulationCfg` has no `dt`
(the timestep is `cfg.sim.mujoco.timestep`), so every run to date has been at
the inherited `decimation = 4` → **50 Hz**, not the 60 Hz the comment claims.
IsaacLab's `sim.dt` does exist, so this code would begin working after a port —
which is a silent behaviour change in the other direction. Set the control rate
explicitly rather than letting it depend on which framework is underneath.
For reference, the hardware loop tops out at **~22–25 Hz**.

**Event term names and parameters all differ:**

| mjlab | IsaacLab |
|---|---|
| `foot_friction` | `randomize_rigid_body_material` |
| `base_com` | `randomize_rigid_body_com` |
| `push_robot` | `push_by_setting_velocity` |
| `reset_base` | `reset_root_state_uniform` |
| `reset_robot_joints` | `reset_joints_by_scale` |

Note that on flat terrain at sim2real stage 0 — the configuration every existing
policy was trained in — only `encoder_bias` (±0.0015 rad) and
`ik_consistent_reset` are active; `foot_friction`, `base_com` and `push_robot`
are all removed. So there is very little domain randomisation to port.

**Env assembly and registration.** `make_velocity_env_cfg()` is an mjlab factory
that supplies ~456 lines of velocity-task configuration for free, mutated
afterwards by name; IsaacLab uses class inheritance from
`LocomotionVelocityRoughEnvCfg` instead. `register_mjlab_task(...)` × 9 becomes
`gymnasium.register(...)`, though eight of the nine task ids are aliases of the
same two configurations. `rl_cfg.py` (53 lines): `RslRlModelCfg(hidden_dims=,
distribution_cfg=)` → `RslRlPpoActorCriticCfg(actor_hidden_dims=,
critic_hidden_dims=, init_noise_std=)`; same `rsl_rl` underneath.

---

## 5. Observations, reward, actions

**The 09-20 rigid observation contract** (33-dim actor) is the one to port, and
the order is part of the contract:

```
base_ang_vel 3 · projected_gravity 3 · joint_pos 8 (biased) ·
joint_vel 8 · actions 8 · command 3                            = 33
```

`base_lin_vel` and `height_scan` are deliberately **removed from the actor** —
no hardware provides either — and remain critic-only. `_validate_observation_contract()`
raises at config-construction time if the term names or order drift, if the
actor's `joint_pos` is not biased, if the critic's is, or if any of
`base_lin_vel` / `height_scan` / the four `foot_*` terms leak into the actor.
**Port that validator early**; it is the cheapest guard against the exact class
of mistake that made the existing rough policies undeployable.

Critic (flat) is 60-dim: the actor's terms plus `base_lin_vel` 3, `foot_height`
4, `foot_air_time` 4, `foot_contact` 4, `foot_contact_forces` 12, with unbiased
`joint_pos`. Dropping `foot_height` with `foot_height_scan` (§3) takes it to 56.

**Reward** — a single composite term, `microtaur_scaled_walk`, weight 1.0, with
18 internally-weighted sub-terms for rigid. 298 lines, of which only **22 read
simulator state**. Three renames (`actuator_force` → `applied_torque`,
`root_link_lin_vel_b` → `root_lin_vel_b`, `root_link_ang_vel_b` →
`root_ang_vel_b`) and one rewrite (contact booleans, §3) cover it.

**Action** — 8-dim, `requested = stand + gain · radians(30°) · action + bias`,
then a three-stage predictive limit filter: per-joint clamp at stand ± 0.75 rad
with 8° margin; `common` and `diff` each limited to 0.52 rad with 10° margin;
and an L1 diamond `|common| + |diff| ≤ 0.70 rad`, all projected forward over a
0.10 s horizon. One fixed step of action delay.

---

## 6. What ports unchanged

This is the part worth knowing before estimating anything:

| what | lines | why it ports free |
|---|---|---|
| `microtaur_kinematics.py` — closed-form five-bar FK and IK, numpy and batched torch | **601** | imports only `math`, `numpy`, `torch`; zero simulator references; its `self_test()` runs as-is |
| three-stage predictive limit filter | **217** | operates on `(position, velocity, requested)` tensors and config scalars, nothing else |
| reward math | **188** | pure tensor algebra on already-fetched state |
| IK-reset sampling and root-height math | **128** | uniform sampling, an IK call, zero-mean jitter, a quaternion from yaw |
| tuning constants (motor table, filter limits, reward scales, command stages, IK ranges, tracking sigmas) | ~370 | numbers; only the actuator field *names* change |
| `tools/terrain_geometry.py`, `tools/terrain_content.py` (this branch) | ~330 | pure numpy by construction |

So roughly **1500 lines of algorithm survive any port untouched** — effectively
the whole intellectual content of the environment. What has to be rebuilt is the
plumbing between that algorithm and a simulator.

---

## 7. Teacher–student

The study needs a map-privileged teacher and a deployable student. IsaacLab and
mjlab both drive `rsl_rl`, and `rsl_rl 5.4.2` already ships the machinery:
`algorithms/distillation.py` (`Distillation` — the student drives, the teacher
labels, i.e. on-policy DAgger at β = 0, with `gradient_length = 15` TBPTT),
`runners/distillation_runner.py`, and `models/rnn_model.py` so the student can be
a GRU or LSTM. A PPO checkpoint loads directly as the teacher.

mjlab has **no** distillation config dataclasses (`grep distillation mjlab/`
→ zero hits), so they would have to be written there. IsaacLab is likely to ship
them in `isaaclab_rl.rsl_rl` — **verify this on the Isaac machine**, since it is
one of the few places where the newer framework plausibly saves real work.

A recurrent student is not a nicety here. The 09-20 contract removed
`base_lin_vel` from the actor and replaced it with nothing: no observation
history (`delay_min_lag = delay_max_lag = 0`, `history_length` at its default of
0), no recurrence, no state estimator. The actor therefore has **no way to
estimate its own forward speed**, which is plausibly why a 128-point lookup
table reproduces it so closely (R² = 1.00 against RL leg angles on pitch).

**Do not use `src/microtaur_velocity/distill_reliable/`** (1821 lines, 20 files):
it is complete but has never been executed — no datasets, no student checkpoints,
no logs in any branch — its student is a stateless ~7k-parameter MLP, and its
DAgger loop is human-in-the-loop by design. Keep only `mjlab_utils.py`, which
several analysis scripts import for teacher loading. The four scripts under
`scripts/` named `*student*` import `microtaur_velocity.distill.student_policy`,
a module that exists in no branch, and crash on import.

---

## 8. Ordered checklist

1. **Loop-closure feasibility spike** (§0). One robot standing, measured
   constraint drift and jitter at the real scale. Everything else is contingent
   on this.
2. MJCF → USD for rigid, with the 56 lines of spec-patching relocated (§1), and
   verified against the MJCF rather than trusted.
3. Actuator as `DCMotorCfg` with the seven values from §4.
4. Contact sensing rebuilt on `net_forces_w` thresholding; `terrain_scan` as
   `RayCasterCfg`; `foot_height_scan` dropped (§3).
5. `encoder_bias` reimplemented on both the observation and the action side (§4)
   — the item most likely to be silently lost.
6. Observation contract and its validator (§5), then the reward with its three
   renames.
7. Terrain: the existing menu re-expressed (§2a) with the ladder re-scaled, and
   the two new families wrapped from `tools/terrain_geometry.py` (§2b).
8. Teacher–student with a recurrent student, after verifying which distillation
   config classes the installed IsaacLab provides (§7).

Items 2–7 were estimated at **6–11 weeks** of engineering, assuming Isaac Sim is
already installed, and producing no new results on their own. Item 1 is what
determines whether that estimate is meaningful.
