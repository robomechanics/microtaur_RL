# Microtaur

## Files to copy

Project root assumed:

```powershell
C:\Users\prome\robotics\microtaur
```

Copy these files:

```powershell
Copy-Item .\src\microtaur_velocity\__init__.py C:\Users\prome\robotics\microtaur\src\microtaur_velocity\__init__.py -Force
Copy-Item .\src\microtaur_velocity\env_cfgs.py C:\Users\prome\robotics\microtaur\src\microtaur_velocity\env_cfgs.py -Force
Copy-Item .\src\microtaur_velocity\microtaur_constants.py C:\Users\prome\robotics\microtaur\src\microtaur_velocity\microtaur_constants.py -Force
Copy-Item .\src\microtaur_velocity\minitaur_pair_action.py C:\Users\prome\robotics\microtaur\src\microtaur_velocity\minitaur_pair_action.py -Force
Copy-Item .\src\microtaur_velocity\rl_cfg.py C:\Users\prome\robotics\microtaur\src\microtaur_velocity\rl_cfg.py -Force
Copy-Item .\microtaur_xmls\rigid_microtaur\robot_modified.xml C:\Users\prome\robotics\microtaur\microtaur_xmls\rigid_microtaur\robot_modified.xml -Force
```

## What changed

1. `minitaur_pair_action.py` interprets the 8 policy outputs as:

```text
leg1_swing, leg1_extension,
leg2_swing, leg2_extension,
leg3_swing, leg3_extension,
leg4_swing, leg4_extension
```

and converts them to motor targets:

```text
a_motor = swing + extension
e_motor = swing - extension
```

2. `env_cfgs.py` uses a nonzero forward command range:

```python
MICROTAUR_LIN_VEL_X_RANGE = (0.08, 0.25)
```

so standing still is no longer a good solution and it doesnt end up learning shuffly motions.

3. `action_rate_l2` is scaled down so periodic leg motion is allowed.

4. The XML disables detailed mesh collisions by default and keeps simple body/foot collision primitives active.

## First run

```powershell
cd C:\Users\prome\robotics\microtaur

$env:MICROTAUR_VARIANT="rigid_microtaur"
$env:PYTHONPATH="C:\Users\prome\robotics\microtaur\src"

uv pip install -e .

uv run --active --no-sync play Mjlab-Velocity-Flat-microtaur_velocity --agent zero
```

If the robot stands correctly, train:

```powershell
uv run --active --no-sync train Mjlab-Velocity-Flat-microtaur_velocity --gpu-ids "[0]" --env.scene.num-envs 512 --agent.max-iterations 1000
```

Then scale up:

```powershell
uv run --active --no-sync train Mjlab-Velocity-Flat-microtaur_velocity --gpu-ids "[0]" --env.scene.num-envs 2048 --agent.max-iterations 3000
```

## If legs move in the wrong direction

Flip signs without editing code:

```powershell
$env:MICROTAUR_SWING_SIGNS="1,-1,1,-1"
$env:MICROTAUR_EXTENSION_SIGNS="1,1,1,1"
```

or edit these in `microtaur_constants.py`:

```python
MINITAUR_SWING_SIGNS
MINITAUR_EXTENSION_SIGNS
```

## If it still stands still

Increase the paired action scales:

```powershell
$env:MICROTAUR_PAIR_SWING_SCALE="0.30"
$env:MICROTAUR_PAIR_EXTENSION_SCALE="0.34"
```

and rerun a fresh training.

## If it jitters/explodes

Reduce the action scales:

```powershell
$env:MICROTAUR_PAIR_SWING_SCALE="0.16"
$env:MICROTAUR_PAIR_EXTENSION_SCALE="0.20"
```

or reduce entropy in `rl_cfg.py` from `0.003` to `0.001`.
