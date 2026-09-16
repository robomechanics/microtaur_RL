# Microtaur handoff setup

This project is an MJLab velocity task for the Microtaur / Minitaur-style robot. The important part is that the package must be installed in editable mode so MJLab can discover the task entry point from `pyproject.toml`.

## Expected repository layout

From the project root, keep this layout:

```text
microtaur/
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── .env.example
├── README.md
├── README_MINITAUR_DROPIN.md
├── microtaur_xmls/
│   └── rigid_microtaur/
│       └── robot_modified.xml
└── src/
    └── microtaur_velocity/
        ├── __init__.py
        ├── env_cfgs.py
        ├── env_cfgs_standby.py
        ├── microtaur_config.py
        ├── microtaur_constants.py
        ├── minitaur_pair_action.py
        ├── rl_cfg.py
        └── trot_test.py
```

Do not forget to commit `microtaur_xmls/rigid_microtaur/robot_modified.xml`; the task will fail at import time if the XML is missing.

## Fresh install: Windows PowerShell

```powershell
git clone <YOUR_GITHUB_REPO_URL>
cd microtaur

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .

$env:MICROTAUR_VARIANT="rigid_microtaur"
```

If PyTorch needs a specific CUDA wheel on the teammate's machine, install that PyTorch wheel first, then rerun:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Fresh install: uv option

```powershell
git clone <YOUR_GITHUB_REPO_URL>
cd microtaur

uv venv --python 3.11
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
uv pip install -e .

$env:MICROTAUR_VARIANT="rigid_microtaur"
```

## Sanity checks

Run these from the project root after installing:

```powershell
python -c "import microtaur_velocity; print('microtaur_velocity import OK')"
python -c "from microtaur_velocity.microtaur_constants import MICROTAUR_XML; print(MICROTAUR_XML)"
```

Then check the MJLab task loads:

```powershell
play Mjlab-Velocity-Flat-microtaur_velocity --agent zero
```

If using `uv run` with an active venv:

```powershell
uv run --active --no-sync play Mjlab-Velocity-Flat-microtaur_velocity --agent zero
```

## Training commands

Start small:

```powershell
train Mjlab-Velocity-Flat-microtaur_velocity --gpu-ids "[0]" --env.scene.num-envs 512 --agent.max-iterations 1000
```

Scale up after the sanity run works:

```powershell
train Mjlab-Velocity-Flat-microtaur_velocity --gpu-ids "[0]" --env.scene.num-envs 2048 --agent.max-iterations 3000
```

Rough terrain / gait variants registered by the package include:

```text
Mjlab-Velocity-Flat-microtaur_velocity
Mjlab-Velocity-Rough-microtaur_velocity
Mjlab-Velocity-Flat-microtaur_gait
Mjlab-Velocity-Rough-microtaur_gait
Mjlab-Velocity-Flat-microtaur_trot
Mjlab-Velocity-Rough-microtaur_trot
Mjlab-Velocity-Flat-microtaur_bound
Mjlab-Velocity-Rough-microtaur_bound
```

## Useful environment variables

```powershell
# Required/default morphology variant
$env:MICROTAUR_VARIANT="rigid_microtaur"

# Optional explicit XML override
$env:MICROTAUR_XML="C:\path\to\microtaur\microtaur_xmls\rigid_microtaur\robot_modified.xml"

# Switch the generic gait task between trot and bound
$env:MICROTAUR_BOUND_GAIT="0"   # trot
$env:MICROTAUR_BOUND_GAIT="1"   # bound

# Action/sign debugging
$env:MICROTAUR_SWING_SIGNS="1,-1,-1,1"
$env:MICROTAUR_EXTENSION_SIGNS="-1,1,1,-1"

# Conservative action scale debugging
$env:MICROTAUR_PAIR_SWING_SCALE="0.26"
$env:MICROTAUR_PAIR_EXTENSION_SCALE="0.21"
```

## Common failure fixes

### `ModuleNotFoundError: No module named 'microtaur_velocity'`

Run this from the project root:

```powershell
python -m pip install -e .
```

### Missing Microtaur XML assertion

Check that this exists:

```text
microtaur_xmls/rigid_microtaur/robot_modified.xml
```

Or set an explicit XML path:

```powershell
$env:MICROTAUR_XML="C:\path\to\robot_modified.xml"
```

### `play` or `train` command not found

Make sure the venv is activated and MJLab installed:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

### CUDA/PyTorch mismatch

Install the PyTorch build that matches the teammate's CUDA/driver setup, then install this project again in editable mode:

```powershell
python -m pip install -e .
```
