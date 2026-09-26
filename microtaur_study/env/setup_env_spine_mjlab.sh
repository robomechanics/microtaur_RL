#!/usr/bin/env bash
# conda env "spine-mjlab": mjlab 1.6 (rsl-rl-lib 5.4.2) + microtaur_study
set -euo pipefail
source /home/rml3/anaconda3/etc/profile.d/conda.sh
conda create -y -n spine-mjlab python=3.11
conda activate spine-mjlab
pip install "mjlab==1.6.0" matplotlib
pip install --no-deps -e /home/rml3/Documents/ben/spine/microtaur_RL/microtaur_study
python -c "import mjlab, importlib.metadata as m; print('mjlab', m.version('mjlab'), 'rsl-rl-lib', m.version('rsl-rl-lib'), 'mujoco', m.version('mujoco'))"
