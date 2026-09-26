#!/usr/bin/env bash
# conda env "spine": Isaac Sim 5.1 + IsaacLab v2.3.0 (rsl-rl-lib 3.0.1) + mujoco + microtaur_study
set -euo pipefail
ROOT=/home/rml3/Documents/ben/spine
source /home/rml3/anaconda3/etc/profile.d/conda.sh
conda create -y -n spine python=3.11
conda activate spine
pip install --upgrade pip
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
if [ ! -d "$ROOT/IsaacLab" ]; then
  git clone https://github.com/isaac-sim/IsaacLab.git "$ROOT/IsaacLab"
fi
cd "$ROOT/IsaacLab" && git checkout -q v2.3.0
export OMNI_KIT_ACCEPT_EULA=YES
./isaaclab.sh --install rsl_rl
pip install mujoco
pip install --no-deps -e "$ROOT/microtaur_RL/microtaur_study"
python -c "import torch, importlib.metadata as m; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); [print(p, m.version(p)) for p in ('isaacsim','isaaclab','isaaclab_rl','rsl-rl-lib','mujoco','numpy')]"
