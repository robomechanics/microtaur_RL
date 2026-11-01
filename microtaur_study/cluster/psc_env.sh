# PSC Bridges-2 paths for the Microtaur jobs, sourced by the cluster scripts. Everything lives under
# $PROJECT/microtaur (persistent, not backed up). The allocation is the user's default SLURM account, so no account
# id is written in any file or printed in the shell.
export MT_PROJECT="${MT_PROJECT:-${PROJECT:?PROJECT unset: run on Bridges-2}/microtaur}"
export MT_REPO="${MT_REPO:-$MT_PROJECT/microtaur_RL}"                     # git clone (GitHub works from PSC)
export MT_RUNS="${MT_RUNS:-$MT_PROJECT/runs}"                             # pipeline --runs root (logs/rsl_rl/... inside)
# Shared read-only Isaac Lab image of the PSC lab guide: Isaac Sim 5.1, isaaclab 0.54.2 (Isaac Lab 2.3.2),
# rsl-rl-lib 3.1.2, torch 2.7.0+cu128, Python 3.11 (local stack: Isaac Lab 2.3.0, rsl-rl-lib 3.0.1).
export MT_SIF="${MT_SIF:-/ocean/containers/ngc/isaac/isaac-lab_2.3.2.sif}"
export MT_PYDEPS="${MT_PYDEPS:-$MT_PROJECT/pydeps}"   # optional pip --target overrides, first on PYTHONPATH (empty = image)
export MT_KIT_CACHE="${MT_KIT_CACHE:-$MT_PROJECT/kit-cache}"              # shader cache, copied back after clean exits
export MT_GPU="${MT_GPU:-l40s-48}"   # L40S 1 SU per GPU-hour (default); h100-80 2 SU/h; v100 does not run Kit
case "$MT_GPU" in l40s*) _mt_cpus=8 ;; *) _mt_cpus=5 ;; esac
export MT_CPUS="${MT_CPUS:-$_mt_cpus}"
