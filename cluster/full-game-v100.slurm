#!/bin/bash
#SBATCH --job-name=pokerl-full-v100
#SBATCH --account=hir@v100
#SBATCH --constraint=v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=pokerl-full-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
# Reuse the exact same smoke/train/eval implementation; SBATCH lines are comments here.
export POKERL_GPU_ARCH=v100
bash cluster/full-game.slurm "$@"
