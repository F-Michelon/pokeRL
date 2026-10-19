#!/bin/bash
#SBATCH --job-name=pokerl-vision
#SBATCH --account=hir@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=pokerl-vision-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
if [[ -n "${POKERL_VENV:-}" ]]; then source "$POKERL_VENV/bin/activate"; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
srun python -m training.train_vision_student --device cuda --data "${POKERL_DATA:-results/teacher}" --epochs "${POKERL_EPOCHS:-30}" --output "results/vision-${SLURM_JOB_ID}/student"
