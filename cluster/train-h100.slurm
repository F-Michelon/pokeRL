#!/bin/bash
#SBATCH --job-name=pokerl-h100
#SBATCH --account=hir@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=pokerl-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
if [[ -n "${POKERL_VENV:-}" ]]; then source "$POKERL_VENV/bin/activate"; fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pokerl-mpl-${SLURM_JOB_ID}"
node --version
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name()); print((torch.ones(1, device="cuda") + 1).cpu())'
python scripts/bootstrap.py
srun python -m training.train_ppo --device cuda --envs 16 --steps "${POKERL_STEPS:-1000000}" --config "${POKERL_CONFIG:-configs/arena_easy.json}" --output "results/h100-${SLURM_JOB_ID}"
python -m training.evaluate --config "${POKERL_CONFIG:-configs/arena_easy.json}" --model "results/h100-${SLURM_JOB_ID}/model.zip" --output "results/h100-${SLURM_JOB_ID}/evaluation.json"
