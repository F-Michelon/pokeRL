#!/bin/bash
#SBATCH --job-name=pokerl-full
#SBATCH --account=hir@h100
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=pokerl-full-%j.log

# Usage: sbatch cluster/full-game.slurm smoke|train|eval [training-run-directory]
set -eo pipefail
# Load the user's H100 module stack. The V100 wrapper inherits its own loaded stack.
if [[ "${POKERL_GPU_ARCH:-h100}" == h100 ]]; then
  module purge
  module load arch/h100
  module load pytorch-gpu/py3/2.8.0
fi
# Bootstrap and training metadata both need git; module purge may remove it.
if ! command -v git >/dev/null 2>&1; then
  module load git
fi
command -v git >/dev/null 2>&1 || { echo 'git is unavailable; load a Git module.' >&2; exit 2; }
set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
venv_dir="${POKERL_VENV:-$PWD/.venv}"
[[ -f "$venv_dir/bin/activate" && -x "$venv_dir/bin/python" ]] || {
  echo "Missing virtual environment: $venv_dir. Build it with the matching GPU modules first." >&2
  exit 2
}
source "$venv_dir/bin/activate"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$PWD/.playwright-browsers}"
python -c 'import sys; print("Python:", sys.executable, flush=True)'
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pokerl-mpl-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1
mode="${1:-smoke}"
case "$mode" in (smoke|train|eval) ;; (*) echo 'Expected smoke, train, or eval' >&2; exit 2;; esac
# No dependency downloads or cloning from compute nodes.
[[ -d upstream/.git && -f rl/upstream-source.js && -f full_game/index.html ]] || { echo 'Run python scripts/bootstrap.py before submitting.' >&2; exit 2; }
python scripts/bootstrap.py
python -c 'import torch, playwright, sb3_contrib; assert torch.cuda.is_available(), "No CUDA GPU"; print(torch.__version__, torch.cuda.get_device_name()); print(torch.ones(1,device="cuda").sum().cpu())'

case "$mode" in
  (smoke)
    out="results/full-smoke-${SLURM_JOB_ID}"
    srun python -m tests.full_training_smoke
    srun python -m training.train_full_game --device cuda --envs 2 --steps 512 --duration 30 --max-seconds 300 --output "$out"
    srun python -m training.evaluate_full_game --model "$out/model.zip" --episodes 2 --output "$out/evaluation.json" --baselines
    echo "SMOKE PASSED: $out"
    ;;
  (train)
    out="results/full-train-${SLURM_JOB_ID}"
    srun python -m training.train_full_game --device cuda --envs "${POKERL_ENVS:-4}" --steps "${POKERL_STEPS:-100000}" --duration "${POKERL_DURATION:-120}" --max-seconds "${POKERL_MAX_SECONDS:-6000}" --output "$out"
    echo "TRAINING SAVED: $out/model.zip"
    ;;
  (eval)
    run="${2:?Pass the training run directory as the second argument}"
    [[ -f "$run/model.zip" && -f "$run/run.json" ]] || { echo "Missing checkpoint or run metadata: $run" >&2; exit 2; }
    srun python -m training.evaluate_full_game --model "$run/model.zip" --episodes "${POKERL_EPISODES:-20}" --baselines --output "$run/evaluation-${SLURM_JOB_ID}.json"
    ;;
esac
