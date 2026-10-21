#!/bin/bash
#SBATCH --job-name=pokerl-full
#SBATCH --account=hir@h100
#SBATCH --constraint=h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=pokerl-full-%j.log

# Usage: ... smoke | train [config.json] | eval|watch [training-run-directory]
set -eo pipefail
set -u
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pokerl-mpl-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1

source .venv/bin/activate

mode="${1:-smoke}"
case "$mode" in (smoke|train|eval|watch) ;; (*) echo 'Expected smoke, train, eval, or watch' >&2; exit 2;; esac
# Bootstrap once before submission; compute nodes need neither Git nor downloads.
[[ -d upstream/js && -f rl/upstream-source.js && -f full_game/index.html && -f full_game/main.js && -f full_game/game.js ]] || { echo 'Missing prepared game files. Run python scripts/bootstrap.py on the login/preparation node before submitting.' >&2; exit 2; }
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
    train_args=(--device cuda --max-seconds "${POKERL_MAX_SECONDS:-6000}" --output "$out")
    config="${2:-${POKERL_CONFIG:-}}"
    if [[ -n "$config" ]]; then train_args+=(--config "$config"); fi
    # Only explicit overrides beat a config (or inherited resume settings).
    for key in ENVS STEPS DURATION SEED GAMMA ENT_COEF N_STEPS BATCH_SIZE N_EPOCHS LEARNING_RATE TARGET_KL CHECKPOINT_STEPS; do
      variable="POKERL_${key}"
      if [[ -n "${!variable:-}" ]]; then
        option="${key,,}"
        train_args+=("--${option//_/-}" "${!variable}")
      fi
    done
    if [[ -n "${POKERL_RESUME:-}" ]]; then train_args+=(--resume "$POKERL_RESUME"); fi
    srun python -m training.train_full_game "${train_args[@]}"
    echo "TRAINING SAVED: $out/model.zip"
    ;;
  (eval)
    run="${2:?Pass the training run directory as the second argument}"
    model="${POKERL_EVAL_MODEL:-$run/model.zip}"
    [[ -f "$model" && -f "$run/run.json" ]] || { echo "Missing checkpoint or run metadata: $model / $run/run.json" >&2; exit 2; }
    eval_args=(--model "$model" --episodes "${POKERL_EPISODES:-20}"
      --seed-start "${POKERL_EVAL_SEED:-1000000000}" --output "$run/evaluation-${SLURM_JOB_ID}.json")
    if [[ "${POKERL_BASELINES:-1}" == 1 ]]; then eval_args+=(--baselines); fi
    if [[ "${POKERL_EVAL_CHECKPOINTS:-0}" == 1 ]]; then eval_args+=(--checkpoints); fi
    if [[ -n "${POKERL_EVAL_DURATION:-}" ]]; then eval_args+=(--duration "$POKERL_EVAL_DURATION"); fi
    srun python -m training.evaluate_full_game "${eval_args[@]}"
    ;;
  (watch)
    run="${2:?Pass the training run directory as the second argument}"
    srun python -m training.watch_full_game --model "$run/model.zip" \
      --headless --record "$run/watch-${SLURM_JOB_ID}" \
      --episodes "${POKERL_EPISODES:-1}" --seed "${POKERL_WATCH_SEED:-1000000000}" \
      --speed "${POKERL_WATCH_SPEED:-1}"
    ;;
esac
