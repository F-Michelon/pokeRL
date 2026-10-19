# Full-game training on Jean Zay

[README](../README.md) · [Pipeline](pipeline.md) · [Log reference](training-logs.md)

## Prepare before submitting

Run commands from the repository root on compute-visible storage. The current
scripts activate `.venv/bin/activate` and do not load modules themselves. Use a
Python/PyTorch environment compatible with the selected GPU. If your venv depends
on site modules for shared libraries, load those before submission so the job
inherits the environment. A venv does not replace required CUDA/system libraries.

If your environment and browser are already installed, do not reinstall them.
Preparation on a node allowed to download dependencies is:

```bash
source .venv/bin/activate
# Only if dependencies are missing, with the intended GPU torch already installed:
python -m pip install -r requirements.txt
export PLAYWRIGHT_BROWSERS_PATH="$PWD/.playwright-browsers"
python -m playwright install chromium
python scripts/bootstrap.py
mkdir -p results
python -m pip freeze > results/jz-packages.txt
```

Creating the GPU-compatible venv is site/environment dependent; do not copy a
laptop CPU venv. For offline compute nodes, stage packages, browser binaries and
game files using a node that can download them. Keep `PLAYWRIGHT_BROWSERS_PATH`
exported when submitting if you used a custom installation path. Alternatively,
set `POKERL_CHROME` to a compatible installed Chromium executable.

Bootstrap is a one-time preparation step for unchanged inputs. Its successful
revision/hash output means it completed. It is no longer called by the full-game
batch script. There is no automatic Git pull: submit from the checkout you have
already prepared and updated.

Keep these available to the job:

- The repository source, `.venv/` and installed Chromium.
- `upstream/`, including `.git` metadata and all game assets/modules.
- Generated `rl/upstream-source.js` and `full_game/index.html`, `main.js`, `game.js`.

These generated/downloaded files are ignored by Git, so cloning this project alone
does not transfer them. For an upstream worktree, its referenced Git metadata must
also remain accessible. A standalone upstream checkout is simpler to transfer.

## Choose the batch script

| Script | Account / GPU | CPUs | Explicit QoS | Default time |
| --- | --- | --- | --- | --- |
| `cluster/full-game-v100.slurm` | `hir@v100`, `v100-32g` | 10 | `qos_gpu-dev` | 2 hours |
| `cluster/full-game.slurm` | `hir@h100`, `h100` | 16 | `qos_gpu_h100-dev` | 2 hours |

These are the values in this checkout, not a statement of current site limits.
Use `sbatch --qos=<allowed-qos> --time=<allowed-time>` to override them for your
allocation. In particular, a development QoS may not allow the default two-hour
training request. Confirm the applicable limits before a longer run.

The V100 wrapper invokes the common H100-named script with Bash. Its embedded
`#SBATCH` lines are then comments; the V100 allocation is retained. Both scripts
use the same `.venv`, so verify compatibility when changing GPU architecture.
The older `train-v100.slurm` and `train-h100.slurm` train the separate arena.

## 1. Smoke test

```bash
smoke_job=$(sbatch --parsable --time=00:15:00 cluster/full-game-v100.slurm smoke)
echo "$smoke_job"
squeue -j "$smoke_job"
```

The job checks CUDA with a real tensor operation, runs deterministic replay and
real menu/reset/terminal checks, trains for 512 transitions in two environments
with a 30-second episode horizon, then evaluates two episodes per policy.
Its learning timer is 300 seconds; browser checks and evaluation also need time.

After completion:

```bash
sacct -j "$smoke_job" --format=JobID,State,ExitCode,Elapsed
cat "pokerl-full-${smoke_job}.log"
cat "results/full-smoke-${smoke_job}/evaluation.json"
```

Require `COMPLETED`, exit `0:0` and `SMOKE PASSED`. Some menu cases are deliberately
injected by the smoke test to exercise callbacks; this is not evidence that the
policy learned to discover or use those features naturally.

## 2. Train

```bash
train_job=$(sbatch --parsable --dependency="afterok:${smoke_job}" \
  cluster/full-game-v100.slurm train)
echo "$train_job"
```

If smoke already completed successfully, the dependency can be omitted.
The job prints `TRAINING SAVED` after saving the final model.

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `POKERL_ENVS` | 4 | Parallel environments |
| `POKERL_STEPS` | 100000 | Requested total transitions across all environments |
| `POKERL_DURATION` | 120 | Maximum simulated seconds per episode |
| `POKERL_MAX_SECONDS` | 6000 | Learning wall-time budget; 0 disables it |
| `POKERL_EPISODES` | 20 | Evaluation episodes per policy |

Example of a larger experiment, after reviewing the first evaluation:

```bash
POKERL_STEPS=500000 POKERL_ENVS=4 POKERL_DURATION=120 \
  sbatch cluster/full-game-v100.slurm train
```

Increasing requested steps does not extend the learning timer or Slurm allocation.
Change those together if needed, leaving startup/save time. A Slurm hard timeout
can kill the job before the final save; intermediate checkpoints may still exist.

Monitor progress:

```bash
tail -f "pokerl-full-${train_job}.log"
```

Inspect `results/full-train-${train_job}/run.json` after completion for the actual
`new_timesteps`, which can differ from requested steps. Ctrl-C stops `tail`, not
the submitted job.

## 3. Evaluate

```bash
eval_job=$(sbatch --parsable --dependency="afterok:${train_job}" \
  cluster/full-game-v100.slurm eval "results/full-train-${train_job}")
echo "$eval_job"
```

After completion:

```bash
cat "results/full-train-${train_job}/evaluation-${eval_job}.json"
```

Evaluation runs the final model and three baselines on paired held-out seeds.
It uses CPU inference but the shared batch script still reserves and checks a GPU.
Compare `survival`, `dead`, `xp`, `damage` and `return`, not just training reward.
`damage` means HP lost. Preserve the per-episode results as well as averages.

## Common failures

| Symptom | What to check |
| --- | --- |
| `No such file or directory: 'git'` | Current full-game jobs do not invoke Git. An old script or old trainer still might; update both. Bootstrap on the preparation node still needs Git. |
| Missing `.git/HEAD` or unresolved revision | Keep upstream Git metadata, including referenced worktree metadata, accessible. |
| Upstream revision mismatch | Compare with `upstream.lock.json`; prepare the correct checkout instead of bypassing the check. |
| Missing prepared game files | Run bootstrap on the preparation node, then keep generated files on shared storage. |
| `.venv/bin/activate` missing | Submit from the repository root and prepare `.venv` there. |
| CUDA unavailable or unsupported architecture | Check the GPU allocation and the venv's PyTorch/CUDA compatibility. |
| Chromium executable missing | Export the browser installation path used during preparation, or set `POKERL_CHROME`. |
| Browser shared-library error | Use a compatible site browser/environment; downloading Chromium alone does not install its system libraries. |
| Job rejected for QoS/time | Override the script's development QoS and time with values allowed for your allocation. |
| Output already contains a run | Select a new output directory; the Python trainer refuses to overwrite `run.json`. |

Slurm snapshots the submitted script, but this project's wrapper and Python
modules are read from shared storage when executed. Keep a running experiment's
checkout stable; use separate prepared copies when experimenting with code changes.
