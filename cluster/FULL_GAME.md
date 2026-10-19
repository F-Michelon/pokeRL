# Full-game training on Jean Zay (project hir)

Use `full-game.slurm`, not the older `train-h100.slurm` / `train-v100.slurm`:
those older jobs still train the small arena. No jobs have been submitted by Codex.

## Prepare once, before submitting

The scripts activate `$PWD/.venv` after loading the GPU modules. Override the
path with `POKERL_VENV=/absolute/path/to/venv`. Build the venv on Jean Zay using
the same module stack as the job; do not copy the laptop's CPU environment.
The H100 script loads `arch/h100` and `pytorch-gpu/py3/2.8.0`. For V100, load
your V100-compatible modules before creating the venv and before submitting.

```bash
# H100 setup. For V100, substitute your V100-compatible module stack.
module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0
module load git
python -m venv --system-site-packages .venv
source .venv/bin/activate
# Reuse the module's CUDA-enabled PyTorch; install the RL/browser dependencies.
export UV_CACHE_DIR="$PWD/.cache/uv"
export PIP_CACHE_DIR="$PWD/.cache/pip"
python -c 'import torch; print(torch.__version__, torch.__file__)'
python -m pip install 'gymnasium==1.2.3' 'stable-baselines3==2.7.1' 'sb3-contrib==2.7.1' 'playwright>=1.55,<2' 'pillow>=11,<13'
export PLAYWRIGHT_BROWSERS_PATH="$PWD/.playwright-browsers"
python -m playwright install chromium
python scripts/bootstrap.py
mkdir -p results
module list 2> results/jz-modules.txt
python -m pip freeze > results/jz-packages.txt
```

The scripts load the Git module if `git` is missing: bootstrap and training metadata
both use it. The observed job failures were missing `playwright` and missing `git`;
the venv addresses the Python dependency, while the Git module addresses the latter.
Do not run an unconstrained `uv pip install torch`: it may download another CUDA
stack into your home cache. A home-cache `Disk quota exceeded` error is a storage
failure, not a GPU failure. Put dependency caches on compute-visible project/work
storage with sufficient quota (as above). `uv cache clean` can remove disposable
uv cache entries from the default home cache when no other uv install is running;
changing the cache location alone does not reclaim the already-used home space.
No packages are installed inside a job. If you use both GPU families, keep separate
venvs built with their respective module stacks and select one with `POKERL_VENV`.
The default browser installation path is `$PWD/.playwright-browsers`; an explicit
`PLAYWRIGHT_BROWSERS_PATH` overrides it.

If the login node cannot download packages/assets, stage these on a permitted
transfer/prepost node. Transfer `upstream/`, the generated bundles, installed packages,
and Chromium installation to compute-visible storage. Browser system libraries
must already be installed by the site; a missing-library error in the smoke job
needs a compatible site browser (`POKERL_CHROME=/absolute/path/to/chromium`) or
an appropriate site container. Do not attempt package downloads inside the job.
`POKERL_VENV`, `PLAYWRIGHT_BROWSERS_PATH` and optional `POKERL_CHROME` are inherited
by sbatch. Submit all commands from the repository root.

## 1. Short test job (H100)

```bash
smoke_job=$(sbatch --parsable --qos=qos_gpu_h100-dev --time=00:15:00 cluster/full-game.slurm smoke)
echo "$smoke_job"
squeue -j "$smoke_job"
# After completion:
sacct -j "$smoke_job" --format=JobID,State,ExitCode,Elapsed
cat "results/full-smoke-${smoke_job}/evaluation.json"
```

Requires job `COMPLETED`, exit `0:0`, and `SMOKE PASSED` in
`pokerl-full-<jobid>.log`. It checks CUDA, Chromium startup, independent replay,
real upgrade/reroll, move switch, item and two-step fusion callbacks, reset/terminal
behavior; then runs 512 transitions through two environments, saves/reloads a
checkpoint, and evaluates it against three baselines. These are infrastructure
checks, not an expected performance win.

## 2. First training

```bash
train_job=$(sbatch --parsable --dependency="afterok:${smoke_job}" cluster/full-game.slurm train)
echo "$train_job"
```

Default: one H100, four independent full-game Chromium workers, 100,000 transitions,
120 simulated seconds per episode, fresh guest Squirtle each episode, no persistent
upgrades. It calls the actual upstream update loop at fixed 60 Hz, six frames per
movement action. Rendering and audio are disabled. This is accelerated structured
state training, not screenshot distillation or normal-speed keyboard automation.

Checkpoint: `results/full-train-<jobid>/model.zip`; metadata: `run.json`;
per-worker episode logs: `env-*.monitor.csv`; intermediate checkpoints every
approximately 10,000 transitions. A 6,000-second training timer saves a final model
before the default two-hour allocation ends. Read `new_timesteps` in `run.json`:
a wall-limited run may finish fewer than the requested transitions. Environment
startup/reset time and slow individual operations still count toward Slurm's limit.

Optional changes before submission:

```bash
POKERL_STEPS=200000 POKERL_ENVS=4 POKERL_DURATION=180 \
  sbatch --dependency="afterok:${smoke_job}" cluster/full-game.slurm train
```

Do not resume old arena or screenshot models: their observations/actions differ.
For future continuation of a full-game checkpoint, `training.train_full_game`
supports `--resume <model.zip>` with a **new** output directory. It does not preserve
mid-episode trajectories or exact RNG state across a resumed training run.

## 3. Evaluate the first training

```bash
eval_job=$(sbatch --parsable --dependency="afterok:${train_job}" \
  cluster/full-game.slurm eval "results/full-train-${train_job}")
echo "$eval_job"
# Once completed:
cat "results/full-train-${train_job}/evaluation-${eval_job}.json"
```

20 paired held-out seeds, starting at 1,000,000,000, with the training run's horizon
and Pokémon. Evaluates PPO, standing + first menu choice, circular movement +
first menu choice, and random valid actions. Reports per-episode and mean survival,
XP, damage, deaths, reward, upgrades, active-move switches, item choices and fusions.
Evaluation inference currently uses CPU, though this common job reserves one GPU
for a consistent first smoke/train/evaluation deployment. Release/adjust that GPU
allocation for later large evaluations after the first workflow is validated.

The environment implements `action_masks()` in each worker, and evaluation passes
the mask to every prediction as required by
[MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).

## V100 alternative and scheduler options

Replace `cluster/full-game.slurm` with `cluster/full-game-v100.slurm` in the three
submission commands. It uses `hir@v100`, a V100 32 GB constraint, ten CPUs and the
same training code. Load your V100-compatible PyTorch modules before submitting;
the wrapper inherits them and does not load the H100 modules. Do not mix smoke on one GPU architecture with an untested
Python/CUDA environment on the other.

The H100 script uses `hir@h100` and `--constraint=h100`, following the
[IDRIS guide](https://www.idris.fr/media/eng/ia/guide_nouvel_utilisateur_ia-eng.pdf).
The scripts inherit the default QoS. The smoke command explicitly selects your
`qos_gpu_h100-dev` QoS; omit or replace it when using V100. Keeping that choice on
the smoke command avoids applying a development QoS to the two-hour training job. Use `sbatch --qos=<your-allowed-qos>` if your
allocation requires one. Check your current allocation with `sinfo`/site guidance.
No specific module version or optional development QoS is assumed.

## What the policy controls

50 masked actions, with unavailable actions excluded from sampling:

- 0–44: nine movement directions (including idle), combined with keeping the
  current attack or choosing active slot 1–4.
- 45–48: choose a displayed upgrade/item/fusion option. Only available options
  are enabled, only when a menu event is pending.
- 49: reroll level-up choices, only while a reroll is available.

Approaching/staying at chests, springs, altars and rifts uses normal game movement
and interaction logic. Choice observations include upgrade kind/stat, move stats,
type/kind and pinned-catalog identity, item identity, and all attacks in fused slots.
Other observations include player/enemies/projectiles/pickups, nearby world objects
and their used/charge/open state, collision/water grid, hazard geometry/timers, rift
location/state, weather, equipped item and action mask. Nearest entity lists are
bounded at 20: this is still a partial observation, not omniscience.

This is a first joint tactical/strategic policy with event-gated actions, not two
separately trained hierarchical policies. The first short horizon will not naturally
exercise every rare boss/legendary event; the smoke test injects real choice events
to check callbacks, and longer training is needed for strategic competence. Picking
a Pokémon and buying persistent account upgrades are not learned in this first run.
Online play and screenshot transfer are not part of these jobs.
