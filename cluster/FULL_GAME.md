# Full-game training on Jean Zay (project hir)

Use `full-game.slurm`, not the older `train-h100.slurm` / `train-v100.slurm`:
those older jobs still train the small arena. No jobs have been submitted by Codex.

## Prepare once, before submitting

From the repository root on Jean Zay, activate a site-compatible CUDA PyTorch
Python 3.12 environment and install `requirements.txt` into your own virtualenv.
For a site module containing PyTorch, a venv with `--system-site-packages` can reuse
that build. Choose the module according to the GPU: the V100 build must support
Volta; an H100-only CUDA build is not interchangeable. Do not install the laptop's
CPU lockfile on Jean Zay. Record the actual module list and package versions.

```bash
# Load your site's compatible CUDA/PyTorch module first (module name is site-specific).
python -m venv --system-site-packages .venv-jz
source .venv-jz/bin/activate
python -m pip install -r requirements.txt
# Use a shared, writable location accessible from compute nodes:
export PLAYWRIGHT_BROWSERS_PATH="$PWD/.playwright-browsers"
python -m playwright install chromium
python scripts/bootstrap.py
export POKERL_VENV="$PWD/.venv-jz"
mkdir -p results
module list 2> results/jz-modules.txt
python -m pip freeze > results/jz-packages.txt
```

If the login node cannot download packages/assets, stage these on a permitted
transfer/prepost node. Transfer `upstream/`, the generated bundles, environment,
and Chromium installation to compute-visible storage. Browser system libraries
must already be installed by the site; a missing-library error in the smoke job
needs a compatible site browser (`POKERL_CHROME=/absolute/path/to/chromium`) or
an appropriate site container. Do not attempt package downloads inside the job.
`PLAYWRIGHT_BROWSERS_PATH`, `POKERL_VENV`, and optional `POKERL_CHROME` are inherited
by sbatch. Submit all commands from the repository root.

## 1. Short test job (H100)

```bash
smoke_job=$(sbatch --parsable --time=00:15:00 cluster/full-game.slurm smoke)
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
same training code. Do not mix smoke on one GPU architecture with an untested
Python/CUDA environment on the other.

The H100 script uses `hir@h100` and `--constraint=h100`, following the
[IDRIS guide](https://www.idris.fr/media/eng/ia/guide_nouvel_utilisateur_ia-eng.pdf).
The scripts inherit the default QoS; use `sbatch --qos=<your-allowed-qos>` if your
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
