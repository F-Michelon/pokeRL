# PokéSurvivor RL

Train a reinforcement-learning agent on the pinned PokéSurvivor single-player
game, running locally in headless Chromium. The current pipeline uses structured
game observations and MaskablePPO to learn movement, active-move selection and
in-game choices. It supports offline GPU jobs on Jean Zay.

The repository also includes a playable local game and earlier arena/vision
experiments. Their checkpoints and environment contracts are separate.

## Documentation

| Guide | Contents |
| --- | --- |
| [Project pipeline](docs/pipeline.md) | Source preparation, browser simulation, observations, actions, reward, PPO and evaluation |
| [Jean Zay workflow](docs/jean-zay.md) | Offline preparation, venv, V100/H100 smoke → train → eval, outputs and troubleshooting |
| [Understanding training logs](docs/training-logs.md) | Episodes vs iterations, parallel games, simulated speed and PPO diagnostics |
| [Improvement experiments](docs/full-game-experiments.md) | Pickup/danger observations, reward and network comparisons, SLURM configs, longer training |
| [Watch a trained model](docs/watch-model.md) | Live model playback and headless video recording, locally or on Jean Zay |
| [Arena and vision experiments](docs/arena-and-vision.md) | Legacy arena, curriculum, screenshot distillation and historical results |

## Full-game training on Jean Zay

Run from the repository root. Prepare the pinned game, a GPU-compatible `.venv`
and Chromium first; see the [preparation instructions](docs/jean-zay.md).
The batch scripts activate `.venv/bin/activate` and do not load modules, pull Git,
install packages or rerun bootstrap. The trainer reads local Git metadata to
verify the upstream revision without invoking Git.

```bash
# V100 smoke test: infrastructure checks, short training and evaluation.
smoke_job=$(sbatch --parsable --time=00:15:00 cluster/full-game-v100.slurm smoke)

# Start training only if smoke succeeds.
train_job=$(sbatch --parsable --dependency="afterok:${smoke_job}" \
  cluster/full-game-v100.slurm train)

# Evaluate the saved model and three baselines after successful training.
eval_job=$(sbatch --parsable --dependency="afterok:${train_job}" \
  cluster/full-game-v100.slurm eval "results/full-train-${train_job}")
```

For H100, use `cluster/full-game.slurm` with an H100-compatible environment.
The scripts currently select development QoS values; check that your requested
wall time is allowed and override `--qos` when needed.

Default training uses four parallel games, 1,000,000 requested transitions, a 120-second
simulated episode horizon and a 6,000-second learning timer. The default partner
is Squirtle. Results are saved in `results/full-train-<jobid>/`; evaluation writes
`evaluation-<eval-jobid>.json` there. A successful smoke run validates the pipeline,
not the agent's playing strength.

## Local setup and validation

Use Python 3.12. Node 22+ is used by the npm commands and legacy arena tools.
For a local CPU environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/bootstrap.py
python -m playwright install chromium
python -m tests.full_training_smoke
python -m training.train_full_game --steps 512 --envs 2 --duration 30 \
  --output results/full-local-smoke
python -m training.evaluate_full_game --model results/full-local-smoke/model.zip \
  --episodes 2 --baselines --output results/full-local-smoke/evaluation.json
```

Use a fresh output directory for each run. The training command starts its own
localhost server; no separate `npm run serve` is needed. Do not use the CPU
installation above for a GPU cluster job.

## Play locally

```bash
npm run serve
```

Open <http://127.0.0.1:8000/full_game/>. The npm startup hook runs bootstrap,
which needs Git and downloads upstream only when it is absent. With prepared
files, you can instead start the server directly:

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Choose **Play as guest** or create a local account, complete onboarding and press
Play. WASD/arrows move; 1–4 or Q/E select moves; Esc pauses. Local accounts persist
in this browser under an isolated `pokerl.fullgame.` storage prefix. Cloud saves,
online multiplayer and public rankings are disabled.

## Scope and reproducibility

[upstream.lock.json](upstream.lock.json) pins upstream to
`e4ba0e173cb7e415f9815c924a48c5c25f516962`. Bootstrap checks the revision,
verifies the bundled arena modules and builds the local full-game entrypoint.
Keep the prepared `upstream/` checkout, including its Git metadata, and generated
files when moving to offline storage; these files are ignored by Git.

Full-game training uses fresh guest progression per episode. It does not learn
account purchases or Pokémon selection, and it is not screenshot-based control
of the public website. The user has reported a successful cluster smoke run;
full-game playing strength must be measured with held-out evaluation. Historical
arena results do not establish full-game performance. See [plan.md](plan.md) for
the original research plan.

The upstream code uses **PolyForm Noncommercial 1.0.0**. Preserve its license and
attribution when sharing. Required Notice: Copyright 2026 JPTER78
(https://github.com/JPTER78/pokesurvivor). The upstream license is included in the
prepared checkout as `upstream/LICENSE-NONCOMMERCIAL.md`.
