# Full-game improvement experiments

These presets isolate successive changes. They are experiments, not claims that
any particular setting is stronger. All use Squirtle, a 120-second training
episode horizon, four environments, seed 123 and one million requested transitions.
Training time is still bounded by the job's learning timer.

| Config in `configs/` | Change from the preceding row | Observation | Network | Gamma |
| --- | --- | --- | --- | --- |
| `full_baseline.json` | Original observation and reward; fresh control run | v1 | 256, 256 | .99 |
| `full_pickups.json` | Typed pickups with values and separate healing/utility slots | v2-pickups | 256, 256 | .99 |
| `full_danger.json` | Enemy motion/attack state, dedicated hostile shots and live hazards | v2 | 256, 256 | .99 |
| `full_survival.json` | More weight on survival and avoiding damage; less on kills | v2 | 256, 256 | .99 |
| `full_horizon.json` | Slower discounting of future rewards | v2 | 256, 256 | .997 |
| `full_large.json` | Larger policy/value networks | v2 | 512, 512 | .997 |

PPO's other settings remain the original `n_steps=128`, `batch_size=64`,
`n_epochs=4`, `learning_rate=0.0003`, `ent_coef=0`. The JSON files override shared
defaults; every output `run.json` records the complete resolved configuration,
observation size, source hashes and actual completed steps. Explicit CLI or
SLURM environment overrides take priority over the selected file.

## What the new observations contain

The original 2,366-value observation remains an unchanged prefix. `v2-pickups`
adds 560 values (2,926 total); `v2` adds 1,722 (4,088 total).

- 20 XP slots, five healing slots and ten utility slots are selected independently.
  Each has position, numeric value, pickup type, pulled status, healing fraction
  or equipped-item catalog identity where relevant. Healing restores 30% of max HP
  before the missing-HP cap. XP no longer hides a healing drop from these slots.
- Enemy motion is measured between actual simulation ticks, after collisions and
  enemy separation. New enemies and large teleport discontinuities have an
  unavailable-velocity flag. Repeated observation reads do not update velocity.
- Each of 20 enemies also has collision radius, contact/shot/skill timers,
  charge phase and remaining phase time, committed dash direction, boss cast
  identity/direction/windup and jump destination/time where present. Readiness
  timers are not promises: range and game logic still decide when an attack occurs.
- Twenty hostile-projectile slots cannot be displaced by friendly projectiles.
  They include velocity, collision radius, damage and remaining lifetime.
- Twenty live hazard slots exclude already-detonated effects and include geometry,
  activation time and whether the player currently overlaps them.

All lists remain bounded; this is not a complete world map. The original terrain
grid and object observations are retained. Old models are evaluated and played
back using v1 automatically. Changed observation layouts or network sizes require
fresh training; resume rejects changes to those or other experimental settings.

## Reward experiment

| Contribution | Original | Survival preset |
| --- | ---: | ---: |
| Per simulated second | +0.1 | +0.2 |
| Per XP point gained | +0.1 | +0.1 |
| Per kill | +0.5 | +0.1 |
| Per HP lost | −0.1 | −0.2 |
| Death | −10 | −20 |

The original values make one kill worth five seconds of survival; the candidate
makes it worth half a second. This is a hypothesis to test. It could also make
the agent too cautious, so compare XP, level and kills alongside survival.
Healing is recorded but receives no extra direct reward, avoiding an incentive
to repeatedly take damage just to heal. Damage penalties are not erased by healing.

Episode logs now include healing recovered, healing/XP pickup counts, kills and
level. Evaluation includes each reward contribution, survival standard error,
horizon completion, and a fixed `referenceReturn` calculated with original reward
weights. Do not rank different reward presets by their training return.

## Submit the experiments

Use the prepared environment described in [Jean Zay setup](jean-zay.md).
The commands below use the scripts' existing two-hour allocation and a 6,000-second
learning timer. These are local script settings, not a claim about current site
limits; add your allocation's allowed `--qos` and `--time` if needed.
For H100, replace `cluster/full-game-v100.slurm` with `cluster/full-game.slurm`.

First check the updated adapter:

```bash
smoke_job=$(sbatch --parsable --time=00:15:00 cluster/full-game-v100.slurm smoke)
```

Submit separate fresh runs. Each command prints a job ID:

```bash
sbatch --dependency="afterok:${smoke_job}" cluster/full-game-v100.slurm train configs/full_baseline.json
sbatch --dependency="afterok:${smoke_job}" cluster/full-game-v100.slurm train configs/full_pickups.json
sbatch --dependency="afterok:${smoke_job}" cluster/full-game-v100.slurm train configs/full_danger.json
sbatch --dependency="afterok:${smoke_job}" cluster/full-game-v100.slurm train configs/full_survival.json
sbatch --dependency="afterok:${smoke_job}" cluster/full-game-v100.slurm train configs/full_horizon.json
sbatch --dependency="afterok:${smoke_job}" cluster/full-game-v100.slurm train configs/full_large.json
```

If smoke already passed, omit the dependency. Clear previously exported
`POKERL_*` overrides if you intend to use each preset unchanged. To save compute,
start with the baseline and pickups, compare, then advance through the other rows.

To submit a run and automatically queue its paired evaluation:

```bash
train_job=$(sbatch --parsable cluster/full-game-v100.slurm train configs/full_horizon.json)
POKERL_EPISODES=20 POKERL_BASELINES=0 \
  sbatch --dependency="afterok:${train_job}" \
  cluster/full-game-v100.slurm eval "results/full-train-${train_job}"
```

Evaluation uses seeds 1,000,000,000 through 1,000,000,019 by default. Baselines are
enabled by default; `POKERL_BASELINES=0` avoids repeating them for every experiment.
Use the same seed range, episode count, partner and horizon for all comparisons.
Results are `results/full-train-JOB/evaluation-EVAL_JOB.json`.

```bash
python -m training.compare_full_game \
  results/full-train-BASELINE_JOB/evaluation-BASELINE_EVAL_JOB.json \
  results/full-train-CANDIDATE_JOB/evaluation-CANDIDATE_EVAL_JOB.json
```

Replace the uppercase job placeholders with actual IDs. The comparison command
rejects different evaluation seeds/horizons. Check its **Steps** column:
one million requested steps is not one million completed steps when a timer stops
training early. Larger models may complete fewer steps in the same wall time.
For a fair capacity comparison, evaluate checkpoints at matched step counts.

## Train longer and inspect learning curves

Checkpoints are saved every 100,000 transitions by default; `progress.csv`
contains PPO training diagnostics. To evaluate all checkpoints plus the final
model on the same seeds:

```bash
POKERL_EVAL_CHECKPOINTS=1 POKERL_BASELINES=0 POKERL_EPISODES=20 \
  sbatch cluster/full-game-v100.slurm eval results/full-train-TRAIN_JOB
```

This costs roughly one evaluation per checkpoint; allow enough job time or
evaluate one selected checkpoint instead:

```bash
POKERL_EVAL_MODEL=results/full-train-TRAIN_JOB/checkpoints/full_500000_steps.zip \
  POKERL_BASELINES=0 \
  sbatch cluster/full-game-v100.slurm eval results/full-train-TRAIN_JOB
```

When the stronger configuration is still improving, resume it into a new run.
The settings are inherited from its metadata; `POKERL_STEPS` requests **additional**
transitions. Pick a fresh training seed to avoid replaying the same initial seed
sequence. Each job still saves before its learning timer expires:

```bash
POKERL_RESUME=results/full-train-TRAIN_JOB/model.zip \
  POKERL_STEPS=2000000 POKERL_SEED=456 POKERL_MAX_SECONDS=6000 \
  sbatch cluster/full-game-v100.slurm train
```

Resume more chunks if needed, using each new run's model. Increasing steps alone
does not extend the learning timer or allocation. For one longer allocation, set
`--time` and `POKERL_MAX_SECONDS` together, leaving time for startup and saving.
Checkpoint resume recovers policy/optimizer state, not the exact old game states.

To measure training-seed variability, train the shortlisted configuration fresh
with seeds 123, 456 and 789, then evaluate each on identical episode seeds:

```bash
for seed in 123 456 789; do
  POKERL_SEED="$seed" sbatch cluster/full-game-v100.slurm train configs/full_horizon.json
done
```

After selecting a configuration, confirm it on an unused evaluation seed range
to avoid selecting only for the first twenty games:

```bash
POKERL_EPISODES=50 POKERL_EVAL_SEED=2000000000 POKERL_BASELINES=0 \
  sbatch cluster/full-game-v100.slurm eval results/full-train-WINNER_JOB
```

If most runs reach the 120-second cap, compare all finalists with
`POKERL_EVAL_DURATION=300`. To train for 300-second episodes, start a fresh run
with `POKERL_DURATION=300`; this is a separate experiment and resume deliberately
rejects the horizon change.

Other exposed overrides include `POKERL_ENVS`, `POKERL_GAMMA`, `POKERL_ENT_COEF`,
`POKERL_N_STEPS`, `POKERL_BATCH_SIZE`, `POKERL_N_EPOCHS`, `POKERL_LEARNING_RATE`,
`POKERL_TARGET_KL`, and `POKERL_CHECKPOINT_STEPS`. Batch size must divide `envs × n_steps`.
Change one factor at a time after the initial comparisons.
