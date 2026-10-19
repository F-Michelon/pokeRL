# Arena and vision experiments

This page preserves the earlier arena workflow and historical local results.
Its action space, checkpoints, curriculum and vision student are separate from
the current full-game MaskablePPO workflow. Commands run from the repository root.
Cluster statements below concern the legacy arena scripts only.

## Arena environment contract

The API and historical policies below target the **training arena**. For the
full game, use `training.full_game_env`, `training.train_full_game` and
`training.evaluate_full_game`, described in the [pipeline guide](pipeline.md).

```js
import { Environment } from './rl/environment.js';
const env = new Environment({stage: 0, duration: 60});
const observation = env.reset(123);
const transition = env.step(4, 6); // right, 6 frames = 100 ms
// {observation, reward, terminated, truncated, info}
console.log(env.getMetrics());
```

Actions: `0 idle`, `1 up`, `2 down`, `3 left`, `4 right`, `5 up-left`,
`6 up-right`, `7 down-left`, `8 down-right`. Diagonals are normalized.
`reset(seed)` rebuilds all game closures, entity IDs, RNG and timers. Environments
have independent RNG streams; rendering never advances the simulation RNG.
`step` after an episode ends raises an error. Death terminates; duration truncates.
The Node JSON-lines protocol in `rl/server.js` has configure/reset/step/upgrade/
metrics/close operations. Each Gymnasium environment owns one Node process.

The observation has **397 float32 values**, with stable nearest-first slots:

| Segment | Contents |
| --- | --- |
| 12 player values | position, HP fraction, XP fraction, level, velocity, elapsed fraction, cooldown, dex, upgrade pending, attack multiplier |
| 20 × 8 enemies | presence, relative position, velocity, HP fraction, ranged flag, speed |
| 20 × 6 projectiles | presence, relative position, velocity, friendly flag |
| 20 × 4 pickups | presence, relative position, XP flag |
| 5 × 5 terrain | nearby collision occupancy at 40-unit spacing |

Distances and speeds are scaled (see `rl/observations.js`); absent slots are zero.
The reward is `0.1 × survival seconds + 0.1 × XP + 0.5 × kills − 0.1 × damage`,
with an additional `−10` on death. Rewards use differences per step, never totals.
Metrics include survival, XP/minute, damage/minute, deaths, kills and level.

The bounded rectangular arena deliberately replaces upstream world generation,
spawner, rendering, weather, effects, progression, and services. Collision radii
are fixed at 12 units. Enemy stats and spawn cadence are controlled by config;
player stats, attack execution, damage/type calculations and entity updates are
upstream code. This is a **controlled gameplay subset**, not a full-game emulator.
The default browser renderer uses original Pokémon sprites, tiles and projectile
artwork in a separate presentation context. It only reads simulation state;
rendering does not replace collision radii or consume the simulation's RNG.
Pathfinding around the optional rectangular obstacles uses collision handling
and upstream stuck recovery, without the full upstream navigation system.

## Train and evaluate

```bash
python -m training.train_ppo --config configs/arena_easy.json \
  --steps 1000000 --envs 8 --device cpu --output results/ppo
python -m training.evaluate --config configs/arena_easy.json \
  --model results/ppo/model.zip --episodes 30 --output results/evaluation.json
```

Evaluation uses identical held-out seeds starting at 1,000,000,000 for PPO,
stand-still, circular movement and nearest-enemy avoidance. Training draws seeds
below that range. Reports retain per-episode metrics and paired return differences.
The movement gate requires at least 30 episodes and a positive lower 95% normal
confidence bound against **each** baseline. It is a working experiment criterion,
not proof of generalization. Check survival and damage separately as well.

```bash
python -m training.curriculum --device cuda --envs 16 --steps 1000000
```

Curriculum: sparse → dense → ranged enemies → obstacles → varied starter/attack.
It stops if a stage fails the movement gate. To extend a run, pass
`--resume results/ppo/model.zip` to `training.train_ppo`. Checkpoints are saved
approximately every 50,000 transitions. `run.json` records configuration, seed,
source pin and hardware. SB3 rounds the requested step count to complete rollouts.
`requirements-local.lock.txt` records this workspace's exact CPU smoke-test
packages; use `requirements.txt` with the cluster's compatible CUDA PyTorch for
GPU jobs, and freeze that environment with each production run.

## Jean Zay: project hir

Prepared scripts (no jobs submitted):

- `cluster/train-v100.slurm`: `hir@v100`, one GPU, 8 parallel environments.
- `cluster/train-h100.slurm`: `hir@h100`, H100 constraint, 16 environments.
- `cluster/vision-h100.slurm`: CNN distillation from recorded local episodes.

From a checked-out copy on Jean Zay, load a compatible **site-provided** PyTorch
module/environment and put Node on PATH. Alternatively set `POKERL_VENV` to your
prepared virtualenv. Dependencies and the pinned upstream must already be present;
compute jobs should not need internet access. The jobs verify CUDA by executing a
small tensor operation before training. V100 needs a PyTorch/CUDA build that still
supports its architecture; do not copy the CPU environment from this laptop.

```bash
sbatch cluster/train-v100.slurm
# OR, independently:
sbatch cluster/train-h100.slurm
# Configuration overrides:
POKERL_CONFIG=configs/arena_dense.json POKERL_STEPS=2000000 sbatch cluster/train-h100.slurm
# After copying the teacher dataset to the cluster:
POKERL_DATA=results/teacher sbatch cluster/vision-h100.slurm
```

Account suffixes follow the [IDRIS user guide](https://www.idris.fr/media/eng/ia/guide_nouvel_utilisateur_ia-eng.pdf).
Verify available CPU counts, constraints and QoS for your allocation using the
[IDRIS partition documentation](http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-exec_partition_slurm-eng.html)
and `sinfo` before submission. Scripts intentionally inherit the site's default
QoS; override `sbatch --qos=...` if your allocation requires it. These scripts have
been syntax-checked locally, not executed on Jean Zay.

The small movement MLP may be limited by environment/IPC throughput; benchmark
CPU versus GPU rather than assuming H100 is faster. CNN training is a more natural
GPU workload. One GPU suffices for these models; these scripts do not implement DDP.

## Strategy and visual control

`configs/arena_strategic.json` enables a controlled level-up event: attack, speed,
or maximum HP. Movement pauses until `chooseUpgrade(0|1|2)`. The Python bridge
uses a separate event-driven heuristic (health when HP is low, attack otherwise).
This is a first strategic baseline, **not a trained strategic policy**. Active move
selection, chests, springs, altars and persistent upgrades are not implemented.
Add them only after the movement gate succeeds, as required by the plan.

The visual pipeline is executable locally:

**Existing vision checkpoints from the initial experiment were trained on debug
shapes.** Use `--graphics debug` with those checkpoints. The structured PPO teacher
remains usable with either renderer. Record a fresh dataset and distill a new
student before evaluating the new artwork with a vision policy; the old vision
results do not carry over to the new camera and appearance.

```bash
# Keep npm run serve running in a separate terminal.
python -m playwright install chromium
python -m training.record_teacher_data --model results/ppo/model.zip \
  --episodes 50 --config arena_easy --output results/teacher
python -m training.train_vision_student --data results/teacher \
  --epochs 30 --device cuda --output results/student
python -m training.run_browser --model results/student.zip --visible
# Optional normal-speed local screenshot-only PPO fine-tuning:
python -m training.run_browser --model results/student.zip --finetune-steps 10000 \
  --output results/student_finetuned
```

If Chromium is already installed, pass `--executable /path/to/chrome` to recording
or browser-run commands. Browser training rejects non-local URLs and blocks
external page requests. It holds WASD keys, captures only the canvas and detects
the local start/level-up/death/completion panels through their rendered color
swatches. No game-state reads occur in student action selection or reward.
The readiness flag is used only for initial page load synchronization. The teacher
recorder alone reads structured observations to label screenshots. Screenshots
and labels are acquired sequentially while the game runs, so capture latency can
introduce a small labeling offset. Fine-tuning uses only survival/death reward;
it does not recover XP or HP from screenshots.

Canvas frames are resized to 160×90 RGB for a PPO CNN policy. Distillation splits
by episode, keeping validation episodes separate. Scale, brightness, particle
overlays, backing resolution and render FPS vary by episode. Matching the full
upstream HUD, menus, world and effects remains transfer work. Browser capture and inference overhead can
make action holds longer than 100 ms; the simulation continues at normal speed.
The fixed local status swatches do **not** detect the official site's menus.

## Run a GPU-trained model on a lightweight laptop

Copy the `.zip` checkpoint back, install CPU dependencies, and use `--device cpu`
(the browser runner's default). Training on V100/H100 does not require that GPU
for inference. The model needs only one forward pass per action, without training
rollouts or gradients. Benchmark on the target laptop:

```bash
python -m training.benchmark results/student.zip
python -m scripts.benchmark_capture
python -m training.run_browser --model results/student.zip --visible --device cpu
```

The benchmark reports median and p95 **inference-only** milliseconds. For the
intended roughly 10 Hz control loop, also measure browser capture latency and CPU
load. A model that fits the latency budget locally can then be considered for a
short supervised online evaluation; GPU training alone does not establish this.

## Validation and plan status

- Phase 0: pinned source, isolated branch and local launch implemented.
- Phases 1–2: deterministic arena/API implemented; 1,000 random full-length
  episodes exercised across all five stages. See `results/validation.json`.
- Phase 3: parallel PPO, baseline evaluation, checkpoints and gated curriculum
  implemented. After 327,680 local training transitions, the easy-arena policy
  passed the gate on 30 held-out seeds: mean return 54.51 vs circles 51.39,
  standing 41.85 and fleeing 39.96. All survived 60 seconds; PPO collected
  126.82 XP/minute vs circles 99.05. See `results/baselines.json` and the local
  checkpoint `results/ppo_validated/model.zip`. Harder curriculum stages are not
  validated; these historical results do not validate full-game GPU training.
- Phase 4: first event-driven upgrade baseline implemented; remaining strategy
  types and trained strategic policy await movement validation.
- Phase 5: local recording, distillation, screenshot/WASD evaluation and RL
  fine-tuning implemented and smoke-tested. Two completed local teacher episodes
  yielded a small pilot dataset; its student reached only 33% held-out action
  accuracy. It completed one 60-second screenshot-only episode on held-out seed
  1000000000 (`results/student_local_evaluation.json`). Standing still also
  survives this easy arena, so this does not establish a competent visual policy
  or public-game transfer. These results used the legacy debug renderer; the new
  default artwork renderer requires fresh teacher recording and student training.
- Phase 6: no online automation performed. Full upstream visual compatibility,
  public-version comparison and supervised guest evaluation remain future work.

Tests: `npm test`, `npm run stress`, `python -m tests.python_smoke` and (with the
local server running) `python -m tests.browser_smoke`. The browser test uses
`POKERL_CHROME` if set, otherwise Playwright's installed Chromium.
`python -m tests.full_game_smoke` additionally checks full-game onboarding,
terrain/objects, live enemy spawning, no external requests, storage isolation
and local account persistence across reloads. See `results/full-game-validation.json`.

The plan's end-to-end success criterion is **not yet achieved**. A working pipeline
and passing integrity checks are distinct from a trained agent outperforming
baselines and surviving on the official site. Online training is excluded. Before
any online evaluation, match the public version, controls, artwork and menus;
use a guest/local save, manually supervise a short run and keep human-like key
holds. The current local-only runner intentionally has no online mode.
