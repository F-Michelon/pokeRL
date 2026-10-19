# Local validation, 2026-10-19

Pinned game: e4ba0e173cb7e415f9815c924a48c5c25f516962.

- Node deterministic replay, reset isolation, action validation, frame grouping,
  termination and curriculum checks passed.
- 1,000 random episodes across stages 0–4 completed; 977 deaths are valid episode
  outcomes, not crashes. Final stress runtime 29.19 seconds.
- Gymnasium check_env and Python bridge reset/cleanup checks passed.
- Chromium localhost canvas capture and WASD displacement checked without page errors.
- Movement PPO: 262,144 transitions, followed by 65,536 transitions after adding
  enemy velocity observations. CPU, four parallel environments, seed 123.
- Held-out seeds 1000000000–1000000029: easy-arena PPO passed the paired return
  gate against all three baselines. All policies survived the full minute.

| Policy | Mean return | XP/min | Damage/min |
| --- | ---: | ---: | ---: |
| PPO | 54.5063 | 126.8167 | 0.4205 |
| Circle | 51.3911 | 99.0500 | 1.4719 |
| Stand | 41.8500 | 0 | 0 |
| Flee | 39.9569 | 6.5333 | 0.6308 |

PPO minus baseline lower 95% normal bounds: stand +12.0767, circle +2.4150,
flee +13.7189. See baselines.json for the complete paired episodes.

Vision smoke: two 2-second episodes, distillation checkpoint save/load and CPU
inference benchmark passed. This tiny smoke dataset is not policy-quality evidence.
The CNN has 3,751,594 parameters; median inference 1.083 ms and p95 1.611 ms on
this execution host with one CPU thread, excluding browser capture. Benchmark
again on the intended laptop. The saved SB3 zip also includes optimizer state.

Jean Zay scripts were syntax-checked only. No cluster job or online game session
was submitted. See README for outstanding curriculum, strategy and transfer work.

Additional checks:

- Screenshot-only PPO fine-tuning completed a 128-transition update and saved
  `student_finetune_smoke.zip`.
- Two completed local teacher episodes produced a 34-frame pilot dataset. The
  10-epoch student achieved 33% held-out episode action accuracy. Its single
  screenshot-only evaluation completed the 60-second easy arena (86 actions,
  62.50 seconds wall time). This is an integration check, not evidence that the
  student is better than standing still. See student_local_evaluation.json.
- Capture profiling found roughly 60–116 ms for a fixed canvas clip. Recording
  now reuses the previous step's frame, avoiding duplicate screenshots, and
  subtracts capture cost from the next hold delay. Actual end-to-end action rate
  still needs benchmarking on the user's laptop; 10 Hz is a target, not a guarantee.
- Five episodes of the tougher strategic configuration: easy-trained PPO survived
  26.67 seconds on average vs circle baseline 41.27. No strategy/generalization
  success is claimed. Further curriculum training is required.
- Final optimized recorder check, including randomized backing resolution and
  render FPS: 18 samples in a 2-second episode (1.995 seconds wall time), roughly
  9 Hz on this host. This short check validates the timing fix, not sustained
  laptop performance. Updated trainer metadata/source-hash smoke run also passed.

## Renderer update

The default local arena now uses the pinned upstream Pokémon sprites, forest
pixel-art tiles and projectile graphics, with a following camera and custom HUD.
Browser smoke checks verified sprite loading, movement and no page/asset errors;
Node determinism tests still pass. Simulation code was not modified.

All vision-policy results above refer to the **legacy debug renderer**. Reproduce
those using `--graphics debug`. The structured PPO teacher can record the new
renderer, but new image data and student training are required before claiming
vision performance on it. Full upstream UI/world equivalence remains unfinished.

## Full single-player game

The server root now opens `/full_game/`: the complete pinned single-player game,
using its original world, entity, spawning, combat, progression and UI modules.
The generated entrypoint selects local-only database mode, isolates browser
storage, disables service workers and prevents external connections. Upstream
source files remain unchanged. The RL arena is still a separate entrypoint.

Browser validation passed guest onboarding and a live run, found floor/wall/liquid
terrain plus trees, grass, rocks, chests, locked chests, springs, traps and altars,
and observed original enemy spawning/HP. A local account's coins and partner
survived reload; unrelated storage was untouched. No page errors, missing assets
or external network requests occurred. Node arena tests still pass.

At that stage this was local play only. The historical policy scores above still
apply only to the training arena; the new adapter is validated separately below.

## Full-game masked PPO pipeline (2026-10-19)

Added a fixed-step browser adapter around the pinned original update loop, with
2,366 structured features and 50 masked movement/move/menu actions. Independent
fresh-context replay passed. Smoke tests cover upgrade selection and reroll,
active-move switching, item selection, two-step fusion, reset, truncation and death.
Simulation randomness is isolated from asynchronous audio initialization.

Local CPU run `results/full_training_ready` completed 512 transitions with two
Chromium workers, saved a MaskablePPO checkpoint and reloaded it for evaluation.
Evaluation used two held-out seeds (1,000,000,000 and 1,000,000,001), fresh Squirtle,
and a 120-second horizon. Mean survival: PPO 42.72 s, stand-first 43.57 s,
circle-first 57.04 s, random-valid 53.78 s. Natural upgrades and move switching
occurred. These tiny samples verify the pipeline only: the smoke model does not
beat the movement baselines. Normal full-game onboarding/account persistence and
the existing Node regression tests also passed after adding the training hook.

Jean Zay smoke/train/evaluation scripts are prepared for hir H100 and V100. No
remote job or CUDA training has been run here. Site CUDA/browser compatibility
must pass the short cluster smoke job before starting the first training.
