## Goal

Build a reproducible local RL environment for PokéSurvivor, train a movement policy, then transfer it to browser-style visual control before testing it cautiously on the online game.

## Phase 0 — Pin the game version

- Fork or clone `JPTER78/pokesurvivor`.
- Record the commit hash used for experiments.
- Run it locally with a static server—not `file://`—for predictable browser automation.
- Create a separate RL branch; keep normal gameplay intact.
- Use only offline/local saves and disable Firebase/cloud features during training.

**Deliverable:** reproducible local game launch command and a pinned source revision.

## Phase 1 — Make a minimal training arena

Do not begin with the full game.

- Add an “RL arena” mode with:
  - one fixed Pokémon;
  - one fixed automatic attack;
  - no menus, collection, gacha, achievements, rifts, or persistent upgrades;
  - a simple map and fixed duration, e.g. 60 seconds;
  - controlled enemy spawning.
- Add deterministic randomness: `reset(seed)` must reproduce an episode exactly.
- Allow rendering, sound and DOM/UI updates to be disabled.

**Deliverable:** same seed → same enemy spawns, damage, and outcome.

## Phase 2 — Expose an environment API

Add a lightweight interface around the existing update loop:

```js
env.reset(seed)
env.step(action, frames = 6)
env.getObservation()
env.getMetrics()
```

Start with simple discrete actions:

```text
0 idle
1 up        2 down
3 left      4 right
5 up-left   6 up-right
7 down-left 8 down-right
```

Use an internal structured observation, such as:

- player position, HP, XP, level, movement velocity;
- up to the nearest 20 enemies: relative position, velocity, HP/type;
- nearby projectiles and pickups;
- local terrain/collision grid;
- active attack and cooldown state.

Return a reward such as:

```text
+ survival time
+ XP collected
+ enemy kills
− damage received
− strong terminal penalty on death
```

**Deliverable:** an RL-compatible environment that can run 1,000 random episodes without crashes or state leakage.

## Phase 3 — Connect it to training

I would use Python for training and JavaScript for the game:

- Game environment: JavaScript/Node.
- RL: PyTorch + CleanRL or Stable-Baselines3.
- Communication: one Node process per environment, ideally over local WebSocket or stdin/stdout JSON.
- Algorithm: start with **PPO**.
- Run several environments in parallel.

Begin with curriculum learning:

1. sparse, slow enemies;
2. denser enemies;
3. enemies with projectiles;
4. terrain obstacles;
5. varied Pokémon and attack types.

Track:

- mean survival time;
- XP/minute;
- damage/minute;
- death rate;
- performance across held-out random seeds.

**Success criterion:** outperform simple baselines such as “stand still,” “always move in a circle,” and “move away from nearest enemy.”

## Phase 4 — Add strategic decisions

Only once movement works, add decision types one at a time:

1. choose level-up reward among three;
2. choose active move;
3. decide whether to open chests/heal at springs/use altars;
4. later, choose Pokémon and persistent upgrades.

Use a hierarchical design:

```text
Tactical policy: movement every ~100 ms
Strategic policy: upgrades and route choices when an event occurs
```

This is much easier to train and debug than a single model making every decision.

**Deliverable:** agent that both survives and selects sensible upgrades in a controlled subset of the game.

## Phase 5 — Build browser-style control locally

Do not go directly to the public website.

Create a browser runner for your own local copy:

- Playwright launches the local game at fixed 1280×720 resolution and 100% zoom.
- Capture only the game canvas.
- Send `keydown` / `keyup` events for WASD.
- Detect level-up and death menus visually.
- Run at normal browser speed.

Then train a vision model:

```text
canvas frame(s) → action
```

The best route is **teacher–student distillation**:

- Your structured-state PPO agent is the teacher.
- Record `(canvas image, teacher action)` while it plays.
- Train a vision model to reproduce its actions.
- Fine-tune the vision model with RL in the local browser game.

During local training, randomize scale, brightness, particles, UI overlays, resolution and FPS to reduce visual overfitting.

**Deliverable:** a policy that plays through screenshots and keyboard inputs on your own local browser instance.

## Phase 6 — Online transfer, cautiously

Before trying the official site:

- compare the public game against your pinned version;
- confirm controls, UI layout and gameplay mechanics still match;
- test only manually supervised short runs;
- use a guest/local account, never your main progression;
- rate-limit actions to human-like key holds—no mass automated sessions.

Initially, treat this as **evaluation**, not training. The online game should not be part of the RL training loop.

**Success criterion:** the local browser-vision agent survives comparably on the official game without access to JavaScript state.

## Suggested repository layout

```text
pokesurvivor/
  rl/
    arena.js
    environment.js
    observations.js
    rewards.js
    seeds.js
    browser_runner/
  training/
    train_ppo.py
    evaluate.py
    record_teacher_data.py
    train_vision_student.py
  configs/
    arena_easy.yaml
    arena_projectiles.yaml
    browser_vision.yaml
  results/
```