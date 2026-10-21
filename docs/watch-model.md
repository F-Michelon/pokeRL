# Watch a trained model

[README](../README.md) · [Cluster workflow](jean-zay.md)

The viewer loads a full-game MaskablePPO checkpoint, opens the rendered game,
and lets the model control movement, moves, upgrades and other choices. It uses
the same fixed-step simulation, observations, masks and deterministic predictions
as evaluation. Audio is disabled. Menu choices are handled automatically; an
overlay shows survival, HP, level, kills, XP, return and the latest action.

## Watch locally

Run from the repository root with the prepared Python environment and Chromium.
The viewer automatically uses the project's `.playwright-browsers/` directory
when present. An explicit `PLAYWRIGHT_BROWSERS_PATH` overrides that default;
otherwise Playwright's normal cache is used if no project cache exists.
Rebuild the generated adapter once after updating to this viewer:

```bash
source .venv/bin/activate
python scripts/build_full_game.py
python -m training.watch_full_game \
  --model results/full-train-1292524/model.zip
```

The browser runs at approximately real-time speed and closes after the game
ends. Close its window or press Ctrl-C in the terminal to stop early. The viewer
starts its own local server. It does not require `npm run serve`.

Use `--speed 2` for double speed, `--episodes 3` for successive games, and
`--seed 1000000005` to inspect a particular evaluation episode. Seeds increment
between games. The saved `run.json` supplies the Pokémon and episode limit;
keep it beside `model.zip` when transferring models to your desktop. Intermediate
models in the run's `checkpoints/` directory are also supported.

The view refreshes after each policy action (normally ten times per simulated
second). Rendering and inference may reduce the achieved playback speed. Brief
pauses make zero-time menu decisions and the final result easier to see. Native
choice dialogs are replaced by the training adapter, so the viewer shows their
decision type and selected action rather than clickable cards.

## Record without a desktop

```bash
python -m training.watch_full_game \
  --model results/full-train-1292524/model.zip \
  --headless --record results/my-first-playback
```

This writes one `episode-…-seed-….webm` per completed game and `report.json` with
its metrics. Open the WebM in a browser or video player. Choose a new recording
directory each time. Omit `--headless` to watch and record simultaneously.
Playwright recording requires its FFmpeg binary; the usual
`python -m playwright install chromium` preparation installs it as well.
Interrupted episodes may leave a recording with Playwright's generated filename;
only completed episodes appear in the report.

## Record on Jean Zay

On the preparation node, rebuild the adapter as above. Then submit:

```bash
sbatch --time=00:15:00 cluster/full-game-v100.slurm watch \
  results/full-train-1292524
```

For H100 use `cluster/full-game.slurm`. This runs without a display and saves
videos under `results/full-train-1292524/watch-<jobid>/`. Copy that directory to
your desktop to watch. The common batch script still reserves and checks a GPU;
playback inference uses CPU. Optional environment variables are
`POKERL_EPISODES` (default 1), `POKERL_WATCH_SEED` (default 1000000000) and
`POKERL_WATCH_SPEED` (default 1).

For comparable quantitative results, continue using `training.evaluate_full_game`.
Playback is for inspecting behavior, and its default single game is not a
performance estimate.
