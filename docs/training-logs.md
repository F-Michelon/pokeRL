# Understanding training logs

[README](../README.md) · [Pipeline](pipeline.md) · [Cluster workflow](jean-zay.md)

## Episodes, steps and iterations

An **episode** is one game from reset until death or truncation. A **transition**
is one decision and its resulting observation/reward. An **iteration** collects
one rollout across all environments before updating the policy.

The default four workers run four independent games. Each contributes 128
transitions per iteration, for 512 total. Rollout boundaries do not restart games.
A game can span several iterations, or end and restart inside one rollout.
When one worker resets, the other games continue.

```text
Collect 128 decisions per worker → optimize PPO for 4 epochs → collect again
             4 workers = 512 transitions per rollout
```

For a fresh default run, iteration 47 gives `47 × 4 × 128 = 24,064` transitions.
A resumed run can retain earlier timestep/update counters. Changing the number
of environments changes the amount of experience per rollout.

## Example: rollout and timing

| Logged field | Example | Interpretation |
| --- | --- | --- |
| `rollout/ep_len_mean` | 425 | Mean decision count over recent completed episodes, normally a buffer of up to 100 episodes |
| `rollout/ep_rew_mean` | 8.86 | Mean total reward over those completed episodes |
| `time/fps` | 126 | Aggregate transitions per real second, including training overhead |
| `time/iterations` | 47 | Rollout iterations in the learning call |
| `time/time_elapsed` | 190 | Approximately 190 real seconds since learning began |
| `time/total_timesteps` | 24064 | Transitions across all workers; about 24% of a fresh 100,000-step target |

These episode statistics summarize completed games, not the partially completed
games currently running. They can lag behind recent policy changes.

An ordinary movement decision advances six simulation ticks at 60 Hz, or 0.1
simulated seconds. Menu decisions advance no game time, and interrupted movement
steps can advance less. Therefore 425 decisions represent at most about 42.5
simulated seconds, not 425 seconds. Use the monitor CSV's `survival` column for
actual game-time survival; its `l` field counts decisions.

Reward is a weighted sum of survival, XP and kills, minus damage taken and a death
penalty. `8.86` is neither a percentage nor a direct survival measurement. See
[the reward formula](pipeline.md#3-observations-actions-and-reward).

## Is the game accelerated?

Yes. Training manually advances the game clock without waiting for real-time
rendering. It is not speeding up a video: it is executing simulation ticks as
fast as the browser/Python/training pipeline can process them.

At 126 transitions per second, assuming nearly all decisions advance 0.1 seconds:

```text
126 × 0.1 = 12.6 simulated seconds per real second across all games
12.6 / 4  = 3.15 simulated seconds per real second per worker on average
```

This is an approximate upper estimate, not a measured speed multiplier. Menu
choices and partial steps reduce simulated time per transition. Workers also
pause for optimization and resets. FPS is an aggregate throughput statistic,
not rendered frames per second or the speed of any particular episode.

At the same throughput, 100,000 transitions would take roughly 794 seconds
(13 minutes 14 seconds) of learning time. Startup, changing game complexity,
resets, saves and later evaluation make actual job duration different.

## PPO optimization diagnostics

| Field | Example | How to read it |
| --- | --- | --- |
| `approx_kl` | 0.01315 | Approximate change between the rollout policy and the updated policy. Watch for abrupt growth; there is no universal good threshold. |
| `clip_fraction` | 0.151 | About 15.1% of sampled action probability ratios were outside the clipping interval during optimization. |
| `clip_range` | 0.2 | PPO's configured clipping tolerance, corresponding to ratio bounds 0.8–1.2. This is not a hard bound on every policy change. |
| `entropy_loss` | -2.64 | Negative policy entropy. Values closer to zero usually mean more deterministic choices; the number of valid masked actions also matters. |
| `explained_variance` | 0.905 | Value predictions explain about 90.5% of the variance in rollout return targets. A good fit is not proof of good actions. |
| `learning_rate` | 0.0003 | Optimizer learning rate. |
| `loss` | 0.0829 | Combined loss reported from the last optimization minibatch, not an episode performance score. |
| `n_updates` | 184 | Completed optimization epochs; here 46 previous rollouts × 4 epochs. |
| `policy_gradient_loss` | -0.0203 | PPO policy objective. Its sign alone does not establish improvement. |
| `value_loss` | 0.625 | Mean squared error between value estimates and return targets, dependent on reward scale. |

Logging happens after collecting a rollout and before optimizing that rollout.
Consequently iteration 47 can show training diagnostics from iteration 46 and
`n_updates = 184`. An epoch makes multiple minibatch optimizer steps, so this
counter is not the number of individual gradient steps.

Explained variance near 1 indicates a strong fit, near 0 indicates little predictive
value, and negative values indicate worse variance-based fit than a constant
prediction. It can be undefined when targets have no variance. Entropy is a
useful diagnostic here, but the fresh-model configuration uses the library's
default zero entropy coefficient: logging entropy does not imply an explicit
entropy bonus is driving training.

The example is consistent with a functioning PPO update. A single table cannot
establish convergence, good exploration or an agent that beats baselines.

## What to watch over a run

- Compare rolling reward and episode length over many iterations. Expect noise.
- Inspect `env-*.monitor.csv` for actual survival, XP, damage and choice counts.
  Higher return with worse survival may reveal a reward tradeoff.
- Check FPS trends before increasing the number of workers. Browser simulation
  can bottleneck the run even when GPU utilization is low.
- Use `run.json` after completion to confirm actual transitions and timing.
- Judge performance with held-out evaluation against the standing, circular and
  random-valid baselines. Do not choose a model based solely on low training loss.

Longer episodes alone are not sufficient: standing still may survive easy early
sections. Longer training alone also does not guarantee better play. Compare
paired evaluation episodes and keep the task/horizon fixed when comparing runs.
