# Full-game experiment report — 21 October 2026

Six single-seed runs trained for the same 6,000-second learning budget. None
reached the requested one million transitions, and no paired held-out evaluation
was submitted for these runs. The table below is therefore diagnostic: it uses
the last 100 completed *training* episodes, while the policy was still changing.

| Job | Setup | Steps | Survival, seconds (mean ± SE) | XP | Damage | Original-weight reference return |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1330453 | Legacy baseline | 319,892 | 47.3 ± 1.4 | 209.8 | 109.3 | 15.4 |
| 1330454 | Typed/dedicated pickup slots | 293,120 | 45.6 ± 0.9 | 200.7 | 106.1 | 14.1 |
| 1330455 | Full danger observation | 583,080 | 47.3 ± 1.3 | 217.7 | 105.7 | 16.6 |
| 1330456 | Survival-oriented reward | 579,988 | 48.7 ± 1.1 | 177.7 | 106.1 | 12.4 |
| 1330457 | Gamma 0.997 | 577,764 | **49.6 ± 1.4** | 212.3 | 114.9 | 15.8 |
| 1330458 | 512 × 512 network | 578,608 | 45.6 ± 1.2 | 196.2 | 110.9 | 12.3 |

No run reached the 120-second horizon in its final 100 training episodes. The
differences are small relative to episode variation and learning curves were not
monotonic, so this does not establish a winner. Training return cannot compare
the reward configurations because its scale and objective changed.

## Provisional conclusion

Gamma 0.997 is the most promising change. The survival reward improves survival
slightly but reduces XP, which suggests a more cautious policy rather than a
uniformly stronger one. Explicit danger features improve XP and the fixed
reference score, but not survival yet. Dedicated pickups alone show no measurable
benefit in this run. The larger network is worse and its final PPO update is much
more aggressive (`approx_kl` 0.071), so raw model size is not the current limit.

The evidence points to learning stability, credit assignment, and the flat entity
representation before model capacity. PPO itself has not been ruled out: these
runs are short, single-seed, still fluctuating, and have no held-out paired test.

## Comparison metric

Use held-out **horizon completion rate** as the primary metric. Until completion
is common, use paired mean/median survival time. Report XP per survived second,
damage per survived second, level and kills as secondary metrics. A fixed
reference return can summarize trade-offs, but it must use identical weights for
all models. Do not compare each configuration's native training return.

Evaluate identical 200,000-step checkpoints for all six observation/configuration
comparisons. Evaluate 500,000-step checkpoints only among jobs 1330455–1330458.
Use the same held-out seeds and at least 50 episodes per model. Choose from paired
per-seed survival differences, then repeat the shortlist with three training seeds.
