# Held-out evaluation report — 21 October 2026

All results use the same 50 held-out seeds (`1000000000`–`1000000049`), a
120-second horizon, deterministic policy actions and Squirtle. This makes the
results directly comparable. No evaluated model completed the full horizon.

## Results

| Setup | Checkpoint | Survival, mean ± SE | Median | XP | Damage | Reference return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy baseline | 200k | 41.5 ± 1.4 s | 40.7 s | 125.0 | 100.1 | 3.7 |
| Pickup observation | 200k | 42.2 ± 1.8 s | 41.5 s | **192.9** | 112.7 | 10.2 |
| Danger observation | 200k | 46.7 ± 1.7 s | 44.8 s | 186.7 | 109.9 | 11.2 |
| Survival reward | 200k | **51.5 ± 1.4 s** | **51.2 s** | 189.1 | 109.7 | **12.9** |
| Gamma 0.997 | 200k | 40.6 ± 1.5 s | 39.6 s | 135.4 | 103.9 | 4.1 |
| Larger network | 200k | 43.9 ± 1.5 s | 44.8 s | 151.1 | 102.5 | 7.1 |
| Danger observation | 500k | **50.3 ± 1.6 s** | **49.2 s** | **193.0** | 114.6 | **11.9** |
| Survival reward | 500k | 45.2 ± 1.5 s | 42.3 s | 144.1 | 100.0 | 7.3 |
| Gamma 0.997 | 500k | 40.3 ± 1.9 s | 40.7 s | 129.3 | 101.0 | 3.9 |
| Larger network | 500k | 44.4 ± 1.5 s | 42.9 s | 167.2 | 112.1 | 7.5 |

## Paired findings

- At 200k, the survival-reward model beats the baseline by **10.0 seconds** on
  average and wins on 41 of 50 identical seeds. Its approximate 95% confidence
  interval for the paired improvement is +6.5 to +13.4 seconds.
- Pickup features add only 0.7 seconds versus baseline, but add 67.9 XP. The
  model clearly uses the pickup information, primarily for progression.
- Danger features add 4.5 seconds over pickup-only observations. The uncertainty
  is large enough that this individual step is borderline, although its direction
  is consistent with the 500k result.
- Survival-oriented rewards add 4.8 seconds over danger features at 200k, while
  reducing damage per survived second from 2.42 to 2.16.
- Gamma 0.997 loses **10.9 seconds** versus the otherwise identical survival
  configuration and wins only 11 of 50 paired seeds. This is a clear regression.
- At 500k, danger observation is best. It beats survival reward by 5.1 seconds,
  gamma 0.997 by 9.9 seconds and the larger model by 5.9 seconds.

## Conclusion

The best evaluated checkpoint is job `1330456` at 200,000 steps. The best model
that continues improving through 500,000 steps is job `1330455`, the full danger
observation with the original reward and gamma 0.99.

These evaluations confirm that network capacity is not the main bottleneck: the
512 × 512 model is worse at both matched checkpoints. They also show policy
degradation with additional PPO training. The next priority should be stable
updates, periodic held-out evaluation and selection of the best checkpoint.
The evidence does not support gamma 0.997 for the current environment.

Training return should not select checkpoints, especially across reward schemes.
Until any model reaches 120 seconds, use paired held-out survival as the primary
metric, with XP/second and damage/second as secondary measures. Once horizon
completion becomes nonzero, make completion rate the primary metric.
