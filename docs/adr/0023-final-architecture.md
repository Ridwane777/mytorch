# ADR-0023: Final network architecture, 901 → 256 → 128 → 5 with the default initialisation

- **Status**: Accepted
- **Date**: 2026-10-27
- **Deciders**: Ridwane777
- **Issue**: #17

## Context

The final network uses `fen-attacks-901` (ADR-0016), the natural class distribution (ADR-0018),
augmentation (ADR-0019) and the dataset split (ADR-0017). Its hidden layers and its weight
initialisation (ADR-0020) were still open. A full training takes hours on one core, so the choice
was made on trial runs: 10 % of the data, scored on the validation split only, so that the test
split stays an unbiased measure of the final network (`docs/performance/09-trials.md`).

## Options considered

| Option | Validation macro F1 | Pros | Cons |
| --- | --- | --- | --- |
| 128, default initialisation | 0.812 (one seed) | Smallest and fastest | 1.7 points below the best |
| 256, default | 0.825 (one seed) | One hidden layer | 0.4 point below the two-layer networks |
| **256 → 128, default** | **0.829 (mean of 5 seeds, ±0.3)** | Within noise of the best; stable across seeds; 264,453 parameters; about 1.5 times faster to train than 512 → 256 | 0.3 point below the 512 → 256 mean, a gap the measurement cannot confirm |
| 512 → 256, default | 0.832 (mean of 5 seeds, ±0.6) | Best mean | Not significantly better (Welch's t ≈ 1.0); twice as variable; 594,437 parameters, a 2.2 times larger network file, longer training |
| Any of these with `he_uniform` | 0.5 to 1 point lower than the default, in all four architectures | — | Consistently worse here |

## Decision

The final network is 901 → 256 → 128 → 5 (ReLU, Softmax) with the default initialisation of
`Dense`. When two options are within noise of each other, the smaller and more stable one is kept.
`configs/final_network.conf` is updated accordingly.

## Consequences

- ✅ The choice rests on five seeds per finalist, not on one lucky run; the test split was never
  used.
- ✅ Shorter final training and a smaller network file than 512 → 256.
- ❌ Possibly 0.3 point of macro F1 left on the table, if the gap of the means is real.
- ❌ The trials used 10 % of the data; with ten times more, a larger network may gain more than it
  did here.
- Follow-up: if the final training shows that 256 → 128 is limited by its capacity (training
  accuracy close to validation accuracy), measure 512 → 256 on the full data and supersede this ADR
  if it wins.
