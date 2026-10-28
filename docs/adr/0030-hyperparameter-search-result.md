# ADR-0030: Candidate configuration from the hyperparameter search

- **Status**: Proposed
- **Date**: 2026-10-28
- **Deciders**: Ridwane777 (issue #10)

## Context
The final network (ADR-0023) was trained with hand-picked hyperparameters: Adam 0.001, batches of
64, hidden layers 256 → 128, no dropout. Issue #10 asked whether they are the best of a wider space.
The search of `docs/performance/15-hyperparameters.md` (module of ADR-0028) trained the 36
combinations of 3 learning rates × 3 batch sizes × 2 hidden-layer sizes × 2 dropout rates at 10 % of
the data, scored on the validation split, then trained the 3 best again with two more seeds.

Results: the current configuration ranks 21st of 36 (macro F1 0.829, one seed). Combination 15,
learning rate 0.001, batches of 32, hidden layers 512 → 256, dropout 0.2, is first (0.848) and keeps
the best mean over three seeds (0.846, spread 0.0045). These are scores at 10 % of the data; whether
the gap holds at 100 % is not measured.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| Keep the current configuration | Nothing to retrain; the pushed network is already measured on the test split | 21st of 36 at 10 %, 0.017 below the three-seed mean of combination 15 |
| **Retain combination 15 as the candidate, retrain it at 100 % and compare on the test split before replacing the network (chosen)** | The decision rests on the full data and an unbiased score, not on a 10 % ranking selected on validation | One more full training (hours); the final network does not change yet |
| Replace the final network with combination 15 now | Immediate | The 10 % ranking may not hold at 100 %; the best of 36 on validation is optimistic |
| Combination 19 (batches of 64) | About half the time per epoch of combination 15 | Mean 0.843 over three seeds with a wide spread (0.834 to 0.848) |

## Decision
Combination 15 (Adam 0.001, batches of 32, hidden layers 512 → 256, dropout 0.2 after each hidden
ReLU) is the candidate configuration. It replaces the final network only if, trained on 100 % of the
data, it beats the current network on the test split; that replacement will be a new ADR superseding
ADR-0023. Until then, the pushed network and ADR-0023 stay unchanged.

## Consequences
- ✅ The hand-picked configuration is now checked against 35 others: dropout 0.2 and the 512 → 256
  layers gain on average, the batch size barely matters, 0.001 is the best learning rate of the three.
- ✅ The seed check rules out a winner by luck: combination 21, second in the grid, falls to third.
- ❌ The gain is measured at 10 % of the data only; at 100 % it may shrink or vanish.
- ❌ Batches of 32 are about twice as slow per epoch as batches of 64 (139 s against 72 s on average
  over the grid at 10 %), so the full training takes longer.
- ❌ Selected on the validation split among 36 scores, so the 0.848 is optimistic.
- Follow-up: train combination 15 on 100 % of the data, evaluate both networks on the test split,
  then write the superseding ADR or abandon this one.
