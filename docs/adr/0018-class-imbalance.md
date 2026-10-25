# ADR-0018: Class imbalance, four options compared on per-class F1 before choosing

- **Status**: Proposed
- **Date**: 2026-10-25
- **Deciders**: to be decided by the pair once the benchmark has run on the real dataset
- **Issue**: #16

## Context

The dataset is imbalanced. The counts reported in issue #16 (and measured in
[reading-a-fen.md](../reading-a-fen.md)) are 300,000 positions `Nothing`, 249,300 `Check` (both
colours) and 173,719 `Checkmate` (both colours), out of 723,019. A network trained on the natural
distribution can reach a good accuracy by favouring the majority class, which the subject's
per-class view (and [ADR-0011](0011-classification-metrics.md): precision, recall and F1 per class)
is meant to reveal. The imbalance is moderate (the largest of the three families is under twice
the smallest), but the five labels are finer than the three families, and the rarest label is
`Checkmate Black` (79,424).

The training procedure itself already supports one remedy: `loss cross_entropy weights=w1,…,w5`
scales the loss of each class (`CrossEntropy(labelSmoothing, classWeights)`, described in
[configuration.md](../configuration.md) and saved with the network file). The other two usual
remedies, oversampling and undersampling, change the training data, not the network. The decision
has to be taken on measurements, per class, not on overall accuracy. **The benchmark described
here has not been run on the real dataset yet** (`dataset/` is not available where this was
written); the script was only exercised on a synthetic dataset to check that it works. No result is
claimed below.

## Options considered

| Option | Pros | Cons |
| --- | --- | --- |
| **None** (natural distribution) | The reference every other option is compared with; the network sees the real frequencies, so its output probabilities stay calibrated to them | A rare class can be under-predicted; the overall accuracy hides it |
| **Loss weighting** (inverse class frequency, normalised to mean 1) | Already in the library, no data change, no duplicated or discarded position, kept in the network file | Weights computed from the training sample's counts; a rare class with a large weight makes its few examples dominate each batch (noisier gradients); probabilities no longer reflect real frequencies |
| **Oversampling** (minority positions duplicated, with replacement, up to the majority count) | Every class is seen as often as the majority; uses all the data | Training time grows (each epoch is longer); the network sees exact duplicates and can memorise them (overfitting); a validation split drawn *after* duplication would contain copies of training positions |
| **Undersampling** (each class drawn, without replacement, down to the minority count) | Smaller and faster training; no duplicates | Discards most of the majority class (information lost); with the real counts, the smallest class would set the size of the whole training set |

## Decision

**Proposed, not yet taken.** Run the four options on the same sample and the same network, and keep
the one that gives the best macro F1 without sacrificing the rarest classes (`Checkmate`
especially), the choice and the measured table to be written here by a new ADR or by moving this
one to Accepted once the real dataset is available. The method decisions already fixed:

- **No C++ change**: weighting uses the existing `weights=` key; resampling is done at the data
  level by `scripts/resample_imbalance.py`, which writes a resampled training file, so the
  analyzer and the library stay as they are.
- **The test set is drawn first, from the natural distribution, and never resampled**: it is
  the last `TEST` positions of one seeded sample, identical for the four options; only the training
  part is resampled. Resampling the test set would measure another problem.
- **Same network, same training for every option** (`configs/imbalance/*.conf`: one-hot 768 → 128
  → 5, ReLU, Softmax, cross-entropy, Adam 0.001, 20 epochs, batches of 64, L2 0.0003), the same
  three initial weights per option (`init seed`), the same shuffling seed. Only the loss line
  (weighted) or the training file (oversample, undersample) changes.
- **`validation=0 patience=0` in all four**: `my_torch_analyzer --train` holds out a fraction of
  the training file for early stopping; on an oversampled file that fraction would contain
  duplicates of training positions and give an optimistic validation loss. Early stopping is
  therefore off for every option, so that they are compared fairly. It costs the early-stopping
  safeguard of [ADR-0010](0010-overfitting-remedies.md) for this comparison.
- **Weights** = inverse frequency of the five labels in the training sample, normalised to mean 1,
  injected by the script into a copy of `configs/imbalance/weighted.conf` (the committed file
  holds a placeholder of ones), in the order of the labels of the configuration.
- **Resampling** uses Python's `random.Random(SEED)`: the same dataset and seed always give the
  same files.
- **Metrics**: F1 for each of the five labels, F1 for the three families (`Nothing`, `Check` = both
  colours, `Checkmate` = both colours), macro F1 and accuracy, on the test set, mean and spread
  over three initial weights.

### Results (to be measured)

| Option | F1 Nothing | F1 Check White | F1 Check Black | F1 Checkmate White | F1 Checkmate Black | Macro F1 | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| None | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured |
| Loss weighting | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured |
| Oversampling | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured |
| Undersampling | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured | to be measured |

Reproduce, with the dataset extracted at the root of the repository:

```sh
sh scripts/bench_imbalance.sh
```

It writes `docs/performance/data/imbalance_runs.csv` (one row per network),
`docs/performance/data/imbalance_f1.png` (F1 per class), `docs/performance/data/imbalance_family_f1.png`
(F1 per family) and the training and resampling logs in `docs/performance/data/imbalance_logs/`.
Method: [performance/imbalance.md](../performance/imbalance.md).

## Consequences

- ✅ The choice will rest on per-class F1 measured on an untouched, naturally distributed test set,
  not on accuracy; the four options share everything but the imbalance handling.
- ✅ No change to the library, the analyzer or the network file format: weighting reuses
  `weights=`, resampling lives in a script that is part of the reproducible training recipe.
- ❌ **Resampling cannot be configured from the network file**: only weighting is described by the
  configuration, so a network trained with oversampling or undersampling is reproduced by
  the script that resamples its training file, not by `my_torch_generator` and the network file
  alone. If a resampling option wins, the recipe has to be kept in `scripts/` (AGENTS.md: keep
  every training script).
- ❌ Early stopping is disabled in the comparison (`validation=0 patience=0`): the best option may
  behave differently when early stopping is turned back on, and 20 fixed epochs may over-train the
  oversampled file.
- ❌ One sample of 20,000 training positions (the default of the script, memory-bound as noted in
  [stalemate.md](../performance/stalemate.md)) and three initial weights: no confidence interval;
  the ranking of close options could change with another seed (`SEED=…`).
- ❌ The benchmark has not been run on the real dataset at the time of writing: until it is, this
  ADR documents a method, not a result.
- Follow-up: run `sh scripts/bench_imbalance.sh`, fill the table and the graphs, then set the
  status to Accepted (or supersede with a new ADR); revisit if the 385-wide encoding of the
  encoding benchmark is adopted (change `input` and the first dense layer of the four
  configurations), or if early stopping is wanted (resample after the validation split, not
  before).
