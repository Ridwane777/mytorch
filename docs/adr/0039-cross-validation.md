# ADR-0039: Cross-validation helper

- **Status**: Proposed
- **Date**: 2026-11-04
- **Deciders**: to be decided by the pair

## Context
Every number published so far comes from one split and one training ([20-cnn.md](../performance/20-cnn.md),
[10-final-network.md](../performance/10-final-network.md)): it says nothing about how much the result
would move with another split, so two models a point apart cannot be told apart from noise. Issue
#39 asks for stratified k-fold cross-validation, each metric reported as a mean and a standard
deviation, and benchmark results shown with an interval. [ADR-0027](0027-generic-data-splits.md)
already provides the folds (`stratifiedKFold`) and the views on their rows (`SubsetSource`); what
is missing is the loop around them and the summary of the metrics, which every script would
otherwise rewrite.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| A loop in each benchmark script | Nothing to add to the library | The loop, the sample standard deviation and the fold bookkeeping are copied in every script, and each copy can get them differently wrong |
| **`crossValidate(source, k, seed, trainer)` in `my_torch/Validation.hpp`, the training left to a callback (chosen)** | One tested implementation; works on any `DataSource` and any model, with its own loss, optimizer and early-stopping policy; the library does not need to know how a fold is trained | The caller writes the callback (about twenty lines) |
| `crossValidate` that trains a `Network` itself from a configuration | Shorter call | Fixes one way of training in the library (early stopping, augmentation, scoring) and does not fit a network that is not a `Network` |
| Folds trained in parallel threads | Up to k times faster | k copies of the network and of the batches in memory at once, on a machine with little free RAM; the matrix product is already threaded ([ADR-0024](0024-parallel-blocked-matmul.md)), so the cores are busy anyway |
| Repeated k-fold (several seeds) | A tighter estimate | r × k trainings; for the CNN the cost is out of reach on the project's machine |

## Decision
`my_torch/Validation.hpp` adds `summarize`, `MetricSummary`, `CrossValidation` and `crossValidate`.
`crossValidate` reads the class of every row (`targetClasses`), splits with `stratifiedKFold`,
builds two `SubsetSource` per fold and calls the trainer once per fold, in order, sequentially. The
trainer returns the `ClassificationReport` of the held-out part; accuracy, macro precision, macro
recall and macro F1 are summarized over the folds.

- The standard deviation is the **sample** one (n - 1 in the denominator): the k folds are a
  sample of the possible splits, not the whole population, and with k = 5 dividing by n would
  understate the spread by about 11 %. It is 0 for a single value.
- The benchmark ([21-cross-validation.md](../performance/21-cross-validation.md)) trains each fold
  for a **fixed number of epochs** (`patience = 0`) and **without early stopping on the held-out
  fold**: stopping on, or restoring the best epoch of, the data a fold is scored on would choose
  the model with the held-out labels and bias the estimate upwards. There is **no augmentation**
  inside the folds either: the symmetric images of a position would land in the training part
  while the position itself is held out, a leak ([ADR-0019](0019-symmetry-augmentation.md)).
- The benchmark reads only the training part of the project's split ([ADR-0017](0017-dataset-split.md)),
  never the test split.

## Consequences
- ✅ Any experiment can report a mean ± standard deviation with a few lines, and the test covers
  the helper once (trainer called k times in order, complementary parts, every row held out
  once, summary equal to `summarize` of the per-fold values).
- ✅ The library stays independent of the way a model is trained.
- ❌ k trainings cost k times the time of one: the CNN benchmark is therefore run on a small
  fraction of the data and for a few epochs, and its numbers are lower than those of a full
  training (stated in the document).
- ❌ With k = 5 the interval is approximate: the five training sets overlap by 75 %, so the fold
  scores are not independent and the standard deviation underestimates the spread of a fresh
  split; the 95 % interval from a Student t distribution with 4 degrees of freedom is wide and
  still optimistic.
- ❌ The folds are trained one after the other, so the wall time is the sum of the folds.
- Follow-up: revisit (repeated k-fold, or folds in parallel) if a machine with enough memory is
  available for the full-data comparison of the dense network and the CNN.
